"""Structural change: Beam Search over partial tours with exact DP tail."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0  # seconds (10s safety margin before 150s kill)

def _held_karp_tail(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Computes the exact minimum cost to visit all unvisited nodes and return to destination.
    Returns (min_cost, first_node_index_in_unvisited).
    Uses Held-Karp DP.
    """
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    
    if n == 0:
        return 0.0, -1
    if n == 1:
        return distance_matrix[current_node, unv[0]] + distance_matrix[unv[0], destination_node], 0

    # Map unvisited nodes to local indices 0..n-1
    full_mask = (1 << n) - 1
    INF = float('inf')
    
    # dp[mask][i] = min cost to start at current_node, visit nodes in mask, end at unv[i]
    # We only need the final cost, so we can optimize memory if n is small, but for n<=15 standard is fine.
    # For speed, we use a dictionary or numpy array. Numpy array is faster for dense access.
    dp = np.full((1 << n, n), INF, dtype=float)
    
    # Base cases: start at current_node, go to unv[i]
    for i in range(n):
        dp[1 << i, i] = distance_matrix[current_node, unv[i]]
        
    # Fill DP table
    # Iterate over all masks
    for mask in range(1 << n):
        # If mask is 0, skip
        if mask == 0:
            continue
        # Optimization: if we've exceeded time, break early? No, we need the exact tail.
        # But we should check if this specific DP call is taking too long.
        # For n=15, 32768 * 15 * 15 ~ 7.3M ops. In Python this is slow (~0.5s).
        # We will only call this for n <= 12 to be safe, or if time permits.
        
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            current_cost = dp[mask, i]
            if current_cost == INF:
                continue
            
            # Try extending to j not in mask
            for j in range(n):
                if mask & (1 << j):
                    continue
                
                new_mask = mask | (1 << j)
                step_cost = distance_matrix[unv[i], unv[j]]
                new_cost = current_cost + step_cost
                
                if new_cost < dp[new_mask, j]:
                    dp[new_mask, j] = new_cost

    # Find the best ending node i in full_mask such that dp[full_mask][i] + dist(unv[i], dest) is min
    best_cost = INF
    best_last_i = -1
    for i in range(n):
        total_cost = dp[full_mask, i] + distance_matrix[unv[i], destination_node]
        if total_cost < best_cost:
            best_cost = total_cost
            best_last_i = i
            
    if best_last_i == -1:
        # Fallback
        return distance_matrix[current_node, unv[0]] + distance_matrix[unv[0], destination_node], 0
        
    # Backtrack to find the first node
    current_mask = full_mask
    current_i = best_last_i
    first_i = -1
    
    while current_mask != 0:
        if current_mask & (current_mask - 1) == 0:
            # Only one bit set, this is the first node
            first_i = current_i
            break
        
        # Find prev_i
        # We didn't store parent pointers in this version for speed/memory, 
        # so we have to backtrack by checking all possible predecessors.
        # This is O(n) per step, total O(n^2).
        prev_i = -1
        for k in range(n):
            if k == current_i:
                continue
            if not (current_mask & (1 << k)):
                continue
            # Check if dp[current_mask ^ (1<<current_i), k] + dist(unv[k], unv[current_i]) == dp[current_mask, current_i]
            prev_mask = current_mask ^ (1 << current_i)
            if dp[prev_mask, k] + distance_matrix[unv[k], unv[current_i]] == dp[current_mask, current_i]:
                prev_i = k
                break
        
        if prev_i == -1:
            # Error in backtracking, fallback
            break
            
        current_mask = prev_mask
        current_i = prev_i
        
    if first_i == -1:
        first_i = 0
        
    return best_cost, first_i


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE:
        # Fallback to nearest neighbor if we're running out of time
        if n == 0:
            return int(destination_node)
        return int(unv[np.argmin(distance_matrix[current_node, unv])])
    
    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(unv[0])
    
    # Strategy: Beam Search
    # We maintain a set of partial tours. Each state is (last_node, remaining_nodes_mask, cost_so_far).
    # However, the interface is `select_next_node`, which implies we are in the middle of a construction.
    # The caller passes `current_node` and `unvisited_nodes`.
    # This means the "history" is fixed. We are only choosing the NEXT node.
    # Wait, the prompt says "picks the next city while a traveling-salesman tour is built step by step".
    # This implies a greedy or heuristic step-by-step process where the state is just (current, unvisited).
    # A true Beam Search would require the caller to maintain multiple states. 
    # But the interface is fixed: `select_next_node(current_node, ...) -> int`.
    # This forces a "myopic" decision: given the current state, which single node to pick next?
    # We cannot maintain multiple beams across calls because the caller only keeps one path.
    # 
    # RE-READING TASK: "Write a Python select_next_node... that picks the next city... scored by mean closed-tour length".
    # "Keep the function names and signatures".
    # If the evaluator uses a standard greedy loop:
    #   tour = [0]
    #   unvisited = [1..N-1]
    #   while unvisited:
    #       next_node = select_next_node(current, dest, unvisited, dm)
    #       tour.append(next_node)
    #       unvisited.remove(next_node)
    #   tour.append(0)
    # 
    # Then we CANNOT do a global beam search because the state is collapsed to a single path.
    # The "Beam Search" strategy in the directive might be interpreted as:
    # "Simulate a beam search locally to determine the best next step?"
    # Or perhaps the directive implies a different interpretation?
    # "Replace the sequential greedy selection with a dynamic programming search (Held-Karp) over the last 15 unvisited nodes... while using a fast insertion heuristic for the initial nodes"
    # 
    # Actually, the directive says: "Replace the sequential greedy selection with a dynamic programming search (Held-Karp) over the last 15 unvisited nodes to compute the exact optimal suffix, while using a fast insertion heuristic for the initial nodes to ensure the exact search remains within the time limit."
    # 
    # This is exactly what the CURRENT program does (Held-Karp for n<=13).
    # The "INSPIRATION 1" (score 1.0334) suggests a "structural change" away from blended scores.
    # The "INSPIRATION 3" (score 0.9982) uses the same structure but scored worse? No, 0.9982 is worse than 1.0102?
    # Wait, score is seed/value. Higher is better. 1.0334 > 1.0102.
    # So Inspiration 1 is better.
    # 
    # The directive for THIS proposal is: "Replace the sequential greedy selection with a dynamic programming search (Held-Karp) over the last 15 unvisited nodes...".
    # This sounds like the current program.
    # BUT, the "Strategy to try in this change" is explicitly: "Replace the sequential greedy selection with a dynamic programming search (Held-Karp) over the last 15 unvisited nodes to compute the exact optimal suffix, while using a fast insertion heuristic for the initial nodes..."
    # 
    # If the current program already does this, why is it stagnating?
    # Maybe the current program's DP is too slow or not used often enough?
    # Current program: `if elapsed < _DEADLINE - 5.0 and n <= 13: return _held_karp_next(...)`
    # 
    # Let's look at the "Structural Change" requirement.
    # "Make a STRUCTURAL change: a different design or modelling family...".
    # 
    # Alternative Interpretation of "Beam Search":
    # Since we can't maintain state across calls, we can perform a **local beam search** or **lookahead search** within the single call.
    # Instead of just picking the best next node greedily, we can explore a few steps ahead (e.g., 3-5 steps) using a beam of width W, and then evaluate the resulting partial tours using a fast heuristic (or DP if small) to score the initial choice.
    # This is a "Lookahead Beam Search" within the `select_next_node` function.
    # 
    # Strategy:
    # 1. Generate top-K candidates for the next node based on a quick heuristic (e.g., nearest neighbor + destination).
    # 2. For each candidate, simulate a short beam search (width B, depth D) to see where it leads.
    # 3. Score the resulting states using a fast estimate (e.g., 2-step lookahead or small DP).
    # 4. Return the candidate that leads to the best estimated final tour.
    # 
    # This is materially different from the current "Blended Score" (which is a single-step weighted sum) and "Exact DP" (which is exact but limited to n<=13).
    # This is a "Lookahead Beam Search".
    
    # Parameters
    BEAM_WIDTH = 4
    LOOKAHEAD_DEPTH = 3
    
    # Step 1: Get top candidates for immediate next step
    # Use a simple heuristic to filter candidates
    d_cur = distance_matrix[current_node, unv]
    d_dest = distance_matrix[unv, destination_node]
    
    # Quick score: dist_to_current + alpha * dist_to_dest
    # We want to pick nodes that are close to current and not too far from dest?
    # Actually, for TSP, we want to avoid "crossing" the tour.
    # Let's just take the top K by distance to current.
    # Or top K by (dist_to_current + dist_to_dest) / 2?
    # Let's use a simple nearest neighbor ranking.
    
    # To be safe, take top 10 candidates by distance to current.
    # If n < 10, take all.
    k_cand = min(10, n)
    top_candidates = np.argsort(d_cur)[:k_cand]
    
    # If n is small enough, we can use exact DP on the whole set?
    # The current program does this for n<=13.
    # If n <= 13, let's stick to exact DP as it's optimal.
    if n <= 13:
        elapsed = time.time() - _START_TIME
        if elapsed < _DEADLINE - 5.0:
            return _held_karp_tail(current_node, destination_node, unv, distance_matrix)[1]
    
    # For n > 13, use Lookahead Beam Search
    
    best_candidate = -1
    best_score = float('inf')
    
    # For each candidate, run a small beam search
    for c_idx in top_candidates:
        next_node = int(unv[c_idx])
        
        # State: (last_node, remaining_set, cost_so_far)
        # remaining_set is a set of indices into unv, excluding c_idx
        remaining_indices = [i for i in range(n) if i != c_idx]
        remaining_set = set(remaining_indices)
        
        # Beam: list of (last_node_global, remaining_set, cost)
        # last_node_global is the actual node ID
        beam = [(next_node, remaining_set, d_cur[c_idx])]
        
        # Expand beam for LOOKAHEAD_DEPTH steps
        for step in range(LOOKAHEAD_DEPTH):
            if not beam:
                break
                
            next_beam = []
            for last_node, rem_set, cost in beam:
                if not rem_set:
                    # No more nodes to visit in lookahead
                    next_beam.append((last_node, rem_set, cost))
                    continue
                
                rem_list = list(rem_set)
                # Get distances from last_node to all remaining
                d_next = distance_matrix[last_node, unv[rem_list]]
                
                # Get top B candidates for this state
                b_width = min(BEAM_WIDTH, len(rem_list))
                top_next_indices = np.argsort(d_next)[:b_width]
                
                for t_idx in top_next_indices:
                    next_node_local = rem_list[t_idx]
                    next_node_global = int(unv[next_node_local])
                    new_cost = cost + d_next[t_idx]
                    new_rem_set = rem_set - {next_node_local}
                    next_beam.append((next_node_global, new_rem_set, new_cost))
            
            # Prune next_beam to width BEAM_WIDTH * 2 (to allow some diversity)
            if len(next_beam) > BEAM_WIDTH * 2:
                # Sort by cost and keep top
                next_beam.sort(key=lambda x: x[2])
                next_beam = next_beam[:BEAM_WIDTH * 2]
                
            beam = next_beam
            
            # Check time
            if time.time() - _START_TIME > _DEADLINE - 2.0:
                # Fallback
                return int(next_node)

        # Score the final beam states
        # For each state in the final beam, estimate the remaining cost.
        # If remaining_set is small, use DP. Else, use heuristic.
        state_score = float('inf')
        for last_node, rem_set, cost in beam:
            rem_list = list(rem_set)
            if not rem_list:
                # All visited in lookahead
                # Cost to return to destination
                total_cost = cost + distance_matrix[last_node, destination_node]
                if total_cost < state_score:
                    state_score = total_cost
            else:
                # Estimate remaining cost
                # Simple heuristic: sum of nearest neighbor + return
                # Or better: 1-step lookahead min
                rem_arr = np.array(rem_list)
                d_rem = distance_matrix[last_node, unv[rem_arr]]
                min_next = d_rem.min()
                # Cost to visit all remaining approximated by min_next + ...
                # Let's use a simple bound: min_next + (len(rem_list)-1)*min_edge + dist_to_dest
                # Actually, just use the min distance to destination from any remaining node as a lower bound?
                # No, we need an estimate.
                # Heuristic: cost + min_next + distance_matrix[unv[rem_arr[np.argmin(d_rem)]], destination_node]
                # This is a 1-step lookahead estimate.
                est_cost = cost + min_next + distance_matrix[unv[rem_arr[np.argmin(d_rem)]], destination_node]
                if est_cost < state_score:
                    state_score = est_cost
        
        if state_score < best_score:
            best_score = state_score
            best_candidate = next_node
            
    if best_candidate != -1:
        return best_candidate
    else:
        # Fallback
        return int(unv[np.argmin(d_cur)])
# EVOLVE-BLOCK-END
