import numpy as np
import time

# Global deadline guard
_DEADLINE = time.time() + 145  # Safety margin for 150s limit

def _estimate_remaining_cost(current, unvisited, destination, dist_matrix):
    """
    Estimate the cost to complete the tour from 'current' visiting all 'unvisited' and returning to 'destination'
    using a Nearest Neighbor heuristic. This provides a tighter 'upper bound' estimate than a simple lower bound.
    """
    if len(unvisited) == 0:
        return dist_matrix[current, destination]
    
    remaining = list(unvisited)
    total_cost = 0.0
    curr = current
    
    while remaining:
        # Find nearest node in remaining
        # Since N is small (max ~50), a simple loop or np.argmin is fine
        dists = dist_matrix[curr, remaining]
        min_idx = np.argmin(dists)
        next_node = remaining[min_idx]
        total_cost += dists[min_idx]
        curr = next_node
        remaining.pop(min_idx)
        
    total_cost += dist_matrix[curr, destination]
    return total_cost

def _held_karp_next(current, unvisited, destination, dist_matrix):
    """
    Exact solution for the next node when the number of unvisited nodes is small.
    Uses Held-Karp DP to find the shortest path from 'current' visiting all 'unvisited' and ending at 'destination'.
    Returns the first node in that optimal path.
    """
    n = len(unvisited)
    if n == 0:
        return -1
    if n == 1:
        return int(unvisited[0])
    
    # Map unvisited nodes to 0..n-1
    # dp[mask][i] = min cost to start at current, visit set 'mask', and end at node 'i' (index in unvisited)
    # We want to find the best first step.
    
    # Optimization: We only need the first node.
    # Let's compute the cost of each possible first node 'v' by solving the subproblem:
    # Start at 'current', go to 'v', visit all others, end at 'destination'.
    # This is equivalent to finding the min over all permutations of unvisited\{v} of:
    # dist(current, v) + dist(v, p1) + ... + dist(pk, destination)
    
    # Standard Held-Karp finds the min cost for a fixed end node.
    # We can compute the full DP table for the set of unvisited nodes.
    # State: (mask, last_node_index)
    # dp[mask][i] = min cost to visit set 'mask' (subset of unvisited) ending at unvisited[i]
    
    size = 1 << n
    dp = [[float('inf')] * n for _ in range(size)]
    
    # Base case: visit only node i
    for i in range(n):
        dp[1 << i][i] = dist_matrix[current, unvisited[i]]
        
    # Fill DP
    for mask in range(size):
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            if dp[mask][i] == float('inf'):
                continue
            for j in range(n):
                if mask & (1 << j):
                    continue
                new_mask = mask | (1 << j)
                cost = dp[mask][i] + dist_matrix[unvisited[i], unvisited[j]]
                if cost < dp[new_mask][j]:
                    dp[new_mask][j] = cost
                    
    # Find the best full path
    full_mask = size - 1
    best_total = float('inf')
    best_first_node = -1
    
    # We need to know which node was the *first* one.
    # We can backtrace, or we can just evaluate the cost for each candidate first node 'k'.
    # Cost if first node is k:
    # dist(current, k) + (min cost to visit all others starting from k and ending at destination)
    # The "min cost to visit all others starting from k" is not directly in dp[full_mask][last].
    # dp[full_mask][last] is cost from current to ... to last, visiting all.
    # So, if we fix the first node k, we are looking for:
    # dist(current, k) + min_{perm of rest} (dist(k, p1) + ... + dist(pk, destination))
    
    # Let's just compute the total cost for each possible first node k by running a mini-Held-Karp or using the dp table.
    # Actually, we can just iterate over all possible 'last' nodes in the full_mask state.
    # But we need to identify the first node.
    
    # Alternative: Just compute the min cost to go from current, visit all, end at destination.
    # min_last dp[full_mask][last] + dist_matrix[unvisited[last], destination]
    # This gives the optimal total cost. But which node is first?
    
    # Let's store parent pointers to reconstruct the path.
    parent = [[-1] * n for _ in range(size)]
    
    # Re-run DP with parent tracking
    dp = [[float('inf')] * n for _ in range(size)]
    for i in range(n):
        dp[1 << i][i] = dist_matrix[current, unvisited[i]]
        
    for mask in range(size):
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            if dp[mask][i] == float('inf'):
                continue
            for j in range(n):
                if mask & (1 << j):
                    continue
                new_mask = mask | (1 << j)
                cost = dp[mask][i] + dist_matrix[unvisited[i], unvisited[j]]
                if cost < dp[new_mask][j]:
                    dp[new_mask][j] = cost
                    parent[new_mask][j] = i
                    
    # Find best end node
    best_end = -1
    best_cost = float('inf')
    for i in range(n):
        total = dp[full_mask][i] + dist_matrix[unvisited[i], destination]
        if total < best_cost:
            best_cost = total
            best_end = i
            
    # Backtrace to find the first node
    # Start from (full_mask, best_end)
    cur_mask = full_mask
    cur_node = best_end
    # We need to go back to the state with mask having only 1 bit set.
    while (cur_mask & (cur_mask - 1)) != 0: # while mask has more than 1 bit
        prev_node = parent[cur_mask][cur_node]
        cur_mask = cur_mask & ~(1 << cur_node)
        cur_node = prev_node
        
    # cur_node is the index of the first node in unvisited
    return int(unvisited[cur_node])

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _DEADLINE
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Time check
    if time.time() > _DEADLINE - 1.0:
        dists = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists)
        return int(unvisited_nodes[best_idx])
    
    # Use exact Held-Karp for small subproblems
    if n_unvisited <= 10:
        return _held_karp_next(current_node, unvisited_nodes, destination_node, distance_matrix)
    
    if n_unvisited == 2:
        a = int(unvisited_nodes[0])
        b = int(unvisited_nodes[1])
        cost_a = distance_matrix[current_node, a] + distance_matrix[a, b] + distance_matrix[b, destination_node]
        cost_b = distance_matrix[current_node, b] + distance_matrix[b, a] + distance_matrix[a, destination_node]
        return a if cost_a < cost_b else b

    # For n_unvisited >= 3, use a hybrid approach:
    # 1. Use a fast heuristic to identify a subset of candidate nodes to consider
    # 2. Apply exact DP on that subset
    
    # Strategy: Use a 2-step lookahead to score all nodes, then take top-K candidates
    # and apply exact DP on those K nodes plus the remaining structure
    
    # First, compute a quick score for each node using 2-step lookahead
    dist_v_to_unvisited = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    dist_v_to_dest = distance_matrix[unvisited_nodes, destination_node]
    dist_current_to_v = distance_matrix[current_node, unvisited_nodes]
    
    # 2-step lookahead: cost to go from v to dest via one intermediate node
    M_temp = dist_v_to_unvisited + dist_v_to_dest[None, :]
    np.fill_diagonal(M_temp, np.inf)
    best_cost_2step = np.min(M_temp, axis=1)
    
    # Quick score for each node
    quick_score = dist_current_to_v + best_cost_2step
    
    # Determine how many nodes to include in exact DP
    # Held-Karp is O(2^k * k^2), so for k=15 that's ~32768*225 ≈ 7.4M ops
    # We need to be careful with time
    
    # Use exact DP for up to 12 nodes (2^12 * 144 ≈ 590K ops)
    # For more nodes, use a reduced set
    
    # Sort by quick score and take top candidates
    sorted_indices = np.argsort(quick_score)
    
    # Decide K based on time budget and n_unvisited
    if n_unvisited <= 12:
        K = n_unvisited
    elif n_unvisited <= 20:
        K = 12
    else:
        # For larger n, use a smarter selection
        # Take top 8 by quick score, but ensure we don't exclude nodes that are
        # close to current or destination
        K = 10
    
    # Select K best candidates
    selected_local_indices = sorted_indices[:K]
    selected_nodes = unvisited_nodes[selected_local_indices]
    
    # Build sub-problem for exact DP
    # The sub-problem: we must visit all K selected nodes and then go to destination
    # But wait - this isn't quite right. We need to visit ALL unvisited nodes eventually.
    # The DP should model the optimal order to visit the selected nodes,
    # but we also need to account for the non-selected nodes.
    
    # Actually, a better approach: use the DP to find the optimal next node
    # by considering a lookahead window of the next few nodes.
    
    # Let's use a different approach: compute the exact optimal suffix for
    # a small window, and use heuristic for the rest.
    
    # Simpler approach: Use the quick score but refine it with a 3-step lookahead
    # for the top candidates
    
    # 3-step lookahead for top K candidates
    K_refine = min(K, n_unvisited)
    refine_indices = sorted_indices[:K_refine]
    refine_nodes = unvisited_nodes[refine_indices]
    
    if K_refine <= 1:
        return int(unvisited_nodes[refine_indices[0]])
    
    if K_refine == 2:
        a = int(refine_nodes[0])
        b = int(refine_nodes[1])
        cost_a = distance_matrix[current_node, a] + distance_matrix[a, b] + distance_matrix[b, destination_node]
        cost_b = distance_matrix[current_node, b] + distance_matrix[b, a] + distance_matrix[a, destination_node]
        return a if cost_a < cost_b else b
    
    # For K_refine >= 3, use DP
    k = K_refine
    # Map local indices to global
    # DP state: (mask, last_node) -> min cost
    # mask is a bitmask of which of the k nodes have been visited
    # last_node is the index of the last visited node (0 to k-1)
    
    # Initialize DP
    INF = float('inf')
    dp = np.full((1 << k, k), INF, dtype=np.float64)
    
    # Base case: visit one node
    for i in range(k):
        dp[1 << i, i] = distance_matrix[current_node, refine_nodes[i]]
    
    # Fill DP
    for mask in range(1, 1 << k):
        for last in range(k):
            if dp[mask, last] == INF:
                continue
            if not (mask & (1 << last)):
                continue
            # Try adding each unvisited node
            for nxt in range(k):
                if mask & (1 << nxt):
                    continue
                new_mask = mask | (1 << nxt)
                cost = dp[mask, last] + distance_matrix[refine_nodes[last], refine_nodes[nxt]]
                if cost < dp[new_mask, nxt]:
                    dp[new_mask, nxt] = cost
    
    # Find the best final state
    # After visiting all k nodes, we go to destination
    full_mask = (1 << k) - 1
    best_total = INF
    best_last = -1
    for last in range(k):
        total = dp[full_mask, last] + distance_matrix[refine_nodes[last], destination_node]
        if total < best_total:
            best_total = total
            best_last = last
    
    # Now we need to find the FIRST node in the optimal path
    # Reconstruct path from DP
    # We need to find which node was visited first in the optimal path
    
    # Backtrack
    path = []
    mask = full_mask
    last = best_last
    while mask != 0:
        path.append(last)
        prev_mask = mask & ~(1 << last)
        if prev_mask == 0:
            break
        # Find the previous node
        best_prev_cost = INF
        best_prev = -1
        for prev in range(k):
            if not (prev_mask & (1 << prev)):
                continue
            cost = dp[prev_mask, prev] + distance_matrix[refine_nodes[prev], refine_nodes[last]]
            if cost < best_prev_cost - 1e-9:
                best_prev_cost = cost
                best_prev = prev
        if best_prev == -1:
            break
        last = best_prev
        mask = prev_mask
    
    path.reverse()
    if len(path) > 0:
        first_local_idx = path[0]
        return int(refine_nodes[first_local_idx])
    else:
        # Fallback
        return int(unvisited_nodes[refine_indices[0]])
