import time
import random
import math

DEADLINE = 115  # safety margin before 120s hard kill

def _compute_score_fast(n, adj_masks, degs):
    """
    Compute the largest degree sum over triangles using bitmask intersection.
    adj_masks: list of integers, adj_masks[i] has bit j set if i-j is an edge.
    degs: list of degrees.
    """
    max_sum = 0
    for u in range(n):
        adj_u = adj_masks[u]
        for v in range(u + 1, n):
            if (adj_u >> v) & 1:
                # Common neighbors of u and v
                common = adj_u & adj_masks[v]
                # Iterate through set bits in common
                m = common
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    s = degs[u] + degs[v] + degs[w]
                    if s > max_sum:
                        max_sum = s
                    m ^= lsb
    return max_sum

def _graph_to_state(edges):
    """Convert edge list to adj_masks and degs."""
    n = 16
    adj_masks = [0] * n
    degs = [0] * n
    for u, v in edges:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        degs[u] += 1
        degs[v] += 1
    return adj_masks, degs

def _state_to_edges(adj_masks):
    """Convert adj_masks back to edge list."""
    n = 16
    edges = []
    for i in range(n):
        mask = adj_masks[i]
        for j in range(i + 1, n):
            if (mask >> j) & 1:
                edges.append([i, j])
    return edges

def construct():
    start_time = time.time()
    n = 16
    m_target = 65
    
    # Precompute all potential pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_score = float('inf')
    best_adj_masks = None
    best_degs = None
    
    # Helper to update best
    def update_best(adj_masks, degs, score):
        nonlocal best_score, best_adj_masks, best_degs
        if score < best_score:
            best_score = score
            best_adj_masks = adj_masks[:]
            best_degs = degs[:]
    
    # Simulated Annealing parameters
    T_start = 1.0
    T_end = 0.01
    cooling_rate = 0.9995
    
    # Initial construction: K_{8,8} + one internal edge
    # This is a good starting point as it's close to optimal structure
    init_edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    
    # We will run multiple SA restarts
    # Each restart starts from the best solution found so far (or initial if none)
    # and perturbs it slightly
    
    num_restarts = 50 # Will be limited by time
    
    for r in range(num_restarts):
        if time.time() - start_time > DEADLINE:
            break
        
        # Initialize state
        if r == 0:
            adj_masks, degs = _graph_to_state(init_edges)
        else:
            # Start from best found
            adj_masks = best_adj_masks[:]
            degs = best_degs[:]
            # Perturb: swap a few random edges
            edge_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in _state_to_edges(adj_masks))
            current_edges_list = list(edge_set)
            for _ in range(5):
                if not current_edges_list:
                    break
                rem_u, rem_v = random.choice(current_edges_list)
                # Pick random non-edge
                while True:
                    add_u, add_v = random.choice(all_pairs)
                    if (min(add_u, add_v), max(add_u, add_v)) not in edge_set:
                        break
                # Apply swap
                adj_masks[rem_u] &= ~(1 << rem_v)
                adj_masks[rem_v] &= ~(1 << rem_u)
                degs[rem_u] -= 1
                degs[rem_v] -= 1
                edge_set.remove((min(rem_u, rem_v), max(rem_u, rem_v)))
                
                adj_masks[add_u] |= (1 << add_v)
                adj_masks[add_v] |= (1 << add_u)
                degs[add_u] += 1
                degs[add_v] += 1
                edge_set.add((min(add_u, add_v), max(add_u, add_v)))
                current_edges_list = list(edge_set)

        current_score = _compute_score_fast(n, adj_masks, degs)
        update_best(adj_masks, degs, current_score)
        
        T = T_start
        steps = 0
        max_steps_per_restart = 100000 # Limit per restart
        
        # Maintain list of edges and non-edges for fast random selection
        edge_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in _state_to_edges(adj_masks))
        current_edges_list = list(edge_set)
        
        while steps < max_steps_per_restart and time.time() - start_time < DEADLINE:
            # Heuristic move selection:
            # 1. Pick an edge to remove. Bias towards edges incident to high-degree vertices.
            #    We can pick a random vertex with probability proportional to its degree,
            #    then pick a random neighbor.
            #    Simpler: Just pick random edge for now, but we can weight it.
            #    Let's try: Pick u with prob ~ deg(u), then pick v from neighbors of u.
            
            # Weighted vertex selection
            total_deg = 2 * m_target
            r_val = random.random() * total_deg
            cum = 0
            u = 0
            for i in range(n):
                cum += degs[i]
                if r_val <= cum:
                    u = i
                    break
            
            # Pick random neighbor v of u
            if degs[u] == 0:
                continue
            # Get neighbors of u
            mask = adj_masks[u]
            # Pick a random set bit
            # Efficient way: pick random int, find bit
            # Or just iterate? degs[u] is small.
            # Let's use a list of neighbors for speed? 
            # Rebuilding neighbor lists is expensive.
            # Bitmask trick:
            # We want a random v such that (mask >> v) & 1 is 1.
            # Just pick random v and check?
            # Or:
            # m = mask
            # while m:
            #    lsb = m & -m
            #    ...
            # This iterates all neighbors.
            
            # Let's just pick random pair for simplicity first, then optimize if needed.
            # Given n=16, picking random pair and checking is O(1).
            # But we want bias.
            # Let's stick to random edge removal for now, but maybe remove one of the edges
            # from the worst triangle?
            
            # Let's find the worst triangle to guide the move.
            # This is expensive if done every step.
            # Let's do it every 100 steps or if stuck.
            
            # For now, standard random swap with SA.
            rem_u, rem_v = random.choice(current_edges_list)
            
            # Pick random non-edge
            # Number of non-edges = 120 - 65 = 55
            # Pick random pair, check if edge.
            attempts = 0
            while True:
                add_u, add_v = random.choice(all_pairs)
                if (min(add_u, add_v), max(add_u, add_v)) not in edge_set:
                    break
                attempts += 1
                if attempts > 10:
                    # Fallback: iterate all pairs to find non-edge
                    for p in all_pairs:
                        if p not in edge_set:
                            add_u, add_v = p
                            break
                    else:
                        break # No non-edges? Should not happen.
                    break
            
            # Perform swap
            # Remove rem
            adj_masks[rem_u] &= ~(1 << rem_v)
            adj_masks[rem_v] &= ~(1 << rem_u)
            degs[rem_u] -= 1
            degs[rem_v] -= 1
            edge_set.remove((min(rem_u, rem_v), max(rem_u, rem_v)))
            
            # Add add
            adj_masks[add_u] |= (1 << add_v)
            adj_masks[add_v] |= (1 << add_u)
            degs[add_u] += 1
            degs[add_v] += 1
            edge_set.add((min(add_u, add_v), max(add_u, add_v)))
            
            new_score = _compute_score_fast(n, adj_masks, degs)
            
            # Acceptance criterion
            delta = new_score - current_score
            if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                current_score = new_score
                # Update current_edges_list
                current_edges_list = list(edge_set)
                # Update best
                if current_score < best_score:
                    best_score = current_score
                    best_adj_masks = adj_masks[:]
                    best_degs = degs[:]
            else:
                # Revert
                # Remove add
                adj_masks[add_u] &= ~(1 << add_v)
                adj_masks[add_v] &= ~(1 << add_u)
                degs[add_u] -= 1
                degs[add_v] -= 1
                edge_set.remove((min(add_u, add_v), max(add_u, add_v)))
                
                # Add rem
                adj_masks[rem_u] |= (1 << rem_v)
                adj_masks[rem_v] |= (1 << rem_u)
                degs[rem_u] += 1
                degs[rem_v] += 1
                edge_set.add((min(rem_u, rem_v), max(rem_u, rem_v)))
            
            steps += 1
            T *= cooling_rate
            
            # If T is too low, restart
            if T < T_end:
                break

    # Return the best edges found
    if best_adj_masks is None:
        # Fallback to initial
        return {'n': n, 'edges': init_edges}
    
    best_edges = _state_to_edges(best_adj_masks)
    return {'n': n, 'edges': best_edges}
