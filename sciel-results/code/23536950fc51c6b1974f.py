"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _calc_distinct_degrees(subset, adj):
    """Calculate distinct degrees for a subset."""
    s = set(subset)
    degs = set()
    for v in s:
        d = 0
        for u in adj[v]:
            if u in s:
                d += 1
        degs.add(d)
    return len(degs)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin within 90s limit
    adj = _build_adj()
    n = P['n']
    
    best_score = -1
    best_subset = list(range(8))
    
    # 1. Brute force for size 8
    for subset in itertools.combinations(range(n), 8):
        score = _calc_distinct_degrees(subset, adj)
        if score > best_score:
            best_score = score
            best_subset = list(subset)
    
    # 2. Local search for sizes 9-17
    rng = random.Random(42)
    
    # Precompute adjacency lists for faster iteration
    adj_lists = [list(adj[i]) for i in range(n)]
    
    for target_size in range(9, 18):
        if time.time() > deadline:
            break
        
        # Multiple restarts
        # We use a mix of random starts and perturbed best solutions
        num_restarts = 500 if target_size <= 12 else 200
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            if restart == 0 and len(best_subset) == target_size:
                current = list(best_subset)
            elif restart == 1 and len(best_subset) >= target_size:
                # Perturb best solution
                current = list(rng.sample(best_subset, target_size))
            else:
                current = rng.sample(range(n), target_size)
            
            current_set = set(current)
            current_score = _calc_distinct_degrees(current, adj)
            
            improved = True
            while improved and time.time() < deadline:
                improved = False
                
                in_set = current_set
                out_set = set(range(n)) - in_set
                
                if not out_set:
                    break
                
                best_candidate_score = current_score
                best_swap = None
                
                # Evaluate all swaps to find the best one
                # To speed up, we can pre-calculate some things, but simple evaluation is robust.
                # Optimization: Only check swaps that might improve distinct degrees.
                # Actually, just checking all is fine if we are fast.
                
                for v_in in out_set:
                    for v_out in in_set:
                        # Create new set for quick check
                        # new_set = (in_set - {v_out}) | {v_in}
                        
                        # Incremental degree calculation is complex to implement correctly in a single pass without state.
                        # Given the small n (17), full recalculation is O(k^2) or O(k * deg).
                        # k is ~10. k^2 is 100. 100 * 100 swaps = 10k ops. Very fast.
                        
                        new_set = set(current)
                        new_set.remove(v_out)
                        new_set.add(v_in)
                        
                        # Calculate distinct degrees
                        degs = set()
                        for v in new_set:
                            d = 0
                            for u in adj[v]:
                                if u in new_set:
                                    d += 1
                            degs.add(d)
                        
                        score = len(degs)
                        
                        if score > best_candidate_score:
                            best_candidate_score = score
                            best_swap = (v_in, v_out)
                            # Early exit if we hit the theoretical max for this size?
                            # Max distinct degrees is k.
                            if best_candidate_score == target_size:
                                break
                    if best_candidate_score == target_size:
                        break
                        
                if best_swap:
                    v_in, v_out = best_swap
                    # Update current
                    current = [v if v != v_out else v_in for v in current]
                    current_set = set(current)
                    current_score = best_candidate_score
                    improved = True
            
            if current_score > best_score:
                best_score = current_score
                best_subset = list(current)
                
                # If we found a perfect score (distinct degrees == size), we can stop early?
                # Max distinct degrees for size k is k.
                # If best_score == target_size, it's optimal for this size.
                # But maybe a larger size has a higher score? No, max is size.
                # So if we get size 17 with 17 distinct degrees, that's global max.
                # But we are iterating sizes.
    
    # Convert to proper types
    result = [int(v) for v in best_subset]
    return result

# EVOLVE-BLOCK-END
