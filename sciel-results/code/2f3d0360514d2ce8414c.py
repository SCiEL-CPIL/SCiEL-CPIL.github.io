import time
import random
import math

def _compute_sizes(A):
    A_list = list(A)
    n = len(A_list)
    sums = set()
    prods = set()
    for i in range(n):
        for j in range(i, n):
            sums.add(A_list[i] + A_list[j])
            prods.add(A_list[i] * A_list[j])
    return len(sums), len(prods)

def _score(A):
    s, p = _compute_sizes(A)
    return max(s, p)

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    # Candidate pool
    max_candidate = 400
    candidates = list(range(1, max_candidate + 1))
    
    # Precompute logs for multiplicative compatibility
    logs = [math.log(c) for c in candidates]
    
    # We will try multiple graph configurations (temperatures)
    # and pick the best resulting set.
    
    best_A = set()
    best_score = float('inf')
    
    # Parameters for the compatibility graph
    # T_add controls how close numbers need to be for additive compatibility
    # T_mult controls how close logs need to be for multiplicative compatibility
    t_add_values = [5, 10, 20, 50]
    t_mult_values = [0.1, 0.5, 1.0, 2.0]
    
    for t_add in t_add_values:
        for t_mult in t_mult_values:
            if time.time() - start_time > time_limit - 5.0:
                break
            
            # Build the graph implicitly by calculating weights on the fly or precomputing a matrix
            # Since N=400, a 400x400 matrix is 160k floats, which is fine.
            # But we can also just compute the greedy step efficiently.
            
            # Greedy Clustering
            A = set()
            # To speed up greedy selection, we can maintain a score for each candidate
            # score[u] = sum of weights to nodes in A
            scores = [0.0] * (max_candidate + 1) # 1-indexed
            
            for _ in range(20):
                best_u = -1
                best_score_u = -1.0
                
                for u in range(1, max_candidate + 1):
                    if u in A:
                        continue
                    
                    # Calculate weight sum to current A
                    # We can update scores incrementally, but for simplicity and given small N,
                    # recalculating or using a precomputed matrix is fine.
                    # Let's use a precomputed weight matrix for this configuration.
                    # Actually, let's just compute the weight to the *last added* node? No, that's not correct.
                    # We need sum of weights to ALL nodes in A.
                    
                    # Optimization: Precompute weight matrix W for this (t_add, t_mult)
                    # This is expensive inside the loop. 
                    # Let's precompute W once per (t_add, t_mult) pair.
                    pass # Placeholder for logic below

            # The above loop is inefficient if we recompute weights.
            # Let's restructure:
            # 1. Precompute W for current t_add, t_mult.
            # 2. Run Greedy.
            
            W = [[0.0] * (max_candidate + 1) for _ in range(max_candidate + 1)]
            for i in range(1, max_candidate + 1):
                for j in range(i + 1, max_candidate + 1):
                    diff_add = abs(i - j)
                    diff_mult = abs(logs[i-1] - logs[j-1])
                    w = math.exp(-diff_add / t_add) + math.exp(-diff_mult / t_mult)
                    W[i][j] = w
                    W[j][i] = w
            
            A = set()
            scores = [0.0] * (max_candidate + 1)
            
            for _ in range(20):
                best_u = -1
                best_val = -1.0
                for u in range(1, max_candidate + 1):
                    if u in A:
                        continue
                    if scores[u] > best_val:
                        best_val = scores[u]
                        best_u = u
                
                if best_u == -1:
                    break
                
                A.add(best_u)
                # Update scores for all other nodes
                for v in range(1, max_candidate + 1):
                    if v not in A:
                        scores[v] += W[v][best_u]
            
            # Evaluate this A
            current_score = _score(A)
            if current_score < best_score:
                best_score = current_score
                best_A = set(A)
                
                # Refinement: Graph-Guided Local Search
                # Try to improve best_A
                improved = True
                while improved and (time.time() - start_time) < time_limit:
                    improved = False
                    A_list = list(best_A)
                    
                    # Generate candidates for swap: nodes outside A with high connectivity to A
                    # We can use the W matrix from the last run, or just pick randoms?
                    # To stay true to "graph guided", let's use the W matrix.
                    # But W depends on t_add, t_mult. 
                    # Let's just do a simple local search here: 
                    # Try swapping a random element in A with a random element in P\A.
                    # Wait, the directive said "rather than performing random swaps".
                    # So I should avoid random swaps.
                    # Instead, I will try swapping elements in A with elements in P\A 
                    # that have the highest weight to A in the *current* graph W.
                    
                    # Compute "attractiveness" of each outside node
                    attractiveness = [0.0] * (max_candidate + 1)
                    for u in range(1, max_candidate + 1):
                        if u in best_A:
                            continue
                        for v in best_A:
                            attractiveness[u] += W[u][v]
                    
                    # Sort outside nodes by attractiveness
                    outside_nodes = [u for u in range(1, max_candidate + 1) if u not in best_A]
                    outside_nodes.sort(key=lambda x: attractiveness[x], reverse=True)
                    
                    # Try swapping the top few outside nodes with each inside node
                    top_outside = outside_nodes[:10] # Only try the most compatible ones
                    
                    for new_val in top_outside:
                        if (time.time() - start_time) > time_limit:
                            break
                        for old_val in A_list:
                            if old_val == new_val:
                                continue
                            new_A = set(best_A)
                            new_A.discard(old_val)
                            new_A.add(new_val)
                            
                            new_score = _score(new_A)
                            if new_score < best_score:
                                best_A = new_A
                                best_score = new_score
                                improved = True
                                # Update A_list for next iteration? 
                                # It's complex to update W/scores dynamically.
                                # Let's just break and restart the inner loop logic if we want to be strict.
                                # Or just break out of swap loops.
                                break
                        if improved:
                            break
                            
    return sorted(best_A)
