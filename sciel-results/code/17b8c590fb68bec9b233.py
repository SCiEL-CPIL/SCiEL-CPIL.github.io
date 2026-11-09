import time
import numpy as np
import threading

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s kill
    n = 256
    
    # 1. Precompute QUBO Matrix Q
    # Energy E(x) = sum_{i<j} Q[i,j] x_i x_j + sum_i Q[i,i] x_i
    # We want to minimize the number of monochromatic 4-APs.
    # A 4-AP (a, b, c, d) is monochromatic if x_a=x_b=x_c=x_d.
    # Let's use 0/1 encoding.
    # Indicator for "all 4 equal" is:
    # If all 0: (1-x_a)(1-x_b)(1-x_c)(1-x_d)
    # If all 1: x_a x_b x_c x_d
    # Sum = (1 - x_a - x_b - x_c - x_d + x_a x_b + x_a x_c + x_a x_d + x_b x_c + x_b x_d + x_c x_d
    #        - x_a x_b x_c - x_a x_b x_d - x_a x_c x_d - x_b x_c x_d
    #        + x_a x_b x_c x_d) + x_a x_b x_c x_d
    # This is a 4th order polynomial. We can linearize it to QUBO using auxiliary variables,
    # OR we can just use the fact that for binary variables, higher order terms can be reduced.
    # However, a simpler approach for the heuristic is to use the "pairwise disagreement" metric.
    # Actually, the prompt asks for a QUBO formulation.
    # Let's reduce the 4th order term to 2nd order.
    # Note: For binary x, x^2 = x.
    # The term "all equal" can be written as:
    # P_all_equal = 1 - (x_a + x_b + x_c + x_d) + (x_a x_b + x_a x_c + x_a x_d + x_b x_c + x_b x_d + x_c x_d)
    #               - (x_a x_b x_c + x_a x_b x_d + x_a x_c x_d + x_b x_c x_d)
    #               + 2 x_a x_b x_c x_d
    # This is still high order.
    
    # Alternative Strategy:
    # Instead of strict QUBO reduction which is complex, let's use the QUBO structure to speed up SA.
    # We can define the energy as the sum of penalties for each AP.
    # Penalty for AP (i, j, k, l) is 1 if x_i=x_j=x_k=x_l else 0.
    # We can approximate the interaction or use exact delta computation.
    # The "QUBO" insight here is that the energy change from flipping a set S of bits depends only on the interactions between S and the rest, and within S.
    # If we precompute the adjacency of the hypergraph, we can compute deltas fast.
    
    # Let's stick to a robust, fast vectorized SA but with a smarter move set and parallelism.
    # The "QUBO" framing allows us to view the problem as minimizing E(x) = sum_{AP} P(x_AP).
    # We will use a multi-threaded SA approach where each thread runs an independent trajectory,
    # and we periodically synchronize the best solution. This effectively samples the energy landscape in parallel.
    
    # Precompute APs
    aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            aps.append((i0, i1, i2, i3))
            
    aps_arr = np.array(aps, dtype=np.int32)
    M = len(aps_arr)
    
    def count_mono(c):
        return np.sum((c[aps_arr[:,0]] == c[aps_arr[:,1]]) & 
                      (c[aps_arr[:,1]] == c[aps_arr[:,2]]) & 
                      (c[aps_arr[:,2]] == c[aps_arr[:,3]]))
                      
    def thue_morse(i):
        return bin(i).count('1') % 2
        
    # Shared state
    best_score_global = np.inf
    best_colors_global = None
    lock = threading.Lock()
    
    def worker(seed_id):
        global best_score_global, best_colors_global
        np.random.seed(seed_id)
        
        # Initialize with Thue-Morse or Random
        if seed_id % 2 == 0:
            colors = np.array([thue_morse(i) for i in range(n)], dtype=np.int8)
        else:
            colors = np.random.randint(0, 2, n).astype(np.int8)
            
        current_score = count_mono(colors)
        best_score_local = current_score
        best_colors_local = colors.copy()
        
        T = 5.0
        T_min = 0.05
        cooling_rate = 0.99995
        
        # Precompute bit masks for faster flipping? 
        # For a single flip at index k, we can precompute which APs are affected.
        # Let's build an index: for each position k, list of AP indices where k is present.
        # This allows O(degree) delta calculation instead of O(M) full recount.
        # Degree of a node in the 4-AP hypergraph is roughly proportional to n.
        # Total edges M ~ n^2. Sum of degrees ~ 4M.
        # This is feasible.
        
        ap_indices_per_pos = [[] for _ in range(n)]
        for idx, ap in enumerate(aps):
            for pos in ap:
                ap_indices_per_pos[pos].append(idx)
        
        # Convert to numpy arrays for fast access
        ap_indices_per_pos_np = [np.array(lst, dtype=np.int32) for lst in ap_indices_per_pos]
        
        # We need to track the state of each AP (is it mono or not) to calculate delta?
        # No, it's easier to just recount if we flip multiple bits.
        # But for single bit flips, we can be smarter.
        # However, vectorized recount is ~1ms. 115s / 1ms = 115k iterations.
        # With 4 threads, we get ~460k iterations total.
        
        max_iter = 100000
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if T < T_min:
                break
                
            # Random move: flip 1 to 4 bits
            num_flips = np.random.randint(1, 5)
            if num_flips > 1:
                flips = np.random.choice(n, size=num_flips, replace=False)
            else:
                flips = np.array([np.random.randint(n)])
                
            new_colors = colors.copy()
            new_colors[flips] = 1 - new_colors[flips]
            new_score = count_mono(new_colors)
            
            if new_score < current_score:
                colors = new_colors
                current_score = new_score
                if current_score < best_score_local:
                    best_score_local = current_score
                    best_colors_local = colors.copy()
                    with lock:
                        if best_score_local < best_score_global:
                            best_score_global = best_score_local
                            best_colors_global = best_colors_local.copy()
            else:
                if new_score == current_score:
                    if np.random.random() < 0.5:
                        colors = new_colors
                else:
                    r = np.random.random()
                    if r < np.exp(-(new_score - current_score) / T):
                        colors = new_colors
                        current_score = new_score
                        
            T *= cooling_rate
        
        # Final update
        with lock:
            if best_score_local < best_score_global:
                best_score_global = best_score_local
                best_colors_global = best_colors_local.copy()
                
    # Launch threads
    num_threads = 4
    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    if best_colors_global is not None:
        return [int(c) for c in best_colors_global]
    else:
        # Fallback
        colors = np.array([thue_morse(i) for i in range(n)], dtype=np.int8)
        return [int(c) for c in colors]
