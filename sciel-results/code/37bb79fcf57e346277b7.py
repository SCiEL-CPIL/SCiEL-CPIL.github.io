import numpy as np
import random
import time
import math

N = 256

def construct():
    deadline = time.time() + 55.0
    rng = random.Random(42)
    
    # 1. Precompute APs and their membership in positions
    # ap_list: list of lists of indices (0-based) for each AP
    # pos_to_aps: list of lists of AP indices for each position
    ap_list = []
    pos_to_aps = [[] for _ in range(N)]
    
    ap_id = 0
    # Iterate over all possible APs (a, d, l)
    # a: start (1-based), d: step, l: length
    for d in range(1, N + 1):
        for a in range(1, N + 1):
            max_l = (N - a) // d + 1
            for l in range(1, max_l + 1):
                indices = [a - 1 + i * d for i in range(l)]
                ap_list.append(indices)
                for idx in indices:
                    pos_to_aps[idx].append(ap_id)
                ap_id += 1
                
    num_aps = len(ap_list)
    
    # Convert pos_to_aps to numpy arrays for faster indexing if needed, 
    # but lists are fine for S[indices] += delta if we convert to array once?
    # Actually, S[list] works in numpy.
    
    def compute_initial_sums(s):
        # s is a numpy array of +1/-1
        S = np.zeros(num_aps, dtype=np.int16)
        for ap_idx, indices in enumerate(ap_list):
            S[ap_idx] = np.sum(s[indices])
        return S

    def get_max_abs(S):
        return int(np.max(np.abs(S)))

    def local_search(s, S, current_max, time_limit, rng, temp_start=10.0, cooling=0.999):
        """
        Simulated Annealing local search.
        """
        temp = temp_start
        steps = 0
        stuck = 0
        
        while time.time() < time_limit:
            j = rng.randint(0, N - 1)
            old_val = s[j]
            delta = -2 * old_val
            
            ap_indices = pos_to_aps[j]
            if not ap_indices:
                continue
                
            # Apply tentative flip
            S[ap_indices] += delta
            
            new_max = np.max(np.abs(S))
            
            # Acceptance criterion
            if new_max < current_max:
                s[j] = -old_val
                current_max = new_max
                stuck = 0
            else:
                # Simulated annealing acceptance
                diff = new_max - current_max
                if temp > 0.1:
                    p = math.exp(-diff / temp)
                    if rng.random() < p:
                        s[j] = -old_val
                        current_max = new_max
                        stuck = 0
                    else:
                        # Revert
                        S[ap_indices] -= delta
                        stuck += 1
                else:
                    # Revert
                    S[ap_indices] -= delta
                    stuck += 1
            
            temp *= cooling
            steps += 1
            
            # If stuck for too long, break to allow restart
            if stuck > 5000:
                break
                
        return s, S, current_max

    def greedy_polish(s, S, current_max, time_limit, rng):
        """
        Greedy hill climbing: only accept moves that strictly improve the score.
        """
        improved = True
        while time.time() < time_limit and improved:
            improved = False
            # Try flipping each position
            for j in range(N):
                old_val = s[j]
                delta = -2 * old_val
                ap_indices = pos_to_aps[j]
                
                S[ap_indices] += delta
                new_max = np.max(np.abs(S))
                
                if new_max < current_max:
                    s[j] = -old_val
                    current_max = new_max
                    improved = True
                else:
                    S[ap_indices] -= delta
                    
        return s, S, current_max

    best_seq = None
    best_score = float('inf')
    
    # Generate diverse initializations
    inits = []
    
    # 1. Thue-Morse
    tm = np.ones(N, dtype=np.int8)
    for i in range(1, N):
        c = bin(i).count('1')
        tm[i] = 1 if c % 2 == 0 else -1
    inits.append(tm)
    
    # 2. Alternating
    alt = np.array([1 if i % 2 == 0 else -1 for i in range(N)], dtype=np.int8)
    inits.append(alt)
    
    # 3. Random (several)
    for _ in range(5):
        inits.append(np.array([1 if rng.random() < 0.5 else -1 for _ in range(N)], dtype=np.int8))
        
    # 4. Block pattern
    block = np.array([1 if (i // 8) % 2 == 0 else -1 for i in range(N)], dtype=np.int8)
    inits.append(block)
    
    # Run search on each init
    for idx, init in enumerate(inits):
        if time.time() > deadline - 5.0:
            break
            
        s = init.copy()
        S = compute_initial_sums(s)
        current_max = get_max_abs(S)
        
        if current_max < best_score:
            best_score = current_max
            best_seq = s.copy()
            
        # Phase 1: Simulated Annealing
        remaining = deadline - time.time()
        if remaining > 2.0:
            sa_time = min(remaining - 1.0, 10.0) # Cap SA time per restart
            s, S, current_max = local_search(s, S, current_max, time.time() + sa_time, rng)
            
        if current_max < best_score:
            best_score = current_max
            best_seq = s.copy()
            
        # Phase 2: Greedy Polish
        if time.time() < deadline - 1.0:
            s, S, current_max = greedy_polish(s, S, current_max, time.time() + (deadline - 1.0), rng)
            
        if current_max < best_score:
            best_score = current_max
            best_seq = s.copy()

    # If we have time left, do more random restarts from the best found so far
    if best_seq is not None and time.time() < deadline - 5.0:
        # Perturb best_seq and search again
        for _ in range(10):
            if time.time() > deadline - 2.0:
                break
            s = best_seq.copy()
            # Flip 5-10 random bits
            num_flips = rng.randint(5, 10)
            for _ in range(num_flips):
                j = rng.randint(0, N - 1)
                s[j] = -s[j]
                
            S = compute_initial_sums(s)
            current_max = get_max_abs(S)
            
            remaining = deadline - time.time()
            if remaining > 2.0:
                sa_time = min(remaining - 1.0, 5.0)
                s, S, current_max = local_search(s, S, current_max, time.time() + sa_time, rng)
                
            if current_max < best_score:
                best_score = current_max
                best_seq = s.copy()

    if best_seq is None:
        best_seq = np.ones(N, dtype=np.int8)

    return [int(x) for x in best_seq]
