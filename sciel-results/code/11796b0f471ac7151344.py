import numpy as np
import time
import itertools

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin

    # Compute the objective: ||h||_2^2 / (||h||_1 * sup|autoconv(h)|)
    def _compute_ratio(h):
        h = np.asarray(h, dtype=float)
        if np.all(h < 1e-15):
            return 0.0
        l2_sq = np.sum(h ** 2)
        l1 = np.sum(h)
        if l1 < 1e-15 or l2_sq < 1e-15:
            return 0.0
        conv = np.convolve(h, h, mode='full')
        sup_conv = np.max(conv)
        if sup_conv < 1e-15:
            return 0.0
        return l2_sq / (l1 * sup_conv)

    # Greedy Sidon set construction
    def _greedy_sidon(candidates, existing_sums=None):
        S = []
        sums = set()
        if existing_sums:
            sums = set(existing_sums)
            # Rebuild sums from S if provided
        for k in candidates:
            ok = True
            for s in S:
                if (k + s) in sums:
                    ok = False
                    break
            if ok and (k + k) in sums:
                ok = False
            if ok:
                S.append(k)
                for s in S[:-1]:
                    sums.add(k + s)
                sums.add(k + k)
        return S

    # Local search: try replacing one element with another to maintain Sidon property
    def _local_search(S, max_rounds=50):
        S = list(S)
        current_sums = set()
        for a, b in itertools.combinations_with_replacement(S, 2):
            current_sums.add(a + b)
        
        improved = True
        rounds = 0
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1
            for i in range(len(S)):
                old = S[i]
                # Try removing old and adding a new candidate
                S_temp = [x for x in S if x != old]
                sums_temp = set()
                for a, b in itertools.combinations_with_replacement(S_temp, 2):
                    sums_temp.add(a + b)
                
                # Try all possible new candidates not in S_temp
                for new_k in range(N):
                    if new_k in S_temp:
                        continue
                    ok = True
                    for s in S_temp:
                        if (new_k + s) in sums_temp:
                            ok = False
                            break
                    if ok and (new_k + new_k) in sums_temp:
                        ok = False
                    if ok:
                        # Check if this improves the objective (more elements or better spread)
                        S_new = S_temp + [new_k]
                        # We just accept if it maintains Sidon (size is same)
                        # The key is to find a better Sidon set structure
                        S = S_new
                        current_sums = sums_temp.copy()
                        for s in S_temp:
                            current_sums.add(new_k + s)
                        current_sums.add(new_k + new_k)
                        improved = True
                        break
        return S

    # Simulated annealing on Sidon sets
    def _simulated_annealing_sidon(S_init, time_budget=5.0):
        S = list(S_init)
        current_sums = set()
        for a, b in itertools.combinations_with_replacement(S, 2):
            current_sums.add(a + b)
        
        best_S = list(S)
        temp = 1.0
        cooling = 0.995
        t0 = time.time()
        
        while time.time() - t0 < time_budget and temp > 0.01:
            # Perturbation: remove one, add another
            if len(S) == 0:
                break
            i = np.random.randint(len(S))
            old = S[i]
            S_temp = [x for x in S if x != old]
            sums_temp = set()
            for a, b in itertools.combinations_with_replacement(S_temp, 2):
                sums_temp.add(a + b)
            
            # Pick a random new candidate
            candidates = [k for k in range(N) if k not in S_temp]
            if not candidates:
                temp *= cooling
                continue
            new_k = np.random.choice(candidates)
            
            ok = True
            for s in S_temp:
                if (new_k + s) in sums_temp:
                    ok = False
                    break
            if ok and (new_k + new_k) in sums_temp:
                ok = False
            
            if ok:
                S_new = S_temp + [new_k]
                # Accept based on some metric (for now, just accept valid Sidon sets)
                # In the future, we could weight by spread
                S = S_new
                current_sums = sums_temp.copy()
                for s in S_temp:
                    current_sums.add(new_k + s)
                current_sums.add(new_k + new_k)
                best_S = list(S)
            temp *= cooling
        
        return best_S

    # Run multiple random greedy + local search + SA
    best_S = []
    np.random.seed(42)
    
    # First, try a few different random seeds for greedy
    for trial in range(30):
        if time.time() > deadline - 5.0:
            break
        candidates = list(range(N))
        np.random.shuffle(candidates)
        S = _greedy_sidon(candidates)
        S = _local_search(S, max_rounds=20)
        if len(S) > len(best_S):
            best_S = list(S)
    
    # Then use simulated annealing from the best found
    if time.time() < deadline - 10.0:
        best_S = _simulated_annealing_sidon(best_S, time_budget=min(30.0, deadline - time.time() - 5.0))
    
    # Final local search
    if time.time() < deadline - 3.0:
        best_S = _local_search(best_S, max_rounds=100)
    
    # If we still have time, try more random restarts
    if time.time() < deadline - 5.0:
        for trial in range(50):
            if time.time() > deadline - 3.0:
                break
            candidates = list(range(N))
            np.random.shuffle(candidates)
            S = _greedy_sidon(candidates)
            S = _local_search(S, max_rounds=10)
            if len(S) > len(best_S):
                best_S = list(S)
    
    # Construct the height vector
    h = np.zeros(N)
    if len(best_S) > 0:
        for idx in best_S:
            h[idx] = 1.0 / len(best_S)
    else:
        h[0] = 1.0
    
    return list(h)
