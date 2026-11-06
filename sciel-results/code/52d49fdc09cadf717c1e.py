import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    max_val = 4096
    max_sum = 3 * max_val + 1  # 12289
    
    best_set = []
    best_size = 0
    
    def run_greedy_numpy(candidates):
        """
        Greedy construction using NumPy for efficient collision checking.
        candidates: list of integers to try in order.
        """
        selected = []
        S = np.array([], dtype=np.int32)
        used_sums = np.zeros(max_sum, dtype=bool)
        
        for x in candidates:
            if time.time() > deadline:
                break
                
            if not selected:
                selected.append(x)
                S = np.array([x], dtype=np.int32)
                used_sums[3*x] = True
                continue
            
            k = len(selected)
            
            # Generate new triple sums involving x
            # Type 1: x + x + x
            s1 = 3 * x
            
            # Type 2: x + x + s_i
            s2 = 2 * x + S
            
            # Type 3: x + s_i + s_j (i <= j)
            i_idx, j_idx = np.triu_indices(k)
            s3 = x + S[i_idx] + S[j_idx]
            
            all_new_sums = np.concatenate(([s1], s2, s3))
            
            # Check for duplicates within new sums
            unique_new = np.unique(all_new_sums)
            if len(unique_new) != len(all_new_sums):
                continue
                
            # Check against used sums
            if np.any(used_sums[all_new_sums]):
                continue
                
            # If valid, update
            selected.append(x)
            S = np.append(S, x)
            used_sums[all_new_sums] = True
            
        return selected
    
    def run_greedy_python(candidates):
        """
        Fallback greedy construction using pure Python sets.
        """
        S = []
        sums = set()
        
        for x in candidates:
            if time.time() > deadline:
                break
                
            new_sums = set()
            
            # (x, x, x)
            s = 3 * x
            if s in sums:
                continue
            new_sums.add(s)
            
            # (x, x, s_i)
            valid = True
            for s_i in S:
                s = 2 * x + s_i
                if s in sums or s in new_sums:
                    valid = False
                    break
                new_sums.add(s)
            if not valid:
                continue
            
            # (x, s_i, s_j)
            n = len(S)
            for i in range(n):
                s_i = S[i]
                for j in range(i, n):
                    s_j = S[j]
                    s = x + s_i + s_j
                    if s in sums or s in new_sums:
                        valid = False
                        break
                    new_sums.add(s)
                if not valid:
                    break
            if not valid:
                continue
            
            S.append(x)
            sums.update(new_sums)
            
        return S
    
    # 1. Deterministic Greedy with NumPy (Fast baseline)
    candidates = list(range(1, max_val + 1))
    S = run_greedy_numpy(candidates)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S[:]
        
    # 2. Randomized Greedy with NumPy
    # We use multiple seeds to explore different orderings.
    # NumPy version is significantly faster, allowing more restarts.
    
    rng = random.Random(42)
    restarts = 0
    while time.time() < deadline - 2.0:  # Leave 2s buffer
        restarts += 1
        candidates = list(range(1, max_val + 1))
        rng.shuffle(candidates)
        
        S = run_greedy_numpy(candidates)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
        # Optional: Local search / improvement
        # For now, simple restarts are robust and fast with NumPy.
        
    return best_set
