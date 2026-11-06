import numpy as np
import random
import time
import math

N = 256

def construct():
    deadline = time.time() + 55.0
    rng = random.Random(12345)
    
    # Precompute all APs: for each (start, step, length), store the 0-based indices
    # To save memory and time, we'll store APs as lists of index arrays
    ap_list = []  # list of numpy arrays of 0-based indices
    ap_set = set()  # to avoid duplicates
    
    for d in range(1, N + 1):
        for a in range(1, N + 1):
            max_l = (N - a) // d + 1
            for l in range(1, max_l + 1):
                # Check if this AP is already stored (as a tuple)
                key = (a, d, l)
                if key not in ap_set:
                    ap_set.add(key)
                    indices = np.array([a - 1 + i * d for i in range(l)], dtype=np.int32)
                    ap_list.append(indices)
    
    num_aps = len(ap_list)
    
    # Build position-to-APs mapping
    pos_to_ap = [[] for _ in range(N)]
    for ap_idx, indices in enumerate(ap_list):
        for idx in indices:
            pos_to_ap[idx].append(ap_idx)
    
    # Convert to numpy arrays
    pos_to_ap_np = [np.array(lst, dtype=np.int32) if lst else np.empty(0, dtype=np.int32) for lst in pos_to_ap]
    
    def compute_sums(s):
        S = np.zeros(num_aps, dtype=np.int16)
        for ap_idx in range(num_aps):
            S[ap_idx] = np.sum(s[ap_list[ap_idx]])
        return S
    
    def get_max_abs(S):
        return int(np.max(np.abs(S)))
    
    def local_search(s, S, current_max, deadline, rng, max_steps=50000):
        """Fast local search with single flips."""
        temp = 2.0
        cooling = 0.9995
        steps = 0
        stuck_count = 0
        
        while time.time() < deadline and steps < max_steps:
            j = rng.randint(0, N - 1)
            old_val = s[j]
            delta = -2 * old_val
            
            ap_indices = pos_to_ap_np[j]
            if len(ap_indices) == 0:
                steps += 1
                continue
            
            old_max = current_max
            new_max = old_max
            improved = False
            
            for ap_idx in ap_indices:
                old_sum = S[ap_idx]
                new_sum = old_sum + delta
                S[ap_idx] = new_sum
                new_abs = abs(new_sum)
                if new_abs > new_max:
                    new_max = new_abs
            
            cost_diff = new_max - old_max
            
            if cost_diff <= 0:
                s[j] = -old_val
                current_max = new_max
                stuck_count = 0
            else:
                if temp > 0.05:
                    p = math.exp(-cost_diff / temp)
                    if rng.random() < p:
                        s[j] = -old_val
                        current_max = new_max
                        stuck_count = 0
                    else:
                        # Revert
                        for ap_idx in ap_indices:
                            S[ap_idx] -= delta
                        stuck_count += 1
                else:
                    # Revert
                    for ap_idx in ap_indices:
                        S[ap_idx] -= delta
                    stuck_count += 1
            
            temp *= cooling
            steps += 1
            
            if temp < 0.01:
                break
        
        return s, S, current_max
    
    best_seq = None
    best_score = float('inf')
    
    # Try several structured initializations
    initializations = []
    
    # 1. Thue-Morse-like sequence
    tm = np.ones(N, dtype=np.int8)
    for i in range(1, N):
        # Count bits
        c = bin(i).count('1')
        tm[i] = 1 if c % 2 == 0 else -1
    initializations.append(tm.copy())
    
    # 2. Quadratic residue-like
    qr = np.ones(N, dtype=np.int8)
    for i in range(N):
        x = (i + 1)
        qr[i] = 1 if (x * x) % 4 == 1 else -1
    initializations.append(qr.copy())
    
    # 3. Random
    initializations.append(np.array([1 if rng.random() < 0.5 else -1 for _ in range(N)], dtype=np.int8))
    
    # 4. Another random
    initializations.append(np.array([1 if rng.random() < 0.5 else -1 for _ in range(N)], dtype=np.int8))
    
    # 5. Alternating
    alt = np.ones(N, dtype=np.int8)
    for i in range(N):
        alt[i] = 1 if i % 2 == 0 else -1
    initializations.append(alt.copy())
    
    # 6. Block pattern
    block = np.ones(N, dtype=np.int8)
    for i in range(N):
        block[i] = 1 if (i // 4) % 2 == 0 else -1
    initializations.append(block.copy())
    
    for init in initializations:
        if time.time() > deadline - 5.0:
            break
        
        s = init.copy()
        S = compute_sums(s)
        current_max = get_max_abs(S)
        
        if current_max < best_score:
            best_score = current_max
            best_seq = s.copy()
        
        remaining = deadline - time.time()
        if remaining > 3.0:
            s, S, current_max = local_search(s, S, current_max, time.time() + remaining - 2.0, rng)
        
        if current_max < best_score:
            best_score = current_max
            best_seq = s.copy()
        
        # Try a perturbation and re-search
        if time.time() > deadline - 5.0:
            break
        
        s2 = s.copy()
        # Flip a few random positions
        for _ in range(rng.randint(5, 15)):
            j = rng.randint(0, N - 1)
            s2[j] = -s2[j]
        
        S2 = compute_sums(s2)
        current_max2 = get_max_abs(S2)
        
        if current_max2 < best_score:
            best_score = current_max2
            best_seq = s2.copy()
        
        remaining2 = deadline - time.time()
        if remaining2 > 3.0:
            s2, S2, current_max2 = local_search(s2, S2, current_max2, time.time() + remaining2 - 2.0, rng)
        
        if current_max2 < best_score:
            best_score = current_max2
            best_seq = s2.copy()
    
    # Final verification
    if best_seq is None:
        best_seq = np.ones(N, dtype=np.int8)
    
    return [int(x) for x in best_seq]
