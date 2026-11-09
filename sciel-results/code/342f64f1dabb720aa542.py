import time
import random
import numpy as np

def construct():
    n = 256
    t0 = time.time()
    deadline = t0 + 55.0  # safety margin before 60s kill

    # 1. Precompute APs
    ap_indices = []
    for d in range(1, n + 1):
        for a in range(n):
            # Indices: a, a+d, ... < n
            idx = np.arange(a, n, d)
            if len(idx) > 0:
                ap_indices.append(idx)
    
    M = len(ap_indices)
    
    # Precompute which APs affect which position
    # affected_aps[i] is a list of AP indices k such that i is in ap_indices[k]
    affected_aps = [[] for _ in range(n)]
    for k, idx in enumerate(ap_indices):
        for pos in idx:
            affected_aps[pos].append(k)
    
    # Convert affected_aps to numpy arrays for faster iteration? 
    # Lists are fine for small sizes, but numpy might be faster for indexing if we vectorize the update.
    # However, the update is `ap_sums[k] += delta`. 
    # If we have a list of k's, we can do `ap_sums[np.array(ks)] += delta`.
    affected_aps_np = [np.array(ks, dtype=np.int32) if ks else np.array([], dtype=np.int32) for ks in affected_aps]

    # 2. Initialize with Thue-Morse
    def thue_morse_signs(n):
        signs = np.zeros(n, dtype=np.int8)
        for i in range(n):
            cnt = bin(i).count('1')
            signs[i] = 1 if cnt % 2 == 0 else -1
        return signs

    s = thue_morse_signs(n)
    
    # 3. Initialize ap_sums
    ap_sums = np.zeros(M, dtype=np.int32)
    for k in range(M):
        ap_sums[k] = np.sum(s[ap_indices[k]])
        
    current_max = np.max(np.abs(ap_sums))
    
    # 4. Constraint-directed local search
    # Instead of shuffling all positions, identify the worst AP(s) and focus on flipping within them
    rng = random.Random(42)
    best_s = s.copy()
    best_max = current_max
    
    stagnation_count = 0
    max_stagnation = 5
    
    while time.time() < deadline - 5.0:
        # Find the APs with maximum absolute sum
        abs_sums = np.abs(ap_sums)
        max_val = np.max(abs_sums)
        
        # Get indices of APs achieving the maximum
        worst_aps = np.where(abs_sums == max_val)[0]
        
        # Collect all positions in these worst APs
        candidate_positions = set()
        for k in worst_aps:
            for pos in ap_indices[k]:
                candidate_positions.add(pos)
        
        # Shuffle candidate positions
        candidates = list(candidate_positions)
        rng.shuffle(candidates)
        
        improved_this_pass = False
        
        for i in candidates:
            if time.time() > deadline - 5.0:
                break
            
            old_sign = s[i]
            new_sign = -old_sign
            delta = -2 * old_sign
            
            s[i] = new_sign
            
            # Update affected APs
            ks = affected_aps_np[i]
            if len(ks) > 0:
                ap_sums[ks] += delta
            
            new_max = np.max(np.abs(ap_sums))
            
            if new_max < current_max:
                current_max = new_max
                improved_this_pass = True
            else:
                # Revert
                s[i] = old_sign
                if len(ks) > 0:
                    ap_sums[ks] -= delta
        
        if improved_this_pass:
            stagnation_count = 0
            if current_max < best_max:
                best_max = current_max
                best_s = s.copy()
        else:
            stagnation_count += 1
            
            # If stuck, try a random perturbation
            if stagnation_count >= max_stagnation:
                # Perturb: flip a few random positions
                for _ in range(5):
                    p = rng.randint(0, n - 1)
                    old_sign = s[p]
                    s[p] = -old_sign
                    delta = -2 * old_sign
                    ks = affected_aps_np[p]
                    if len(ks) > 0:
                        ap_sums[ks] += delta
                
                current_max = np.max(np.abs(ap_sums))
                stagnation_count = 0
    
    return [int(x) for x in best_s]
