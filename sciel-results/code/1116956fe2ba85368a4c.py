import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10s margin for 60s limit
    
    n = 30
    
    # Precompute LCM of 1..30
    L = 1
    for i in range(1, n + 1):
        L = L * i // gcd(L, i)
    
    # We work in integer weights W[i] = L // (i+1) for i in 0..29
    W = [L // (i + 1) for i in range(n)]
    
    best_abs_K = float('inf')
    best_coeffs = None
    
    def try_update(coeffs):
        nonlocal best_abs_K, best_coeffs
        K = sum(coeffs[i] * W[i] for i in range(n))
        if K == 0:
            return
        abs_K = abs(K)
        if abs_K < best_abs_K:
            best_abs_K = abs_K
            best_coeffs = list(coeffs)
    
    # Phase 1: Meet-in-the-middle search
    mid = n // 2  # 15
    
    def generate_half(start, end):
        """Generate all (sum, mask) pairs for indices start to end-1."""
        length = end - start
        sums = [0]
        masks = [0]
        
        for i in range(length):
            w = W[start + i]
            new_sums = []
            new_masks = []
            multiplier = 3 ** i
            
            for k in range(len(sums)):
                s = sums[k]
                m = masks[k]
                
                new_sums.append(s - w)
                new_masks.append(m)
                
                new_sums.append(s)
                new_masks.append(m + multiplier)
                
                new_sums.append(s + w)
                new_masks.append(m + 2 * multiplier)
                
            sums = new_sums
            masks = new_masks
        
        return sums, masks
    
    sums_a, masks_a = generate_half(0, mid)
    sums_b, masks_b = generate_half(mid, n)
    
    pairs_b = list(zip(sums_b, masks_b))
    pairs_b.sort(key=lambda x: x[0])
    
    arr_b_sums = [p[0] for p in pairs_b]
    arr_b_masks = [p[1] for p in pairs_b]
    
    import bisect
    
    for i in range(len(sums_a)):
        if time.time() > deadline:
            break
            
        a_sum = sums_a[i]
        target = -a_sum
        
        idx = bisect.bisect_left(arr_b_sums, target)
        
        candidates = []
        if idx < len(arr_b_sums):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
            
        for c_idx in candidates:
            b_sum = arr_b_sums[c_idx]
            total = a_sum + b_sum
            if total != 0:
                abs_total = abs(total)
                if abs_total < best_abs_K:
                    best_abs_K = abs_total
                    coeffs = [0] * n
                    for j in range(mid):
                        digit = (masks_a[i] // (3 ** j)) % 3
                        coeffs[j] = digit - 1
                    for j in range(mid):
                        digit = (arr_b_masks[c_idx] // (3 ** j)) % 3
                        coeffs[mid + j] = digit - 1
                    best_coeffs = coeffs
    
    if best_coeffs is None:
        best_coeffs = [0] * n
        best_coeffs[0] = 1
        best_coeffs[1] = -1
        K = sum(best_coeffs[i] * W[i] for i in range(n))
        best_abs_K = abs(K)
    
    # Phase 2: Local search with 2-opt moves
    random.seed(42)
    current = list(best_coeffs)
    current_K = sum(current[i] * W[i] for i in range(n))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = list(range(n))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
                
            old_c = current[i]
            
            for new_c in [-1, 0, 1]:
                if new_c == old_c:
                    continue
                
                new_K = current_K + (new_c - old_c) * W[i]
                
                if new_K == 0:
                    continue
                
                new_abs_K = abs(new_K)
                if new_abs_K < best_abs_K:
                    current[i] = new_c
                    current_K = new_K
                    best_abs_K = new_abs_K
                    best_coeffs = list(current)
                    improved = True
                    old_c = new_c
                    break
        
        # Try 2-opt moves: flip two positions simultaneously
        if time.time() < deadline:
            pairs = list(zip(indices, indices[1:]))
            random.shuffle(pairs)
            
            for i, j in pairs[:20]:  # Limit number of 2-opt attempts per pass
                if time.time() > deadline:
                    break
                    
                old_ci = current[i]
                old_cj = current[j]
                
                for new_ci in [-1, 0, 1]:
                    if new_ci == old_ci:
                        continue
                    for new_cj in [-1, 0, 1]:
                        if new_cj == old_cj:
                            continue
                        
                        new_K = current_K + (new_ci - old_ci) * W[i] + (new_cj - old_cj) * W[j]
                        
                        if new_K == 0:
                            continue
                        
                        new_abs_K = abs(new_K)
                        if new_abs_K < best_abs_K:
                            current[i] = new_ci
                            current[j] = new_cj
                            current_K = new_K
                            best_abs_K = new_abs_K
                            best_coeffs = list(current)
                            improved = True
                            break
                    else:
                        continue
                    break
    
    # Phase 3: Random restarts with local search
    if time.time() < deadline - 5:
        for restart in range(200):
            if time.time() > deadline:
                break
            
            current = list(best_coeffs)
            current_K = sum(current[i] * W[i] for i in range(n))
            
            num_perturb = random.randint(1, 4)
            for _ in range(num_perturb):
                idx = random.randint(0, n - 1)
                current[idx] = random.choice([-1, 0, 1])
            
            current_K = sum(current[i] * W[i] for i in range(n))
            if current_K == 0:
                continue
            
            improved_local = True
            while improved_local and time.time() < deadline:
                improved_local = False
                indices = list(range(n))
                random.shuffle(indices)
                
                for i in indices:
                    if time.time() > deadline:
                        break
                        
                    old_c = current[i]
                    
                    for new_c in [-1, 0, 1]:
                        if new_c == old_c:
                            continue
                        
                        new_K = current_K + (new_c - old_c) * W[i]
                        
                        if new_K == 0:
                            continue
                        
                        new_abs_K = abs(new_K)
                        if new_abs_K < best_abs_K:
                            current[i] = new_c
                            current_K = new_K
                            best_abs_K = new_abs_K
                            best_coeffs = list(current)
                            improved_local = True
                            old_c = new_c
                            break
                
                # 2-opt in restarts
                if time.time() < deadline:
                    pairs = list(zip(indices, indices[1:]))
                    random.shuffle(pairs)
                    
                    for i, j in pairs[:10]:
                        if time.time() > deadline:
                            break
                            
                        old_ci = current[i]
                        old_cj = current[j]
                        
                        for new_ci in [-1, 0, 1]:
                            if new_ci == old_ci:
                                continue
                            for new_cj in [-1, 0, 1]:
                                if new_cj == old_cj:
                                    continue
                                
                                new_K = current_K + (new_ci - old_ci) * W[i] + (new_cj - old_cj) * W[j]
                                
                                if new_K == 0:
                                    continue
                                
                                new_abs_K = abs(new_K)
                                if new_abs_K < best_abs_K:
                                    current[i] = new_ci
                                    current[j] = new_cj
                                    current_K = new_K
                                    best_abs_K = new_abs_K
                                    best_coeffs = list(current)
                                    improved_local = True
                                    break
                            else:
                                continue
                            break
    
    return best_coeffs
