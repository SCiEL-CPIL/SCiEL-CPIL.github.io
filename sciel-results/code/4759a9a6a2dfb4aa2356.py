import time
import random

def construct():
    N = 51975
    # Get divisors
    divisors = [d for d in range(3, N + 1, 2) if N % d == 0]
    
    # Precompute base masks S_m
    S_masks = {}
    for m in divisors:
        mask = 0
        for k in range(N // m):
            mask |= (1 << (k * m))
        S_masks[m] = mask
        
    # Precompute all shifted masks
    ALL_ONES = (1 << N) - 1
    Masks = {}
    for m in divisors:
        S = S_masks[m]
        masks_for_m = []
        for r in range(m):
            if r == 0:
                mask = S
            else:
                mask = (S << r) | (S >> (N - r))
                mask &= ALL_ONES
            masks_for_m.append(mask)
        Masks[m] = masks_for_m

    start_time = time.time()
    deadline = start_time + 110.0
    
    def greedy_construct():
        uncovered = ALL_ONES
        remaining = set(divisors)
        classes = []
        
        while remaining and time.time() < deadline:
            best_count = N + 1
            best_m = None
            best_r = None
            best_new_uncov = None
            
            for m in remaining:
                if time.time() > deadline:
                    break
                masks_for_m = Masks[m]
                for r in range(m):
                    if time.time() > deadline:
                        break
                    mask = masks_for_m[r]
                    new_uncov = uncovered & ~mask
                    c = new_uncov.bit_count()
                    if c < best_count:
                        best_count = c
                        best_m = m
                        best_r = r
                        best_new_uncov = new_uncov
                if time.time() > deadline:
                    break
            
            if best_m is None:
                break
                
            uncovered = best_new_uncov
            remaining.remove(best_m)
            classes.append([best_m, best_r])
        
        return classes
    
    def compute_coverage(classes):
        total_covered = 0
        for m, r in classes:
            total_covered |= Masks[m][r]
        return total_covered
    
    def local_search(classes):
        n_classes = len(classes)
        total_covered = compute_coverage(classes)
        uncovered = ALL_ONES & ~total_covered
        current_score = uncovered.bit_count()
        
        # Precompute coverage without each class
        cov_without = [0] * n_classes
        for i in range(n_classes):
            cov_without[i] = 0
            for j in range(n_classes):
                if j != i:
                    cov_without[i] |= Masks[classes[j][0]][classes[j][1]]
        
        # Try to improve by replacing one class at a time
        max_ls_iters = 15
        ls_iter = 0
        improved = True
        while improved and ls_iter < max_ls_iters and time.time() < deadline:
            improved = False
            best_improvement = None
            best_improvement_score = current_score
            ls_iter += 1
            
            for i in range(n_classes):
                if time.time() > deadline:
                    break
                
                m_curr, r_curr = classes[i]
                cov_w = cov_without[i]
                
                # Try other moduli
                for m_new in divisors:
                    if m_new == m_curr:
                        continue
                    if time.time() > deadline:
                        break
                    masks_for_m = Masks[m_new]
                    # Sample residues to save time
                    step = max(1, m_new // 50)
                    for r_new in range(0, m_new, step):
                        if time.time() > deadline:
                            break
                        new_covered = cov_w | masks_for_m[r_new]
                        new_uncov = ALL_ONES & ~new_covered
                        c = new_uncov.bit_count()
                        if c < best_improvement_score:
                            best_improvement_score = c
                            best_improvement = (i, m_new, r_new)
            
            if best_improvement:
                i, m_new, r_new = best_improvement
                classes[i] = [m_new, r_new]
                current_score = best_improvement_score
                improved = True
                # Recompute total_covered and cov_without
                total_covered = compute_coverage(classes)
                for i in range(n_classes):
                    cov_without[i] = 0
                    for j in range(n_classes):
                        if j != i:
                            cov_without[i] |= Masks[classes[j][0]][classes[j][1]]
        
        return classes
    
    # Run greedy
    classes = greedy_construct()
    
    if time.time() > deadline:
        return classes
    
    # Run local search
    classes = local_search(classes)
    
    return classes
