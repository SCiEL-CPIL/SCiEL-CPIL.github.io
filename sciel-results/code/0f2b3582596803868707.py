import time
import numpy as np

def construct():
    N = 51975
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # Generate divisors > 1 of 51975 = 3^3 * 5^2 * 7 * 11
    divisors = []
    for a in range(4):          # 3^a
        for b in range(3):      # 5^b
            for c in range(2):  # 7^c
                for d in range(2):  # 11^d
                    v = (3 ** a) * (5 ** b) * (7 ** c) * (11 ** d)
                    if v > 1:
                        divisors.append(v)
    divisors.sort()
    n_divs = len(divisors)
    
    # Precompute positions for each divisor and residue class
    pos_arrays = []
    for m in divisors:
        arr = []
        for r in range(m):
            arr.append(np.arange(r, N, m, dtype=np.int32))
        pos_arrays.append(arr)
    
    # Sort indices by coverage (small m first covers more residues)
    cov_counts = [N // m for m in divisors]
    idx_by_cov = sorted(range(n_divs), key=lambda i: cov_counts[i], reverse=True)
    idx_by_cov_rev = idx_by_cov[::-1]
    
    def greedy_pass(idx_order, deadline):
        """Greedy: for each divisor index in idx_order, pick best residue."""
        covered = np.zeros(N, dtype=np.uint8)
        classes = []
        
        for idx in idx_order:
            if time.time() > deadline:
                break
            m = divisors[idx]
            pos_arr = pos_arrays[idx]
            
            best_r = 0
            best_gain = -1
            for r in range(m):
                pos = pos_arr[r]
                gain = int((covered[pos] == 0).sum())
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            if best_gain > 0:
                classes.append([m, best_r])
                covered[pos_arr[best_r]] = 1
        
        uncovered = int((covered == 0).sum())
        return classes, covered, uncovered
    
    def local_search(classes, covered, deadline):
        """Try to improve by changing residues or swapping moduli."""
        best_classes = [c[:] for c in classes]
        best_uncovered = int((covered == 0).sum())
        used_mods = set(m for m, r in best_classes)
        
        max_iters = 20
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try adding unused moduli
            for idx in idx_by_cov:
                if time.time() > deadline:
                    break
                m = divisors[idx]
                if m in used_mods:
                    continue
                if cov_counts[idx] < 2:
                    continue
                
                pos_arr = pos_arrays[idx]
                best_r = 0
                best_gain = -1
                for r in range(m):
                    pos = pos_arr[r]
                    gain = int((covered[pos] == 0).sum())
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r
                
                if best_gain > 0:
                    new_classes = best_classes + [[m, best_r]]
                    new_cov = covered.copy()
                    new_cov[pos_arr[best_r]] = 1
                    new_uncovered = int((new_cov == 0).sum())
                    if new_uncovered < best_uncovered:
                        best_classes = new_classes
                        covered = new_cov
                        best_uncovered = new_uncovered
                        used_mods.add(m)
                        improved = True
            
            # Try changing residue for existing classes
            for i, (m, r) in enumerate(best_classes):
                if time.time() > deadline:
                    break
                
                idx = divisors.index(m)
                pos_arr = pos_arrays[idx]
                
                # For small m, try all residues; for large m, sample
                if m <= 40:
                    candidates = list(range(m))
                else:
                    candidates = [0, m // 2, (m - 1) // 2, min(1, m - 1), max(0, m - 2)]
                    candidates = [c for c in candidates if 0 <= c < m and c != r]
                
                best_r_new = r
                best_uncovered_new = best_uncovered
                
                for r_new in candidates:
                    if r_new == r:
                        continue
                    
                    old_pos = pos_arr[r]
                    new_pos = pos_arr[r_new]
                    
                    # Compute new covered state
                    new_cov = covered.copy()
                    # Remove old class coverage (set to 0 where only this class covers)
                    # Then add new class coverage
                    # Simple approach: recompute from scratch for correctness
                    temp_cov = np.zeros(N, dtype=np.uint8)
                    for j, (mm, rr) in enumerate(best_classes):
                        if j == i:
                            continue
                        jdx = divisors.index(mm)
                        temp_cov[pos_arrays[jdx][rr]] = 1
                    temp_cov[new_pos] = 1
                    
                    new_uncovered = int((temp_cov == 0).sum())
                    if new_uncovered < best_uncovered_new:
                        best_uncovered_new = new_uncovered
                        best_r_new = r_new
                
                if best_r_new != r:
                    best_classes[i] = [m, best_r_new]
                    # Recompute covered
                    covered = np.zeros(N, dtype=np.uint8)
                    for mm, rr in best_classes:
                        idx_m = divisors.index(mm)
                        covered[pos_arrays[idx_m][rr]] = 1
                    best_uncovered = best_uncovered_new
                    improved = True
            
            if not improved:
                break
        
        return best_classes, covered, best_uncovered
    
    best_classes = []
    best_uncovered = N
    
    # Strategy 1: Greedy from smallest moduli up
    classes, cov, unc = greedy_pass(idx_by_cov, deadline - 30.0)
    if unc < best_uncovered:
        best_uncovered = unc
        best_classes = classes[:]
    
    # Strategy 2: Greedy from largest moduli down
    if time.time() < deadline - 40.0:
        classes, cov, unc = greedy_pass(idx_by_cov_rev, deadline - 30.0)
        if unc < best_uncovered:
            best_uncovered = unc
            best_classes = classes[:]
    
    # Strategy 3: Local search improvement
    if time.time() < deadline - 10.0 and best_classes:
        best_classes, cov, best_uncovered = local_search(best_classes, cov, deadline - 5.0)
    
    return [[m, r] for m, r in best_classes]
