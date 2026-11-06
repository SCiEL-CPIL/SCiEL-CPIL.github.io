import numpy as np
import time

def construct():
    N = 51975
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    # Generate divisors of 51975 (odd divisors > 1)
    # 51975 = 3^3 * 5^2 * 7 * 11
    divisors = []
    for d in range(3, N + 1, 2):
        if N % d == 0:
            divisors.append(d)
    
    covered = np.zeros(N, dtype=bool)
    classes = []
    available_moduli = divisors.copy()
    
    # Helper: vectorized find_best_residue
    def find_best_residue(m, covered_arr):
        """Find residue r that maximizes uncovered count in class r mod m."""
        # Create index array for each residue class
        # For residue r, indices are r, r+m, r+2m, ...
        # We can use: np.arange(r, N, m) but that's a loop
        # Better: use the fact that we want sum of ~covered_arr[r::m]
        # Vectorize by computing all counts at once
        max_count = 0
        best_r = 0
        for r in range(m):
            c = np.count_nonzero(~covered_arr[r::m])
            if c > max_count:
                max_count = c
                best_r = r
        return best_r, max_count
    
    # Helper: compute gain of adding (m, r) given current covered
    def compute_gain(m, r, covered_arr):
        return np.count_nonzero(~covered_arr[r::m])
    
    # 1. Greedy Initialization
    while available_moduli:
        if time.time() > deadline:
            break
            
        best_gain = 0
        best_m = None
        best_r = None
        
        for m in available_moduli:
            max_count = 0
            best_r_m = 0
            for r in range(m):
                c = np.count_nonzero(~covered[r::m])
                if c > max_count:
                    max_count = c
                    best_r_m = r
            
            if max_count > best_gain:
                best_gain = max_count
                best_m = m
                best_r = best_r_m
        
        if best_m is None or best_gain == 0:
            break
            
        classes.append([best_m, best_r])
        covered[best_r::best_m] = True
        available_moduli.remove(best_m)
        
        if np.all(covered):
            break

    # 2. Local Search: Coordinate Descent + Swaps + Add
    selected_moduli = {c[0] for c in classes}
    unselected_moduli = [m for m in divisors if m not in selected_moduli]
    
    improved = True
    polish_rounds = 0
    while improved and time.time() < deadline:
        improved = False
        current_uncovered = np.count_nonzero(~covered)
        
        # A. Coordinate Descent: For each selected m, try to find a better r
        for i in range(len(classes)):
            if time.time() > deadline:
                break
            m = classes[i][0]
            r_old = classes[i][1]
            
            # Temporarily remove
            covered[r_old::m] = False
            
            # Find best new r
            r_new, gain_new = find_best_residue(m, covered)
            
            # Apply new r
            covered[r_new::m] = True
            uncovered_after_add = np.count_nonzero(~covered)
            
            # Check if improvement
            if uncovered_after_add < current_uncovered:
                classes[i][1] = r_new
                current_uncovered = uncovered_after_add
                improved = True
            else:
                # Revert
                covered[r_new::m] = False
                covered[r_old::m] = True
            
            if current_uncovered == 0:
                return classes

        # B. Swaps: Try to swap a selected modulus with an unselected one
        for m_out in list(selected_moduli):
            if time.time() > deadline:
                break
            
            r_out = None
            idx_out = None
            for j, c in enumerate(classes):
                if c[0] == m_out:
                    r_out = c[1]
                    idx_out = j
                    break
            
            # Temporarily remove m_out
            covered[r_out::m_out] = False
            new_uncovered = np.count_nonzero(~covered)
            
            best_swap_gain = 0
            best_m_in = None
            best_r_in = None
            
            for m_in in unselected_moduli:
                r_in, gain_in = find_best_residue(m_in, covered)
                if gain_in > best_swap_gain:
                    best_swap_gain = gain_in
                    best_m_in = m_in
                    best_r_in = r_in
            
            # Check if swap is beneficial
            lost = new_uncovered - current_uncovered
            if best_m_in is not None and best_swap_gain > lost:
                # Perform swap
                classes[idx_out][0] = best_m_in
                classes[idx_out][1] = best_r_in
                covered[best_r_in::best_m_in] = True
                
                selected_moduli.remove(m_out)
                selected_moduli.add(best_m_in)
                unselected_moduli.remove(best_m_in)
                unselected_moduli.append(m_out)
                
                current_uncovered = np.count_nonzero(~covered)
                improved = True
                if current_uncovered == 0:
                    return classes
            else:
                # Revert
                covered[r_out::m_out] = True

        # C. Add unselected moduli if they help
        for m in unselected_moduli[:]:
            if time.time() > deadline:
                break
            r, gain = find_best_residue(m, covered)
            if gain > 0:
                classes.append([m, r])
                covered[r::m] = True
                unselected_moduli.remove(m)
                selected_moduli.add(m)
                current_uncovered = np.count_nonzero(~covered)
                improved = True
                if current_uncovered == 0:
                    return classes
        
        # D. Second pass: try swaps again after additions
        if improved:
            for m_out in list(selected_moduli):
                if time.time() > deadline:
                    break
                
                r_out = None
                idx_out = None
                for j, c in enumerate(classes):
                    if c[0] == m_out:
                        r_out = c[1]
                        idx_out = j
                        break
                
                # Temporarily remove m_out
                covered[r_out::m_out] = False
                new_uncovered = np.count_nonzero(~covered)
                
                best_swap_gain = 0
                best_m_in = None
                best_r_in = None
                
                for m_in in unselected_moduli:
                    r_in, gain_in = find_best_residue(m_in, covered)
                    if gain_in > best_swap_gain:
                        best_swap_gain = gain_in
                        best_m_in = m_in
                        best_r_in = r_in
                
                # Check if swap is beneficial
                lost = new_uncovered - current_uncovered
                if best_m_in is not None and best_swap_gain > lost:
                    # Perform swap
                    classes[idx_out][0] = best_m_in
                    classes[idx_out][1] = best_r_in
                    covered[best_r_in::best_m_in] = True
                    
                    selected_moduli.remove(m_out)
                    selected_moduli.add(best_m_in)
                    unselected_moduli.remove(best_m_in)
                    unselected_moduli.append(m_out)
                    
                    current_uncovered = np.count_nonzero(~covered)
                    improved = True
                    if current_uncovered == 0:
                        return classes
                else:
                    # Revert
                    covered[r_out::m_out] = True

        # E. Final polish: multiple coordinate descent passes
        polish_rounds += 1
        if polish_rounds <= 3:
            for _ in range(2):  # Two extra CD passes
                if time.time() > deadline:
                    break
                cd_improved = False
                for i in range(len(classes)):
                    if time.time() > deadline:
                        break
                    m = classes[i][0]
                    r_old = classes[i][1]
                    
                    # Temporarily remove
                    covered[r_old::m] = False
                    
                    # Find best new r
                    r_new, gain_new = find_best_residue(m, covered)
                    
                    # Apply new r
                    covered[r_new::m] = True
                    uncovered_after_add = np.count_nonzero(~covered)
                    
                    # Check if improvement
                    if uncovered_after_add < current_uncovered:
                        classes[i][1] = r_new
                        current_uncovered = uncovered_after_add
                        cd_improved = True
                    else:
                        # Revert
                        covered[r_new::m] = False
                        covered[r_old::m] = True
                    
                    if current_uncovered == 0:
                        return classes
                
                if not cd_improved:
                    break

    return classes
