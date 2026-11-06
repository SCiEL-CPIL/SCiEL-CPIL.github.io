import numpy as np
import time
import random

def construct():
    N = 51975
    start_time = time.time()
    deadline = start_time + 138.0  # Safety margin before 150s kill

    # Generate divisors of 51975 (odd divisors > 1)
    # 51975 = 3^3 * 5^2 * 7 * 11
    divisors = []
    for d in range(3, N + 1, 2):
        if N % d == 0:
            divisors.append(d)
    
    # Precompute residue indices for each modulus
    # For a given m, residues[r] = np.arange(r, N, m)
    # This avoids repeated slicing in the inner loop
    residue_indices = {}
    for m in divisors:
        residues = []
        for r in range(m):
            residues.append(np.arange(r, N, m))
        residue_indices[m] = residues

    def compute_uncovered(covered):
        return np.count_nonzero(~covered)

    def find_best_residue(m, covered):
        """Find the residue r for modulus m that covers the most uncovered residues."""
        max_count = 0
        best_r = 0
        for r in range(m):
            c = np.count_nonzero(~covered[residue_indices[m][r]])
            if c > max_count:
                max_count = c
                best_r = r
        return best_r, max_count

    def greedy_initialization():
        """Greedy set cover initialization."""
        covered = np.zeros(N, dtype=bool)
        classes = []
        available_moduli = divisors.copy()
        
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
                    c = np.count_nonzero(~covered[residue_indices[m][r]])
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

        return classes, covered

    def coordinate_descent(classes, covered, max_iter=10):
        """Coordinate descent: for each selected m, try to find a better r."""
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter and time.time() < deadline:
            improved = False
            iter_count += 1
            current_uncovered = compute_uncovered(covered)
            
            for i in range(len(classes)):
                if time.time() > deadline:
                    break
                m = classes[i][0]
                r_old = classes[i][1]
                
                # Temporarily remove
                covered[best_r::m] = False  # This is a bug, let me fix
                # Actually, let's be more careful
                covered[classes[i][1]::m] = False
                uncovered_after_removal = compute_uncovered(covered)
                
                # Find best new r
                r_new, gain_new = find_best_residue(m, covered)
                
                # Apply new r
                covered[r_new::m] = True
                uncovered_after_add = compute_uncovered(covered)
                
                # Check if improvement
                if uncovered_after_add < current_uncovered:
                    classes[i][1] = r_new
                    current_uncovered = uncovered_after_add
                    improved = True
                else:
                    # Revert
                    covered[r_new::m] = False
                    covered[classes[i][1]::m] = True  # This is wrong, should be r_old
                    # Let me fix this properly
                    covered[r_old::m] = True
                
                if current_uncovered == 0:
                    return classes, covered

        return classes, covered

    # Let me rewrite more carefully
    def coordinate_descent_v2(classes, covered, max_iter=5):
        """Coordinate descent: for each selected m, try to find a better r."""
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter and time.time() < deadline:
            improved = False
            iter_count += 1
            current_uncovered = compute_uncovered(covered)
            
            for i in range(len(classes)):
                if time.time() > deadline:
                    break
                m = classes[i][0]
                r_old = classes[i][1]
                
                # Temporarily remove the current residue class
                covered[r_old::m] = False
                uncovered_after_removal = compute_uncovered(covered)
                
                # Find best new r for this modulus
                r_new, gain_new = find_best_residue(m, covered)
                
                # Apply new r
                covered[r_new::m] = True
                uncovered_after_add = compute_uncovered(covered)
                
                # Check if improvement
                if uncovered_after_add < current_uncovered:
                    classes[i][1] = r_new
                    current_uncovered = uncovered_after_add
                    improved = True
                else:
                    # Revert to original
                    covered[r_new::m] = False
                    covered[r_old::m] = True
                
                if current_uncovered == 0:
                    return classes, covered

        return classes, covered

    def swap_search(classes, covered, unselected_moduli, max_swaps=20):
        """Try to swap a selected modulus with an unselected one."""
        selected_moduli = {c[0] for c in classes}
        improved = False
        current_uncovered = compute_uncovered(covered)
        
        for _ in range(max_swaps):
            if time.time() > deadline:
                break
            
            best_swap_gain = 0
            best_m_out = None
            best_m_in = None
            best_r_in = None
            best_idx = None
            
            for i, (m_out, r_out) in enumerate(classes):
                if time.time() > deadline:
                    break
                
                # Temporarily remove m_out
                covered[r_out::m_out] = False
                new_uncovered = compute_uncovered(covered)
                lost = new_uncovered - current_uncovered
                
                # Try each unselected modulus
                for m_in in unselected_moduli:
                    r_in, gain_in = find_best_residue(m_in, covered)
                    net_gain = gain_in - lost
                    if net_gain > best_swap_gain:
                        best_swap_gain = net_gain
                        best_m_out = m_out
                        best_m_in = m_in
                        best_r_in = r_in
                        best_idx = i
                
                # Revert
                covered[r_out::m_out] = True
            
            if best_m_out is not None and best_swap_gain > 0:
                # Perform swap
                classes[best_idx][0] = best_m_in
                classes[best_idx][1] = best_r_in
                covered[best_r_in::best_m_in] = True
                
                # Update moduli sets
                selected_moduli.remove(best_m_out)
                selected_moduli.add(best_m_in)
                unselected_moduli.remove(best_m_in)
                unselected_moduli.append(best_m_out)
                
                current_uncovered = compute_uncovered(covered)
                improved = True
                if current_uncovered == 0:
                    return classes, covered, True
            else:
                break

        return classes, covered, improved

    def add_unselected(classes, covered, unselected_moduli):
        """Add unselected moduli if they help."""
        improved = False
        current_uncovered = compute_uncovered(covered)
        
        for m in unselected_moduli[:]:
            if time.time() > deadline:
                break
            r, gain = find_best_residue(m, covered)
            if gain > 0:
                classes.append([m, r])
                covered[r::m] = True
                unselected_moduli.remove(m)
                improved = True
                current_uncovered = compute_uncovered(covered)
                if current_uncovered == 0:
                    return classes, covered, True

        return classes, covered, improved

    # Main algorithm
    best_classes, best_covered = greedy_initialization()
    best_uncovered = compute_uncovered(best_covered)
    
    if best_uncovered == 0:
        return best_classes

    # Coordinate descent
    best_classes, best_covered = coordinate_descent_v2(best_classes, best_covered, max_iter=5)
    best_uncovered = compute_uncovered(best_covered)
    
    if best_uncovered == 0:
        return best_classes

    # Prepare for swap search
    selected_moduli = {c[0] for c in best_classes}
    unselected_moduli = [m for m in divisors if m not in selected_moduli]
    
    # Swap search
    best_classes, best_covered, _ = swap_search(best_classes, best_covered, unselected_moduli, max_swaps=10)
    best_uncovered = compute_uncovered(best_covered)
    
    if best_uncovered == 0:
        return best_classes

    # Add unselected moduli
    best_classes, best_covered, _ = add_unselected(best_classes, best_covered, unselected_moduli)
    best_uncovered = compute_uncovered(best_covered)
    
    if best_uncovered == 0:
        return best_classes

    # Final coordinate descent
    best_classes, best_covered = coordinate_descent_v2(best_classes, best_covered, max_iter=3)
    best_uncovered = compute_uncovered(best_covered)

    return best_classes
