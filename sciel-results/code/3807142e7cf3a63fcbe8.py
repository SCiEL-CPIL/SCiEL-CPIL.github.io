import numpy as np
import time
import random

def construct():
    N = 51975
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Generate divisors of 51975 (odd divisors > 1)
    divisors = []
    for d in range(3, N + 1, 2):
        if N % d == 0:
            divisors.append(d)
    
    # coverage_count[i] = number of classes covering residue i
    covered_count = np.zeros(N, dtype=np.int32)
    uncovered = N
    classes = []
    selected_moduli = set()
    available_moduli = divisors.copy()
    
    # Helper to update uncovered count and covered_count when adding a class
    def add_class(m, r):
        nonlocal uncovered
        # Count how many were 0
        gains = np.sum(covered_count[r::m] == 0)
        covered_count[r::m] += 1
        uncovered -= gains
        classes.append([m, r])
        selected_moduli.add(m)
        if r in [c[1] for c in classes if c[0] == m]: # Just in case, but we manage lists
            pass
        available_moduli.remove(m)
        
    # Helper to remove a class
    def remove_class(m, r):
        nonlocal uncovered
        # Count how many were 1 (become 0) and how many were 0 (become -1, invalid)
        # Actually, we just decrement.
        # Uncovered increases by count of elements that become 0.
        # Elements that become 0 are those that were 1.
        # Elements that were 0 become -1 (should not happen if valid).
        # Let's just compute the change.
        old_slice = covered_count[r::m]
        # New slice is old_slice - 1
        # New zeros are old_slice == 1
        # Old zeros are old_slice == 0
        delta = np.sum(old_slice == 1) - np.sum(old_slice == 0)
        covered_count[r::m] -= 1
        uncovered += delta
        classes.remove([m, r])
        selected_moduli.remove(m)
        available_moduli.append(m)

    # 1. Greedy Initialization
    while available_moduli:
        if time.time() > deadline:
            break
        best_gain = 0
        best_m = None
        best_r = None
        
        for m in available_moduli:
            # Find best r for m
            # We can vectorize this slightly, but loop is fine for N=51975
            for r in range(m):
                gain = np.sum(covered_count[r::m] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_m = m
                    best_r = r
        if best_m is None or best_gain == 0:
            break
        add_class(best_m, best_r)
        if uncovered == 0:
            return classes

    # 2. Simulated Annealing
    best_uncovered = uncovered
    best_classes = [c[:] for c in classes]
    best_covered_count = covered_count.copy()
    
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    # Precompute indices for faster access?
    # Not really needed with numpy slices.
    
    while time.time() < deadline and T > min_T:
        # Pick a random move
        move_type = random.choice(['change_res', 'swap', 'add', 'remove'])
        
        delta_U = 0
        move_applied = False
        move_info = None
        
        if move_type == 'change_res' and classes:
            idx = random.randint(0, len(classes) - 1)
            m, r_old = classes[idx]
            r_new = random.randint(0, m - 1)
            if r_new != r_old:
                # Remove old
                old_slice = covered_count[r_old::m]
                delta_rem = np.sum(old_slice == 1) - np.sum(old_slice == 0)
                # Add new
                new_slice = covered_count[r_new::m]
                delta_add = -np.sum(new_slice == 0)
                
                delta_U = delta_rem + delta_add
                move_info = ('change_res', idx, m, r_old, r_new)
                move_applied = True
                
        elif move_type == 'swap' and classes and available_moduli:
            idx = random.randint(0, len(classes) - 1)
            m_out, r_out = classes[idx]
            m_in = random.choice(available_moduli)
            r_in = random.randint(0, m_in - 1)
            
            # Remove out
            old_slice_out = covered_count[r_out::m_out]
            delta_rem = np.sum(old_slice_out == 1) - np.sum(old_slice_out == 0)
            # Add in
            new_slice_in = covered_count[r_in::m_in]
            delta_add = -np.sum(new_slice_in == 0)
            
            delta_U = delta_rem + delta_add
            move_info = ('swap', idx, m_out, r_out, m_in, r_in)
            move_applied = True

        elif move_type == 'add' and available_moduli:
            m = random.choice(available_moduli)
            r = random.randint(0, m - 1)
            new_slice = covered_count[r::m]
            delta_U = -np.sum(new_slice == 0)
            move_info = ('add', m, r)
            move_applied = True

        elif move_type == 'remove' and classes:
            idx = random.randint(0, len(classes) - 1)
            m, r = classes[idx]
            old_slice = covered_count[r::m]
            delta_U = np.sum(old_slice == 1) - np.sum(old_slice == 0)
            move_info = ('remove', idx, m, r)
            move_applied = True

        if not move_applied:
            continue

        # Acceptance criterion
        if delta_U < 0 or random.random() < np.exp(-delta_U / T):
            # Apply move
            if move_info[0] == 'change_res':
                _, idx, m, r_old, r_new = move_info
                covered_count[r_old::m] -= 1
                covered_count[r_new::m] += 1
                classes[idx][1] = r_new
            elif move_info[0] == 'swap':
                _, idx, m_out, r_out, m_in, r_in = move_info
                covered_count[r_out::m_out] -= 1
                covered_count[r_in::m_in] += 1
                classes[idx][0] = m_in
                classes[idx][1] = r_in
                selected_moduli.remove(m_out)
                selected_moduli.add(m_in)
                available_moduli.remove(m_in)
                available_moduli.append(m_out)
            elif move_info[0] == 'add':
                _, m, r = move_info
                covered_count[r::m] += 1
                classes.append([m, r])
                selected_moduli.add(m)
                available_moduli.remove(m)
            elif move_info[0] == 'remove':
                _, idx, m, r = move_info
                covered_count[r::m] -= 1
                classes.pop(idx)
                selected_moduli.remove(m)
                available_moduli.append(m)
            
            uncovered += delta_U
            
            if uncovered < best_uncovered:
                best_uncovered = uncovered
                best_classes = [c[:] for c in classes]
                best_covered_count = covered_count.copy()
                if best_uncovered == 0:
                    return best_classes

        T *= cooling_rate

    # Restore best solution
    # The function returns classes, so we return best_classes
    return best_classes
