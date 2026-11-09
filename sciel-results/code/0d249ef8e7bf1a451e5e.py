# EVOLVE-BLOCK-START
"""
Improved approach: Greedy + Aggressive Local Search with Multi-Move and Residue Optimization.
Key improvements:
1. 2-opt moves (remove 1, add up to 2) to escape local optima.
2. Residue re-optimization to maximize slack and enable future additions.
3. Smarter greedy initialization with backtracking.
4. Time-budgeted search with safety margins.
"""
import time
import math
from math import gcd
from fractions import Fraction
from itertools import combinations
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Moduli range
    moduli = list(range(20, 201))
    n_mod = len(moduli)
    mod_idx = {m: i for i, m in enumerate(moduli)}
    
    # Precompute gcd table
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i, n_mod):
            g = gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute weights
    weights = {m: 1.0 / m for m in moduli}
    
    def is_valid(classes, m, r):
        """Check if class (m, r) is disjoint from all classes in the set."""
        mi = mod_idx[m]
        for (m2, r2) in classes:
            mj = mod_idx[m2]
            g = gcd_table[mi][mj]
            if (r - r2) % g == 0:
                return False
        return True
    
    def find_valid_residue(classes, m):
        """Find any valid residue for m, or None."""
        for r in range(m):
            if is_valid(classes, m, r):
                return r
        return None
    
    def find_best_residue_slack(classes, m):
        """
        Find residue r for m that maximizes the minimum 'distance' to conflict 
        with existing classes. This helps in packing.
        """
        mi = mod_idx[m]
        best_r = -1
        best_slack = -1
        for r in range(m):
            if not is_valid(classes, m, r):
                continue
            min_dist = m
            for (m2, r2) in classes:
                mj = mod_idx[m2]
                g = gcd_table[mi][mj]
                if g > 1:
                    diff = (r - r2) % g
                    dist = min(diff, g - diff)
                    min_dist = min(min_dist, dist)
            if min_dist > best_slack:
                best_slack = min_dist
                best_r = r
        return best_r
    
    def score(classes):
        """Compute sum of 1/m for the set of classes."""
        return sum(weights[m] for m, r in classes)
    
    def greedy_construct(order):
        """Greedy construction: add classes in the given order if possible."""
        classes = []
        used_moduli = set()
        for m in order:
            if m in used_moduli:
                continue
            r = find_best_residue_slack(classes, m)
            if r is not None and r != -1:
                classes.append([m, r])
                used_moduli.add(m)
        return classes
    
    def try_add_modulus(classes, m_new):
        """Try to add modulus m_new to classes. Returns new classes or None."""
        r_new = find_valid_residue(classes, m_new)
        if r_new is not None:
            return classes + [[m_new, r_new]]
        return None
    
    def shift_to_add(classes, m_new, max_shifts=2):
        """
        Try to add m_new by shifting existing residues.
        Returns new classes if successful, None otherwise.
        """
        # Find the residue of m_new with minimum conflicts
        best_r_candidate = -1
        min_conflicts = m_new
        blockers = []
        
        for r_new in range(m_new):
            confs = []
            for idx, (m_old, r_old) in enumerate(classes):
                g = gcd_table[mod_idx[m_new]][mod_idx[m_old]]
                if (r_new - r_old) % g == 0:
                    confs.append(idx)
            if len(confs) < min_conflicts:
                min_conflicts = len(confs)
                best_r_candidate = r_new
                blockers = confs
                if min_conflicts == 0:
                    return classes + [[m_new, r_new]]
                if min_conflicts <= max_shifts:
                    break
        
        if min_conflicts == 0:
            return classes + [[m_new, best_r_candidate]]
        if min_conflicts > max_shifts:
            return None
        
        # Try to shift the blockers
        if min_conflicts == 1:
            idx_b = blockers[0]
            m_b, r_b_old = classes[idx_b]
            other_classes = [classes[i] for i in range(len(classes)) if i != idx_b]
            
            for r_b_new in range(m_b):
                # Check against m_new
                g_nm = gcd_table[mod_idx[m_new]][mod_idx[m_b]]
                if (best_r_candidate - r_b_new) % g_nm == 0:
                    continue
                # Check against others
                valid = True
                for m_o, r_o in other_classes:
                    g_bo = gcd_table[mod_idx[m_b]][mod_idx[m_o]]
                    if (r_b_new - r_o) % g_bo == 0:
                        valid = False
                        break
                if valid:
                    new_classes = [list(c) for c in classes]
                    new_classes[idx_b][1] = r_b_new
                    new_classes.append([m_new, best_r_candidate])
                    return new_classes
        elif min_conflicts == 2:
            idx_b1, idx_b2 = blockers
            m_b1, _ = classes[idx_b1]
            m_b2, _ = classes[idx_b2]
            other_classes = [classes[i] for i in range(len(classes)) if i not in blockers]
            
            # Limit search for performance
            max_iter = min(m_b1 * m_b2, 5000)
            count = 0
            for r_b1_new in range(m_b1):
                g_nm1 = gcd_table[mod_idx[m_new]][mod_idx[m_b1]]
                if (best_r_candidate - r_b1_new) % g_nm1 == 0:
                    continue
                for r_b2_new in range(m_b2):
                    count += 1
                    if count > max_iter:
                        break
                    
                    g_b1b2 = gcd_table[mod_idx[m_b1]][mod_idx[m_b2]]
                    if (r_b1_new - r_b2_new) % g_b1b2 == 0:
                        continue
                    g_nm2 = gcd_table[mod_idx[m_new]][mod_idx[m_b2]]
                    if (best_r_candidate - r_b2_new) % g_nm2 == 0:
                        continue
                    
                    valid = True
                    for m_o, r_o in other_classes:
                        g_bo = gcd_table[mod_idx[m_b1]][mod_idx[m_o]]
                        if (r_b1_new - r_o) % g_bo == 0:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    for m_o, r_o in other_classes:
                        g_bo = gcd_table[mod_idx[m_b2]][mod_idx[m_o]]
                        if (r_b2_new - r_o) % g_bo == 0:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    new_classes = [list(c) for c in classes]
                    new_classes[idx_b1][1] = r_b1_new
                    new_classes[idx_b2][1] = r_b2_new
                    new_classes.append([m_new, best_r_candidate])
                    return new_classes
                if count > max_iter:
                    break
        return None
    
    def residue_polish(classes, max_iter=100):
        """
        Polish residues to maximize slack, enabling future additions.
        This is a neutral move (score unchanged) but improves packing.
        """
        best_classes = [list(c) for c in classes]
        best_slack = float('inf')
        
        for _ in range(max_iter):
            if time.time() - start_time > deadline - 5:
                break
            
            improved = False
            for idx in range(len(best_classes)):
                m, r_old = best_classes[idx]
                other_classes = [best_classes[i] for i in range(len(best_classes)) if i != idx]
                
                # Find best residue for m given others
                best_r = r_old
                best_min_dist = -1
                for r in range(m):
                    if not is_valid(other_classes, m, r):
                        continue
                    min_dist = m
                    for m2, r2 in other_classes:
                        g = gcd_table[mod_idx[m]][mod_idx[m2]]
                        if g > 1:
                            diff = (r - r2) % g
                            dist = min(diff, g - diff)
                            min_dist = min(min_dist, dist)
                    if min_dist > best_min_dist:
                        best_min_dist = min_dist
                        best_r = r
                
                if best_r != r_old:
                    best_classes[idx][1] = best_r
                    improved = True
            
            if not improved:
                break
        
        return best_classes
    
    def local_search(classes, max_iter=2000):
        """
        Improve by:
        1. Adding missing moduli.
        2. 2-opt moves: remove 1, add up to 2.
        3. Shift to add.
        4. Residue polish.
        """
        best_classes = [list(c) for c in classes]
        best_score = score(best_classes)
        used_moduli = set(m for m, r in best_classes)
        
        missing_moduli = [m for m in sorted(moduli, key=weights.get, reverse=True) if m not in used_moduli]
        
        stuck_count = 0
        for iteration in range(max_iter):
            if time.time() - start_time > deadline - 5:
                break
            
            improved = False
            new_classes = [list(c) for c in best_classes]
            new_used = set(m for m, r in new_classes)
            
            # --- Move 1: Try to add a missing modulus directly ---
            for m_new in missing_moduli[:30]:
                if m_new in new_used:
                    continue
                result = try_add_modulus(new_classes, m_new)
                if result is not None:
                    new_classes = result
                    new_used.add(m_new)
                    improved = True
                    break
            
            # --- Move 2: Shift to add ---
            if not improved:
                for m_new in missing_moduli[:15]:
                    if m_new in new_used:
                        continue
                    result = shift_to_add(new_classes, m_new, max_shifts=2)
                    if result is not None:
                        new_classes = result
                        new_used.add(m_new)
                        improved = True
                        break
            
            # --- Move 3: 2-opt: remove one low-value, add up to two high-value ---
            if not improved:
                # Sort by weight ascending
                sorted_classes = sorted(enumerate(best_classes), key=lambda x: weights[x[1][0]])
                for idx_remove, (m_old, r_old) in sorted_classes[:8]:
                    temp_classes = [c for i, c in enumerate(best_classes) if i != idx_remove]
                    temp_used = set(m for m, r in temp_classes)
                    
                    # Try adding two missing moduli
                    added = []
                    for m_new in missing_moduli:
                        if m_new in temp_used or m_new in added:
                            continue
                        result = try_add_modulus(temp_classes + added, m_new)
                        if result is not None:
                            added.append([m_new, result[-1][1]])
                            temp_used.add(m_new)
                            if len(added) == 2:
                                break
                    
                    if len(added) >= 1:
                        new_score_candidate = score(temp_classes + added)
                        if new_score_candidate > best_score + 1e-9:
                            new_classes = temp_classes + added
                            new_used = set(m for m, r in new_classes)
                            improved = True
                            break
            
            if improved:
                new_score = score(new_classes)
                if new_score > best_score:
                    best_classes = new_classes
                    best_score = new_score
                    used_moduli = new_used
                    missing_moduli = [m for m in sorted(moduli, key=weights.get, reverse=True) if m not in used_moduli]
                    stuck_count = 0
                else:
                    new_classes = [list(c) for c in best_classes]
                    new_used = set(m for m, r in new_classes)
            else:
                stuck_count += 1
                # Periodic residue polish
                if stuck_count % 20 == 0:
                    polished = residue_polish(best_classes, max_iter=30)
                    ps = score(polished)
                    if ps >= best_score - 1e-9:
                        best_classes = polished
                        best_score = ps
                if stuck_count > 200:
                    break
        
        return best_classes
    
    # --- Main Execution ---
    
    # Start with greedy solution
    order = sorted(moduli, key=lambda m: weights[m], reverse=True)
    initial_classes = greedy_construct(order)
    
    best_classes = initial_classes
    best_score = score(best_classes)
    
    # Local search
    improved_classes = local_search(best_classes, max_iter=1500)
    improved_score = score(improved_classes)
    if improved_score > best_score:
        best_classes = improved_classes
        best_score = improved_score
    
    # Random restarts
    random.seed(42)
    for restart in range(12):
        if time.time() - start_time > deadline - 12:
            break
        
        remaining = list(moduli)
        order = []
        while remaining:
            total_w = sum(1.0/m for m in remaining)
            r = random.random() * total_w
            cum = 0
            chosen_idx = 0
            for i, m in enumerate(remaining):
                cum += 1.0/m
                if r <= cum:
                    chosen_idx = i
                    break
            order.append(remaining[chosen_idx])
            remaining.pop(chosen_idx)
        
        classes = greedy_construct(order)
        classes = local_search(classes, max_iter=800)
        s = score(classes)
        if s > best_score:
            best_score = s
            best_classes = classes
    
    # Final residue polish
    if time.time() - start_time < deadline - 8:
        polished = residue_polish(best_classes, max_iter=50)
        ps = score(polished)
        if ps >= best_score - 1e-9:
            best_classes = polished
            best_score = ps
    
    # One more local search after polish
    if time.time() - start_time < deadline - 5:
        final = local_search(best_classes, max_iter=500)
        fs = score(final)
        if fs > best_score:
            best_classes = final
            best_score = fs
    
    return best_classes

# EVOLVE-BLOCK-END
