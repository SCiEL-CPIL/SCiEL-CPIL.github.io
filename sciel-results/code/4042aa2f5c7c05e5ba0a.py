import time
import numpy as np
import random
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 150s kill
    
    # Precompute all odd divisors of 51975 > 1
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    divisors.sort()  # ascending order
    n_div = len(divisors)
    
    # Precompute residue index arrays for each divisor
    # For divisor m, residues[r] = array of positions congruent to r mod m
    residue_positions = {}
    for m in divisors:
        residue_positions[m] = [np.arange(r, base, m) for r in range(m)]
    
    def score_solution(classes):
        """Count uncovered residues for a list of [m, r] classes."""
        if not classes:
            return base
        covered = np.zeros(base, dtype=np.uint8)
        for m, r in classes:
            covered[residue_positions[m][r]] = 1
        return int(np.sum(covered == 0))
    
    def compute_covered(classes):
        """Compute covered array from classes."""
        if not classes:
            return np.zeros(base, dtype=np.uint8)
        covered = np.zeros(base, dtype=np.uint8)
        for m, r in classes:
            covered[residue_positions[m][r]] = 1
        return covered
    
    def greedy_pass(div_order, base, deadline, covered_init=None):
        """One greedy pass: pick best (m,r) at each step from div_order."""
        covered = np.zeros(base, dtype=np.uint8) if covered_init is None else covered_init.copy()
        classes = []
        available = set(div_order)
        
        while available:
            if time.time() > deadline - 5.0:
                break
            
            best_gain = -1
            best_m = None
            best_r = None
            
            for m in div_order:
                if m not in available:
                    continue
                
                # Max possible coverage for this modulus
                max_possible = (base + m - 1) // m
                if max_possible <= best_gain:
                    continue
                
                # Find best residue for this modulus using vectorized approach
                best_r_for_m = 0
                best_count_for_m = -1
                for r in range(m):
                    cnt = int(np.sum(covered[residue_positions[m][r]] == 0))
                    if cnt > best_count_for_m:
                        best_count_for_m = cnt
                        best_r_for_m = r
                
                if best_count_for_m > best_gain:
                    best_gain = best_count_for_m
                    best_m = m
                    best_r = best_r_for_m
            
            if best_m is None or best_gain <= 0:
                break
            
            classes.append([best_m, best_r])
            covered[residue_positions[best_m][best_r]] = 1
            available.remove(best_m)
        
        return classes, covered
    
    def local_search_full(classes, base, deadline):
        """Intensive local search: try all residues for small moduli, random for large."""
        if not classes:
            return classes, base
        
        best_classes = [c[:] for c in classes]
        best_score = score_solution(best_classes)
        rng = random.Random(7)
        
        improved = True
        iterations = 0
        while improved and time.time() < deadline - 5.0:
            improved = False
            iterations += 1
            
            for idx in range(len(best_classes)):
                if time.time() > deadline - 5.0:
                    break
                m, r = best_classes[idx]
                
                if m <= 200:
                    # Try all residues
                    best_r = r
                    best_cnt = -1
                    for r2 in range(m):
                        if r2 == r:
                            continue
                        # Count new coverage
                        pos = residue_positions[m][r2]
                        old_pos = residue_positions[m][r]
                        # New positions covered: in pos but not in old_pos and not covered by others
                        # Simplified: compute score delta
                        # Remove class idx, add with r2
                        covered_without = np.zeros(base, dtype=np.uint8)
                        for j, (om, or_) in enumerate(best_classes):
                            if j != idx:
                                covered_without[residue_positions[om][or_]] = 1
                        covered_new = covered_without.copy()
                        covered_new[pos] = 1
                        new_score = int(np.sum(covered_new == 0))
                        if new_score < best_cnt:
                            best_cnt = new_score
                            best_r = r2
                    if best_r != r:
                        best_classes[idx][1] = best_r
                        best_score = best_cnt
                        improved = True
                        break
                else:
                    # Try random residues
                    for _ in range(10):
                        r2 = rng.randint(0, m - 1)
                        if r2 == r:
                            continue
                        best_classes[idx][1] = r2
                        new_score = score_solution(best_classes)
                        if new_score < best_score:
                            best_score = new_score
                            improved = True
                            break
                        else:
                            best_classes[idx][1] = r
                    if improved:
                        break
        
        return best_classes, best_score
    
    def milp_residue_optimize(classes, base, deadline):
        """Use MILP to optimize residues for a subset of small moduli."""
        if not classes:
            return classes, base
        
        # Only work with small moduli where MILP is tractable
        small_classes = [(i, m, r) for i, (m, r) in enumerate(classes) if m <= 30]
        if len(small_classes) < 2:
            return classes, score_solution(classes)
        
        # Build MILP: for each small modulus, choose one residue
        # Maximize total coverage
        n_vars = sum(m for _, m, _ in small_classes)
        n_constraints = len(small_classes)
        
        # Objective: maximize coverage of each position
        # c[p] = sum over (i, r) where p in residue_positions[m_i][r] of x[i, r]
        # We want to maximize sum_p min(1, c[p]) which is non-linear
        # Approximate: maximize sum_p c[p] (counts multiple coverage)
        # This is a linear proxy
        
        c_obj = np.zeros(n_vars)
        var_idx = 0
        for ci, (orig_idx, m, r) in enumerate(small_classes):
            for r2 in range(m):
                pos = residue_positions[m][r2]
                c_obj[var_idx:var_idx+len(pos)] += 1.0  # wrong dimension
                var_idx += 1
        
        # Actually, objective is per-variable: each variable contributes to coverage
        # c_obj[j] = number of positions covered by variable j
        c_obj = np.zeros(n_vars)
        var_idx = 0
        for ci, (orig_idx, m, r) in enumerate(small_classes):
            for r2 in range(m):
                c_obj[var_idx] = len(residue_positions[m][r2])
                var_idx += 1
        
        # Constraints: for each modulus, exactly one residue selected
        A_rows = []
        for ci, (orig_idx, m, r) in enumerate(small_classes):
            row = np.zeros(n_vars)
            var_idx = 0
            for cj, (oi2, m2, r2) in enumerate(small_classes):
                if cj == ci:
                    row[var_idx:var_idx+m] = 1.0
                var_idx += m
            A_rows.append(row)
        
        A = np.array(A_rows)
        lb = np.ones(n_constraints)
        ub = np.ones(n_constraints)
        
        try:
            res = milp(
                c=-c_obj,  # minimize negative = maximize
                constraints=LinearConstraint(A, lb, ub),
                integrality=np.ones(n_vars),
                bounds=Bounds(0, 1),
                options={'time_limit': max(1, deadline - time.time() - 10)}
            )
            if res.status == 0 or res.status == 2:  # optimal or time limit with solution
                if res.x is not None and np.any(res.x > 0.5):
                    new_classes = [c[:] for c in classes]
                    var_idx = 0
                    for ci, (orig_idx, m, r) in enumerate(small_classes):
                        for r2 in range(m):
                            if res.x[var_idx] > 0.5:
                                new_classes[orig_idx][1] = r2
                                break
                            var_idx += 1
                    new_score = score_solution(new_classes)
                    old_score = score_solution(classes)
                    if new_score < old_score:
                        return new_classes, new_score
        except Exception:
            pass
        
        return classes, score_solution(classes)
    
    def local_search(classes, base, deadline):
        """Local search: try changing residues and swapping classes."""
        if not classes:
            return classes, base
        
        best_classes = [c[:] for c in classes]
        best_score = score_solution(best_classes)
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # Try changing residue for each class
            for idx in range(len(best_classes)):
                if time.time() > deadline - 5.0:
                    break
                m, r = best_classes[idx]
                # Try a few alternative residues
                candidates = [0, m//2, m-1, (r+1)%m, (r-1)%m]
                if m <= 20:
                    candidates = list(range(m))
                for r2 in candidates:
                    if r2 == r:
                        continue
                    best_classes[idx][1] = r2
                    new_score = score_solution(best_classes)
                    if new_score < best_score:
                        best_score = new_score
                        improved = True
                        break
                    else:
                        best_classes[idx][1] = r
                if improved:
                    break
            
            # Try adding a new class if we have unused divisors
            if not improved:
                used_moduli = set(c[0] for c in best_classes)
                for m in divisors:
                    if m in used_moduli:
                        continue
                    if time.time() > deadline - 5.0:
                        break
                    # Try all residues for this new modulus
                    best_r = None
                    best_new_score = best_score
                    for r in range(m):
                        if m > 100 and r > 20:
                            break
                        best_classes.append([m, r])
                        new_score = score_solution(best_classes)
                        best_classes.pop()
                        if new_score < best_new_score:
                            best_new_score = new_score
                            best_r = r
                    if best_r is not None:
                        best_classes.append([m, best_r])
                        best_score = best_new_score
                        improved = True
                        break
            
            # Try removing a class if it doesn't help
            if not improved and len(best_classes) > 5:
                for idx in range(len(best_classes)):
                    if time.time() > deadline - 5.0:
                        break
                    saved = best_classes.pop(idx)
                    new_score = score_solution(best_classes)
                    if new_score <= best_score:
                        best_score = new_score
                        improved = True
                    else:
                        best_classes.insert(idx, saved)
                    if improved:
                        break
        
        return best_classes, best_score
    
    best_score = base
    best_classes = []
    
    # Strategy 1: Greedy with ascending divisors
    if time.time() < deadline - 40.0:
        classes, covered = greedy_pass(divisors, base, deadline)
        score = int(np.sum(covered == 0))
        if score < best_score:
            best_score = score
            best_classes = classes
    
    # Strategy 2: Greedy with descending divisors
    if time.time() < deadline - 40.0:
        div_desc = list(reversed(divisors))
        classes, covered = greedy_pass(div_desc, base, deadline)
        score = int(np.sum(covered == 0))
        if score < best_score:
            best_score = score
            best_classes = classes
    
    # Strategy 3: Greedy with medium-first order
    if time.time() < deadline - 40.0:
        med_order = sorted(divisors, key=lambda m: abs(m - 100))
        classes, covered = greedy_pass(med_order, base, deadline)
        score = int(np.sum(covered == 0))
        if score < best_score:
            best_score = score
            best_classes = classes
    
    # Strategy 4: Randomized greedy attempts
    rng = random.Random(42)
    attempts = 0
    while time.time() < deadline - 30.0 and attempts < 8:
        attempts += 1
        div_rand = list(divisors)
        rng.shuffle(div_rand)
        classes, covered = greedy_pass(div_rand, base, deadline)
        score = int(np.sum(covered == 0))
        if score < best_score:
            best_score = score
            best_classes = classes
    
    # MILP refinement on small moduli
    if best_classes and time.time() < deadline - 25.0:
        best_classes, best_score = milp_residue_optimize(best_classes, base, deadline)
    
    # Full local search
    if best_classes and time.time() < deadline - 15.0:
        best_classes, best_score = local_search_full(best_classes, base, deadline)
    
    # MILP again after local search
    if best_classes and time.time() < deadline - 10.0:
        best_classes, best_score = milp_residue_optimize(best_classes, base, deadline)
    
    # Final local search
    if best_classes and time.time() < deadline - 5.0:
        best_classes, best_score = local_search(best_classes, base, deadline)
    
    return best_classes if best_classes else []
