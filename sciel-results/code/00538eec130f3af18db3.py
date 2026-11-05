import math
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    deadline = time.time() + 115  # Safety margin
    best_score = 0.0
    best_solution = []
    
    # Moduli range
    all_moduli = list(range(20, 201))
    
    # Strategy: 
    # 1. Use MILP on subsets of moduli to find optimal residue assignments
    # 2. Use greedy + local search as fallback
    # 3. Combine results
    
    def greedy_construct(moduli_subset, random_seed=None):
        """Greedy construction for a given set of moduli."""
        if random_seed is not None:
            random.seed(random_seed)
        
        selected = []
        for m in moduli_subset:
            forbidden = [False] * m
            for m2, r2 in selected:
                g = math.gcd(m, m2)
                start = r2 % g
                for r in range(start, m, g):
                    forbidden[r] = True
            
            valid_residues = [r for r in range(m) if not forbidden[r]]
            if valid_residues:
                r = random.choice(valid_residues)
                selected.append((m, r))
        
        return selected
    
    def local_search(selected, max_iterations=50):
        """Local search to improve a solution by trying to add or swap residues."""
        if not selected:
            return selected
        
        best = list(selected)
        best_score_val = sum(1.0 / m for m, r in best)
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Try adding each modulus from all_moduli that's not in best
            current_moduli = set(m for m, r in best)
            
            for m in all_moduli:
                if m in current_moduli:
                    continue
                
                # Check if we can add m with some residue
                forbidden = [False] * m
                for m2, r2 in best:
                    g = math.gcd(m, m2)
                    start = r2 % g
                    for r in range(start, m, g):
                        forbidden[r] = True
                
                valid_residues = [r for r in range(m) if not forbidden[r]]
                if valid_residues:
                    # Try adding with a random valid residue
                    r = random.choice(valid_residues)
                    new_solution = best + [(m, r)]
                    new_score = best_score_val + 1.0 / m
                    if new_score > best_score_val:
                        best = new_solution
                        best_score_val = new_score
                        current_moduli.add(m)
                        improved = True
            
            # Try swapping: remove one and add another
            if not improved and len(best) > 1:
                for i in range(len(best)):
                    m_remove, r_remove = best[i]
                    remaining = best[:i] + best[i+1:]
                    rem_score = best_score_val - 1.0 / m_remove
                    
                    for m_add in all_moduli:
                        if m_add in set(m for m, r in remaining):
                            continue
                        
                        forbidden = [False] * m_add
                        for m2, r2 in remaining:
                            g = math.gcd(m_add, m2)
                            start = r2 % g
                            for r in range(start, m_add, g):
                                forbidden[r] = True
                        
                        valid_residues = [r for r in range(m_add) if not forbidden[r]]
                        if valid_residues:
                            r_add = random.choice(valid_residues)
                            new_score = rem_score + 1.0 / m_add
                            if new_score > best_score_val:
                                best = remaining + [(m_add, r_add)]
                                best_score_val = new_score
                                improved = True
                                break
                    if improved:
                        break
            
            if not improved:
                break
        
        return best
    
    def milp_solve(moduli_subset, time_limit=10):
        """Use MILP to solve for optimal residues for a subset of moduli."""
        n_mod = len(moduli_subset)
        if n_mod == 0:
            return []
        
        # Build variable mapping: var_idx[m] = start index for modulus m
        var_idx = {}
        total_vars = 0
        for i, m in enumerate(moduli_subset):
            var_idx[m] = total_vars
            total_vars += m
        
        if total_vars == 0:
            return []
        
        # Objective: maximize sum of (1/m) * x_{m,r}
        c = np.zeros(total_vars)
        for m in moduli_subset:
            start = var_idx[m]
            for r in range(m):
                c[start + r] = -1.0 / m  # Negate for minimization
        
        # Constraints:
        # 1. For each modulus m, at most one residue can be selected
        # 2. For incompatible pairs (m,r) and (m',r'), x_{m,r} + x_{m',r'} <= 1
        
        A_rows = []
        b_rows = []
        
        # Constraint 1: For each m, sum of residues <= 1
        for m in moduli_subset:
            start = var_idx[m]
            row = np.zeros(total_vars)
            row[start:start+m] = 1.0
            A_rows.append(row)
            b_rows.append(1.0)
        
        # Constraint 2: Pairwise incompatibility
        for i in range(n_mod):
            m1 = moduli_subset[i]
            for j in range(i+1, n_mod):
                m2 = moduli_subset[j]
                g = math.gcd(m1, m2)
                
                # For each residue r1 of m1 and r2 of m2, if r1 % g == r2 % g, they're incompatible
                start1 = var_idx[m1]
                start2 = var_idx[m2]
                
                for r1 in range(m1):
                    r1_mod_g = r1 % g
                    for r2 in range(m2):
                        if r2 % g == r1_mod_g:
                            row = np.zeros(total_vars)
                            row[start1 + r1] = 1.0
                            row[start2 + r2] = 1.0
                            A_rows.append(row)
                            b_rows.append(1.0)
        
        if not A_rows:
            return []
        
        A = np.array(A_rows)
        b = np.array(b_rows)
        
        constraints = LinearConstraint(A, -np.inf, b)
        bounds = Bounds(0, 1)
        
        try:
            result = milp(c=c, constraints=constraints, bounds=bounds,
                          integrality=np.ones(total_vars),
                          options={'time_limit': time_limit})
            
            if result.status == 0 and result.x is not None:
                solution = []
                for m in moduli_subset:
                    start = var_idx[m]
                    for r in range(m):
                        if result.x[start + r] > 0.5:
                            solution.append((m, r))
                            break
                return solution
        except Exception:
            pass
        
        return []
    
    # Phase 1: Greedy with multiple restarts
    for restart in range(500):
        if time.time() > deadline - 30:
            break
        
        selected = greedy_construct(all_moduli, random_seed=restart)
        score = sum(1.0 / m for m, r in selected)
        
        if score > best_score:
            best_score = score
            best_solution = [[m, r] for m, r in selected]
    
    # Phase 2: Local search on best solution
    if best_solution and time.time() < deadline - 20:
        improved = local_search([(m, r) for m, r in best_solution], max_iterations=30)
        score = sum(1.0 / m for m, r in improved)
        if score > best_score:
            best_score = score
            best_solution = [[m, r] for m, r in improved]
    
    # Phase 3: MILP on promising subsets (smaller moduli have higher weight)
    if time.time() < deadline - 15:
        # Try MILP on subsets of smaller moduli
        subset_sizes = [30, 40, 50]
        for size in subset_sizes:
            if time.time() > deadline - 10:
                break
            
            # Focus on smaller moduli which have higher 1/m
            subset = all_moduli[:size]
            solution = milp_solve(subset, time_limit=8)
            
            if solution:
                score = sum(1.0 / m for m, r in solution)
                if score > best_score:
                    best_score = score
                    best_solution = [[m, r] for m, r in solution]
    
    # Phase 4: Final local search polish
    if best_solution and time.time() < deadline - 5:
        improved = local_search([(m, r) for m, r in best_solution], max_iterations=20)
        score = sum(1.0 / m for m, r in improved)
        if score > best_score:
            best_score = score
            best_solution = [[m, r] for m, r in improved]
    
    return best_solution
