"""
3-SAT solver using Spectral Max-Cut initialization + Efficient WalkSAT.

Structural Change:
- Uses a spectral bisection heuristic derived from a conflict graph (Max-Cut formulation)
  to generate high-quality initial assignments.
- Retains the efficient incremental clause tracking from the baseline WalkSAT.
"""
import random
import time
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["instance_clauses"] if "instance_clauses" in instance else instance["clauses"]
    
    start_time = time.time()
    deadline = start_time + 290  # Safety margin

    # 1. Preprocess: remove tautologies, dedupe
    seen = set()
    filtered_clauses = []
    for c in clauses:
        lits = set(c)
        # Check for tautology
        if any(-l in lits for l in lits):
            continue
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            filtered_clauses.append(c)
    clauses = filtered_clauses
    clause_count = len(clauses)
    
    if clause_count == 0:
        return [True] * n

    # 2. Build indices for efficiency
    var_pos = [[] for _ in range(n)]
    var_neg = [[] for _ in range(n)]
    for i, c in enumerate(clauses):
        for l in c:
            idx = abs(l) - 1
            if l > 0:
                var_pos[idx].append(i)
            else:
                var_neg[idx].append(i)

    # 3. Spectral Initialization (Max-Cut formulation)
    # Construct a weighted graph where edges represent "pressure" to be different (cut) or same.
    # If x_i and x_j appear in a clause with SAME sign, we prefer them to be DIFFERENT 
    # (to avoid 0,0 which is bad for x_i OR x_j). So weight +1 (cut).
    # If x_i and -x_j appear in a clause, we prefer them to be SAME? 
    # Clause (x_i OR -x_j). Bad if x_i=0, -x_j=0 => x_i=0, x_j=1.
    # We want to avoid (0,1). In Max-Cut (different sets), (0,1) is a cut.
    # So we want to AVOID cutting. Weight -1? 
    # Let's use W[i][j] += 1 for same-sign co-occurrence, -1 for opposite.
    
    W = np.zeros((n, n))
    for c in clauses:
        # For 3-SAT, we consider all pairs in the clause
        for k in range(len(c)):
            for m in range(k+1, len(c)):
                l1, l2 = c[k], c[m]
                v1, v2 = abs(l1)-1, abs(l2)-1
                if v1 == v2: continue
                sign1 = 1 if l1 > 0 else -1
                sign2 = 1 if l2 > 0 else -1
                
                if sign1 == sign2:
                    # Same sign: e.g., x, y. Bad: 0,0. We want 1,0 or 0,1 or 1,1.
                    # We want to "cut" the edge (different values) to avoid 0,0? 
                    # If we cut, one is 1, one is 0. This satisfies.
                    # If we don't cut, both 0 (bad) or both 1 (good).
                    # Generally, cutting is safer for "OR" of positives.
                    W[v1][v2] += 1
                    W[v2][v1] += 1
                else:
                    # Opposite sign: e.g., x, -y. Bad: x=0, -y=0 => x=0, y=1.
                    # We want to avoid (0,1).
                    # In Max-Cut, (0,1) is a cut (different sets).
                    # So we want to NOT cut.
                    W[v1][v2] -= 1
                    W[v2][v1] -= 1

    # Compute Laplacian L = D - W
    # Note: W can have negative weights. Standard Laplacian definition holds for symmetric W.
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    
    # Find eigenvectors
    # We want the Fiedler vector (2nd smallest eigenvalue)
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Sort by eigenvalue
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # The smallest eigenvalue is 0 (usually). The next is the Fiedler vector.
        # If the graph is disconnected, there might be multiple 0s.
        # We pick the eigenvector corresponding to the first non-zero eigenvalue, 
        # or just the 2nd smallest if we assume connectivity.
        # For robustness, if eigvals[0] is close to 0, use eigvecs[:, 1].
        if eigvals[1] < 1e-6:
             # Disconnected, try next
             fiedler_idx = 2
        else:
             fiedler_idx = 1
             
        fiedler_vec = eigvecs[:, fiedler_idx]
        
        # Initialize assignment based on Fiedler vector
        # x_i = True if fiedler_vec[i] > 0 else False
        spectral_assignment = [bool(v) for v in fiedler_vec]
        
    except Exception:
        spectral_assignment = [random.random() < 0.5 for _ in range(n)]

    # Helper: Count unsat
    def count_unsat(assignment):
        count = 0
        for c in clauses:
            sat = False
            for l in c:
                idx = abs(l) - 1
                if assignment[idx] == (l > 0):
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    # Helper: Check if single clause is satisfied
    def satisfied(assignment, c):
        for l in c:
            idx = abs(l) - 1
            if assignment[idx] == (l > 0):
                return True
        return False

    rng = random.Random(42)
    best_assignment = None
    best_unsat_count = clause_count + 1
    
    # Phase 1: WalkSAT with Spectral Start
    max_restarts = 3000
    steps_per_restart = 10000
    noise = 0.25

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Initialization Strategy
        if restart == 0:
            assignment = spectral_assignment[:]
        elif restart < 5:
            # Perturb spectral assignment
            assignment = spectral_assignment[:]
            for _ in range(n // 4 + 1):
                assignment[rng.randint(0, n - 1)] = not assignment[rng.randint(0, n - 1)]
        else:
            if best_assignment is not None and rng.random() < 0.7:
                assignment = best_assignment[:]
                for _ in range(n // 4 + 1):
                    assignment[rng.randint(0, n - 1)] = not assignment[rng.randint(0, n - 1)]
            else:
                assignment = [rng.random() < 0.5 for _ in range(n)]

        # Build initial sat_count and unsat list
        sat_count = [0] * clause_count
        unsat_clauses = []
        for i, c in enumerate(clauses):
            cnt = 0
            for l in c:
                idx = abs(l) - 1
                if assignment[idx] == (l > 0):
                    cnt += 1
            sat_count[i] = cnt
            if cnt == 0:
                unsat_clauses.append(i)

        current_unsat = len(unsat_clauses)
        if current_unsat == 0:
            return assignment

        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]

        # WalkSAT Loop
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break

            if not unsat_clauses:
                return assignment

            # Random flip with probability 'noise'
            if rng.random() < noise:
                var = rng.randint(0, n - 1)
                old_val = assignment[var]
                assignment[var] = not old_val

                # Update sat_count and unsat_clauses
                new_unsat = []
                for ci in var_pos[var]:
                    old_sat = sat_count[ci]
                    new_sat = old_sat - 1 if old_val else old_sat + 1
                    sat_count[ci] = new_sat
                    if new_sat == 0:
                        new_unsat.append(ci)
                
                for ci in var_neg[var]:
                    old_sat = sat_count[ci]
                    new_sat = old_sat + 1 if old_val else old_sat - 1
                    sat_count[ci] = new_sat
                    if new_sat == 0:
                        new_unsat.append(ci)
                
                # Rebuild unsat list
                new_unsat_set = set(new_unsat)
                updated_unsat = [ci for ci in unsat_clauses if sat_count[ci] == 0]
                for ci in new_unsat:
                    if ci not in updated_unsat:
                        updated_unsat.append(ci)
                unsat_clauses = updated_unsat
                continue

            # Heuristic flip: Choose random unsat clause, pick variable to flip
            clause_idx = unsat_clauses[rng.randint(0, len(unsat_clauses) - 1)]
            clause = clauses[clause_idx]

            best_var = -1
            best_score = -1
            
            for l in clause:
                var = abs(l) - 1
                score = 0
                target_val = (l > 0)
                
                if assignment[var] != target_val:
                    # Flipping makes this literal true.
                    # Score: how many *other* unsat clauses does this fix?
                    affected = var_pos[var] if l > 0 else var_neg[var]
                    for ci in affected:
                        if sat_count[ci] == 0:
                            # Check if this is the only unsat literal in ci
                            c2 = clauses[ci]
                            only_unsat = True
                            for l2 in c2:
                                if l2 == l:
                                    continue
                                idx2 = abs(l2) - 1
                                if assignment[idx2] == (l2 > 0):
                                    only_unsat = False
                                    break
                            if only_unsat:
                                score += 1

                if score > best_score:
                    best_score = score
                    best_var = var

            if best_var >= 0:
                var_to_flip = best_var
            else:
                l = clause[rng.randint(0, len(clause) - 1)]
                var_to_flip = abs(l) - 1

            # Apply flip
            old_val = assignment[var_to_flip]
            assignment[var_to_flip] = not old_val

            # Update sat_count and unsat_clauses
            new_unsat = []
            for ci in var_pos[var_to_flip]:
                old_sat = sat_count[ci]
                new_sat = old_sat - 1 if old_val else old_sat + 1
                sat_count[ci] = new_sat
                if new_sat == 0:
                    new_unsat.append(ci)
            
            for ci in var_neg[var_to_flip]:
                old_sat = sat_count[ci]
                new_sat = old_sat + 1 if old_val else old_sat - 1
                sat_count[ci] = new_sat
                if new_sat == 0:
                    new_unsat.append(ci)
            
            # Rebuild unsat list
            new_unsat_set = set(new_unsat)
            updated_unsat = [ci for ci in unsat_clauses if sat_count[ci] == 0]
            for ci in new_unsat:
                if ci not in updated_unsat:
                    updated_unsat.append(ci)
            unsat_clauses = updated_unsat

            # Track best
            if step % 100 == 0:
                current_unsat = len(unsat_clauses)
                if current_unsat < best_unsat_count:
                    best_unsat_count = current_unsat
                    best_assignment = assignment[:]

    # Phase 2: Final Greedy Polish on best assignment
    if best_assignment is not None:
        assignment = best_assignment[:]
        for _ in range(5000):
            if time.time() > deadline:
                break
            unsat_clauses = [c for c in clauses if not satisfied(assignment, c)]
            if not unsat_clauses:
                return assignment
            
            # Find variable that fixes most clauses
            best_var = -1
            best_gain = 0
            for c in unsat_clauses[:50]:
                for l in c:
                    var = abs(l) - 1
                    if assignment[var] == (l > 0):
                        continue
                    gain = 0
                    for c2 in unsat_clauses[:50]:
                        for l2 in c2:
                            if abs(l2) - 1 == var and assignment[var] != (l2 > 0):
                                gain += 1
                                break
                    if gain > best_gain:
                        best_gain = gain
                        best_var = var
            
            if best_var >= 0:
                assignment[best_var] = not assignment[best_var]
                current_unsat = count_unsat(assignment)
                if current_unsat < best_unsat_count:
                    best_unsat_count = current_unsat
                    best_assignment = assignment[:]
                if current_unsat == 0:
                    return assignment

    if best_assignment is not None:
        return best_assignment
    return [rng.random() < 0.5 for _ in range(n)]
