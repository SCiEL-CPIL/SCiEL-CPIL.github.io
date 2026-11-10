"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    start_time = time.time()
    DEADLINE = 295.0  # Safety margin before 300s limit

    # Preprocess: convert to 0-based variable indices and signs
    clause_vars = []
    clause_sigs = []
    for c in clauses:
        cv = [abs(l) - 1 for l in c]
        cs = [1 if l > 0 else -1 for l in c]
        clause_vars.append(cv)
        clause_sigs.append(cs)

    # Build var -> list of (clause_idx, pos_in_clause) for efficient updates
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            var_to_clauses[clause_vars[i][j]].append((i, j))

    # Precompute variable degrees for static heuristic
    var_degree = [0] * n
    for i in range(m):
        for j in range(3):
            var_degree[clause_vars[i][j]] += 1

    # Precompute clause weights (sum of degrees of its variables)
    clause_weights = [0] * m
    for i in range(m):
        w = 0
        for j in range(3):
            w += var_degree[clause_vars[i][j]]
        clause_weights[i] = w

    # Precompute numpy arrays for fast vectorized operations
    cv_np = np.array(clause_vars, dtype=np.int32)  # (m, 3)
    cs_np = np.array(clause_sigs, dtype=np.int8)   # (m, 3)
    # lit_val[i, j] = 1.0 if literal (i,j) is satisfied given assignment, else 0.0
    # We'll compute this on the fly using boolean assignment

    rng = random.Random(42)

    def spectral_initialization():
        """
        Formulate as Max-Cut: build graph where nodes are variables,
        edges connect variables that appear together in clauses with
        conflicting signs (i.e., both positive or both negative literals
        in the same clause create a "conflict" if assigned the same value).
        
        For each clause (l1, l2, l3):
        - If l1 and l2 have same sign: they conflict if both True or both False
          -> edge between var1, var2 with weight indicating they should differ
        - If l1 and l2 have opposite signs: they conflict if one is True and other False
          -> edge between var1, var2 with weight indicating they should be same
        
        Actually, for Max-Cut formulation of 3-SAT:
        Each clause (a, b, c) is satisfied unless all three literals are false.
        We want to assign values to minimize unsatisfied clauses.
        
        A simpler approach: use the Laplacian of a graph where edges represent
        "soft constraints" from clauses, then use the Fiedler vector for bisection.
        
        Build a weighted graph:
        - For each clause, add edges between pairs of variables with weights
          that encourage satisfying the clause.
        """
        # Build adjacency matrix for spectral bisection
        # Weight of edge (u,v) indicates how much they should be in different cuts
        # For a clause (x, y, z):
        #   If signs are (+,+,+): clause unsat only if x=0,y=0,z=0
        #   We want to encourage at least one to be True
        #   This is hard to capture directly in Max-Cut, so we use a heuristic:
        #   Add edges with small weights between all pairs in the clause
        
        row = []
        col = []
        data = []
        
        for i in range(m):
            cv = clause_vars[i]
            cs = clause_sigs[i]
            v1, v2, v3 = cv[0], cv[1], cv[2]
            s1, s2, s3 = cs[0], cs[1], cs[2]
            
            # For each pair in the clause, add an edge with weight based on sign compatibility
            # If two literals have the same sign, they "agree" when both True or both False
            # If opposite signs, they "agree" when one True and one False
            
            # Heuristic: add edge weight 1.0 for pairs with same sign (should differ in cut),
            # and 0.5 for pairs with opposite sign (should be same in cut, so negative weight)
            
            # Pair (v1, v2)
            if s1 == s2:
                w = 1.0
            else:
                w = -0.5
            row.append(v1)
            col.append(v2)
            data.append(w)
            row.append(v2)
            col.append(v1)
            data.append(w)
            
            # Pair (v1, v3)
            if s1 == s3:
                w = 1.0
            else:
                w = -0.5
            row.append(v1)
            col.append(v3)
            data.append(w)
            row.append(v3)
            col.append(v1)
            data.append(w)
            
            # Pair (v2, v3)
            if s2 == s3:
                w = 1.0
            else:
                w = -0.5
            row.append(v2)
            col.append(v3)
            data.append(w)
            row.append(v3)
            col.append(v2)
            data.append(w)
        
        if not row:
            return None
        
        A = csr_matrix((data, (row, col)), shape=(n, n))
        
        # Compute degree matrix
        degrees = np.abs(A).sum(axis=1).A1
        D = csr_matrix((degrees, (range(n), range(n))), shape=(n, n))
        
        # Laplacian L = D - A
        L = D - A
        
        # Find Fiedler vector (eigenvector of second smallest eigenvalue)
        try:
            # Use eigsh for sparse eigenvalue computation
            # We want the eigenvector corresponding to the second smallest eigenvalue
            # eigsh finds eigenvalues near sigma; use sigma=0 with which='LM' on L
            # Actually, for Fiedler vector, we want the eigenvector of the smallest
            # non-zero eigenvalue. Use sigma=1e-10 with which='LM'
            vals, vecs = eigsh(L, k=2, sigma=0, which='LM')
            # Sort by eigenvalue
            idx = np.argsort(vals)
            fiedler = vecs[:, idx[1]]  # Second smallest eigenvalue's eigenvector
        except Exception:
            try:
                # Fallback: use dense eigendecomposition for small n
                L_dense = L.toarray()
                vals, vecs = np.linalg.eigh(L_dense)
                fiedler = vecs[:, 1]
            except Exception:
                return None
        
        # Assign variables: True if fiedler[i] > median, False otherwise
        median_val = np.median(fiedler)
        assignment = [fiedler[i] > median_val for i in range(n)]
        
        return assignment

    def make_random_assignment():
        return [rng.random() < 0.5 for _ in range(n)]

    def walk_sat_run(assignment, max_flips, p=0.55):
        """WalkSAT with optimized list-based unsat clause tracking and static clause weighting."""
        
        # Initialize clause satisfaction status
        clause_sat_count = [0] * m
        for i in range(m):
            cv = clause_vars[i]
            cs = clause_sigs[i]
            cnt = 0
            for j in range(3):
                if assignment[cv[j]] == (cs[j] > 0):
                    cnt += 1
            clause_sat_count[i] = cnt

        # Initialize unsat_list and unsat_pos for O(1) selection and updates
        unsat_list = []
        unsat_pos = [-1] * m  # -1 if not in unsat_list, else index in unsat_list
        
        for i in range(m):
            if clause_sat_count[i] == 0:
                unsat_pos[i] = len(unsat_list)
                unsat_list.append(i)

        for flip in range(max_flips):
            if time.time() - start_time > DEADLINE:
                return None, assignment

            if len(unsat_list) == 0:
                return assignment[:], assignment

            # Pick an unsatisfied clause
            # Strategy: With high probability, pick the clause with highest static weight.
            # With low probability, pick randomly to escape local minima.
            if rng.random() < 0.8:
                # Find max weight in unsat_list
                best_ci = -1
                best_w = -1
                for idx in unsat_list:
                    if clause_weights[idx] > best_w:
                        best_w = clause_weights[idx]
                        best_ci = idx
                ci = best_ci
            else:
                idx = rng.randint(0, len(unsat_list) - 1)
                ci = unsat_list[idx]

            # Choose literal to flip
            if rng.random() < p:
                # Random literal in the clause
                li = rng.randint(0, 2)
            else:
                # Weighted: prefer literals that satisfy more other unsat clauses
                best_li = 0
                best_weight = -1
                for li in range(3):
                    var = clause_vars[ci][li]
                    weight = 0
                    for (oi, oj) in var_to_clauses[var]:
                        if oi == ci:
                            continue
                        # Check if oi is currently unsat
                        if unsat_pos[oi] != -1:
                            weight += 1
                    if weight > best_weight:
                        best_weight = weight
                        best_li = li
                li = best_li

            var = clause_vars[ci][li]
            old_val = assignment[var]
            new_val = not old_val
            assignment[var] = new_val

            # Update clause_sat_count and unsat_list for all clauses containing this var
            for (oi, oj) in var_to_clauses[var]:
                osig = clause_sigs[oi][oj]
                was_sat = (old_val == (osig > 0))
                now_sat = (new_val == (osig > 0))
                
                if was_sat and not now_sat:
                    # Clause became unsat
                    clause_sat_count[oi] -= 1
                    if clause_sat_count[oi] == 0:
                        # Add to unsat_list
                        unsat_pos[oi] = len(unsat_list)
                        unsat_list.append(oi)
                elif not was_sat and now_sat:
                    # Clause became sat
                    clause_sat_count[oi] += 1
                    if clause_sat_count[oi] > 0:
                        # Remove from unsat_list if it was there
                        if unsat_pos[oi] != -1:
                            pos = unsat_pos[oi]
                            last_idx = len(unsat_list) - 1
                            last_clause = unsat_list[last_idx]
                            
                            # Swap with last
                            unsat_list[pos] = last_clause
                            unsat_pos[last_clause] = pos
                            
                            # Pop last
                            unsat_list.pop()
                            unsat_pos[oi] = -1

        # Return None if not solved, but return the final assignment for tracking
        if len(unsat_list) == 0:
            return assignment[:], assignment
        return None, assignment

    def count_unsat(assignment):
        """Count unsatisfied clauses using numpy for speed."""
        # Convert assignment to numpy bool array
        a_np = np.array(assignment, dtype=bool)
        # For each literal (i, j): satisfied if a_np[cv_np[i,j]] == (cs_np[i,j] > 0)
        # cs_np[i,j] > 0 is a bool array (m, 3)
        lit_sat = a_np[cv_np] == (cs_np > 0)  # shape (m, 3)
        # Clause is satisfied if any literal is satisfied
        clause_sat = lit_sat.any(axis=1)  # shape (m,)
        return int((~clause_sat).sum())

    def polish(assignment, unsat_clauses):
        """Try to fix a small number of unsat clauses by flipping variables."""
        if not unsat_clauses:
            return assignment
        current_unsat = count_unsat(assignment)
        if current_unsat == 0:
            return assignment
        
        # Collect unique variables from unsat clauses
        vars_to_try = set()
        for ci in unsat_clauses:
            for j in range(3):
                vars_to_try.add(clause_vars[ci][j])
        
        best_assign = assignment[:]
        best_count = current_unsat
        
        for var in vars_to_try:
            # Flip var
            assignment[var] = not assignment[var]
            new_count = count_unsat(assignment)
            if new_count < best_count:
                best_count = new_count
                best_assign = assignment[:]
            # Unflip var
            assignment[var] = not assignment[var]
            
        return best_assign

    def continuous_energy_descent(assignment, max_iters=500, lr=0.1):
        """
        Continuous relaxation: treat variables as real values in [0, 1],
        minimize energy E = sum_c max(0, 1 - sum_j lit_val_cj)
        where lit_val_cj = x_v if sig=+1 else (1-x_v).
        Use gradient descent with soft-max approximation.
        Returns a boolean assignment derived from the continuous solution.
        """
        # Convert to numpy float array
        x = np.array(assignment, dtype=np.float64)
        
        # Precompute for each literal (i, j):
        # lit_val = x[var] if sig > 0 else 1 - x[var]
        # We need efficient gradient computation.
        # E = sum_i max(0, 1 - sum_j lit_val_ij)
        # Use soft version: E_soft = sum_i log(1 + exp(beta * (1 - sum_j lit_val_ij))) / beta
        # Or simpler: just use the hard version with subgradient.
        
        beta = 10.0  # sharpness of soft-max
        
        for it in range(max_iters):
            if time.time() - start_time > DEADLINE:
                break
            
            # Compute literal values
            # lit_vals[i, j] = x[cv[i,j]] if cs[i,j] > 0 else 1 - x[cv[i,j]]
            x_vars = x[cv_np]  # shape (m, 3)
            sig_mask = cs_np > 0  # shape (m, 3)
            lit_vals = np.where(sig_mask, x_vars, 1.0 - x_vars)  # shape (m, 3)
            
            # Clause satisfaction sum
            clause_sum = lit_vals.sum(axis=1)  # shape (m,)
            
            # Deficit: how much each clause is unsatisfied
            deficit = np.maximum(0.0, 1.0 - clause_sum)  # shape (m,)
            
            # If all satisfied, return
            if deficit.sum() < 1e-9:
                break
            
            # Gradient of E = sum deficit_i with respect to x
            # d(deficit_i)/d(lit_val_ij) = -1 if clause_sum_i < 1 else 0
            # d(lit_val_ij)/d(x_v) = +1 if sig > 0 else -1
            # So dE/dx_v = sum_{(i,j) where v appears and clause_i is unsat} (-1) * sig_ij
            
            # For each variable, accumulate gradient
            grad = np.zeros(n, dtype=np.float64)
            
            # Only consider unsatisfied clauses (deficit > 0)
            unsat_mask = deficit > 0  # shape (m,)
            if not unsat_mask.any():
                break
            
            unsat_indices = np.where(unsat_mask)[0]
            
            for i in unsat_indices:
                for j in range(3):
                    var = cv_np[i, j]
                    sig = cs_np[i, j]
                    # d(lit_val_ij)/d(x_var) = sig (since sig is +1 or -1)
                    # d(deficit_i)/d(lit_val_ij) = -1 (since deficit = max(0, 1-sum), and sum < 1)
                    # So contribution to grad[var] = -1 * sig
                    grad[var] -= sig
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                break
            
            # Gradient descent step
            x -= lr * grad / grad_norm
            
            # Clip to [0, 1]
            x = np.clip(x, 0.0, 1.0)
            
            # Optional: threshold and check
            if it % 50 == 0:
                bool_assign = x > 0.5
                if count_unsat(bool_assign) == 0:
                    return bool_assign.tolist()
        
        # Final threshold
        bool_assign = x > 0.5
        return bool_assign.tolist()

    best_assignment = None
    best_unsat = m + 1

    # Luby-like restart sequence
    multipliers = [1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, 1, 1, 2, 1, 1, 2, 4, 8, 16]
    base_flips = 100
    max_restarts = 1000

    # Try spectral initialization first
    spectral_init = spectral_initialization()
    if spectral_init is not None:
        spectral_unsat = count_unsat(spectral_init)
        if spectral_unsat == 0:
            return spectral_init
        best_unsat = min(best_unsat, spectral_unsat)
        best_assignment = spectral_init[:]
        
        # Try WalkSAT from spectral initialization
        result, final_assignment = walk_sat_run(spectral_init, max_flips=2000, p=0.5)
        if result is not None:
            return result
        unsat_count = count_unsat(final_assignment)
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = final_assignment[:]

    for restart in range(max_restarts):
        if time.time() - start_time > DEADLINE:
            break

        # Determine flip budget
        mult = multipliers[restart % len(multipliers)]
        current_base = base_flips + (restart // 10) * 100
        flips = current_base * mult
        
        if flips > 5000:
            flips = 5000

        # Alternate between random and spectral-based initializations
        if restart % 3 == 0 and spectral_init is not None:
            # Use spectral init with some random perturbation
            assignment = spectral_init[:]
            for i in range(n):
                if rng.random() < 0.1:
                    assignment[i] = not assignment[i]
        else:
            assignment = make_random_assignment()
        
        p = 0.55
        result, final_assignment = walk_sat_run(assignment, max_flips=flips, p=p)
        
        if result is not None:
            return result

        # Track best
        unsat_count = count_unsat(final_assignment)
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = final_assignment[:]
            
            # If we are very close, try polishing
            if unsat_count <= 3:
                unsat_clauses = []
                for i in range(m):
                    cv = clause_vars[i]
                    cs = clause_sigs[i]
                    sat = False
                    for j in range(3):
                        if final_assignment[cv[j]] == (cs[j] > 0):
                            sat = True
                            break
                    if not sat:
                        unsat_clauses.append(i)
                
                polished = polish(final_assignment, unsat_clauses)
                polished_unsat = count_unsat(polished)
                if polished_unsat == 0:
                    return polished
                if polished_unsat < best_unsat:
                    best_unsat = polished_unsat
                    best_assignment = polished[:]
        
        # Occasionally try continuous energy descent from the current best
        if restart % 5 == 0 and best_assignment is not None and best_unsat <= 10:
            cont_result = continuous_energy_descent(best_assignment, max_iters=200, lr=0.1)
            cont_unsat = count_unsat(cont_result)
            if cont_unsat == 0:
                return cont_result
            if cont_unsat < best_unsat:
                best_unsat = cont_unsat
                best_assignment = cont_result[:]

    if best_assignment is not None:
        return best_assignment

    return [True] * n
# EVOLVE-BLOCK-END
