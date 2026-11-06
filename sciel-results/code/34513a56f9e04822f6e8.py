# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 285  # safety margin before 330s limit

    # Convert to 0-based
    clauses_internal = []
    for c in clauses:
        clauses_internal.append(tuple((abs(l) - 1, l > 0) for l in c))

    m = len(clauses_internal)

    # Build index: for each variable, which clauses contain it
    var_to_clauses = [[] for _ in range(n)]
    for i, c in enumerate(clauses_internal):
        for var, _ in c:
            var_to_clauses[var].append(i)

    # Pure literal elimination preprocessing
    pure_assignment = [None] * n
    active_clauses = set(range(m))
    changed = True
    while changed:
        changed = False
        lit_counts = [[0, 0] for _ in range(n)]
        for ci in active_clauses:
            c = clauses_internal[ci]
            for var, is_pos in c:
                idx = 1 if is_pos else 0
                lit_counts[var][idx] += 1

        for var in range(n):
            if pure_assignment[var] is not None:
                continue
            neg, pos = lit_counts[var]
            if neg == 0 and pos > 0:
                pure_assignment[var] = True
                changed = True
                for ci in var_to_clauses[var]:
                    if ci in active_clauses:
                        c = clauses_internal[ci]
                        for v, p in c:
                            if pure_assignment[v] is not None and pure_assignment[v] == p:
                                active_clauses.discard(ci)
                                break
            elif pos == 0 and neg > 0:
                pure_assignment[var] = False
                changed = True
                for ci in var_to_clauses[var]:
                    if ci in active_clauses:
                        c = clauses_internal[ci]
                        for v, p in c:
                            if pure_assignment[v] is not None and pure_assignment[v] == p:
                                active_clauses.discard(ci)
                                break

    # Check if all clauses are satisfied by pure assignment
    all_satisfied = True
    for ci in range(m):
        c = clauses_internal[ci]
        sat = False
        for var, is_pos in c:
            if pure_assignment[var] is not None and pure_assignment[var] == is_pos:
                sat = True
                break
        if not sat:
            all_satisfied = False
            break

    if all_satisfied:
        assignment = []
        for var in range(n):
            if pure_assignment[var] is not None:
                assignment.append(pure_assignment[var])
            else:
                assignment.append(False)
        return assignment

    # Build reduced problem
    non_pure_vars = [var for var in range(n) if pure_assignment[var] is None]
    np_count = len(non_pure_vars)
    var_map = {var: i for i, var in enumerate(non_pure_vars)}

    reduced_clauses = []
    for ci in range(m):
        c = clauses_internal[ci]
        sat = False
        new_clause = []
        for var, is_pos in c:
            if pure_assignment[var] is not None:
                if pure_assignment[var] == is_pos:
                    sat = True
                    break
            else:
                new_clause.append((var_map[var], is_pos))
        if not sat and new_clause:
            reduced_clauses.append(tuple(new_clause))

    rm = len(reduced_clauses)

    if rm == 0:
        assignment = []
        for var in range(n):
            if pure_assignment[var] is not None:
                assignment.append(pure_assignment[var])
            else:
                assignment.append(False)
        return assignment

    # Build index for reduced problem
    rvar_to_clauses = [[] for _ in range(np_count)]
    for i, c in enumerate(reduced_clauses):
        for var, _ in c:
            rvar_to_clauses[var].append(i)

    # Precompute clause literal arrays for fast access
    # clause_vars[i] = list of (var, is_pos) for clause i
    # We'll use numpy arrays for vectorized operations
    clause_vars_arr = np.zeros((rm, 3), dtype=np.int8)  # variable indices
    clause_signs_arr = np.zeros((rm, 3), dtype=np.int8)  # 1 for positive, 0 for negative
    clause_lens = np.zeros(rm, dtype=np.int8)

    for i, c in enumerate(reduced_clauses):
        for j, (var, is_pos) in enumerate(c):
            clause_vars_arr[i, j] = var
            clause_signs_arr[i, j] = 1 if is_pos else 0
        clause_lens[i] = len(c)

    # Precompute for each variable, the clauses it appears in and its sign
    var_clause_info = [[] for _ in range(np_count)]
    for i in range(rm):
        for j in range(clause_lens[i]):
            var = clause_vars_arr[i, j]
            sign = clause_signs_arr[i, j]
            var_clause_info[var].append((i, sign))

    rng = random.Random(42)

    def count_unsat_reduced(assignment):
        """Count unsatisfied clauses in reduced problem using vectorized ops."""
        # Convert assignment to numpy array
        assign_arr = np.array(assignment, dtype=np.int8)
        # For each clause, check if any literal is satisfied
        # A literal is satisfied if (var value == sign)
        # Create a matrix: for each clause-clause position, is the literal satisfied?
        # lit_satisfied[i,j] = 1 if assign[clause_vars[i,j]] == clause_signs[i,j]
        
        # Get variable values for all positions
        var_vals = assign_arr[clause_vars_arr]  # shape (rm, 3)
        # Check satisfaction: var_vals == clause_signs_arr
        sat_mask = (var_vals == clause_signs_arr)  # shape (rm, 3)
        # A clause is satisfied if any literal in it is satisfied
        # But we need to only consider valid positions (j < clause_lens[i])
        # Create mask for valid positions
        valid_mask = np.zeros((rm, 3), dtype=bool)
        for i in range(rm):
            valid_mask[i, :clause_lens[i]] = True
        sat_mask = sat_mask & valid_mask
        # Clause is satisfied if any valid literal is satisfied
        clause_sat = sat_mask.any(axis=1)  # shape (rm,)
        # Count unsatisfied
        return int((~clause_sat).sum())

    def get_unsat_clause_indices(assignment):
        """Get indices of unsatisfied clauses."""
        assign_arr = np.array(assignment, dtype=np.int8)
        var_vals = assign_arr[clause_vars_arr]
        sat_mask = (var_vals == clause_signs_arr)
        valid_mask = np.zeros((rm, 3), dtype=bool)
        for i in range(rm):
            valid_mask[i, :clause_lens[i]] = True
        sat_mask = sat_mask & valid_mask
        clause_sat = sat_mask.any(axis=1)
        return np.where(~clause_sat)[0]

    def build_full_assignment(reduced_assign):
        """Build full assignment from reduced assignment."""
        full = [False] * n
        for var in range(n):
            if pure_assignment[var] is not None:
                full[var] = pure_assignment[var]
            else:
                full[var] = reduced_assign[var_map[var]]
        return full

    best_full_assignment = None
    best_unsat_count = m + 1

    # Main search loop with adaptive restarts
    max_restarts = 2000
    base_steps = 3000
    step_growth = 1.08

    restart = 0
    while restart < max_restarts and time.time() < deadline:
        # Random initial assignment for non-pure variables
        reduced_assign = [rng.random() < 0.5 for _ in range(np_count)]

        # Count initial unsatisfied
        unsat_count = count_unsat_reduced(reduced_assign)

        if unsat_count == 0:
            return build_full_assignment(reduced_assign)

        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_full_assignment = build_full_assignment(reduced_assign)

        # Dynamic noise parameter
        initial_noise = 0.4
        min_noise = 0.05
        noise_decay = 0.998

        steps = int(base_steps * (step_growth ** restart))
        steps = min(steps, 80000)

        noise = initial_noise
        prev_unsat = unsat_count
        no_improve_count = 0
        best_in_restart = unsat_count

        for step in range(steps):
            if time.time() >= deadline:
                break

            # Find unsatisfied clauses
            unsat_indices = get_unsat_clause_indices(reduced_assign)
            
            if len(unsat_indices) == 0:
                return build_full_assignment(reduced_assign)

            cur_unsat = len(unsat_indices)
            
            if cur_unsat < best_unsat_count:
                best_unsat_count = cur_unsat
                best_full_assignment = build_full_assignment(reduced_assign)
            
            if cur_unsat < best_in_restart:
                best_in_restart = cur_unsat
                no_improve_count = 0
            else:
                no_improve_count += 1

            # If we made progress, reduce noise slightly
            if cur_unsat < prev_unsat:
                noise = max(min_noise, noise * 0.97)
            prev_unsat = cur_unsat

            # Decay noise
            noise = max(min_noise, noise * noise_decay)

            # If no improvement for a while, increase noise to escape local minima
            if no_improve_count > 500:
                noise = min(0.8, noise * 1.5)
                no_improve_count = 0

            # Select a variable to flip
            if rng.random() < noise:
                # Random variable from any unsatisfied clause
                ci = unsat_indices[rng.randint(0, len(unsat_indices) - 1)]
                c = reduced_clauses[ci]
                false_vars = [var for var, is_pos in c if reduced_assign[var] != is_pos]
                if false_vars:
                    var_to_flip = false_vars[rng.randint(0, len(false_vars) - 1)]
                else:
                    var_to_flip = rng.randint(0, np_count - 1)
            else:
                # Glare strategy: pick clause with most false literals, then pick weighted variable
                # First, compute variable weights (how many unsat clauses each var is in with false literal)
                var_weights = [0] * np_count
                for ci in unsat_indices:
                    c = reduced_clauses[ci]
                    for var, is_pos in c:
                        if reduced_assign[var] != is_pos:
                            var_weights[var] += 1

                # Pick clause with most false literals (glare)
                best_ci = unsat_indices[0]
                best_false_count = -1
                for ci in unsat_indices:
                    c = reduced_clauses[ci]
                    false_count = sum(1 for var, is_pos in c if reduced_assign[var] != is_pos)
                    if false_count > best_false_count:
                        best_false_count = false_count
                        best_ci = ci
                    elif false_count == best_false_count and rng.random() < 0.3:
                        # Random tie-breaking to add diversity
                        best_ci = ci

                c = reduced_clauses[best_ci]
                false_vars = [(var, var_weights[var]) for var, is_pos in c if reduced_assign[var] != is_pos]
                if not false_vars:
                    var_to_flip = rng.randint(0, np_count - 1)
                else:
                    # Weighted selection among false literals
                    total_fw = sum(w for _, w in false_vars)
                    r = rng.random() * total_fw
                    cum = 0
                    var_to_flip = false_vars[0][0]
                    for var, w in false_vars:
                        cum += w
                        if r <= cum:
                            var_to_flip = var
                            break

            # Flip the variable
            reduced_assign[var_to_flip] = not reduced_assign[var_to_flip]

            # Periodic unit clause propagation
            if step % 50 == 0:
                # Try to satisfy unit clauses (clauses with exactly one false literal)
                for ci in unsat_indices:
                    if time.time() >= deadline:
                        break
                    c = reduced_clauses[ci]
                    false_lits = [(var, is_pos) for var, is_pos in c if reduced_assign[var] != is_pos]
                    if len(false_lits) == 1:
                        var, is_pos = false_lits[0]
                        if reduced_assign[var] != is_pos:
                            reduced_assign[var] = is_pos

        restart += 1

    # Return best found assignment
    if best_full_assignment is not None:
        return best_full_assignment

    # Fallback: random assignment
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
