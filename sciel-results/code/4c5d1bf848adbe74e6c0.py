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
    # A literal is pure if its variable only appears with the same sign
    # We can fix such variables to satisfy all clauses containing them
    pure_assignment = [None] * n  # None = unassigned, True/False = assigned
    active_clauses = set(range(m))
    changed = True
    while changed:
        changed = False
        # Count occurrences of each literal
        lit_counts = [[0, 0] for _ in range(n)]  # [negative_count, positive_count]
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
                # Remove satisfied clauses
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
        # Build full assignment
        assignment = []
        for var in range(n):
            if pure_assignment[var] is not None:
                assignment.append(pure_assignment[var])
            else:
                assignment.append(False)
        return assignment

    # Build reduced problem: only variables and clauses not fully determined
    # For remaining clauses, some literals may be fixed
    # We'll work with the full problem but pre-assign pure literals
    # and only search over non-pure variables

    non_pure_vars = [var for var in range(n) if pure_assignment[var] is None]
    np_count = len(non_pure_vars)
    var_map = {var: i for i, var in enumerate(non_pure_vars)}

    # Build reduced clauses: remove literals that are already satisfied,
    # and replace remaining literals with indices into non_pure_vars
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
                # else: literal is false, skip it
            else:
                new_clause.append((var_map[var], is_pos))
        if not sat and new_clause:
            reduced_clauses.append(tuple(new_clause))

    rm = len(reduced_clauses)

    if rm == 0:
        # All remaining clauses are satisfied
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

    # Compute clause weights for variable selection
    # Weight of a variable = number of unsatisfied clauses it appears in
    # We'll compute this dynamically

    rng = random.Random(42)

    def count_unsat_reduced(assignment):
        """Count unsatisfied clauses in reduced problem."""
        cnt = 0
        for c in reduced_clauses:
            sat = False
            for var, is_pos in c:
                if assignment[var] == is_pos:
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt

    def full_count_unsat(full_assignment):
        """Count unsatisfied clauses in full problem."""
        cnt = 0
        for c in clauses_internal:
            sat = False
            for var, is_pos in c:
                if full_assignment[var] == is_pos:
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt

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
    max_restarts = 1000
    base_steps = 2000
    step_growth = 1.12

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

        # Dynamic noise parameter: starts high, decreases over steps
        # noise controls probability of flipping a random variable vs. weighted selection
        initial_noise = 0.5
        min_noise = 0.1
        noise_decay = 0.995

        steps = int(base_steps * (step_growth ** restart))
        steps = min(steps, 50000)

        noise = initial_noise
        prev_unsat = unsat_count

        for step in range(steps):
            if time.time() >= deadline:
                break

            # Find unsatisfied clauses
            unsat_clauses = []
            for ci, c in enumerate(reduced_clauses):
                sat = False
                for var, is_pos in c:
                    if reduced_assign[var] == is_pos:
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(ci)

            if not unsat_clauses:
                return build_full_assignment(reduced_assign)

            cur_unsat = len(unsat_clauses)
            if cur_unsat < best_unsat_count:
                best_unsat_count = cur_unsat
                best_full_assignment = build_full_assignment(reduced_assign)

            # If we made progress, reduce noise slightly
            if cur_unsat < prev_unsat:
                noise = max(min_noise, noise * 0.95)
            prev_unsat = cur_unsat

            # Decay noise
            noise = max(min_noise, noise * noise_decay)

            # Select a variable to flip
            if rng.random() < noise:
                # Random variable from any unsatisfied clause
                ci = rng.choice(unsat_clauses)
                c = reduced_clauses[ci]
                false_vars = [var for var, is_pos in c if reduced_assign[var] != is_pos]
                if false_vars:
                    var_to_flip = rng.choice(false_vars)
                else:
                    var_to_flip = rng.randint(0, np_count - 1)
            else:
                # Weighted selection: pick a clause weighted by its "frustration"
                # then pick a variable in that clause weighted by how many unsat clauses it's in
                # First, compute variable weights (how many unsat clauses each var is in)
                var_weights = [0] * np_count
                for ci in unsat_clauses:
                    c = reduced_clauses[ci]
                    for var, is_pos in c:
                        if reduced_assign[var] != is_pos:
                            var_weights[var] += 1

                # Pick a clause weighted by sum of weights of its false literals
                clause_weights = []
                for ci in unsat_clauses:
                    c = reduced_clauses[ci]
                    w = 0
                    for var, is_pos in c:
                        if reduced_assign[var] != is_pos:
                            w += var_weights[var]
                    clause_weights.append(w)

                total_w = sum(clause_weights)
                if total_w == 0:
                    ci = rng.choice(unsat_clauses)
                else:
                    r = rng.random() * total_w
                    cum = 0
                    ci = unsat_clauses[0]
                    for i, w in enumerate(clause_weights):
                        cum += w
                        if r <= cum:
                            ci = unsat_clauses[i]
                            break

                c = reduced_clauses[ci]
                false_vars = [(var, var_weights[var]) for var, is_pos in c if reduced_assign[var] != is_pos]
                if not false_vars:
                    var_to_flip = rng.randint(0, np_count - 1)
                else:
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

            # Periodic local polish: try to satisfy unit clauses
            if step % 100 == 0:
                for ci in unsat_clauses:
                    c = reduced_clauses[ci]
                    false_lits = [(var, is_pos) for var, is_pos in c if reduced_assign[var] != is_pos]
                    if len(false_lits) == 1:
                        var, is_pos = false_lits[0]
                        if reduced_assign[var] != is_pos:
                            reduced_assign[var] = is_pos
                            # Recheck if this made other clauses unit
                            break  # One flip at a time to avoid cascading issues

        restart += 1

    # Return best found assignment
    if best_full_assignment is not None:
        return best_full_assignment

    # Fallback: random assignment
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
