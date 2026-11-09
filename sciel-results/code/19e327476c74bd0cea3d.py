"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    """Satisfying assignment for a locally generated planted satisfiable 3-CNF.

    instance: dict with 'n' (int) and 'clauses' (list of lists of signed 1-based ints).
    Returns: list of booleans (length n) such that every clause is satisfied.
    """
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    # Deadline guard: evaluator kills at 330s; keep a safety margin.
    start_time = time.time()
    deadline = start_time + 290.0  # 40s margin

    # Precompute literal arrays for efficient evaluation
    # For each clause, store (var_index, sign) where sign is +1 or -1
    clause_vars = []  # list of arrays of variable indices (0-based)
    clause_signs = []  # list of arrays of signs (+1 or -1)
    for clause in clauses:
        vars_ = np.array([abs(l) - 1 for l in clause], dtype=np.int32)
        signs = np.array([1 if l > 0 else -1 for l in clause], dtype=np.int8)
        clause_vars.append(vars_)
        clause_signs.append(signs)

    def is_satisfied(assignment):
        """Check if assignment satisfies all clauses. Returns True if all satisfied."""
        for i in range(m):
            # For each literal in clause i, check if it's satisfied
            # A literal is satisfied if assignment[var] * sign == 1
            # i.e., assignment[var] == (sign == 1)
            vals = assignment[clause_vars[i]]
            # Convert assignment (booleans) to +/-1
            vals_int = vals.astype(np.int8) * 2 - 1  # True->1, False->-1
            if not np.any(vals_int == clause_signs[i]):
                return False
        return True

    def count_unsat(assignment):
        """Count number of unsatisfied clauses."""
        count = 0
        for i in range(m):
            vals = assignment[clause_vars[i]]
            vals_int = vals.astype(np.int8) * 2 - 1
            if not np.any(vals_int == clause_signs[i]):
                count += 1
        return count

    def walksat(assignment, rng, max_steps=50000):
        """Run WalkSAT with the given assignment and RNG. Returns (assignment, solved)."""
        steps = 0
        while steps < max_steps:
            # Check deadline
            if time.time() > deadline:
                return assignment, False

            # Find unsatisfied clauses
            unsat_clauses = []
            for i in range(m):
                vals = assignment[clause_vars[i]]
                vals_int = vals.astype(np.int8) * 2 - 1
                if not np.any(vals_int == clause_signs[i]):
                    unsat_clauses.append(i)

            if not unsat_clauses:
                return assignment, True

            # Pick a random unsatisfied clause
            ci = rng.choice(unsat_clauses)
            vars_ = clause_vars[ci]
            signs = clause_signs[ci]

            # Score each variable in the clause: how many clauses would be satisfied if flipped
            best_var = -1
            best_score = -1
            for j, var in enumerate(vars_):
                # Flip this variable and count how many currently-unsat clauses become sat
                # and how many currently-sat clauses become unsat
                # We want to maximize: (clauses that become sat) - (clauses that become unsat)
                # Simplified: count clauses where this variable is the only unsat literal
                flip_var = var
                # Check all clauses containing this variable
                score = 0
                for k in range(m):
                    if flip_var not in clause_vars[k]:
                        continue
                    # Check if this clause is currently unsat
                    vals_k = assignment[clause_vars[k]]
                    vals_k_int = vals_k.astype(np.int8) * 2 - 1
                    clause_k_unsat = not np.any(vals_k_int == clause_signs[k])
                    
                    if clause_k_unsat:
                        # If we flip flip_var, does this clause become sat?
                        # Only if flip_var is the only unsat literal in this clause
                        # and flipping it satisfies it
                        idx_in_clause = np.where(clause_vars[k] == flip_var)[0]
                        if len(idx_in_clause) > 0:
                            sign_k = clause_signs[k][idx_in_clause[0]]
                            # After flip, assignment[flip_var] changes
                            new_val = not assignment[flip_var]
                            new_val_int = (1 if new_val else -1)
                            if new_val_int == sign_k:
                                # This literal would be satisfied after flip
                                # Check if all other literals in this clause are already sat
                                other_indices = [x for x in range(len(clause_vars[k])) if x != idx_in_clause[0]]
                                all_others_sat = True
                                for oi in other_indices:
                                    v = clause_vars[k][oi]
                                    s = clause_signs[k][oi]
                                    v_int = (1 if assignment[v] else -1)
                                    if v_int != s:
                                        all_others_sat = False
                                        break
                                if all_others_sat:
                                    score += 1
                    else:
                        # Clause is currently sat. If we flip flip_var, does it become unsat?
                        # Only if flip_var is the only satisfied literal
                        idx_in_clause = np.where(clause_vars[k] == flip_var)[0]
                        if len(idx_in_clause) > 0:
                            sign_k = clause_signs[k][idx_in_clause[0]]
                            # Currently satisfied means assignment[flip_var]*sign_k == 1
                            # After flip, it would be unsatisfied
                            # Check if all other literals are unsat
                            other_indices = [x for x in range(len(clause_vars[k])) if x != idx_in_clause[0]]
                            all_others_unsat = True
                            for oi in other_indices:
                                v = clause_vars[k][oi]
                                s = clause_signs[k][oi]
                                v_int = (1 if assignment[v] else -1)
                                if v_int == s:
                                    all_others_unsat = False
                                    break
                            if all_others_unsat:
                                score -= 1
                if score > best_score:
                    best_score = score
                    best_var = var

            if best_var >= 0:
                assignment[best_var] = not assignment[best_var]
            else:
                # Random flip from the clause
                var = vars_[rng.randrange(len(vars_))]
                assignment[var] = not assignment[var]

            steps += 1

        # Final check
        if is_satisfied(assignment):
            return assignment, True
        return assignment, False

    def local_search(assignment, rng, max_steps=30000):
        """Run local search (min-conflicts style). Returns (assignment, solved)."""
        steps = 0
        while steps < max_steps:
            if time.time() > deadline:
                return assignment, False

            # Find all unsatisfied clauses
            unsat_clauses = []
            for i in range(m):
                vals = assignment[clause_vars[i]]
                vals_int = vals.astype(np.int8) * 2 - 1
                if not np.any(vals_int == clause_signs[i]):
                    unsat_clauses.append(i)

            if not unsat_clauses:
                return assignment, True

            # Pick random unsat clause
            ci = rng.choice(unsat_clauses)
            vars_ = clause_vars[ci]

            # Find the variable in this clause that appears in the most unsat clauses
            # (or just pick randomly with some heuristic)
            best_var = -1
            best_count = -1
            for var in vars_:
                # Count how many unsat clauses contain this variable
                count = 0
                for uc in unsat_clauses:
                    if var in clause_vars[uc]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_var = var

            if best_var >= 0:
                assignment[best_var] = not assignment[best_var]
            else:
                var = vars_[rng.randrange(len(vars_))]
                assignment[var] = not assignment[var]

            steps += 1

        if is_satisfied(assignment):
            return assignment, True
        return assignment, False

    # Strategy: try multiple approaches with different random seeds
    best_assignment = None
    best_unsat_count = m + 1

    # Try WalkSAT with different seeds
    seeds_to_try = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for seed in seeds_to_try:
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        assignment, solved = walksat(assignment, rng, max_steps=30000)
        if solved:
            return assignment
        # Track best
        uc = count_unsat(assignment)
        if uc < best_unsat_count:
            best_unsat_count = uc
            best_assignment = assignment[:]

    # Try local search with different seeds
    for seed in seeds_to_try:
        if time.time() > deadline:
            break
        rng = random.Random(seed + 100)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        assignment, solved = local_search(assignment, rng, max_steps=20000)
        if solved:
            return assignment
        uc = count_unsat(assignment)
        if uc < best_unsat_count:
            best_unsat_count = uc
            best_assignment = assignment[:]

    # If we have a best assignment, try to improve it with more WalkSAT
    if best_assignment is not None and time.time() < deadline:
        for seed in range(10, 30):
            if time.time() > deadline:
                break
            rng = random.Random(seed)
            assignment = best_assignment[:]
            assignment, solved = walksat(assignment, rng, max_steps=20000)
            if solved:
                return assignment
            uc = count_unsat(assignment)
            if uc < best_unsat_count:
                best_unsat_count = uc
                best_assignment = assignment[:]

    # Return best assignment found
    if best_assignment is not None:
        return best_assignment
    else:
        # Fallback: random assignment
        rng = random.Random(0)
        return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
