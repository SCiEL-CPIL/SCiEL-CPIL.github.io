<<<<<<< SEARCH
# EVOLVE-BLOCK-START
import random


def solve(instance):
    # WalkSAT with a fixed flip budget.
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    assignment = [rng.random() < 0.5 for _ in range(n)]
    def satisfied(clause):
        return any(assignment[abs(l) - 1] == (l > 0) for l in clause)
    for _ in range(20000):
        unsat = [c for c in clauses if not satisfied(c)]
        if not unsat:
            break
        clause = rng.choice(unsat)
        literal = rng.choice(clause)
        assignment[abs(literal) - 1] = not assignment[abs(literal) - 1]
    return assignment
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    deadline = time.time() + 280  # safety margin before 300s limit

    # Build index: var -> list of (clause_idx, literal)
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var_to_clauses[abs(lit) - 1].append((ci, lit))

    # Precompute literal polarity for each clause entry
    # For each clause, store (var_idx, is_positive)
    clause_vars = []
    for clause in clauses:
        clause_vars.append([(abs(l) - 1, l > 0) for l in clause])

    def is_clause_satisfied(ci):
        for var_idx, is_pos in clause_vars[ci]:
            if assignment[var_idx] == is_pos:
                return True
        return False

    def count_unsat_in_clause(ci):
        count = 0
        for var_idx, is_pos in clause_vars[ci]:
            if assignment[var_idx] != is_pos:
                count += 1
        return count

    best_assignment = None
    best_unsat = float('inf')

    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() >= deadline:
            break

        assignment = [rng.random() < 0.5 for _ in range(n)]

        # Build unsat set and clause satisfaction status
        unsat_set = set()
        for ci in range(len(clauses)):
            if not is_clause_satisfied(ci):
                unsat_set.add(ci)

        if not unsat_set:
            return assignment

        # Track unsat count per unsatisfied clause
        unsat_counts = {ci: count_unsat_in_clause(ci) for ci in unsat_set}

        # Frequency heuristic: count how many times each variable appears in unsat clauses
        var_freq = [0] * n
        for ci in unsat_set:
            for var_idx, _ in clause_vars[ci]:
                var_freq[var_idx] += 1

        steps_this_restart = 0
        max_steps = 200000  # generous per-restart budget, time will limit

        while unsat_set and steps_this_restart < max_steps:
            if time.time() >= deadline:
                break

            # Pick a random unsatisfied clause
            ci = rng.choice(list(unsat_set))

            # Score each literal in the clause
            best_var = -1
            best_score = -1
            for var_idx, is_pos in clause_vars[ci]:
                # If we flip var_idx, how many other unsat clauses become satisfied?
                # For each clause containing var_idx, check if flipping helps
                gain = 0
                for (other_ci, other_lit) in var_to_clauses[var_idx]:
                    if other_ci not in unsat_set:
                        continue
                    # After flipping var_idx, does other_ci become satisfied?
                    # other_ci is currently unsat. Flipping var_idx:
                    # If other_lit matches the new value of var_idx, and no other lit already satisfies...
                    # Actually, if var_idx currently does NOT satisfy other_lit's polarity,
                    # and after flip it does, then other_ci might become satisfied
                    # if all other literals in other_ci are already satisfied or this is the last unsat.
                    # Simpler: check if unsat_counts[other_ci] == 1 and this var is the only unsat
                    # OR if flipping makes this var satisfy and unsat_count decreases
                    current_satisfies = assignment[var_idx] == (other_lit > 0)
                    if not current_satisfies:
                        # Flipping will make this var satisfy other_lit
                        # Check if this was the only unsat literal in other_ci
                        if unsat_counts.get(other_ci, 0) == 1:
                            gain += 1
                # Also consider: if flipping var_idx makes clause ci satisfied
                # (it always does since ci is unsat and we're flipping a var in it)
                # The gain is the number of OTHER clauses that become satisfied
                score = gain
                if score > best_score:
                    best_score = score
                    best_var = var_idx

            if best_var < 0:
                # Fallback: random pick
                var_idx, _ = rng.choice(clause_vars[ci])
                best_var = var_idx

            # Flip best_var
            old_val = assignment[best_var]
            assignment[best_var] = not old_val

            # Update unsat_set and unsat_counts
            for (other_ci, other_lit) in var_to_clauses[best_var]:
                other_is_pos = other_lit > 0
                # Did this clause's satisfaction change?
                if other_ci in unsat_set:
                    # It was unsat. Check if it's now satisfied.
                    if assignment[best_var] == other_is_pos:
                        # This var now satisfies. Check if clause is fully satisfied.
                        now_satisfied = True
                        for v_idx, v_pos in clause_vars[other_ci]:
                            if v_idx == best_var:
                                continue
                            if assignment[v_idx] != v_pos:
                                now_satisfied = False
                                break
                        if now_satisfied:
                            unsat_set.discard(other_ci)
                            del unsat_counts[other_ci]
                        else:
                            unsat_counts[other_ci] -= 1
                    else:
                        # This var still doesn't satisfy (wasn't satisfying before either since clause was unsat)
                        # Actually if it was unsat before and we flipped, it might have been the one unsat
                        # or it might still be unsat. Let's just recount if needed.
                        # If it was satisfying before (impossible for unsat clause with this var being flipped to not-satisfy)
                        # Skip complex logic; recount is safer for correctness
                        pass
                else:
                    # It was satisfied. Check if it's now unsatisfied.
                    if not (assignment[best_var] == other_is_pos):
                        # This var no longer satisfies. Check if clause is now unsatisfied.
                        now_unsat = True
                        for v_idx, v_pos in clause_vars[other_ci]:
                            if v_idx == best_var:
                                continue
                            if assignment[v_idx] != v_pos:
                                now_unsat = False
                                break
                        if now_unsat:
                            unsat_set.add(other_ci)
                            unsat_counts[other_ci] = 1
                        else:
                            pass

            # Fix up unsat_counts for clauses that might be stale
            # For clauses in unsat_set, ensure count is accurate
            # This is expensive; do it periodically or trust the updates
            # Let's do a lighter check: for the flipped var's clauses, recount if in unsat_set
            # Already handled above for the flipped var's clauses.

            # Update var_freq lazily (or skip for speed; use simple heuristic)
            # For now, keep freq static per restart for speed

            steps_this_restart += 1

            if not unsat_set:
                return assignment

        # Track best
        current_unsat = len(unsat_set)
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment[:]

    if best_assignment is not None:
        return best_assignment
    return assignment
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I'm concerned about the complexity of the incremental update logic. Let me simplify to a cleaner incremental approach that's more likely to be correct and fast. Let me reconsider.

The key insight from the observed improvements is:
1. Incremental clause-status tracking (unsat_set + var_to_clauses index)
2. Frequency-based literal-selection heuristic
3. This reduces per-step cost from O(num_clauses * clause_len) to O(deg(var))

Let me rewrite with a cleaner approach:

