"""Planted 3-SAT solver using WalkSAT with gain heuristic and restarts."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 280  # safety margin before 300s limit

    # Precompute variable-to-clause index for fast gain computation
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var_to_clauses[abs(lit) - 1].append(ci)

    def clause_satisfied(clause, assignment):
        for lit in clause:
            if assignment[abs(lit) - 1] == (lit > 0):
                return True
        return False

    def count_unsat(clauses, assignment):
        cnt = 0
        for c in clauses:
            if not clause_satisfied(c, assignment):
                cnt += 1
        return cnt

    def walksat_with_gains(n, clauses, var_to_clauses, seed):
        rng = random.Random(seed)
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # Track unsatisfied clauses
        unsat_clauses = set()
        for ci, c in enumerate(clauses):
            if not clause_satisfied(c, assignment):
                unsat_clauses.add(ci)

        max_steps = 50000
        for step in range(max_steps):
            if time.time() > deadline:
                return None
            if not unsat_clauses:
                return assignment

            # Pick a random unsatisfied clause
            ci = rng.choice(list(unsat_clauses))
            clause = clauses[ci]

            # Compute gain for each literal in the clause
            # Gain = (number of other unsat clauses that would become satisfied)
            #       - (number of currently satisfied clauses that would become unsat)
            best_var = -1
            best_gain = -1
            best_val = None
            for lit in clause:
                v = abs(lit) - 1
                target_val = (lit > 0)
                # If variable already has this value, flipping it won't help this clause
                # but we still consider it (though gain will likely be negative)

                gain = 0
                # For each clause containing variable v, check effect of flipping
                for cj in var_to_clauses[v]:
                    if cj == ci:
                        continue
                    other_clause = clauses[cj]
                    # Check if this other clause is currently satisfied by v
                    v_satisfies_other = False
                    for ol in other_clause:
                        ov = abs(ol) - 1
                        if ov == v:
                            if assignment[v] == (ol > 0):
                                v_satisfies_other = True
                            break

                    # Check if other clause is satisfied by any other variable
                    other_satisfied = False
                    for ol in other_clause:
                        ov = abs(ol) - 1
                        if ov != v and assignment[ov] == (ol > 0):
                            other_satisfied = True
                            break

                    # If v currently satisfies other_clause and no other var satisfies it,
                    # flipping v makes it unsat: gain -= 1
                    if v_satisfies_other and not other_satisfied:
                        gain -= 1
                    # If v does NOT currently satisfy other_clause, and no other var
                    # satisfies it (so it's currently unsat), flipping v to target_val
                    # might satisfy it if target_val matches the literal sign
                    elif not v_satisfies_other and not other_satisfied:
                        # Check if setting v to target_val would satisfy this clause
                        for ol in other_clause:
                            ov = abs(ol) - 1
                            if ov == v:
                                if target_val == (ol > 0):
                                    gain += 1
                                break

                if gain > best_gain:
                    best_gain = gain
                    best_var = v
                    best_val = target_val

            if best_var == -1:
                return None

            # Flip the variable
            old_val = assignment[best_var]
            assignment[best_var] = best_val

            # Update unsat_clauses: check clauses containing best_var
            for cj in var_to_clauses[best_var]:
                c = clauses[cj]
                was_unsat = cj in unsat_clauses
                is_satisfied = clause_satisfied(c, assignment)
                if was_unsat and is_satisfied:
                    unsat_clauses.discard(cj)
                elif not was_unsat and not is_satisfied:
                    unsat_clauses.add(cj)

        return None

    # Try multiple restarts with different seeds
    best_result = None
    seed = 0
    while time.time() < deadline - 5:
        result = walksat_with_gains(n, clauses, var_to_clauses, seed)
        if result is not None:
            # Verify
            if all(clause_satisfied(c, result) for c in clauses):
                return result
        seed += 1
        # Increase restart frequency if stuck
        if seed % 50 == 0:
            pass  # already restarting each iteration

    # Final fallback: return best effort or random
    if best_result is not None:
        return best_result
    rng = random.Random(42)
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
