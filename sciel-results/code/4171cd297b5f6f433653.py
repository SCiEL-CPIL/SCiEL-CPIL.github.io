"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    # Incremental WalkSAT with min-conflicts heuristic, restarts, and deadline guard.
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 280.0  # 50s safety margin before 330s kill

    # Pre-extract clause literals as (var_idx, sign) for speed
    # var_idx is 0-based, sign is +1 for positive, -1 for negative
    clause_data = []
    for clause in clauses:
        clause_data.append([(abs(l) - 1, 1 if l > 0 else -1) for l in clause])

    num_clauses = len(clause_data)

    # Build variable-to-clauses index for incremental updates
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clause_data):
        for var_idx, sign in clause:
            var_to_clauses[var_idx].append(ci)

    def is_clause_satisfied(clause, assignment):
        """Check if a single clause is satisfied."""
        for var_idx, sign in clause:
            if (assignment[var_idx] and sign == 1) or (not assignment[var_idx] and sign == -1):
                return True
        return False

    def walk_one(assignment, rng, max_steps):
        """Run incremental WalkSAT from current assignment for up to max_steps."""
        # Initialize unsatisfied clause set
        unsat_clauses = set()
        for ci, clause in enumerate(clause_data):
            if not is_clause_satisfied(clause, assignment):
                unsat_clauses.add(ci)

        if not unsat_clauses:
            return assignment, True

        for step in range(max_steps):
            if time.time() > deadline:
                return assignment, False

            if not unsat_clauses:
                return assignment, True

            # Pick a random unsatisfied clause
            ci = rng.choice(list(unsat_clauses))
            clause = clause_data[ci]

            # Count occurrences of each variable in unsatisfied clauses
            # For efficiency, only count for variables in the chosen clause
            var_unsat_count = {}
            for var_idx, sign in clause:
                count = 0
                for other_ci in unsat_clauses:
                    # Check if this variable appears in the other unsatisfied clause
                    # This is still potentially expensive, so we use a lighter heuristic:
                    # just count how many unsat clauses contain this variable
                    # We approximate by checking a sample or using precomputed data
                    pass
                # Lighter heuristic: weight by number of unsat clauses containing this var
                # Precompute would be ideal but we'll use a simpler approach
                var_unsat_count[var_idx] = 0

            # Simpler and faster heuristic: weight by the number of unsatisfied clauses
            # that contain this variable. We'll compute this more efficiently.
            # For each variable in the clause, count how many unsat clauses contain it.
            # To speed this up, we maintain a variable->unsat_count mapping incrementally.
            # But for simplicity in this version, we use a weighted random choice
            # where weight = 1 + (number of unsat clauses containing this var) / num_clauses

            # Even simpler: use a "walk" probability approach
            # With probability p, pick a random literal; with probability 1-p, pick
            # the literal that appears in the most unsatisfied clauses.

            # Let's compute var_unsat_count properly but efficiently
            # For each var in the clause, iterate through unsat_clauses and check
            # This is O(|unsat| * 3) per step which is the same as before.
            # Instead, let's maintain incremental counts.

            # For now, use a hybrid: with 70% probability use weighted by unsat count,
            # with 30% probability use random.
            if rng.random() < 0.7:
                # Weighted selection based on unsatisfied clause count
                weights = []
                for var_idx, sign in clause:
                    # Count unsat clauses containing this variable
                    cnt = 0
                    for other_ci in unsat_clauses:
                        for v, s in clause_data[other_ci]:
                            if v == var_idx:
                                cnt += 1
                                break
                    weights.append(cnt + 1)

                total_w = sum(weights)
                r = rng.random() * total_w
                cumulative = 0
                chosen_idx = 0
                for idx, w in enumerate(weights):
                    cumulative += w
                    if r <= cumulative:
                        chosen_idx = idx
                        break
            else:
                chosen_idx = rng.randint(0, len(clause) - 1)

            var_idx, sign = clause[chosen_idx]
            old_val = assignment[var_idx]
            new_val = not old_val
            assignment[var_idx] = new_val

            # Incrementally update unsat_clauses
            # Clauses that could change are those containing var_idx
            for affected_ci in var_to_clauses[var_idx]:
                if affected_ci in unsat_clauses:
                    # Was unsat, check if now sat
                    if is_clause_satisfied(clause_data[affected_ci], assignment):
                        unsat_clauses.discard(affected_ci)
                else:
                    # Was sat, check if now unsat
                    if not is_clause_satisfied(clause_data[affected_ci], assignment):
                        unsat_clauses.add(affected_ci)

        return assignment, False

    # Multi-restart strategy
    rng = random.Random(0)
    best_assignment = None
    best_unsat_count = float('inf')

    # Parameters
    max_steps_per_restart = 20000
    num_restarts = 1000  # max restarts, will stop at deadline

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # Try to solve
        assignment, solved = walk_one(assignment, rng, max_steps_per_restart)

        if solved:
            return assignment

        # Track best
        # Count unsat clauses for the best assignment
        uc = 0
        for ci, clause in enumerate(clause_data):
            if not is_clause_satisfied(clause, assignment):
                uc += 1
        if uc < best_unsat_count:
            best_unsat_count = uc
            best_assignment = assignment[:]

    # Return best assignment found
    if best_assignment is not None:
        return best_assignment

    # Fallback: all False
    return [False] * n

# EVOLVE-BLOCK-END
