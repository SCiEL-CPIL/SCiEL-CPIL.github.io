"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    # Incremental WalkSAT with var_unsat_count tracking, restarts, and deadline guard.
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 280.0  # 50s safety margin before 330s kill

    # Pre-extract clause literals as (var_idx, sign) for speed
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

    def init_state(assignment):
        """Initialize unsat_clauses set and var_unsat_count array."""
        unsat_clauses = set()
        var_unsat_count = [0] * n
        for ci, clause in enumerate(clause_data):
            if not is_clause_satisfied(clause, assignment):
                unsat_clauses.add(ci)
                for var_idx, sign in clause:
                    var_unsat_count[var_idx] += 1
        return unsat_clauses, var_unsat_count

    def flip_var(assignment, var_idx, unsat_clauses, var_unsat_count):
        """Flip a variable and incrementally update unsat_clauses and var_unsat_count."""
        old_val = assignment[var_idx]
        assignment[var_idx] = not old_val

        for affected_ci in var_to_clauses[var_idx]:
            clause = clause_data[affected_ci]
            if affected_ci in unsat_clauses:
                # Was unsat, check if now sat
                if is_clause_satisfied(clause, assignment):
                    unsat_clauses.discard(affected_ci)
                    # Decrement var_unsat_count for all vars in this clause
                    for v, s in clause:
                        var_unsat_count[v] -= 1
            else:
                # Was sat, check if now unsat
                if not is_clause_satisfied(clause, assignment):
                    unsat_clauses.add(affected_ci)
                    # Increment var_unsat_count for all vars in this clause
                    for v, s in clause:
                        var_unsat_count[v] += 1

    def walk_one(assignment, rng, max_steps, unsat_clauses, var_unsat_count):
        """Run incremental WalkSAT from current state for up to max_steps."""
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

            # Weighted selection based on var_unsat_count
            # With 70% probability use weighted, 30% random (walk probability)
            if rng.random() < 0.7:
                weights = []
                for var_idx, sign in clause:
                    weights.append(var_unsat_count[var_idx] + 1)
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
            flip_var(assignment, var_idx, unsat_clauses, var_unsat_count)

        return assignment, False

    # Multi-restart strategy
    rng = random.Random(0)
    best_assignment = None
    best_unsat_count = float('inf')

    # Adaptive restart strategy: start with fewer steps, increase over time
    base_steps = 5000
    max_restarts = 2000

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # Initialize state
        unsat_clauses, var_unsat_count = init_state(assignment)

        # Try to solve with adaptive step count
        # Use more steps as we go (geometric increase capped)
        steps = min(base_steps * (1 + restart // 10), 50000)
        assignment, solved = walk_one(assignment, rng, steps, unsat_clauses, var_unsat_count)

        if solved:
            return assignment

        # Track best
        uc = len(unsat_clauses)
        if uc < best_unsat_count:
            best_unsat_count = uc
            best_assignment = assignment[:]

    # Return best assignment found
    if best_assignment is not None:
        return best_assignment

    # Fallback: all False
    return [False] * n

# EVOLVE-BLOCK-END
