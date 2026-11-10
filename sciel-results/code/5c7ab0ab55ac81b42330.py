"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    """Incremental WalkSAT with maintained conflict counts, phase saving, and adaptive walk probability."""
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 285.0  # 45s safety margin before 330s kill

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

    def init_unsat_and_counts(assignment):
        """Initialize unsat clause set and per-variable unsat clause counts."""
        unsat_clauses = set()
        var_unsat_count = [0] * n
        for ci, clause in enumerate(clause_data):
            if not is_clause_satisfied(clause, assignment):
                unsat_clauses.add(ci)
                for var_idx, _ in clause:
                    var_unsat_count[var_idx] += 1
        return unsat_clauses, var_unsat_count

    def walk_one(assignment, rng, max_steps, phase_history=None):
        """Run incremental WalkSAT with dynamic clause weights and perturbed restarts."""
        unsat_clauses, var_unsat_count = init_unsat_and_counts(assignment)

        if not unsat_clauses:
            return assignment, True

        # Dynamic clause weights: clauses that stay unsatisfied get higher weight
        clause_weights = [1.0] * num_clauses
        clause_age = [0] * num_clauses  # how long each clause has been unsatisfied

        # Adaptive walk probability
        walk_prob = 0.3
        best_uc_in_run = len(unsat_clauses)
        best_assignment_in_run = assignment[:]
        steps_since_improve = 0
        improve_threshold = 300

        # Weight decay and growth parameters
        weight_decay = 0.98
        weight_growth = 1.05
        max_weight = 50.0
        weight_update_interval = 100  # update weights every N steps

        for step in range(max_steps):
            if time.time() > deadline:
                return assignment, False

            if not unsat_clauses:
                return assignment, True

            # Periodically update clause weights
            if step % weight_update_interval == 0 and step > 0:
                # Decay weights for all clauses
                for ci in range(num_clauses):
                    clause_weights[ci] *= weight_decay
                # Grow weights for currently unsatisfied clauses
                for ci in unsat_clauses:
                    clause_weights[ci] *= weight_growth
                    clause_weights[ci] = min(clause_weights[ci], max_weight)
                    clause_age[ci] += 1

            # Pick a weighted random unsatisfied clause
            # Build weighted list for selection
            unsat_list = list(unsat_clauses)
            if len(unsat_list) <= 10:
                ci = rng.choice(unsat_list)
            else:
                # Use weighted selection for larger sets
                weights = [clause_weights[ci] for ci in unsat_list]
                total_w = sum(weights)
                r = rng.random() * total_w
                cumulative = 0
                ci = unsat_list[0]
                for i, w in enumerate(weights):
                    cumulative += w
                    if r <= cumulative:
                        ci = unsat_list[i]
                        break

            clause = clause_data[ci]

            # Choose which literal to flip
            # With probability walk_prob: random literal from clause
            # Otherwise: pick literal whose variable appears in most unsat clauses
            # (weighted by clause weights for better signal)
            if rng.random() < walk_prob:
                chosen_idx = rng.randint(0, len(clause) - 1)
            else:
                # Pick variable with highest weighted unsat count among clause variables
                # Compute weighted score for each variable in the clause
                best_var = clause[0][0]
                best_score = var_unsat_count[best_var] * clause_weights[ci]
                for i in range(1, len(clause)):
                    v = clause[i][0]
                    score = var_unsat_count[v] * clause_weights[ci]
                    if score > best_score:
                        best_score = score
                        best_var = v
                # Find index of best_var in clause
                for i in range(len(clause)):
                    if clause[i][0] == best_var:
                        chosen_idx = i
                        break

            var_idx, sign = clause[chosen_idx]
            old_val = assignment[var_idx]
            new_val = not old_val
            assignment[var_idx] = new_val

            # Incrementally update unsat_clauses and var_unsat_count
            for affected_ci in var_to_clauses[var_idx]:
                was_unsat = affected_ci in unsat_clauses
                is_now_sat = is_clause_satisfied(clause_data[affected_ci], assignment)

                if was_unsat and is_now_sat:
                    unsat_clauses.discard(affected_ci)
                    for v, _ in clause_data[affected_ci]:
                        var_unsat_count[v] -= 1
                    # Reset age and reduce weight for satisfied clause
                    clause_age[affected_ci] = 0
                elif not was_unsat and not is_now_sat:
                    unsat_clauses.add(affected_ci)
                    for v, _ in clause_data[affected_ci]:
                        var_unsat_count[v] += 1

            # Track progress and adapt walk probability
            current_uc = len(unsat_clauses)
            if current_uc < best_uc_in_run:
                best_uc_in_run = current_uc
                best_assignment_in_run = assignment[:]
                steps_since_improve = 0
                walk_prob = max(0.15, walk_prob * 0.95)
            else:
                steps_since_improve += 1
                if steps_since_improve > improve_threshold:
                    # Perturbed restart: flip a few random variables to escape local minimum
                    num_flips = rng.randint(1, min(5, n // 10))
                    for _ in range(num_flips):
                        flip_var = rng.randint(0, n - 1)
                        assignment[flip_var] = not assignment[flip_var]
                    
                    # Reinitialize unsat counts after perturbation
                    unsat_clauses, var_unsat_count = init_unsat_and_counts(assignment)
                    if not unsat_clauses:
                        return assignment, True
                    
                    # Reset some state
                    steps_since_improve = 0
                    improve_threshold = max(200, improve_threshold * 0.85)
                    walk_prob = min(0.6, walk_prob * 1.1)
                    # Reset clause ages
                    clause_age = [0] * num_clauses

        return assignment, False

    # Phase saving: remember which value of each variable led to more progress
    rng = random.Random(0)
    best_assignment = None
    best_unsat_count = float('inf')

    # Parameters
    max_steps_per_restart = 20000
    num_restarts = 2000  # max restarts, will stop at deadline

    # Phase history: for each variable, track which value was "good"
    phase_good = [None] * n  # None means undecided

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Initial assignment: use phase history if available, else random
        assignment = []
        for i in range(n):
            if phase_good[i] is not None:
                assignment.append(phase_good[i])
            else:
                assignment.append(rng.random() < 0.5)

        # Try to solve
        assignment, solved = walk_one(assignment, rng, max_steps_per_restart)

        if solved:
            return assignment

        # Update phase history based on best assignment found
        current_uc = sum(1 for ci, c in enumerate(clause_data) if not is_clause_satisfied(c, assignment))
        if current_uc < best_unsat_count:
            best_unsat_count = current_uc
            best_assignment = assignment[:]
            for i in range(n):
                phase_good[i] = assignment[i]

    # Return best assignment found
    if best_assignment is not None:
        return best_assignment

    # Fallback: all False
    return [False] * n

# EVOLVE-BLOCK-END
