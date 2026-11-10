"""Simulated annealing SAT solver with incremental tracking for planted 3-SAT."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290.0  # safety margin before 330s kill

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

    def init_state(assignment):
        """Initialize clause satisfaction state and unsat set."""
        clause_sat = [0] * num_clauses  # count of satisfied literals per clause
        for ci, clause in enumerate(clause_data):
            for var_idx, sign in clause:
                if (assignment[var_idx] and sign == 1) or (not assignment[var_idx] and sign == -1):
                    clause_sat[ci] += 1
        unsat_set = set(ci for ci in range(num_clauses) if clause_sat[ci] == 0)
        return clause_sat, unsat_set

    def flip_variable(assignment, var_idx, clause_sat, unsat_set):
        """Flip variable and update clause states incrementally. Returns delta_unsat."""
        old_val = assignment[var_idx]
        new_val = not old_val
        assignment[var_idx] = new_val

        delta_unsat = 0
        for ci in var_to_clauses[var_idx]:
            clause = clause_data[ci]
            # Find the sign of this variable in this clause
            sign = None
            for v, s in clause:
                if v == var_idx:
                    sign = s
                    break

            # Determine if this literal was satisfied before and after flip
            was_sat = (old_val and sign == 1) or (not old_val and sign == -1)
            is_sat = (new_val and sign == 1) or (not new_val and sign == -1)

            if was_sat and not is_sat:
                clause_sat[ci] -= 1
                if clause_sat[ci] == 0:
                    unsat_set.add(ci)
                    delta_unsat += 1
            elif not was_sat and is_sat:
                clause_sat[ci] += 1
                if clause_sat[ci] == 1:
                    unsat_set.discard(ci)
                    delta_unsat -= 1

        return delta_unsat

    def simulated_annealing(assignment, rng, max_steps, t_initial, t_final, cooling):
        """Run simulated annealing from current assignment."""
        clause_sat, unsat_set = init_state(assignment)
        current_unsat = len(unsat_set)
        best_unsat = current_unsat
        best_assignment = assignment[:]
        best_time = 0.0

        for step in range(max_steps):
            if time.time() > deadline:
                break

            # Compute current temperature
            if step < max_steps - 1:
                t = t_initial * (t_final / t_initial) ** (step / max_steps)
            else:
                t = t_final

            # If no unsatisfied clauses, we're done
            if current_unsat == 0:
                return assignment, True

            # Choose a variable to flip
            # With some probability, pick from unsatisfied clauses (greedy bias)
            # Otherwise, pick a random variable (exploration)
            if unsat_set and rng.random() < 0.7:
                # Pick a random unsatisfied clause
                ci = rng.choice(list(unsat_set))
                clause = clause_data[ci]
                # Pick a random variable from that clause
                var_idx, sign = rng.choice(clause)
            else:
                var_idx = rng.randint(0, n - 1)

            # Flip and compute delta
            old_unsat = current_unsat
            delta_unsat = flip_variable(assignment, var_idx, clause_sat, unsat_set)
            current_unsat = old_unsat + delta_unsat

            # Accept or reject based on simulated annealing
            if delta_unsat <= 0:
                # Always accept improvements
                if current_unsat < best_unsat:
                    best_unsat = current_unsat
                    best_assignment = assignment[:]
                    best_time = time.time()
            else:
                # Deterioration: accept with probability exp(-delta / T)
                if t > 0:
                    prob = math.exp(-delta_unsat / t)
                    if rng.random() < prob:
                        # Accept deterioration
                        pass
                    else:
                        # Reject: flip back
                        flip_variable(assignment, var_idx, clause_sat, unsat_set)
                        current_unsat = old_unsat

        # Return best found during this annealing run
        if best_unsat < current_unsat:
            # Restore best assignment
            assignment[:] = best_assignment
            return assignment, False
        return assignment, False

    # Multi-restart strategy with varying annealing parameters
    rng = random.Random(0)
    best_assignment = None
    best_unsat_count = float('inf')

    # Parameters tuned for planted 3-SAT with 150 variables
    max_steps_per_restart = 80000
    num_restarts = 2000  # max restarts, will stop at deadline

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # Vary annealing parameters across restarts
        # Higher initial temperature for more exploration early on
        t_initial = 5.0 + 3.0 * (restart % 4) / 3.0  # 5, 6, 7, 8
        t_final = 0.01
        cooling = 0.999

        # Run simulated annealing
        assignment, solved = simulated_annealing(assignment, rng, max_steps_per_restart, t_initial, t_final, cooling)

        if solved:
            return assignment

        # Track best
        clause_sat, unsat_set = init_state(assignment)
        uc = len(unsat_set)
        if uc < best_unsat_count:
            best_unsat_count = uc
            best_assignment = assignment[:]

    # Return best assignment found
    if best_assignment is not None:
        return best_assignment

    # Fallback: all False
    return [False] * n

# EVOLVE-BLOCK-END
