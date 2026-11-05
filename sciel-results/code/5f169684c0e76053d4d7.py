"""Synthetic planted 3-SAT solver (50 variables) with WalkSAT + simulated annealing hybrid."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    """WalkSAT with damage-minimizing flip selection and simulated annealing escape."""
    n, clauses = instance["n"], instance["clauses"]
    num_clauses = len(clauses)
    deadline = time.time() + 280  # Safety margin before 300s kill

    # Build variable-to-clause index for fast delta computation
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var_to_clauses[abs(lit) - 1].append(ci)

    def is_satisfied(clause, assignment):
        for lit in clause:
            if assignment[abs(lit) - 1] == (lit > 0):
                return True
        return False

    def count_sat(assignment):
        cnt = 0
        for clause in clauses:
            if is_satisfied(clause, assignment):
                cnt += 1
        return cnt

    def get_unsat_clauses(assignment):
        return [ci for ci in range(num_clauses) if not is_satisfied(clauses[ci], assignment)]

    def compute_damage(var_idx, assignment, clause_sat_flags):
        """
        Compute how many clauses would become unsatisfied if we flip var_idx.
        A clause becomes unsatisfied if it was satisfied ONLY by var_idx with its current value.
        """
        old_val = assignment[var_idx]
        damage = 0
        for ci in var_to_clauses[var_idx]:
            clause = clauses[ci]
            # Check if this clause is currently satisfied
            if not clause_sat_flags[ci]:
                continue
            # Check if it's satisfied by any other literal
            other_sat = False
            for lit in clause:
                v = abs(lit) - 1
                if v != var_idx:
                    if assignment[v] == (lit > 0):
                        other_sat = True
                        break
            if not other_sat:
                # This clause is satisfied only by var_idx, flipping would break it
                damage += 1
        return damage

    def compute_gain(var_idx, assignment, clause_sat_flags):
        """
        Compute how many unsatisfied clauses would become satisfied if we flip var_idx.
        """
        new_val = not assignment[var_idx]
        gain = 0
        for ci in var_to_clauses[var_idx]:
            clause = clauses[ci]
            # Check if this clause is currently unsatisfied
            if clause_sat_flags[ci]:
                continue
            # Check if flipping would satisfy it
            would_be_sat = False
            for lit in clause:
                v = abs(lit) - 1
                if v == var_idx:
                    if new_val == (lit > 0):
                        would_be_sat = True
                        break
                else:
                    if assignment[v] == (lit > 0):
                        would_be_sat = True
                        break
            if would_be_sat:
                gain += 1
        return gain

    def update_clause_flags(assignment, clause_sat_flags, affected_vars):
        """Update clause satisfaction flags for clauses involving affected variables."""
        affected_clauses = set()
        for v in affected_vars:
            for ci in var_to_clauses[v]:
                affected_clauses.add(ci)
        for ci in affected_clauses:
            clause_sat_flags[ci] = is_satisfied(clauses[ci], assignment)

    def walksat_flip(assignment, rng, unsat_indices, clause_sat_flags):
        """
        WalkSAT step: pick a random unsat clause, flip the literal that minimizes net damage.
        Returns (flipped_var, delta_sat, affected_clauses)
        """
        if not unsat_indices:
            return None, 0, []

        # Pick a random unsatisfied clause
        ci = rng.choice(unsat_indices)
        clause = clauses[ci]

        best_var = None
        best_score = -float('inf')

        for lit in clause:
            var_idx = abs(lit) - 1
            # Compute damage and gain for flipping this variable
            damage = compute_damage(var_idx, assignment, clause_sat_flags)
            gain = compute_gain(var_idx, assignment, clause_sat_flags)
            # Net change in satisfied clauses
            delta = gain - damage
            # Add small random noise to break ties
            score = delta + rng.random() * 0.1
            if score > best_score:
                best_score = score
                best_var = var_idx

        if best_var is None:
            return None, 0, []

        # Perform the flip
        assignment[best_var] = not assignment[best_var]

        # Update clause flags for affected clauses
        affected_clauses = list(var_to_clauses[best_var])
        for aci in affected_clauses:
            clause_sat_flags[aci] = is_satisfied(clauses[aci], assignment)

        # Compute actual delta
        old_sat = num_clauses - len(unsat_indices)
        new_unsat = [aci for aci in range(num_clauses) if not clause_sat_flags[aci]]
        new_sat = num_clauses - len(new_unsat)
        actual_delta = new_sat - old_sat

        return best_var, actual_delta, affected_clauses

    # Main search loop with restarts
    best_assignment = None
    best_sat_count = -1

    max_restarts = 40

    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break

        rng = random.Random(42 + restart * 137)
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # Initialize clause satisfaction flags
        clause_sat_flags = [is_satisfied(clauses[ci], assignment) for ci in range(num_clauses)]
        current_sat = sum(clause_sat_flags)

        if current_sat == num_clauses:
            return assignment

        if current_sat > best_sat_count:
            best_sat_count = current_sat
            best_assignment = assignment[:]

        # Adaptive parameters
        remaining_time = deadline - time.time()
        max_steps = int(min(15000, max(3000, remaining_time * 4000)))

        # WalkSAT with occasional random moves to escape local optima
        no_improve_steps = 0
        max_no_improve = 1500
        temperature = 1.0
        t_min = 0.05
        cooling_rate = (t_min / temperature) ** (1.0 / max(max_steps, 1))

        for step in range(max_steps):
            if time.time() > deadline - 5:
                break

            unsat_indices = [ci for ci in range(num_clauses) if not clause_sat_flags[ci]]

            if not unsat_indices:
                return assignment

            if current_sat > best_sat_count:
                best_sat_count = current_sat
                best_assignment = assignment[:]

            # Decide: WalkSAT flip or random escape move
            if no_improve_steps > max_no_improve and temperature > t_min:
                # Escape: flip a random variable (simulated annealing style)
                var_idx = rng.randrange(n)
                old_val = assignment[var_idx]
                assignment[var_idx] = not assignment[var_idx]

                # Update affected clauses
                for ci in var_to_clauses[var_idx]:
                    clause_sat_flags[ci] = is_satisfied(clauses[ci], assignment)

                new_sat = sum(clause_sat_flags)
                delta = new_sat - current_sat
                current_sat = new_sat

                # Accept if improvement, or with probability exp(delta/T) if deterioration
                if delta < 0:
                    prob = math.exp(delta / max(temperature, 1e-10))
                    if rng.random() >= prob:
                        # Reject: undo the flip
                        assignment[var_idx] = old_val
                        for ci in var_to_clauses[var_idx]:
                            clause_sat_flags[ci] = is_satisfied(clauses[ci], assignment)
                        current_sat -= delta
                else:
                    no_improve_steps = 0

                temperature *= cooling_rate
                step += 1
                continue

            # Normal WalkSAT step
            flipped_var, delta, affected = walksat_flip(assignment, rng, unsat_indices, clause_sat_flags)

            if flipped_var is None:
                break

            current_sat += delta

            if delta > 0:
                no_improve_steps = 0
            elif delta == 0:
                no_improve_steps += 1
            else:
                no_improve_steps += 2  # Penalize deterioration more

            temperature *= cooling_rate

            # Early termination if stuck
            if no_improve_steps > max_no_improve and temperature < 0.1:
                break

        # Final polish: try targeted flips on remaining unsat clauses
        if current_sat < num_clauses and time.time() < deadline - 5:
            unsat_indices = [ci for ci in range(num_clauses) if not clause_sat_flags[ci]]
            if unsat_indices:
                # Try each unsat clause, find the flip that gives best net gain
                for _ in range(min(100, len(unsat_indices))):
                    ci = rng.choice(unsat_indices)
                    clause = clauses[ci]
                    best_var = None
                    best_delta = -float('inf')
                    for lit in clause:
                        var_idx = abs(lit) - 1
                        damage = compute_damage(var_idx, assignment, clause_sat_flags)
                        gain = compute_gain(var_idx, assignment, clause_sat_flags)
                        delta = gain - damage
                        if delta > best_delta:
                            best_delta = delta
                            best_var = var_idx
                    if best_var is not None and best_delta > 0:
                        assignment[best_var] = not assignment[best_var]
                        for aci in var_to_clauses[best_var]:
                            clause_sat_flags[aci] = is_satisfied(clauses[aci], assignment)
                        current_sat += best_delta
                        if current_sat == num_clauses:
                            return assignment

            if current_sat > best_sat_count:
                best_sat_count = current_sat
                best_assignment = assignment[:]

    # Return best assignment found
    if best_assignment is not None:
        return best_assignment

    # Fallback
    return assignment
# EVOLVE-BLOCK-END
