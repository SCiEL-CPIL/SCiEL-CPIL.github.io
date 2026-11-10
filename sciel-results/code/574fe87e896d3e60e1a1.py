"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 330s kill

    # Preprocess: build variable frequency for weighting
    var_count = [0] * n
    for clause in clauses:
        for lit in clause:
            var_count[abs(lit) - 1] += 1

    # Efficient clause satisfaction tracking
    # For each clause, we track which literals are satisfied and the count
    # clause_sat[i] = number of satisfied literals in clause i
    # clause_unsat[i] = list of unsatisfied literal indices (or just track count)

    def init_tracking(assignment):
        """Initialize clause satisfaction tracking."""
        clause_sat_count = [0] * len(clauses)
        for i, clause in enumerate(clauses):
            cnt = 0
            for l in clause:
                if assignment[abs(l) - 1] == (l > 0):
                    cnt += 1
            clause_sat_count[i] = cnt
        return clause_sat_count

    def get_unsat_clauses(clause_sat_count):
        """Get list of unsatisfied clause indices."""
        return [i for i, cnt in enumerate(clause_sat_count) if cnt == 0]

    def flip_var(var, assignment, clause_sat_count):
        """Flip a variable and update clause satisfaction counts."""
        old_val = assignment[var]
        new_val = not old_val
        assignment[var] = new_val
        # Update all clauses containing this variable
        for i, clause in enumerate(clauses):
            for l in clause:
                v = abs(l) - 1
                if v == var:
                    # This literal's truth value changed
                    if new_val == (l > 0):
                        # Now satisfied
                        clause_sat_count[i] += 1
                    else:
                        # Now unsatisfied
                        clause_sat_count[i] -= 1
                    break  # Each clause has at most one occurrence of var in 3-SAT typically

    def count_unsat(clause_sat_count):
        return sum(1 for c in clause_sat_count if c == 0)

    def is_fully_satisfied(clause_sat_count):
        return all(c > 0 for c in clause_sat_count)

    # Unit propagation
    def unit_propagate(assignment):
        """Unit propagation: assign literals forced by unit clauses.
        Returns (success, assignment) where success indicates no conflict."""
        assignment = assignment[:]
        changed = True
        while changed:
            changed = False
            for clause in clauses:
                unassigned = [l for l in clause if assignment[abs(l) - 1] is None]
                satisfied = any(assignment[abs(l) - 1] is not None and assignment[abs(l) - 1] == (l > 0) for l in clause)
                if satisfied:
                    continue
                if len(unassigned) == 0:
                    return False, assignment
                if len(unassigned) == 1:
                    l = unassigned[0]
                    if assignment[abs(l) - 1] is None:
                        assignment[abs(l) - 1] = (l > 0)
                        changed = True
        return True, assignment

    def random_init():
        """Generate a random assignment with optional unit propagation."""
        assignment = [rng.random() < 0.5 for _ in range(n)]
        return assignment

    def random_init_with_prop():
        """Generate assignment with unit propagation from a random start."""
        # Start with None for unassigned
        assignment = [None] * n
        # Randomly assign some variables first to trigger propagation
        for i in range(n):
            if rng.random() < 0.3:  # 30% chance to pre-assign
                assignment[i] = rng.random() < 0.5
        success, assignment = unit_propagate(assignment)
        if not success:
            # Conflict, just use random
            return [rng.random() < 0.5 for _ in range(n)]
        # Fill remaining unassigned
        for i in range(n):
            if assignment[i] is None:
                assignment[i] = rng.random() < 0.5
        return assignment

    # WalkSAT with gain/loss heuristic and weighted clause selection
    def walksat(assignment, max_steps=10000, p_gain=0.6):
        """WalkSAT with gain/loss heuristic."""
        clause_sat_count = init_tracking(assignment)

        for step in range(max_steps):
            if time.time() >= deadline:
                break

            unsat_indices = get_unsat_clauses(clause_sat_count)
            if not unsat_indices:
                break

            # Weighted selection of unsatisfied clause: prefer shorter/more constrained
            # Use inverse frequency weighting for clause selection
            weights = []
            for idx in unsat_indices:
                clause = clauses[idx]
                # Lower total frequency of variables in clause -> more likely planted
                w = 1.0
                for l in clause:
                    v = abs(l) - 1
                    w *= (1.0 / (var_count[v] + 1.0))
                weights.append(w)

            total_w = sum(weights)
            if total_w <= 0:
                clause_idx = rng.choice(unsat_indices)
            else:
                r = rng.random() * total_w
                cum = 0
                clause_idx = unsat_indices[-1]
                for i, w in enumerate(weights):
                    cum += w
                    if r <= cum:
                        clause_idx = unsat_indices[i]
                        break

            clause = clauses[clause_idx]

            # Compute gain/loss for each literal in the clause
            best_var = -1
            best_score = -10000

            for l in clause:
                var = abs(l) - 1
                # Simulate flip: count how many unsat clauses would be fixed
                # and how many sat clauses would be broken
                gain = 0
                loss = 0

                # For efficiency, only check clauses containing this variable
                for i, c in enumerate(clauses):
                    has_var = False
                    for ll in c:
                        if abs(ll) - 1 == var:
                            has_var = True
                            break
                    if not has_var:
                        continue

                    if clause_sat_count[i] > 0:
                        # Currently satisfied: would flipping break it?
                        # Only if this var's literal is the ONLY satisfied literal
                        only_sat = True
                        for ll in c:
                            v2 = abs(ll) - 1
                            if v2 == var:
                                # This literal would become unsatisfied
                                continue
                            elif assignment[v2] == (ll > 0):
                                # Another literal is satisfied, so clause stays sat
                                only_sat = False
                                break
                        if only_sat:
                            loss += 1
                    else:
                        # Currently unsatisfied: would flipping satisfy it?
                        # Check if the literal for var in this clause would become true
                        for ll in c:
                            if abs(ll) - 1 == var:
                                # After flip, this literal's value changes
                                # If it was false and becomes true, clause is satisfied
                                if not (assignment[var] == (ll > 0)):
                                    # Currently false, will become true
                                    gain += 1
                                break

                score = gain - loss
                if score > best_score:
                    best_score = score
                    best_var = var

            # Decide which variable to flip
            if best_var >= 0 and (rng.random() < p_gain or best_score > 0):
                var_to_flip = best_var
            else:
                # Random variable from the clause
                l = rng.choice(clause)
                var_to_flip = abs(l) - 1

            flip_var(var_to_flip, assignment, clause_sat_count)

        return assignment

    # Simulated annealing as a secondary method
    def simulated_annealing(assignment, max_iters=50000, T_start=2.0, T_end=0.01):
        """Simulated annealing for 3-SAT."""
        clause_sat_count = init_tracking(assignment)
        curr_cost = count_unsat(clause_sat_count)
        best_assign = assignment[:]
        best_cost = curr_cost

        T = T_start
        alpha = (T_end / T_start) ** (1.0 / max(max_iters, 1))

        for it in range(max_iters):
            if time.time() >= deadline:
                break
            if curr_cost == 0:
                break

            # Pick a random variable to flip
            var = rng.randrange(n)
            old_val = assignment[var]
            flip_var(var, assignment, clause_sat_count)
            new_cost = count_unsat(clause_sat_count)

            delta = new_cost - curr_cost
            if delta < 0 or (T > 0 and rng.random() < math.exp(-delta / max(T, 1e-10))):
                curr_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_assign = assignment[:]
            else:
                # Revert
                flip_var(var, assignment, clause_sat_count)

            T *= alpha

        if best_cost == 0:
            return best_assign
        return best_assign

    # Main search loop with restarts
    best_assignment = None
    best_unsat_count = float('inf')

    max_restarts = 100
    restarts = 0

    while time.time() < deadline and restarts < max_restarts:
        restarts += 1

        # Choose initialization strategy
        if restarts % 3 == 1:
            assignment = random_init_with_prop()
        elif restarts % 3 == 2:
            assignment = random_init()
        else:
            # Use previous best if available
            if best_assignment is not None:
                assignment = best_assignment[:]
                # Perturb slightly
                for _ in range(min(5, n)):
                    v = rng.randrange(n)
                    assignment[v] = not assignment[v]
            else:
                assignment = random_init()

        # Run WalkSAT
        max_steps = 5000 + rng.randint(0, 5000)
        p_gain = 0.5 + 0.3 * rng.random()
        assignment = walksat(assignment, max_steps=max_steps, p_gain=p_gain)

        # Check result
        clause_sat_count = init_tracking(assignment)
        unsat_count = count_unsat(clause_sat_count)

        if unsat_count == 0:
            return assignment

        # Track best
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]

        # If not solved, try simulated annealing from current state
        if time.time() < deadline - 10:
            assignment = simulated_annealing(assignment, max_iters=20000, T_start=1.0, T_end=0.01)
            clause_sat_count = init_tracking(assignment)
            unsat_count = count_unsat(clause_sat_count)
            if unsat_count == 0:
                return assignment
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment[:]

        # Final WalkSAT polish if we have time
        if time.time() < deadline - 5 and best_assignment is not None:
            assignment = walksat(best_assignment, max_steps=3000, p_gain=0.7)
            clause_sat_count = init_tracking(assignment)
            unsat_count = count_unsat(clause_sat_count)
            if unsat_count == 0:
                return assignment
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment[:]

    # Return best found
    if best_assignment is not None:
        return best_assignment
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
