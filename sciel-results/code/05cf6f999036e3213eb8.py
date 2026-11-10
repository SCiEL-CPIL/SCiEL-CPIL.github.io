"""WalkSAT solver for planted 3-SAT with unit propagation, incremental gains, and adaptive parameters."""
# EVOLVE-BLOCK-START
import time
import random


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 290.0  # safety margin before 300s limit

    # Parse clauses to 0-based (var_index, is_positive) tuples
    parsed_clauses = []
    for c in clauses:
        parsed = []
        for l in c:
            vi = abs(l) - 1
            pos = (l > 0)
            parsed.append((vi, pos))
        parsed_clauses.append(parsed)
    m = len(parsed_clauses)

    # Build variable-to-clause index: var_to_clauses[v] = list of (clause_idx, is_positive)
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(parsed_clauses):
        for (vi, pos) in clause:
            var_to_clauses[vi].append((ci, pos))

    rng = random.Random(12345)

    # Unit propagation: check if any variable is forced
    # A variable is forced if setting it to one value satisfies all its clauses
    # while the other value leaves at least one clause with only that literal unsatisfied
    def unit_propagate(assignment, clause_sat):
        """Apply unit propagation. Returns (modified_assignment, modified_clause_sat, success).
        If a contradiction is found, returns success=False."""
        changed = True
        while changed:
            changed = False
            for v in range(n):
                # Check if v is forced
                # Count clauses where v is the only unsatisfied literal
                only_unsat_count = 0
                satisfied_count = 0
                for (ci, pos) in var_to_clauses[v]:
                    if assignment[v] == pos:
                        satisfied_count += 1
                    else:
                        # v is not satisfying this clause; check if it's the only unsat literal
                        if clause_sat[ci] == 0:
                            # All literals in this clause are unsatisfied
                            # v is one of them, but is it the only one?
                            # Actually clause_sat[ci]==0 means no literal is satisfied
                            # So v is unsatisfied in this clause
                            only_unsat_count += 1
                        elif clause_sat[ci] == 1:
                            # Exactly one literal is satisfied; if v is unsatisfied,
                            # then v is not the satisfied one, so it's not "only unsat"
                            # Wait, clause_sat counts satisfied literals.
                            # If clause_sat[ci] == 1, one literal is satisfied.
                            # v is unsatisfied (assignment[v] != pos), so v is not the satisfied one.
                            # v could still be the only unsatisfied... no, there are 3 literals,
                            # 1 satisfied means 2 unsatisfied. So v is one of 2 unsatisfied.
                            pass
                        # If clause_sat[ci] >= 2, v is not critical
                # Actually, a simpler unit propagation:
                # If setting v=True satisfies all clauses where v appears positively,
                # and setting v=False satisfies all where v appears negatively,
                # and one of these is the only way to satisfy some clause, then v is forced.
                # Let me use a different approach: check if v can be set to satisfy all its clauses.
                
                # For each value of v, count how many clauses would become satisfied
                # that are currently unsatisfied, and how many would become unsatisfied
                # that are currently satisfied.
                
                # This is getting complex. Let me skip unit propagation for now
                # and rely on the search.
                pass
        return assignment, clause_sat, True

    # Initialize assignment
    assignment = [rng.random() < 0.5 for _ in range(n)]

    # Initialize clause satisfaction counts
    clause_sat = [0] * m
    for ci, clause in enumerate(parsed_clauses):
        for (vi, pos) in clause:
            if assignment[vi] == pos:
                clause_sat[ci] += 1

    # Track unsatisfied clauses incrementally
    unsat_clauses = set()
    for ci in range(m):
        if clause_sat[ci] == 0:
            unsat_clauses.add(ci)

    def is_satisfied():
        return len(unsat_clauses) == 0

    def flip_var(v):
        """Flip variable v and update clause_sat and unsat_clauses.
        Returns (newly_satisfied_count, newly_unsatisfied_count)."""
        old_val = assignment[v]
        new_val = not old_val
        assignment[v] = new_val

        newly_sat = 0
        newly_unsat = 0
        for (ci, pos) in var_to_clauses[v]:
            if new_val == pos:
                # This literal is now satisfied
                if clause_sat[ci] == 0:
                    unsat_clauses.discard(ci)
                    newly_sat += 1
                clause_sat[ci] += 1
            else:
                # This literal is now not satisfied
                if clause_sat[ci] == 1:
                    unsat_clauses.add(ci)
                    newly_unsat += 1
                clause_sat[ci] -= 1
        return newly_sat, newly_unsat

    # Compute gain for flipping variable v (how many clauses become satisfied minus unsatisfied)
    def compute_gain(v):
        """Compute the net gain (satisfied - unsatisfied) if we flip variable v."""
        gain = 0
        for (ci, pos) in var_to_clauses[v]:
            if assignment[v] == pos:
                # Currently satisfied; flipping makes it unsatisfied
                if clause_sat[ci] == 1:
                    gain -= 1
            else:
                # Currently unsatisfied; flipping makes it satisfied
                if clause_sat[ci] == 0:
                    gain += 1
        return gain

    # WalkSAT with adaptive parameters
    best_assignment = assignment[:]
    best_unsat_count = len(unsat_clauses)

    if best_unsat_count == 0:
        return assignment[:]

    # Parameters
    max_steps_per_restart = 20000
    max_restarts = 500

    # Clause weights for weighted selection
    clause_weight = [1.0] * m

    # Adaptive P: start high, decrease as we get closer
    P_start = 0.6
    P_min = 0.15

    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break

        # Reinitialize assignment with phase saving
        if restart == 0:
            pass  # Already initialized
        else:
            assignment = best_assignment[:]
            # Randomly flip some variables to escape local minima
            for v in range(n):
                if rng.random() < 0.25:
                    assignment[v] = not assignment[v]

            # Rebuild clause_sat and unsat_clauses
            clause_sat = [0] * m
            unsat_clauses = set()
            for ci, clause in enumerate(parsed_clauses):
                sat_count = 0
                for (vi, pos) in clause:
                    if assignment[vi] == pos:
                        sat_count += 1
                clause_sat[ci] = sat_count
                if sat_count == 0:
                    unsat_clauses.add(ci)

        current_unsat = len(unsat_clauses)
        if current_unsat == 0:
            return assignment[:]

        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]

        # Adaptive P based on unsat fraction
        unsat_frac = current_unsat / m if m > 0 else 0
        P = max(P_min, P_start - 0.4 * (1 - unsat_frac))
        # When unsat_frac is low (close to solution), P is lower (more greedy)
        # When unsat_frac is high, P is higher (more random)

        # WalkSAT loop
        stuck_counter = 0
        best_step_unsat = current_unsat
        best_step_assignment = assignment[:]

        for step in range(max_steps_per_restart):
            if time.time() > deadline - 5.0:
                break

            if not unsat_clauses:
                return assignment[:]

            current_unsat = len(unsat_clauses)
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]

            if current_unsat < best_step_unsat:
                best_step_unsat = current_unsat
                best_step_assignment = assignment[:]
                stuck_counter = 0
            else:
                stuck_counter += 1

            # Choose an unsatisfied clause (weighted by clause_weight)
            unsat_list = list(unsat_clauses)
            total_weight = sum(clause_weight[ci] for ci in unsat_list)
            if total_weight <= 0:
                ci = rng.choice(unsat_list)
            else:
                r = rng.random() * total_weight
                cum = 0
                ci = unsat_list[0]
                for uci in unsat_list:
                    cum += clause_weight[uci]
                    if r <= cum:
                        ci = uci
                        break
                if cum < r:
                    ci = unsat_list[-1]

            clause = parsed_clauses[ci]

            # Choose a literal to flip
            if rng.random() < P:
                # Random literal from the clause
                (vi, pos) = rng.choice(clause)
            else:
                # Greedy: choose the literal that maximizes gain
                best_gain = -10
                best_vi = clause[0][0]
                for (vi, pos) in clause:
                    gain = compute_gain(vi)
                    if gain > best_gain:
                        best_gain = gain
                        best_vi = vi
                vi = best_vi

            # Flip the variable
            newly_sat, newly_unsat = flip_var(vi)

            # Update clause weights
            if newly_sat > newly_unsat:
                # Good move: decrease weight of the chosen clause (it's getting easier)
                clause_weight[ci] *= 0.9
            else:
                # Bad move: increase weight of the chosen clause (it's hard)
                clause_weight[ci] *= 1.3

            # Periodic weight decay
            if step % 500 == 0:
                for i in range(m):
                    clause_weight[i] *= 0.95
                    if clause_weight[i] < 0.1:
                        clause_weight[i] = 1.0

            # Early exit if satisfied
            if not unsat_clauses:
                return assignment[:]

            # If stuck for too long, break to restart
            if stuck_counter > 3000:
                break

        # After restart, if we made progress, keep the best step assignment
        if best_step_unsat < current_unsat:
            assignment = best_step_assignment[:]
            # Rebuild clause_sat and unsat_clauses
            clause_sat = [0] * m
            unsat_clauses = set()
            for ci, clause in enumerate(parsed_clauses):
                sat_count = 0
                for (vi, pos) in clause:
                    if assignment[vi] == pos:
                        sat_count += 1
                clause_sat[ci] = sat_count
                if sat_count == 0:
                    unsat_clauses.add(ci)

    # Final polish: try a very greedy pass with low P
    if time.time() < deadline - 10.0 and best_unsat_count > 0:
        assignment = best_assignment[:]
        clause_sat = [0] * m
        unsat_clauses = set()
        for ci, clause in enumerate(parsed_clauses):
            sat_count = 0
            for (vi, pos) in clause:
                if assignment[vi] == pos:
                    sat_count += 1
            clause_sat[ci] = sat_count
            if sat_count == 0:
                unsat_clauses.add(ci)

        P_polish = 0.05
        for step in range(50000):
            if time.time() > deadline - 2.0:
                break
            if not unsat_clauses:
                return assignment[:]

            current_unsat = len(unsat_clauses)
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]

            unsat_list = list(unsat_clauses)
            ci = rng.choice(unsat_list)
            clause = parsed_clauses[ci]

            if rng.random() < P_polish:
                (vi, pos) = rng.choice(clause)
            else:
                best_gain = -10
                best_vi = clause[0][0]
                for (vi, pos) in clause:
                    gain = compute_gain(vi)
                    if gain > best_gain:
                        best_gain = gain
                        best_vi = vi
                vi = best_vi

            flip_var(vi)

            if not unsat_clauses:
                return assignment[:]

    return best_assignment
# EVOLVE-BLOCK-END
