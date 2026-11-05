"""Optimized 3-SAT solver with cached conflict counts and fast WalkSAT."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    deadline = time.time() + 285.0  # safety margin below 330s kill
    num_clauses = len(clauses)

    # Precompute clause info: for each clause, list of (var_idx, target_val)
    # Also build var_to_clauses mapping
    clause_info = []
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        info = []
        for lit in clause:
            var_idx = abs(lit) - 1
            target_val = (lit > 0)
            info.append((var_idx, target_val))
            var_to_clauses[var_idx].append(ci)
        clause_info.append(info)

    # For each clause, precompute which literal position each variable occupies
    # clause_var_pos[ci][var_idx] = position in clause_info[ci], or -1
    clause_var_pos = []
    for ci in range(num_clauses):
        pos_map = {}
        for pos, (v, t) in enumerate(clause_info[ci]):
            pos_map[v] = pos
        clause_var_pos.append(pos_map)

    def is_clause_sat(ci, assignment):
        """Check if clause ci is satisfied."""
        for v, t in clause_info[ci]:
            if assignment[v] == t:
                return True
        return False

    def compute_unsat(assignment):
        """Return list of unsatisfied clause indices."""
        return [ci for ci in range(num_clauses) if not is_clause_sat(ci, assignment)]

    def init_conflict_counts(assignment):
        """
        Compute conflict count for each variable.
        conflict[var] = number of clauses where this var is the ONLY satisfied literal.
        """
        conflict = [0] * n
        for ci in range(num_clauses):
            sat_vars = []
            for v, t in clause_info[ci]:
                if assignment[v] == t:
                    sat_vars.append(v)
            if len(sat_vars) == 1:
                conflict[sat_vars[0]] += 1
        return conflict

    def update_conflict_after_flip(var_idx, assignment, conflict):
        """
        Update conflict counts after flipping var_idx.
        Must be called AFTER the flip has been applied to assignment.
        """
        # For each clause containing var_idx, check if it's now unit-satisfied by var_idx
        for ci in var_to_clauses[var_idx]:
            # Count satisfied literals in this clause
            sat_count = 0
            sat_var = -1
            for v, t in clause_info[ci]:
                if assignment[v] == t:
                    sat_count += 1
                    sat_var = v
                    if sat_count > 1:
                        break
            # If exactly one satisfied literal and it's var_idx, increment
            # If previously it was unit-satisfied by var_idx, decrement
            # We need to track the old state, so let's recompute differently:
            # Actually, let's just recompute from scratch for simplicity if n is small
            pass
        # For 50 vars, recomputing is fast enough
        # But let's do it properly: recompute all conflict counts
        # Actually, for 50 variables, let's just recompute
        new_conflict = [0] * n
        for ci in range(num_clauses):
            sat_vars = []
            for v, t in clause_info[ci]:
                if assignment[v] == t:
                    sat_vars.append(v)
                    if len(sat_vars) > 1:
                        break
            if len(sat_vars) == 1:
                new_conflict[sat_vars[0]] += 1
        return new_conflict

    def make_random_assignment(seed):
        rng = random.Random(seed)
        return [rng.random() < 0.5 for _ in range(n)]

    def walksat_step(assignment, unsat_list, conflict, rng, p_walk):
        """
        Perform one WalkSAT step with min-confl heuristic using cached conflict counts.
        Returns (new_unsat_list, new_conflict, flipped_var).
        """
        if not unsat_list:
            return unsat_list, conflict, -1

        # Pick a random unsatisfied clause
        ci = rng.choice(unsat_list)
        clause = clause_info[ci]

        # Min-confl: pick variable with minimum conflict count
        best_var = None
        best_conf = float('inf')

        for var_idx, target_val in clause:
            conf = conflict[var_idx]
            if conf < best_conf:
                best_conf = conf
                best_var = var_idx

        if best_var is None:
            return unsat_list, conflict, -1

        # WalkSAT: with probability p_walk, pick random literal from clause
        if rng.random() < p_walk:
            best_var = rng.choice(clause)[0]

        # Flip the variable
        assignment[best_var] = not assignment[best_var]

        # Update unsat list
        new_unsat = set(unsat_list)
        for other_ci in var_to_clauses[best_var]:
            if is_clause_sat(other_ci, assignment):
                new_unsat.discard(other_ci)
            else:
                new_unsat.add(other_ci)
        new_unsat_list = list(new_unsat)

        # Update conflict counts
        new_conflict = update_conflict_after_flip(best_var, assignment, conflict)

        return new_unsat_list, new_conflict, best_var

    def final_polish(assignment, unsat_list, rng, max_steps):
        """Final high-precision polish with tighter parameters."""
        cur_unsat = len(unsat_list)
        if cur_unsat == 0:
            return assignment, 0

        conflict = init_conflict_counts(assignment)
        p_walk = 0.02
        steps = 0
        prev_unsat = cur_unsat
        while cur_unsat > 0 and steps < max_steps:
            if time.time() > deadline:
                break
            steps += 1
            new_unsat_list, new_conflict, _ = walksat_step(assignment, unsat_list, conflict, rng, p_walk)
            cur_unsat = len(new_unsat_list)
            unsat_list = new_unsat_list
            conflict = new_conflict
            if cur_unsat == 0:
                return assignment, 0
            if cur_unsat < prev_unsat:
                p_walk = max(0.01, p_walk * 0.9)
            else:
                if steps % 50 == 0:
                    p_walk = min(0.3, p_walk * 1.2)
            prev_unsat = cur_unsat
        return assignment, cur_unsat

    best_assignment = None
    best_unsat = num_clauses + 1

    # Try multiple random restarts with WalkSAT + min-confl
    for attempt in range(8000):
        if time.time() > deadline:
            break

        assignment = make_random_assignment(attempt * 7919 + 13)
        unsat_list = compute_unsat(assignment)
        cur_unsat = len(unsat_list)

        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
        if cur_unsat == 0:
            return assignment

        # Local search: WalkSAT with dynamic parameters
        rng = random.Random(attempt * 31337 + 42)
        steps = 0
        max_steps = 15000

        # Adaptive walk probability: start low, increase if stuck
        p_walk = 0.05
        stuck_counter = 0
        last_improvement_step = 0
        conflict = init_conflict_counts(assignment)

        while cur_unsat > 0 and steps < max_steps:
            if time.time() > deadline:
                break
            steps += 1

            new_unsat_list, new_conflict, _ = walksat_step(assignment, unsat_list, conflict, rng, p_walk)
            cur_unsat = len(new_unsat_list)
            unsat_list = new_unsat_list
            conflict = new_conflict

            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment[:]
                last_improvement_step = steps
            if cur_unsat == 0:
                return assignment

            # Track if we're stuck
            if steps - last_improvement_step > 200:
                stuck_counter += 1
                if stuck_counter > 3:
                    p_walk = min(0.5, p_walk * 1.5)
                    stuck_counter = 0
            else:
                stuck_counter = 0
                if cur_unsat < best_unsat:
                    p_walk = max(0.02, p_walk * 0.85)

            # If close to solution, do a polish pass
            if cur_unsat <= 3 and steps > 5000:
                polish_assignment, polish_unsat = final_polish(assignment, unsat_list, rng, 5000)
                if polish_unsat == 0:
                    return polish_assignment
                if polish_unsat < cur_unsat:
                    assignment = polish_assignment
                    cur_unsat = polish_unsat
                    unsat_list = compute_unsat(assignment)
                    conflict = init_conflict_counts(assignment)

    # Final attempt: if we have a good partial solution, try to polish it
    if best_assignment is not None and best_unsat <= 5:
        rng = random.Random(99999)
        polish_assignment, polish_unsat = final_polish(best_assignment, compute_unsat(best_assignment), rng, 20000)
        if polish_unsat == 0:
            return polish_assignment

    return best_assignment if best_assignment else make_random_assignment(0)
# EVOLVE-BLOCK-END
