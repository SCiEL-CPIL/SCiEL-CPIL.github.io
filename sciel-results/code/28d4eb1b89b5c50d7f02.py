"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import sys

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    if m == 0:
        return [False] * n

    # Deadline guard: 320s safety margin out of 330s limit
    start_time = time.time()
    deadline = start_time + 320.0

    # Pre-process clauses
    clause_vars = []
    clause_polarity = []
    var_to_clauses = [[] for _ in range(n)]
    var_clause_index = {}  # (v, ci) -> index in clause

    for i, c in enumerate(clauses):
        vars_in_c = []
        pols_in_c = []
        for lit in c:
            v = abs(lit) - 1
            p = lit > 0
            vars_in_c.append(v)
            pols_in_c.append(p)
            var_to_clauses[v].append(i)
        clause_vars.append(vars_in_c)
        clause_polarity.append(pols_in_c)

    # State variables
    assignment = [False] * n
    clause_sat_count = [0] * m
    clause_is_sat = [False] * m
    unsat_clauses = []

    def update_clause_sat(ci):
        cvars = clause_vars[ci]
        cpols = clause_polarity[ci]
        count = 0
        for v, p in zip(cvars, cpols):
            if assignment[v] == p:
                count += 1
        clause_sat_count[ci] = count
        clause_is_sat[ci] = (count > 0)

    def initialize_state(assignment):
        unsat_clauses.clear()
        for i in range(m):
            update_clause_sat(i)
            if not clause_is_sat[i]:
                unsat_clauses.append(i)

    def flip_variable(v):
        assignment[v] = not assignment[v]
        for ci in var_to_clauses[v]:
            was_sat = clause_is_sat[ci]
            cvars = clause_vars[ci]
            cpols = clause_polarity[ci]
            new_count = 0
            for vv, pp in zip(cvars, cpols):
                if assignment[vv] == pp:
                    new_count += 1
            clause_sat_count[ci] = new_count
            is_sat = (new_count > 0)
            clause_is_sat[ci] = is_sat
            if was_sat and not is_sat:
                unsat_clauses.append(ci)
            elif not was_sat and is_sat:
                try:
                    unsat_clauses.remove(ci)
                except ValueError:
                    pass

    def calculate_breakage(v):
        """Number of currently-satisfied clauses that would become unsat if v is flipped."""
        breakage = 0
        for ci in var_to_clauses[v]:
            if clause_is_sat[ci]:
                cvars = clause_vars[ci]
                cpols = clause_polarity[ci]
                v_satisfies = False
                for vv, pp in zip(cvars, cpols):
                    if vv == v:
                        if assignment[v] == pp:
                            v_satisfies = True
                        break
                if v_satisfies and clause_sat_count[ci] == 1:
                    breakage += 1
        return breakage

    def calculate_gain(v):
        """Number of currently-unsatisfied clauses that would become sat if v is flipped."""
        gain = 0
        for ci in var_to_clauses[v]:
            if not clause_is_sat[ci]:
                cvars = clause_vars[ci]
                cpols = clause_polarity[ci]
                for vv, pp in zip(cvars, cpols):
                    if vv == v:
                        if (not assignment[v]) == pp:
                            gain += 1
                        break
        return gain

    rng = random.Random(42)
    best_assignment = None
    best_unsat_count = m + 1

    def local_search(noise_prob, max_steps, early_exit_stuck):
        """Run a local search phase. Returns (assignment, unsat_count)."""
        nonlocal assignment, unsat_clauses
        current_unsat = len(unsat_clauses)
        if current_unsat == 0:
            return assignment[:], 0

        steps = 0
        best_local_unsat = current_unsat
        best_local_assignment = assignment[:]
        stuck_counter = 0

        while steps < max_steps:
            if time.time() > deadline - 3.0:
                break
            if not unsat_clauses:
                return assignment[:], 0

            ci = rng.choice(unsat_clauses)
            cvars = clause_vars[ci]

            if rng.random() < noise_prob:
                v = rng.choice(cvars)
            else:
                # Weighted phase: minimize breakage, then maximize gain
                best_v = -1
                best_break = 10**9
                best_gain = -10**9
                for v in cvars:
                    b = calculate_breakage(v)
                    if b < best_break:
                        best_break = b
                        best_gain = calculate_gain(v)
                        best_v = v
                    elif b == best_break:
                        g = calculate_gain(v)
                        if g > best_gain:
                            best_gain = g
                            best_v = v
                v = best_v

            flip_variable(v)
            steps += 1
            current_unsat = len(unsat_clauses)

            if current_unsat == 0:
                return assignment[:], 0

            if current_unsat < best_local_unsat:
                best_local_unsat = current_unsat
                best_local_assignment = assignment[:]
                stuck_counter = 0
            else:
                stuck_counter += 1

            if early_exit_stuck and stuck_counter > early_exit_stuck:
                break

        return best_local_assignment, best_local_unsat

    # Main search loop
    max_restarts = 2000
    max_steps_per_restart = 50000
    noise_prob = 0.3

    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break

        # Random restart
        assignment = [rng.random() < 0.5 for _ in range(n)]
        initialize_state(assignment)
        current_unsat = len(unsat_clauses)

        if current_unsat == 0:
            return assignment[:]

        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]

        # Run local search with moderate noise
        assignment, current_unsat = local_search(noise_prob, max_steps_per_restart, 8000)

        if current_unsat == 0:
            return assignment[:]

        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]

        # Final polish pass: if we're close, do a tighter local search
        if best_unsat_count <= max(3, m // 50) and time.time() < deadline - 15.0:
            assignment = best_assignment[:]
            initialize_state(assignment)
            current_unsat = len(unsat_clauses)
            if current_unsat == 0:
                return assignment[:]
            # Tighter noise, more steps
            assignment, current_unsat = local_search(0.1, 100000, 20000)
            if current_unsat == 0:
                return assignment[:]
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]

    if best_assignment:
        return best_assignment
    return assignment

# EVOLVE-BLOCK-END
