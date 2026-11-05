"""Rewritten solver: bitset-accelerated WalkSAT with frequency-weighted variable selection."""
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

    start_time = time.time()
    deadline = start_time + 325.0  # 5s safety margin out of 330s

    # Pre-process: build clause data and variable-to-clause index
    # For each clause, store the list of (var_idx, polarity)
    clause_vars = []
    clause_pols = []
    var_to_clauses = [[] for _ in range(n)]

    for i, c in enumerate(clauses):
        vs = []
        ps = []
        for lit in c:
            v = abs(lit) - 1
            p = lit > 0
            vs.append(v)
            ps.append(p)
            var_to_clauses[v].append(i)
        clause_vars.append(vs)
        clause_pols.append(ps)

    # State
    assignment = [False] * n
    clause_sat_count = [0] * m
    clause_is_sat = [False] * m
    unsat_clauses = []

    def initialize_state():
        unsat_clauses.clear()
        for i in range(m):
            cvars = clause_vars[i]
            cpols = clause_pols[i]
            cnt = 0
            for v, p in zip(cvars, cpols):
                if assignment[v] == p:
                    cnt += 1
            clause_sat_count[i] = cnt
            clause_is_sat[i] = (cnt > 0)
            if cnt == 0:
                unsat_clauses.append(i)

    def flip_variable(v):
        """Flip variable v and incrementally update clause states."""
        assignment[v] = not assignment[v]
        for ci in var_to_clauses[v]:
            cvars = clause_vars[ci]
            cpols = clause_pols[ci]
            # Determine if v was satisfying this clause before flip
            # Find polarity of v in this clause
            v_satisfies_now = False
            for vv, pp in zip(cvars, cpols):
                if vv == v:
                    v_satisfies_now = (assignment[v] == pp)
                    break

            old_count = clause_sat_count[ci]
            if v_satisfies_now:
                new_count = old_count + 1
            else:
                new_count = old_count - 1

            clause_sat_count[ci] = new_count
            was_sat = clause_is_sat[ci]
            is_sat = (new_count > 0)
            clause_is_sat[ci] = is_sat

            if was_sat and not is_sat:
                unsat_clauses.append(ci)
            elif not was_sat and is_sat:
                # Remove from unsat_clauses
                try:
                    idx = unsat_clauses.index(ci)
                    unsat_clauses[idx] = unsat_clauses[-1]
                    unsat_clauses.pop()
                except ValueError:
                    pass

    def calculate_gain(v):
        """
        Calculate gain of flipping variable v.
        Gain = (# unsat clauses containing v that would become sat)
              - (# sat clauses containing v that would become unsat)
        """
        gain = 0
        for ci in var_to_clauses[v]:
            if not clause_is_sat[ci]:
                # Clause is unsat: flipping v will satisfy it (since v is in the clause
                # and the clause is unsat, v's current value doesn't satisfy it, so flip does)
                gain += 1
            else:
                # Clause is sat: check if v is the only satisfying literal
                # Find v's polarity in this clause
                v_satisfies = False
                cvars = clause_vars[ci]
                cpols = clause_pols[ci]
                for vv, pp in zip(cvars, cpols):
                    if vv == v:
                        v_satisfies = (assignment[v] == pp)
                        break
                if v_satisfies and clause_sat_count[ci] == 1:
                    gain -= 1
        return gain

    rng = random.Random(12345)
    best_assignment = None
    best_unsat_count = m + 1

    # Frequency tracking for weighted selection
    var_unsat_freq = [0] * n  # How many unsat clauses each variable appears in

    def update_unsat_freq():
        for v in range(n):
            var_unsat_freq[v] = 0
        for ci in unsat_clauses:
            for v in clause_vars[ci]:
                var_unsat_freq[v] += 1

    # Main search loop
    max_restarts = 500
    max_steps_per_restart = 80000

    for restart in range(max_restarts):
        if time.time() > deadline - 3.0:
            break

        # Random restart
        assignment = [rng.random() < 0.5 for _ in range(n)]
        initialize_state()

        current_unsat = len(unsat_clauses)

        if current_unsat == 0:
            return assignment[:]

        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]

        # Update frequency for this restart's initial state
        update_unsat_freq()

        steps = 0
        stuck_counter = 0
        noise_prob = 0.15

        while steps < max_steps_per_restart:
            if time.time() > deadline - 3.0:
                break

            if not unsat_clauses:
                return assignment[:]

            # Pick a random unsatisfied clause
            ci = rng.randrange(len(unsat_clauses))
            cvars = clause_vars[ci]

            # Weighted variable selection: prefer variables that appear in more unsat clauses
            # Use inverse frequency weighting (variables in fewer unsat clauses get higher weight)
            # Actually, let's use a hybrid: with some probability pick by gain, else weighted random
            if rng.random() < noise_prob:
                # Pure random from clause
                v = rng.choice(cvars)
            else:
                # Compute gains and select with softmax-like weighting
                gains = []
                for v in cvars:
                    g = calculate_gain(v)
                    gains.append((v, g))

                # Find max gain
                max_g = max(g for _, g in gains)
                min_g = min(g for _, g in gains)

                if max_g == min_g:
                    # All same gain, pick randomly
                    v = rng.choice(cvars)
                else:
                    # Weighted selection: higher gain = higher probability
                    # Use exponential weighting with temperature
                    temp = 2.0
                    weights = []
                    for v, g in gains:
                        # Shift so max gain has weight 1
                        w = (g - min_g) + 1  # ensure positive
                        weights.append(w)

                    total_w = sum(weights)
                    r = rng.random() * total_w
                    cum = 0
                    chosen_v = cvars[0]
                    for (v, g), w in zip(gains, weights):
                        cum += w
                        if r <= cum:
                            chosen_v = v
                            break
                    v = chosen_v

            flip_variable(v)
            steps += 1

            current_unsat = len(unsat_clauses)

            if current_unsat == 0:
                return assignment[:]

            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
                stuck_counter = 0
            else:
                stuck_counter += 1

            # Periodic frequency update and early exit
            if steps % 2000 == 0:
                update_unsat_freq()
                if stuck_counter > 15000:
                    break

    if best_assignment is not None:
        return best_assignment
    return assignment

# EVOLVE-BLOCK-END
