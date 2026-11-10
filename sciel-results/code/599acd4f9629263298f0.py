"""Planted 3-SAT solver: incremental WalkSAT + SA escapes + repair."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 290.0  # safety margin vs 330s kill

    # Pre-index: for each variable, list of (clause_idx, literal)
    var_clauses = [[] for _ in range(n)]
    for ci, cl in enumerate(clauses):
        for lit in cl:
            var_clauses[abs(lit) - 1].append((ci, lit))

    # Incremental state tracking
    def init_state(a):
        """Initialize clause satisfaction counts and unsat list from assignment a."""
        clause_sat_count = [0] * m
        unsat = []
        for ci, cl in enumerate(clauses):
            cnt = 0
            for lit in cl:
                if a[abs(lit) - 1] == (lit > 0):
                    cnt += 1
            clause_sat_count[ci] = cnt
            if cnt == 0:
                unsat.append(ci)
        return clause_sat_count, unsat

    def update_state(v, old_val, new_val, clause_sat_count, unsat, unsat_set):
        """Update clause state after flipping variable v from old_val to new_val."""
        for (ci, lit) in var_clauses[v]:
            old_sat = (old_val == (lit > 0))
            new_sat = (new_val == (lit > 0))
            if old_sat == new_sat:
                continue
            old_cnt = clause_sat_count[ci]
            new_cnt = old_cnt + (1 if new_sat else -1)
            clause_sat_count[ci] = new_cnt
            if old_cnt == 0 and new_cnt > 0:
                # Clause became satisfied
                if ci in unsat_set:
                    unsat_set.discard(ci)
            elif old_cnt > 0 and new_cnt == 0:
                # Clause became unsatisfied
                unsat_set.add(ci)
        # Rebuild unsat list only if size changed (lazy)
        if len(unsat) != len(unsat_set):
            unsat[:] = list(unsat_set)

    best_assignment = None
    best_unsat = m + 1

    def update_best(a):
        nonlocal best_assignment, best_unsat
        clause_sat_count, unsat = init_state(a)
        u = len(unsat)
        if u < best_unsat:
            best_unsat = u
            best_assignment = a[:]
        return u

    rng = random.Random(12345)

    def walk_sat(a, max_steps):
        """Incremental WalkSAT with clause weights."""
        clause_weights = [1.0] * m
        clause_sat_count, unsat = init_state(a)
        unsat_set = set(unsat)

        for step in range(max_steps):
            if time.time() > deadline:
                return a

            # Clean unsat list
            if len(unsat) != len(unsat_set):
                unsat[:] = [ci for ci in unsat if clause_sat_count[ci] == 0]
                unsat_set = set(unsat)

            if not unsat:
                return a

            # Pick a clause weighted by clause_weights
            u_len = len(unsat)
            if u_len <= 0:
                return a
            if u_len == 1:
                ci = unsat[0]
            else:
                total_w = 0.0
                for c in unsat:
                    total_w += clause_weights[c]
                if total_w <= 0:
                    ci = unsat[rng.randrange(u_len)]
                else:
                    r = rng.random() * total_w
                    acc = 0.0
                    ci = unsat[-1]
                    for c in unsat:
                        acc += clause_weights[c]
                        if r <= acc:
                            ci = c
                            break
            cl = clauses[ci]

            # Score each literal in the clause
            best_var = None
            best_score = -1e18
            best_val = None
            for lit in cl:
                v = abs(lit) - 1
                desired = (lit > 0)
                if a[v] == desired:
                    continue
                # Flipping v to desired
                gain = 0.0
                loss = 0.0
                for (ci2, lit2) in var_clauses[v]:
                    old_sat = (a[v] == (lit2 > 0))
                    new_sat = (desired == (lit2 > 0))
                    if old_sat == new_sat:
                        continue
                    old_cnt = clause_sat_count[ci2]
                    new_cnt = old_cnt + (1 if new_sat else -1)
                    if old_cnt == 0 and new_cnt > 0:
                        gain += clause_weights[ci2]
                    elif old_cnt > 0 and new_cnt == 0:
                        loss += clause_weights[ci2]
                score = gain - 1.0 * loss
                if score > best_score:
                    best_score = score
                    best_var = v
                    best_val = desired

            # Add random walk noise: with probability p_walk, pick a random variable
            p_walk = 0.02 + 0.03 * (step / max_steps)
            if best_var is None or rng.random() < p_walk:
                v = abs(rng.choice(cl)) - 1
                old_val = a[v]
                new_val = not old_val
                a[v] = new_val
                update_state(v, old_val, new_val, clause_sat_count, unsat, unsat_set)
                continue

            old_val = a[best_var]
            a[best_var] = best_val
            update_state(best_var, old_val, best_val, clause_sat_count, unsat, unsat_set)

            # Update clause weights
            clause_weights[ci] *= 1.05
            if step % 500 == 0:
                max_w = 0.0
                for w in clause_weights:
                    if w > max_w:
                        max_w = w
                if max_w > 0:
                    for i in range(m):
                        clause_weights[i] /= max_w

        return a

    def sa_escape(a, steps, t0):
        """Simulated annealing escape with incremental updates."""
        clause_sat_count, unsat = init_state(a)
        unsat_set = set(unsat)

        for step in range(steps):
            if time.time() > deadline:
                return a

            if not unsat:
                return a

            ci = rng.randrange(len(unsat))
            ci = unsat[ci]
            cl = clauses[ci]
            v = abs(rng.choice(cl)) - 1
            new_val = not a[v]
            old_val = a[v]

            # Compute delta in # unsat
            old_unsat_count = len(unsat_set)
            a[v] = new_val
            update_state(v, old_val, new_val, clause_sat_count, unsat, unsat_set)
            new_unsat_count = len(unsat_set)
            delta = new_unsat_count - old_unsat_count

            if delta <= 0:
                pass  # accept
            else:
                # Use exp with clamped argument
                arg = -delta / max(t0, 1e-9)
                if arg < -700:
                    prob = 0.0
                else:
                    prob = math.exp(arg)
                if rng.random() < prob:
                    pass  # accept
                else:
                    a[v] = old_val
                    update_state(v, new_val, old_val, clause_sat_count, unsat, unsat_set)

            t0 *= 0.999
            if t0 < 0.1:
                t0 = 0.1
        return a

    # Main loop with restarts
    restart = 0
    base_steps = 8000
    while time.time() < deadline:
        a = [rng.random() < 0.5 for _ in range(n)]
        max_steps = base_steps + restart * 1500
        a = walk_sat(a, max_steps)
        u = update_best(a)
        if u == 0:
            return a

        # If stuck, try SA escape from current state
        if time.time() < deadline - 5.0:
            a2 = a[:]
            sa_steps = 3000 + restart * 500
            t0 = 3.0 + 2.0 * (u / max(m, 1))
            a2 = sa_escape(a2, sa_steps, t0)
            u2 = update_best(a2)
            if u2 == 0:
                return a2

        restart += 1
        if restart > 200:
            break

    # Final: try repair from best assignment
    if best_assignment is not None and time.time() < deadline - 2.0:
        a = best_assignment[:]
        clause_sat_count, unsat = init_state(a)
        unsat_set = set(unsat)
        # Greedy repair: repeatedly flip variable that reduces unsat most
        for _ in range(1000):
            if time.time() > deadline:
                break
            if not unsat:
                return a
            # Find variable flip that reduces unsat the most
            best_v = -1
            best_gain = 0
            best_val = None
            candidates = list(range(n))
            rng.shuffle(candidates)
            for v in candidates[:150]:
                val = not a[v]
                # Simulate flip
                old_val = a[v]
                a[v] = val
                update_state(v, old_val, val, clause_sat_count, unsat, unsat_set)
                u = len(unsat_set)
                gain = best_unsat - u
                a[v] = old_val
                update_state(v, val, old_val, clause_sat_count, unsat, unsat_set)
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
                    best_val = val
            if best_v >= 0:
                old_val = a[best_v]
                a[best_v] = best_val
                update_state(best_v, old_val, best_val, clause_sat_count, unsat, unsat_set)
                update_best(a)
            else:
                break

    if best_assignment is not None:
        return best_assignment
    return [False] * n
# EVOLVE-BLOCK-END
