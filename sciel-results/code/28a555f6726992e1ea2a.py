"""Planted 3-SAT solver: weighted WalkSAT + SA escapes + repair."""
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

    def satisfied(cl, a):
        for lit in cl:
            if a[abs(lit) - 1] == (lit > 0):
                return True
        return False

    def unsat_clauses(a):
        return [ci for ci, cl in enumerate(clauses) if not satisfied(cl, a)]

    def count_unsat(a):
        c = 0
        for cl in clauses:
            if not satisfied(cl, a):
                c += 1
        return c

    best_assignment = None
    best_unsat = m + 1

    def update_best(a):
        nonlocal best_assignment, best_unsat
        u = count_unsat(a)
        if u < best_unsat:
            best_unsat = u
            best_assignment = a[:]
        return u

    # Weighted WalkSAT with restarts and occasional SA escape
    rng = random.Random(12345)

    def walk_sat(a, clause_weights, max_steps):
        # a: list of booleans, clause_weights: list of floats
        for step in range(max_steps):
            if time.time() > deadline:
                return a
            # Find unsat clauses
            unsat = []
            for ci, cl in enumerate(clauses):
                sat = False
                for lit in cl:
                    if a[abs(lit) - 1] == (lit > 0):
                        sat = True
                        break
                if not sat:
                    unsat.append(ci)
            if not unsat:
                return a

            # Pick a clause weighted by clause_weights
            total_w = sum(clause_weights[ci] for ci in unsat)
            if total_w <= 0:
                ci = rng.choice(unsat)
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

            # Score each literal in the clause: gain = (# unsat clauses it would satisfy)
            # minus (# sat clauses it would falsify), weighted by clause weights
            best_var = None
            best_score = -1e18
            best_val = None
            # Precompute which clauses are currently unsat for fast scoring
            unsat_set = set(unsat)
            for lit in cl:
                v = abs(lit) - 1
                desired = (lit > 0)
                if a[v] == desired:
                    # Flipping would make this literal false; it's already satisfying
                    # this clause via this literal? Actually this literal is currently
                    # satisfying the clause. Flipping breaks it.
                    continue
                # Flipping v to desired would make this literal true.
                # Gain: number of unsat clauses containing a literal that becomes true
                # Loss: number of sat clauses containing a literal that becomes false
                gain = 0.0
                loss = 0.0
                for (ci2, lit2) in var_clauses[v]:
                    if ci2 in unsat_set:
                        # Would this flip satisfy clause ci2?
                        if a[v] != desired:
                            # After flip, a[v]=desired
                            if (lit2 > 0) == desired:
                                # This literal becomes true
                                gain += clause_weights[ci2]
                    else:
                        # Clause is currently sat. Would flip falsify it?
                        # Only if this literal is the only one satisfying it.
                        # Check if any other literal in ci2 is satisfied.
                        other_sat = False
                        for lit3 in clauses[ci2]:
                            if abs(lit3) - 1 == v:
                                continue
                            if a[abs(lit3) - 1] == (lit3 > 0):
                                other_sat = True
                                break
                        if not other_sat:
                            # This clause would become unsat
                            if (lit2 > 0) != desired:
                                # After flip, this literal becomes false
                                loss += clause_weights[ci2]
                score = gain - 1.0 * loss
                if score > best_score:
                    best_score = score
                    best_var = v
                    best_val = desired
            if best_var is None:
                # All literals in clause already satisfy it? Shouldn't happen.
                # Pick a random variable in the clause.
                v = abs(cl[0]) - 1
                a[v] = not a[v]
                continue
            a[best_var] = best_val

            # Update clause weights: increase weight of still-unsat clauses slightly
            # (we'll do a lighter update: just bump the chosen clause)
            clause_weights[ci] *= 1.05
            # Occasionally normalize
            if step % 500 == 0:
                max_w = max(clause_weights) if clause_weights else 1.0
                if max_w > 0:
                    clause_weights = [w / max_w for w in clause_weights]

        return a

    # Simulated annealing escape
    def sa_escape(a, clause_weights, steps, t0):
        for step in range(steps):
            if time.time() > deadline:
                return a
            unsat = [ci for ci, cl in enumerate(clauses) if not satisfied(cl, a)]
            if not unsat:
                return a
            ci = rng.choice(unsat)
            cl = clauses[ci]
            v = abs(rng.choice(cl)) - 1
            new_val = not a[v]
            # Compute delta in # unsat
            old_unsat = count_unsat(a)
            a[v] = new_val
            new_unsat = count_unsat(a)
            delta = new_unsat - old_unsat
            if delta <= 0 or rng.random() < math.exp(-delta / max(t0, 1e-9)):
                pass  # accept
            else:
                a[v] = not a[v]  # revert
            t0 *= 0.9995
        return a

    # Main loop with restarts
    restart = 0
    base_steps = 3000
    while time.time() < deadline:
        a = [rng.random() < 0.5 for _ in range(n)]
        clause_weights = [1.0] * m
        max_steps = base_steps + restart * 500
        a = walk_sat(a, clause_weights, max_steps)
        u = update_best(a)
        if u == 0:
            return a

        # If stuck, try SA escape from current state
        if time.time() < deadline - 5.0:
            a2 = a[:]
            a2 = sa_escape(a2, clause_weights, 1000, 5.0)
            u2 = update_best(a2)
            if u2 == 0:
                return a2

        restart += 1
        if restart > 200:
            break

    # Final: try repair from best assignment
    if best_assignment is not None and time.time() < deadline - 2.0:
        a = best_assignment[:]
        # Greedy repair: repeatedly flip variable that reduces unsat most
        for _ in range(500):
            if time.time() > deadline:
                break
            unsat = [ci for ci, cl in enumerate(clauses) if not satisfied(cl, a)]
            if not unsat:
                return a
            # Find variable flip that reduces unsat the most
            best_v = -1
            best_gain = 0
            unsat_set = set(unsat)
            # Sample a subset of variables for speed
            candidates = list(range(n))
            rng.shuffle(candidates)
            for v in candidates[:50]:
                for val in (True, False):
                    if a[v] == val:
                        continue
                    # Simulate flip
                    a[v] = val
                    u = count_unsat(a)
                    a[v] = not val
                    gain = best_unsat - u
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v
                        best_val = val
            if best_v >= 0:
                a[best_v] = best_val
                update_best(a)
            else:
                break

    if best_assignment is not None:
        return best_assignment
    return [False] * n
# EVOLVE-BLOCK-END
