"""Synthetic planted 3-SAT solver (50 vars). Structural redesign:
bias-initialized Weighted WalkSAT with adaptive restarts + deadline guard."""
import random
import time


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    deadline = time.time() + 295.0  # safety margin vs 330s kill

    # Precompute variable occurrence by polarity to bias initial assignment
    # toward the (unknown) planted assignment. In planted 3-SAT, the planted
    # assignment tends to make literals appear consistently, so a literal that
    # appears more often is more likely to be "true" under the planted model.
    pos_count = [0] * n
    neg_count = [0] * n
    for c in clauses:
        for lit in c:
            v = abs(lit) - 1
            if lit > 0:
                pos_count[v] += 1
            else:
                neg_count[v] += 1

    def satisfied(c, a):
        for lit in c:
            if a[abs(lit) - 1] == (lit > 0):
                return True
        return False

    def count_unsat(a):
        return sum(0 if satisfied(c, a) else 1 for c in clauses)

    # Precompute, for each variable, which clauses contain it (for fast delta eval)
    var_to_clauses = [[] for _ in range(n)]
    for ci, c in enumerate(clauses):
        for lit in c:
            var_to_clauses[abs(lit) - 1].append(ci)

    def flip_delta(v, a):
        # Number of clauses that change satisfaction status if we flip v.
        # We want: new_unsat - old_unsat over clauses containing v.
        delta = 0
        for ci in var_to_clauses[v]:
            c = clauses[ci]
            was_sat = satisfied(c, a)
            # simulate flip
            a[v] = not a[v]
            now_sat = satisfied(c, a)
            a[v] = not a[v]  # revert
            if now_sat and not was_sat:
                delta -= 1
            elif was_sat and not now_sat:
                delta += 1
        return delta

    rng = random.Random(0)

    # Initial assignment: bias toward more-frequent polarity per variable
    assignment = []
    for v in range(n):
        if pos_count[v] > neg_count[v]:
            assignment.append(True)
        elif neg_count[v] > pos_count[v]:
            assignment.append(False)
        else:
            assignment.append(rng.random() < 0.5)

    if count_unsat(assignment) == 0:
        return assignment

    best = assignment[:]
    best_unsat = count_unsat(best)

    # Weighted WalkSAT with restarts and adaptive noise.
    max_restarts = 200000
    base_steps = 2000
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        # On restart, re-seed with bias assignment (occasionally random)
        if restart == 0:
            a = assignment[:]
        else:
            if rng.random() < 0.7:
                a = []
                for v in range(n):
                    if pos_count[v] > neg_count[v]:
                        a.append(True)
                    elif neg_count[v] > neg_count[v] - 1 and neg_count[v] > 0:
                        a.append(False)
                    else:
                        a.append(rng.random() < 0.5)
            else:
                a = [rng.random() < 0.5 for _ in range(n)]

        steps = base_steps + rng.randint(0, 500)
        # Adaptive noise: start high, decay
        noise_start = 0.5
        for step in range(steps):
            if time.time() > deadline:
                break
            unsat = [c for c in clauses if not satisfied(c, a)]
            if not unsat:
                best = a[:]
                best_unsat = 0
                return best
            cu = count_unsat(a)
            if cu < best_unsat:
                best = a[:]
                best_unsat = cu
            # Pick a clause: prefer more-violated / shorter (all length 3 here)
            clause = rng.choice(unsat)
            # Decide: flip best variable vs random (noise)
            noise = noise_start * (1.0 - step / steps)
            if rng.random() < noise:
                lit = rng.choice(clause)
                v = abs(lit) - 1
                a[v] = not a[v]
            else:
                # Greedy: pick variable in clause minimizing increase in unsat
                best_v = -1
                best_d = 10**9
                for lit in clause:
                    v = abs(lit) - 1
                    d = flip_delta(v, a)
                    if d < best_d:
                        best_d = d
                        best_v = v
                # small random tie-break among near-best
                if best_v >= 0:
                    a[best_v] = not a[best_v]
                else:
                    lit = rng.choice(clause)
                    a[abs(lit) - 1] = not a[abs(lit) - 1]
            # occasional early-restart if stuck
            if step % 200 == 0 and cu >= best_unsat:
                break

    return best
