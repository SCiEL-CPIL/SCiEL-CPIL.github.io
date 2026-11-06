"""Planted 3-SAT solver: bitmask-accelerated WalkSAT + bias init + O(1) greedy flips."""
# EVOLVE-BLOCK-START
import random
import time

DEADLINE_FRACTION = 0.95  # safety margin: stop at 95% of 330s


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    t_start = time.time()
    deadline = t_start + 330 * DEADLINE_FRACTION

    # --- Precompute bitmask structure ---
    # pos_mask[v]: clauses where +v appears; neg_mask[v]: clauses where -v appears
    pos_mask = [0] * n
    neg_mask = [0] * n
    # clause_vars[ci] = list of (var_idx, is_positive)
    clause_vars = []
    for ci, clause in enumerate(clauses):
        bit = 1 << ci
        for lit in clause:
            v = abs(lit) - 1
            if lit > 0:
                pos_mask[v] |= bit
            else:
                neg_mask[v] |= bit
        clause_vars.append([(abs(l) - 1, l > 0) for l in clause])

    all_mask = (1 << m) - 1

    # Literal frequency for bias-based initialization
    lit_freq = {}
    for clause in clauses:
        for l in clause:
            lit_freq[l] = lit_freq.get(l, 0) + 1

    # --- Fast helpers ---
    def satisfied_mask(assignment):
        sm = 0
        for v in range(n):
            if assignment[v]:
                sm |= pos_mask[v]
            else:
                sm |= neg_mask[v]
        return sm

    def unsat_count(assignment):
        return m - bin(satisfied_mask(assignment)).count('1')

    # O(1) flip gain via precomputed masks:
    #   old_mask = masks satisfied by v's current value
    #   new_mask = masks satisfied by v's flipped value
    #   others_sat = satisfied by some variable other than v
    #   gain = popcount(new_mask & unsat) - popcount(old_mask & sat & ~others_sat)
    def flip_gain(v, assignment, sat):
        if assignment[v]:
            old_mask = pos_mask[v]
            new_mask = neg_mask[v]
        else:
            old_mask = neg_mask[v]
            new_mask = pos_mask[v]
        others_sat = sat & ~(old_mask | new_mask)
        unsat = all_mask ^ sat
        return (new_mask & unsat).bit_count() - (old_mask & sat & ~others_sat).bit_count()

    # --- Initializations ---
    rng = random.Random(42)

    def init_by_bias():
        assign = [False] * n
        for v in range(1, n + 1):
            assign[v - 1] = (lit_freq.get(v, 0) >= lit_freq.get(-v, 0))
        return assign

    def init_random():
        return [rng.random() < 0.5 for _ in range(n)]

    # Pick the best starting point from several candidates
    candidates = [init_by_bias()] + [init_random() for _ in range(5)]
    best_init = min(candidates, key=lambda a: unsat_count(a))
    assignment = best_init[:]

    best_assignment = assignment[:]
    best_unsat_count = unsat_count(assignment)
    if best_unsat_count == 0:
        return best_assignment

    # --- WalkSAT with O(1) greedy flip gain ---
    walksat_restarts = 100000   # high; bounded by deadline
    max_steps_per_restart = 20000
    noise_param = 0.25          # probability of random flip vs greedy

    for restart in range(walksat_restarts):
        if time.time() > deadline:
            break

        if restart == 0:
            assignment = best_init[:]
        else:
            assignment = init_random()

        sat = satisfied_mask(assignment)
        cur_unsat = m - sat.bit_count()

        if cur_unsat == 0:
            return assignment
        if cur_unsat < best_unsat_count:
            best_unsat_count = cur_unsat
            best_assignment = assignment[:]

        for step in range(max_steps_per_restart):
            if time.time() > deadline:
                break

            if cur_unsat == 0:
                return assignment
            if cur_unsat < best_unsat_count:
                best_unsat_count = cur_unsat
                best_assignment = assignment[:]

            if rng.random() < noise_param:
                v = rng.randint(0, n - 1)
            else:
                # Greedy: max gain flip, random tie-break
                best_gain = -10 ** 9
                best_v = 0
                for v in range(n):
                    g = flip_gain(v, assignment, sat)
                    if g > best_gain or (g == best_gain and rng.random() < 0.5):
                        best_gain = g
                        best_v = v
                v = best_v

            # Apply flip and update sat incrementally
            if assignment[v]:
                old_mask = pos_mask[v]
                new_mask = neg_mask[v]
            else:
                old_mask = neg_mask[v]
                new_mask = pos_mask[v]
            others_sat = sat & ~(old_mask | new_mask)
            sat = others_sat | new_mask
            assignment[v] = not assignment[v]
            cur_unsat = m - sat.bit_count()

    # --- Final high-precision polish: steepest-descent local search from best ---
    if time.time() < deadline:
        assignment = best_assignment[:]
        sat = satisfied_mask(assignment)
        cur_unsat = m - sat.bit_count()
        improved = True
        while improved and time.time() < deadline:
            if cur_unsat == 0:
                return assignment
            improved = False
            best_gain = 0
            best_v = -1
            for v in range(n):
                g = flip_gain(v, assignment, sat)
                if g > best_gain:
                    best_gain = g
                    best_v = v
            if best_v >= 0:
                if assignment[best_v]:
                    old_mask = pos_mask[best_v]
                    new_mask = neg_mask[best_v]
                else:
                    old_mask = neg_mask[best_v]
                    new_mask = pos_mask[best_v]
                others_sat = sat & ~(old_mask | new_mask)
                sat = others_sat | new_mask
                assignment[best_v] = not assignment[best_v]
                cur_unsat = m - sat.bit_count()
                improved = True
        if cur_unsat == 0:
            return assignment
        if cur_unsat < best_unsat_count:
            best_unsat_count = cur_unsat
            best_assignment = assignment[:]

    return best_assignment
# EVOLVE-BLOCK-END
