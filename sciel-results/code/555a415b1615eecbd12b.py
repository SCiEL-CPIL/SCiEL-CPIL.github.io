"""Planted 3-SAT solver: clause-weighted Max-SAT heuristic with dynamic weights,
fast bitmask satisfaction tracking, and a final high-precision polish pass."""
# EVOLVE-BLOCK-START
import random
import time
import math

DEADLINE_FRACTION = 0.95


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    t_start = time.time()
    deadline = t_start + 330 * DEADLINE_FRACTION

    # Precompute clause structure and bitmasks
    # For each variable, bitmasks of clauses where +v or -v appears
    pos_mask = [0] * n
    neg_mask = [0] * n
    clause_vars = []  # list of (var_idx, is_positive) per clause

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

    # Fast satisfaction count using bitmasks
    def sat_count(assignment):
        sm = 0
        for v in range(n):
            if assignment[v]:
                sm |= pos_mask[v]
            else:
                sm |= neg_mask[v]
        return sm.bit_count()

    def is_solved(assignment):
        sm = 0
        for v in range(n):
            if assignment[v]:
                sm |= pos_mask[v]
            else:
                sm |= neg_mask[v]
        return sm == all_mask

    # Compute unsatisfied clause bitmask
    def unsat_mask(assignment):
        sm = 0
        for v in range(n):
            if assignment[v]:
                sm |= pos_mask[v]
            else:
                sm |= neg_mask[v]
        return all_mask ^ sm

    # Weighted gain for flipping variable v, given current weights on unsat clauses
    # Weights: list of floats, one per clause
    def weighted_flip_gain(v, assignment, weights):
        """
        Gain = sum over clauses affected by v of:
          if clause becomes satisfied: +weight[ci]
          if clause becomes unsatisfied: -weight[ci]
        Only clauses in pos_mask[v] | neg_mask[v] can change.
        """
        affected = pos_mask[v] | neg_mask[v]
        if affected == 0:
            return 0.0

        cur_sat = 0
        for vv in range(n):
            if assignment[vv]:
                cur_sat |= pos_mask[vv]
            else:
                cur_sat |= neg_mask[vv]
        cur_unsat = all_mask ^ cur_sat

        gain = 0.0
        aff = affected
        while aff:
            lsb = aff & (-aff)
            ci = lsb.bit_length() - 1
            aff ^= lsb
            w = weights[ci]
            if w == 0.0:
                continue

            cvars = clause_vars[ci]
            # Find v's polarity in this clause
            v_pos = None
            for vv, pos in cvars:
                if vv == v:
                    v_pos = pos
                    break
            if v_pos is None:
                continue

            # Check if any other literal in clause satisfies it (independent of v)
            others_sat = False
            for vv, pos in cvars:
                if vv == v:
                    continue
                if assignment[vv] == pos:
                    others_sat = True
                    break

            cur_clause_sat = (assignment[v] == v_pos) or others_sat
            new_clause_sat = ((not assignment[v]) == v_pos) or others_sat

            if new_clause_sat and not cur_clause_sat:
                gain += w
            elif cur_clause_sat and not new_clause_sat:
                gain -= w

        return gain

    # --- Main search: clause-weighted Max-SAT heuristic ---
    rng = random.Random(42)

    best_assignment = None
    best_unsat = m + 1

    # Number of restarts / steps - we'll adapt based on time
    max_restarts = 500
    max_steps = 8000

    # Weight update parameters
    weight_decay = 0.95  # decay factor for weights each step
    weight_boost = 1.5   # boost for unsatisfied clauses

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        cur_unsat = m - sat_count(assignment)

        if cur_unsat == 0:
            return assignment
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]

        # Initialize weights: higher weight for clauses that are harder
        # Use a simple heuristic: start uniform, then boost unsat clauses
        weights = [1.0] * m

        for step in range(max_steps):
            if time.time() > deadline:
                break

            cur_sat_mask = 0
            for vv in range(n):
                if assignment[vv]:
                    cur_sat_mask |= pos_mask[vv]
                else:
                    cur_sat_mask |= neg_mask[vv]
            cur_unsat_mask = all_mask ^ cur_sat_mask
            cur_unsat = bin(cur_unsat_mask).count('1')

            if cur_unsat == 0:
                return assignment
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
                best_assignment = assignment[:]

            # Find best variable to flip using weighted gain
            best_gain = -1e18
            best_v = 0
            for v in range(n):
                g = weighted_flip_gain(v, assignment, weights)
                if g > best_gain:
                    best_gain = g
                    best_v = v

            # With some probability, do a random flip to escape local optima
            if rng.random() < 0.15:
                best_v = rng.randint(0, n - 1)

            assignment[best_v] = not assignment[best_v]

            # Update weights: boost unsatisfied clauses, decay all
            # Only update weights every few steps for efficiency
            if step % 3 == 0:
                new_weights = []
                for ci in range(m):
                    if cur_unsat_mask & (1 << ci):
                        new_weights.append(min(weights[ci] * weight_boost, 100.0))
                    else:
                        new_weights.append(weights[ci] * weight_decay)
                weights = new_weights

    # --- Final high-precision polish pass ---
    if best_assignment is not None and time.time() < deadline:
        assignment = best_assignment[:]
        # Iterated local search with decreasing noise
        for polish_round in range(50):
            if time.time() > deadline:
                break
            improved = True
            while improved and time.time() < deadline:
                improved = False
                cur_unsat = m - sat_count(assignment)
                if cur_unsat == 0:
                    return assignment
                # Try all single flips
                best_g = 0
                best_v = -1
                for v in range(n):
                    g = weighted_flip_gain(v, assignment, [1.0] * m)
                    if g > best_g:
                        best_g = g
                        best_v = v
                if best_v >= 0:
                    assignment[best_v] = not assignment[best_v]
                    improved = True
                    if sat_count(assignment) > sat_count(best_assignment):
                        best_assignment = assignment[:]
                        best_unsat = m - sat_count(assignment)

        # Check if we solved it
        if is_solved(assignment):
            return assignment
        if sat_count(assignment) > sat_count(best_assignment):
            best_assignment = assignment[:]

    return best_assignment
# EVOLVE-BLOCK-END
