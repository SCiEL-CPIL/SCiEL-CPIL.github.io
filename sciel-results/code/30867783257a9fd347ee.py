"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 285.0  # safety margin before the 330s kill

    # Pre-compute clauses as 0-based (idx, pos) tuples and build a variable->clauses index
    clauses_idx = [[(abs(l) - 1, l > 0) for l in c] for c in clauses]
    m = len(clauses_idx)
    var_to_clauses = [[] for _ in range(n)]
    for ci, c in enumerate(clauses_idx):
        for idx, _pos in c:
            var_to_clauses[idx].append(ci)

    # Precompute the flip effect per clause: for each clause, which literal index (0..2)
    # satisfies it, and the (idx, pos) so we can tell if a variable currently satisfies it.
    # sat_mask[ci] = bitmask of literal positions currently satisfied (recomputed on flip)

    def is_sat(assignment, ci):
        c = clauses_idx[ci]
        for idx, pos in c:
            if assignment[idx] == pos:
                return True
        return False

    def sat_mask(assignment, ci):
        c = clauses_idx[ci]
        mask = 0
        for k, (idx, pos) in enumerate(c):
            if assignment[idx] == pos:
                mask |= (1 << k)
        return mask

    rng = random.Random(12345)

    best_assignment = None
    best_unsat = m + 1

    # Adaptive noise: start greedy, add noise as we get stuck.
    max_flips_per_run = 4000

    while time.time() < deadline:
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        masks = [sat_mask(assignment, ci) for ci in range(m)]
        unsat = set(ci for ci in range(m) if masks[ci] == 0)

        if not unsat:
            return assignment

        if len(unsat) < best_unsat:
            best_unsat = len(unsat)
            best_assignment = assignment[:]

        noise = 0.2
        stuck = 0

        for _flip in range(max_flips_per_run):
            if time.time() > deadline:
                break
            if not unsat:
                return assignment

            if len(unsat) < best_unsat:
                best_unsat = len(unsat)
                best_assignment = assignment[:]

            # Pick an unsatisfied clause
            ci = rng.choice(list(unsat))
            c = clauses_idx[ci]

            # Decide which literal to flip:
            # With prob (1-noise) prefer a literal that is currently UNSAT in this clause
            # (flipping it satisfies the clause). With prob noise, flip any literal.
            unsat_positions = [k for k, (idx, pos) in enumerate(c) if not (assignment[idx] == pos)]
            if unsat_positions and rng.random() > noise:
                k = rng.choice(unsat_positions)
            else:
                k = rng.randrange(3)
            idx, pos = c[k]
            assignment[idx] = not assignment[idx]

            # Incrementally update masks and unsat for clauses touching idx
            for cj in var_to_clauses[idx]:
                masks[cj] = sat_mask(assignment, cj)
                if masks[cj] == 0:
                    unsat.add(cj)
                else:
                    unsat.discard(cj)

            # Adaptive noise: if stuck (unsat not decreasing), increase noise
            # (cheap heuristic; recompute a coarse signal every so often)
            stuck += 1
            if stuck % 200 == 0:
                noise = min(0.7, noise + 0.05)

        # End of this run; continue with a fresh restart if time remains
        if time.time() > deadline:
            break

    if best_assignment is not None:
        return best_assignment
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
