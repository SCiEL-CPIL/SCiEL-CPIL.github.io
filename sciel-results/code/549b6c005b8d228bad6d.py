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

    # Precompute literal position per clause for fast mask updates
    # For each clause, store the (idx, pos) pairs

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
    max_flips_per_run = 6000

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
        last_unsat_count = len(unsat)
        no_improve = 0

        for _flip in range(max_flips_per_run):
            if time.time() > deadline:
                break
            if not unsat:
                return assignment

            cur_unsat = len(unsat)
            if cur_unsat < best_unsat:
                best_unsat = cur_unsat
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
            old_val = assignment[idx]
            new_val = not old_val
            assignment[idx] = new_val

            # Incrementally update masks and unsat for clauses touching idx
            for cj in var_to_clauses[idx]:
                old_mask = masks[cj]
                # Find which position in clause cj corresponds to variable idx
                # We need to update the mask for this variable's contribution
                cc = clauses_idx[cj]
                # Find the position k2 where cc[k2] has idx
                for k2, (i2, p2) in enumerate(cc):
                    if i2 == idx:
                        bit = 1 << k2
                        if new_val == p2:
                            masks[cj] = old_mask | bit
                        else:
                            masks[cj] = old_mask & ~bit
                        break
                if masks[cj] == 0:
                    unsat.add(cj)
                else:
                    unsat.discard(cj)

            # Adaptive noise: if stuck (unsat not decreasing), increase noise
            cur_unsat = len(unsat)
            if cur_unsat < last_unsat_count:
                no_improve = 0
                last_unsat_count = cur_unsat
            else:
                no_improve += 1

            if no_improve % 150 == 0:
                noise = min(0.7, noise + 0.05)
            elif no_improve == 0 and noise > 0.2:
                noise = max(0.2, noise - 0.02)

        # End of this run; continue with a fresh restart if time remains
        if time.time() > deadline:
            break

    # Final polish: if we have a best assignment that's close, try a focused local search
    if best_assignment is not None and best_unsat > 0:
        remaining_time = deadline - time.time()
        if remaining_time > 2.0:
            assignment = best_assignment[:]
            masks = [sat_mask(assignment, ci) for ci in range(m)]
            unsat = set(ci for ci in range(m) if masks[ci] == 0)
            noise = 0.1
            for _flip in range(3000):
                if time.time() > deadline:
                    break
                if not unsat:
                    return assignment
                if len(unsat) < best_unsat:
                    best_unsat = len(unsat)
                    best_assignment = assignment[:]
                ci = rng.choice(list(unsat))
                c = clauses_idx[ci]
                unsat_positions = [k for k, (idx2, pos2) in enumerate(c) if not (assignment[idx2] == pos2)]
                if unsat_positions and rng.random() > noise:
                    k = rng.choice(unsat_positions)
                else:
                    k = rng.randrange(3)
                idx2, pos2 = c[k]
                new_val = not assignment[idx2]
                assignment[idx2] = new_val
                for cj in var_to_clauses[idx2]:
                    old_mask = masks[cj]
                    cc = clauses_idx[cj]
                    for k2, (i2, p2) in enumerate(cc):
                        if i2 == idx2:
                            bit = 1 << k2
                            if new_val == p2:
                                masks[cj] = old_mask | bit
                            else:
                                masks[cj] = old_mask & ~bit
                            break
                    if masks[cj] == 0:
                        unsat.add(cj)
                    else:
                        unsat.discard(cj)

    if best_assignment is not None:
        return best_assignment
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
