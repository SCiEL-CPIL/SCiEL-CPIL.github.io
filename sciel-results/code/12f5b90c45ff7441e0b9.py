"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 285.0  # safety margin before the 330s kill

    # 0-based (idx, pos) representation of each clause.
    clauses_idx = [[(abs(l) - 1, l > 0) for l in c] for c in clauses]
    m = len(clauses_idx)

    # variable -> list of (clause_id, position_in_clause)
    var_to_pos = [[] for _ in range(n)]
    for ci, c in enumerate(clauses_idx):
        for k, (idx, _pos) in enumerate(c):
            var_to_pos[idx].append((ci, k))

    # Bit-vector: for each variable and each polarity, a bitmask over clauses.
    # pos_mask[v]  : clauses where the POSITIVE literal of v appears
    # neg_mask[v]  : clauses where the NEGATIVE literal of v appears
    pos_mask = [0] * n
    neg_mask = [0] * n
    for ci, c in enumerate(clauses_idx):
        bit = 1 << ci
        for idx, pos in c:
            if pos:
                pos_mask[idx] |= bit
            else:
                neg_mask[idx] |= bit

    rng = random.Random(12345)

    best_assignment = None
    best_unsat = m + 1

    max_flips_per_run = 4000

    while time.time() < deadline:
        # Random initial assignment.
        assignment = [rng.random() < 0.5 for _ in range(n)]

        # Satisfied-clause mask and unsatisfied set.
        sat_mask = 0
        for v in range(n):
            if assignment[v]:
                sat_mask |= pos_mask[v]
            else:
                sat_mask |= neg_mask[v]
        unsat = (1 << m) - 1
        unsat &= ~sat_mask
        unsat_count = bin(unsat).count("1")

        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]

        noise = 0.2
        no_improve = 0
        last_unsat_count = unsat_count

        for _flip in range(max_flips_per_run):
            if time.time() > deadline:
                break
            if unsat_count == 0:
                return assignment

            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]

            if noise < 0.65 and rng.random() < noise:
                # --- Noisy random flip (escape move) ---
                v = rng.randrange(n)
            else:
                # --- Greedy: pick the flip with maximum net gain ---
                best_gain = -10**9
                best_v = -1
                best_random = False
                for v in range(n):
                    cur = pos_mask[v] if assignment[v] else neg_mask[v]
                    new = neg_mask[v] if assignment[v] else pos_mask[v]
                    gain = bin(new & unsat).count("1") - bin(cur & unsat).count("1")
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v
                        best_random = False
                    elif gain == best_gain:
                        if best_v == -1 or rng.random() < 0.5:
                            best_v = v
                            best_random = True
                v = best_v

            # Apply flip to v.
            old = assignment[v]
            cur = pos_mask[v] if old else neg_mask[v]
            new = neg_mask[v] if old else pos_mask[v]
            assignment[v] = not old

            # Update satisfied mask and unsatisfied set incrementally.
            sat_mask = (sat_mask & ~cur) | new
            unsat = (1 << m) - 1
            unsat &= ~sat_mask
            unsat_count = bin(unsat).count("1")

            # Adaptive noise: increase when stuck, decrease when improving.
            if unsat_count < last_unsat_count:
                no_improve = 0
                last_unsat_count = unsat_count
                noise = max(0.15, noise - 0.05)
            else:
                no_improve += 1
                if no_improve % 100 == 0:
                    noise = min(0.7, noise + 0.05)

        if time.time() > deadline:
            break

    if best_assignment is not None:
        return best_assignment
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
