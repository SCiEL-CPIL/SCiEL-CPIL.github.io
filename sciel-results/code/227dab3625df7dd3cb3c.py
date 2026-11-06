import time
import numpy as np
from itertools import combinations
from math import comb

def construct():
    start_time = time.time()
    deadline = start_time + 112  # safety margin before 150s kill

    def search_for_n(n, time_limit, seed=0):
        M = comb(n, 3)
        triples = list(combinations(range(n), 3))
        tri_to_idx = {tri: i for i, tri in enumerate(triples)}

        # Incidence: 4-sets and 5-sets
        quad_triples = []
        for quad in combinations(range(n), 4):
            idxs = [tri_to_idx[tuple(x for j, x in enumerate(quad) if j != i)] for i in range(4)]
            quad_triples.append(idxs)
        quint_triples = []
        for quint in combinations(range(n), 5):
            idxs = [tri_to_idx[t] for t in combinations(quint, 3)]
            quint_triples.append(idxs)

        n_quads = len(quad_triples)
        n_quints = len(quint_triples)

        quad_triples = np.array(quad_triples, dtype=np.int32)
        quint_triples = np.array(quint_triples, dtype=np.int32)

        # Inverse incidence
        triple_to_4sets = [[] for _ in range(M)]
        for qidx, tidxs in enumerate(quad_triples):
            for t in tidxs:
                triple_to_4sets[t].append(qidx)
        triple_to_5sets = [[] for _ in range(M)]
        for qidx, tidxs in enumerate(quint_triples):
            for t in tidxs:
                triple_to_5sets[t].append(qidx)
        triple_to_4sets = [np.array(x, dtype=np.int32) for x in triple_to_4sets]
        triple_to_5sets = [np.array(x, dtype=np.int32) for x in triple_to_5sets]

        # Bit masks for each quad/quint
        nwords = (M + 63) // 64
        quad_masks = np.zeros((n_quads, nwords), dtype=np.uint64)
        for qidx, tidxs in enumerate(quad_triples):
            for t in tidxs:
                quad_masks[qidx, t >> 6] |= (np.uint64(1) << np.uint64(t & 63))
        quint_masks = np.zeros((n_quints, nwords), dtype=np.uint64)
        for qidx, tidxs in enumerate(quint_triples):
            for t in tidxs:
                quint_masks[qidx, t >> 6] |= (np.uint64(1) << np.uint64(t & 63))

        rng = np.random.RandomState(seed)

        # Modular heuristic initialization
        colors = np.full(M, -1, dtype=np.int8)  # -1 = unassigned, 0 = red, 1 = blue
        for i, (a, b, c) in enumerate(triples):
            colors[i] = 0 if (a + b + c) % 3 == 0 else 1

        # Build red/blue bitsets (only over assigned)
        red_set = np.zeros(nwords, dtype=np.uint64)
        blue_set = np.zeros(nwords, dtype=np.uint64)
        for t in range(M):
            if colors[t] == 0:
                red_set[t >> 6] |= (np.uint64(1) << np.uint64(t & 63))
            elif colors[t] == 1:
                blue_set[t >> 6] |= (np.uint64(1) << np.uint64(t & 63))

        # Learned clauses: list of (frozenset of triple indices, required_color)
        # A clause means: not all these triples have this color.
        # Represent as (mask_words, color)
        learned_clauses = []  # list of (nwords-array, color)

        def count_violations(red_set, blue_set):
            r4 = np.all((quad_masks & red_set[None, :]) == quad_masks, axis=1).sum()
            b5 = np.all((quint_masks & blue_set[None, :]) == quint_masks, axis=1).sum()
            return int(r4) + int(b5)

        def get_violating_quads(red_set):
            return np.where(np.all((quad_masks & red_set[None, :]) == quad_masks, axis=1))[0]

        def get_violating_quints(blue_set):
            return np.where(np.all((quint_masks & blue_set[None, :]) == quad_masks, axis=1))[0]

        def check_learned(red_set, blue_set):
            # Check if any learned clause is violated
            for mask, col in learned_clauses:
                if col == 0:
                    if np.all((mask & red_set[None, :]) == mask, axis=1).any():
                        return True
                else:
                    if np.all((mask & blue_set[None, :]) == mask, axis=1).any():
                        return True
            return False

        def flip_score(t, red_set, blue_set):
            old = colors[t]
            if old == 0:
                qmask = quad_masks[triple_to_4sets[t]]
                broken = np.all((qmask & red_set[None, :]) == qmask, axis=1).sum()
                qmask = quint_masks[triple_to_5sets[t]]
                without_t = qmask & ~(np.uint64(1) << np.uint64(t & 63))
                created = np.all((without_t & blue_set[None, :]) == without_t, axis=1).sum()
                return int(broken) - int(created)
            else:
                qmask = quint_masks[triple_to_5sets[t]]
                broken = np.all((qmask & blue_set[None, :]) == qmask, axis=1).sum()
                qmask = quad_masks[triple_to_4sets[t]]
                without_t = qmask & ~(np.uint64(1) << np.uint64(t & 63))
                created = np.all((without_t & red_set[None, :]) == without_t, axis=1).sum()
                return int(broken) - int(created)

        def apply_flip(t, red_set, blue_set):
            old = colors[t]
            bit = np.uint64(1) << np.uint64(t & 63)
            w = t >> 6
            if old == 0:
                red_set[w] &= ~bit
                blue_set[w] |= bit
                colors[t] = 1
            else:
                blue_set[w] &= ~bit
                red_set[w] |= bit
                colors[t] = 0
            return old

        # Conflict-driven search with clause learning
        # State: colors, red_set, blue_set, depth
        # When we find a violation, learn a clause and backtrack

        # We'll do a hybrid: local search + conflict learning
        # When stuck (no improving move), learn a clause from the current conflict

        best_violations = count_violations(red_set, blue_set)
        best_state = colors.copy()
        best_red = red_set.copy()
        best_blue = blue_set.copy()

        max_conflicts = 2000
        conflicts_found = 0

        for it in range(200000):
            if time.time() > start_time + time_limit:
                break

            viol_quads = get_violating_quads(red_set)
            viol_quints = get_violating_quints(blue_set)

            if len(viol_quads) == 0 and len(viol_quints) == 0:
                # Check learned clauses
                if not check_learned(red_set, blue_set):
                    return colors.tolist()
                else:
                    # Learned clause violated, need to backtrack
                    # Find which clause
                    for mask, col in learned_clauses:
                        if col == 0:
                            viol = np.all((mask & red_set[None, :]) == mask, axis=1)
                        else:
                            viol = np.all((mask & blue_set[None, :]) == mask, axis=1)
                        if viol.any():
                            # Learn a stronger clause: flip one of the triples in the clause
                            # Pick a random one to flip
                            triple_indices = []
                            for t in range(M):
                                if col == 0:
                                    if (red_set[t >> 6] >> (t & 63)) & 1:
                                        triple_indices.append(t)
                                else:
                                    if (blue_set[t >> 6] >> (t & 63)) & 1:
                                        triple_indices.append(t)
                            if triple_indices:
                                t = triple_indices[rng.randint(len(triple_indices))]
                                apply_flip(t, red_set, blue_set)
                            break
                    continue

            # Collect candidates from violations
            candidate_set = set()
            for q in viol_quads:
                for t in quad_triples[q]:
                    candidate_set.add(int(t))
            for q in viol_quints:
                for t in quint_triples[q]:
                    candidate_set.add(int(t))

            if not candidate_set:
                break

            # Score candidates
            best_t = -1
            best_s = 0
            for t in candidate_set:
                s = flip_score(t, red_set, blue_set)
                if s > best_s:
                    best_s = s
                    best_t = t

            if best_t == -1:
                # No improving move: learn a clause from current conflict
                # Use the first violating quad or quint
                if len(viol_quads) > 0:
                    q = viol_quads[0]
                    mask = quad_masks[q].copy()
                    learned_clauses.append((mask, 0))
                elif len(viol_quints) > 0:
                    q = viol_quints[0]
                    mask = quint_masks[q].copy()
                    learned_clauses.append((mask, 1))

                conflicts_found += 1
                if conflicts_found >= max_conflicts:
                    # Give up on this n
                    break

                # Random perturbation to escape
                t = int(rng.randint(0, M))
                apply_flip(t, red_set, blue_set)
            else:
                apply_flip(best_t, red_set, blue_set)

            # Track best
            cur_viol = count_violations(red_set, blue_set)
            if cur_viol < best_violations:
                best_violations = cur_viol
                best_state = colors.copy()
                best_red = red_set.copy()
                best_blue = blue_set.copy()

        # If we didn't find a valid coloring, return best effort if it's valid
        if best_violations == 0 and not check_learned(best_red, best_blue):
            return best_state.tolist()

        return None

    # Fallback valid small coloring
    def fallback(n):
        triples = list(combinations(range(n), 3))
        return [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]

    best_n = 9
    best_colors = fallback(9)

    # Try increasing n with multiple restarts
    for n in range(10, 35):
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
        per_n_budget = min(remaining - 2.0, 35.0)
        if per_n_budget <= 0:
            break
        n_restarts = 4
        per_restart = per_n_budget / n_restarts
        found = None
        for r in range(n_restarts):
            if time.time() > deadline - 2.0:
                break
            t_limit = min(per_restart, deadline - time.time() - 2.0)
            if t_limit <= 0.5:
                break
            res = search_for_n(n, t_limit, seed=n * 1000 + r * 13 + 7)
            if res is not None:
                found = res
                break
        if found is not None:
            best_n = n
            best_colors = found

    return {'n': best_n, 'colors': best_colors}
