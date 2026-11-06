import time
import numpy as np
from itertools import combinations
from math import comb

def construct():
    start_time = time.time()
    deadline = start_time + 112  # safety margin before 120s kill (evaluator kills at 150s)

    def search_for_n(n, time_limit, seed=0):
        M = comb(n, 3)
        triples = list(combinations(range(n), 3))
        tri_to_idx = {tri: i for i, tri in enumerate(triples)}

        # Incidence: which 4-sets and 5-sets contain each triple
        quad_triples = []   # list of 4 triple-indices per 4-set
        for quad in combinations(range(n), 4):
            idxs = [tri_to_idx[tuple(x for j, x in enumerate(quad) if j != i)] for i in range(4)]
            quad_triples.append(idxs)
        quint_triples = []  # list of 10 triple-indices per 5-set
        for quint in combinations(range(n), 5):
            idxs = [tri_to_idx[t] for t in combinations(quint, 3)]
            quint_triples.append(idxs)

        quad_triples = np.array(quad_triples, dtype=np.int32)
        quint_triples = np.array(quint_triples, dtype=np.int32)

        # For each triple, the 4-sets and 5-sets it belongs to (for incremental delta)
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

        # Bit masks: for each 4-set, a bitmask (np.uint64 array over M bits) of its 4 triples
        # Use numpy bitarray as a uint64 array of length (M+63)//64
        nwords = (M + 63) // 64
        quad_masks = np.zeros((len(quad_triples), nwords), dtype=np.uint64)
        for qidx, tidxs in enumerate(quad_triples):
            for t in tidxs:
                quad_masks[qidx, t >> 6] |= (np.uint64(1) << np.uint64(t & 63))
        quint_masks = np.zeros((len(quint_triples), nwords), dtype=np.uint64)
        for qidx, tidxs in enumerate(quint_triples):
            for t in tidxs:
                quint_masks[qidx, t >> 6] |= (np.uint64(1) << np.uint64(t & 63))

        rng = np.random.RandomState(seed)

        # Initial coloring: random with bias, then hill-climb
        colors = rng.randint(0, 2, size=M, dtype=np.int8)

        # Build red/blue bitsets
        red_set = np.zeros(nwords, dtype=np.uint64)
        blue_set = np.zeros(nwords, dtype=np.uint64)
        for t in range(M):
            if colors[t] == 0:
                red_set[t >> 6] |= (np.uint64(1) << np.uint64(t & 63))
            else:
                blue_set[t >> 6] |= (np.uint64(1) << np.uint64(t & 63))

        def count_violations(red_set, blue_set):
            # red K4: quad where all 4 bits set in red_set
            r4 = np.all((quad_masks & red_set[None, :]) == quad_masks, axis=1).sum()
            b5 = np.all((quint_masks & blue_set[None, :]) == quint_masks, axis=1).sum()
            return int(r4) + int(b5)

        def violation_counts(red_set, blue_set):
            r4 = np.all((quad_masks & red_set[None, :]) == quad_masks, axis=1)
            b5 = np.all((quint_masks & blue_set[None, :]) == quint_masks, axis=1)
            return r4, b5

        # Score of flipping triple t: how many violations it removes minus how many it creates
        def flip_score(t, red_set, blue_set):
            old = colors[t]
            # If old is red (0): removing it from red_set can break red K4s; adding to blue_set can create blue K5s
            if old == 0:
                # red K4s broken: quads containing t that are currently all-red
                qmask = quad_masks[triple_to_4sets[t]]
                broken = np.all((qmask & red_set[None, :]) == qmask, axis=1).sum()
                # blue K5s created: quints containing t where all OTHER 9 are already blue
                qmask = quint_masks[triple_to_5sets[t]]
                # a quint becomes all-blue iff (qmask & blue_set) has 9 bits and the t-bit is the missing one
                # i.e. (qmask & blue_set) == qmask with the t-bit cleared
                without_t = qmask & ~(np.uint64(1) << np.uint64(t & 63))
                created = np.all((without_t & blue_set[None, :]) == without_t, axis=1).sum()
                return int(broken) - int(created)
            else:
                # old is blue (1): removing from blue_set breaks blue K5s; adding to red_set can create red K4s
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

        # Greedy steepest descent
        max_iter = 200000
        for it in range(max_iter):
            if time.time() > start_time + time_limit:
                break
            best_t = -1
            best_s = 0
            # Sample a random subset of triples for speed when M is large, else all
            if M <= 4000:
                candidates = range(M)
            else:
                candidates = rng.choice(M, size=min(M, 2000), replace=False)
            for t in candidates:
                s = flip_score(t, red_set, blue_set)
                if s > best_s:
                    best_s = s
                    best_t = t
            if best_t == -1:
                break
            apply_flip(best_t, red_set, blue_set)
            if count_violations(red_set, blue_set) == 0:
                return colors.tolist()

        # Simulated annealing
        T = 1.0
        cooling = 0.99995
        min_T = 0.005
        while time.time() < start_time + time_limit:
            t = int(rng.randint(0, M))
            old = colors[t]
            s = flip_score(t, red_set, blue_set)
            apply_flip(t, red_set, blue_set)
            if s < 0 or (T > min_T and rng.random() < np.exp(-s / T)):
                pass
            else:
                # revert
                apply_flip(t, red_set, blue_set)
            if count_violations(red_set, blue_set) == 0:
                return colors.tolist()
            T *= cooling

        return None

    # Fallback valid small coloring (n=9 with modular heuristic is safe)
    def fallback(n):
        triples = list(combinations(range(n), 3))
        return [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]

    best_n = 9
    best_colors = fallback(9)

    # Try increasing n, with multiple random restarts and a per-n time budget
    for n in range(10, 35):
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
        # Allocate budget: more time for larger n, but keep margin
        per_n_budget = min(remaining - 2.0, 30.0)
        if per_n_budget <= 0:
            break
        # Multiple restarts within the budget
        n_restarts = 3
        per_restart = per_n_budget / n_restarts
        found = None
        for r in range(n_restarts):
            if time.time() > deadline - 2.0:
                break
            t_limit = min(per_restart, deadline - time.time() - 2.0)
            if t_limit <= 0.5:
                break
            res = search_for_n(n, t_limit, seed=n * 1000 + r * 7 + 1)
            if res is not None:
                found = res
                break
        if found is not None:
            best_n = n
            best_colors = found

    return {'n': best_n, 'colors': best_colors}
