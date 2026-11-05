import time
import random
import numpy as np
import math

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin vs 150s kill

    if q != 3 or n != 7 or radius != 1:
        # Trivial valid fallback: return all words (only used off-task)
        total = q ** n
        result = []
        for i in range(total):
            w = []
            x = i
            for _ in range(n):
                w.append(x % q)
                x //= q
            result.append(w)
        return result

    N = 3 ** n  # 2187
    deg = 1 + 2 * n  # 15

    # ---------- Precompute neighbor structure ----------
    # neighbors_flat: shape (N, deg), each row = the word + all 14 single-position changes
    neighbors_flat = np.empty((N, deg), dtype=np.int32)
    for i in range(N):
        w = [0] * n
        x = i
        for p in range(n - 1, -1, -1):
            w[p] = x % 3
            x //= 3
        row = [i]
        for pos in range(n):
            v = w[pos]
            for nv in range(3):
                if nv != v:
                    x2 = i
                    # change position pos to nv
                    base = 3 ** (n - 1 - pos)
                    x2 = i - v * base + nv * base
                    row.append(x2)
        neighbors_flat[i] = row

    # Precompute per-word neighbor sets as python sets for fast membership / repair
    nb_sets = [set(neighbors_flat[i].tolist()) for i in range(N)]

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % 3
            idx //= 3
        return t

    def greedy_construct(seed_val, time_limit):
        rng = random.Random(seed_val)
        cov = np.zeros(N, dtype=np.int32)
        cod = []
        cod_set = set()
        unc_count = N

        # Random start
        start = rng.randint(0, N - 1)
        cod.append(start)
        cod_set.add(start)
        cov[neighbors_flat[start]] += 1
        unc_count -= np.sum(cov[neighbors_flat[start]] == 1)

        # Precompute the flat neighbor indices of all words (for vectorized gain)
        all_nb = neighbors_flat.ravel()  # (N*deg,)

        while unc_count > 0 and time.time() < time_limit:
            # gain[i] = number of currently-uncovered words that are neighbors of i
            # = sum over uncovered words u of [i in neighbors(u)]
            # Build uncovered boolean, then scatter
            unc_mask = (cov == 0)
            # For each uncovered word u, add 1 to gain of all its neighbors.
            # gain[neighbors_flat[u]] += 1  for each uncovered u.
            # Vectorize: gather all neighbor rows of uncovered words, add 1 via np.add.at
            unc_idx = np.where(unc_mask)[0]
            if len(unc_idx) == 0:
                break
            nb_rows = neighbors_flat[unc_idx]  # (len(unc_idx), deg)
            gain = np.zeros(N, dtype=np.int32)
            np.add.at(gain, nb_rows.ravel(), 1)
            gain[cod_set] = 0  # cannot reselect

            max_gain = gain.max()
            if max_gain <= 0:
                break
            best = np.where(gain == max_gain)[0]
            pick = best[rng.randint(0, len(best) - 1)]

            cod.append(pick)
            cod_set.add(pick)
            prev = cov[neighbors_flat[pick]]
            cov[neighbors_flat[pick]] += 1
            # words that became covered for the first time
            unc_count -= np.sum((prev == 0) & (cov[neighbors_flat[pick]] == 1))

        return cod, cod_set, cov

    def strip_redundant(cod, cod_set, cov, time_limit):
        """Remove all redundant codewords (those whose removal keeps coverage valid)."""
        changed = True
        while changed:
            changed = False
            # A codeword c is redundant iff every neighbor has coverage >= 2
            # (i.e., min coverage over its neighbors >= 2).
            # Vectorized: for all codewords, compute min coverage over neighbors.
            if not cod:
                break
            arr = np.array(cod, dtype=np.int32)
            nb_rows = neighbors_flat[arr]  # (len(cod), deg)
            mincov = cov[nb_rows].min(axis=1)
            redundant_mask = mincov >= 2
            if not redundant_mask.any():
                break
            for k in range(len(cod)):
                if redundant_mask[k]:
                    c = cod[k]
                    cod.remove(c)
                    cod_set.remove(c)
                    cov[neighbors_flat[c]] -= 1
                    changed = True
        return cod, cod_set, cov

    def greedy_repair(uncovered_set, cod_set, time_limit):
        """Add minimal codewords to cover uncovered_set. Returns list added."""
        if not uncovered_set:
            return []
        added = []
        temp = set(uncovered_set)
        while temp:
            if time.time() > time_limit:
                break
            # candidate pool = neighbors of uncovered words, not already selected
            best_c = -1
            best_cnt = 0
            for u in temp:
                for c in nb_sets[u]:
                    if c in cod_set:
                        continue
                    cnt = 0
                    # count how many of temp are neighbors of c
                    for t in temp:
                        if t in nb_sets[c]:
                            cnt += 1
                    if cnt > best_cnt:
                        best_cnt = cnt
                        best_c = c
                        if best_cnt == len(temp):
                            break
                if best_cnt == len(temp):
                    break
            if best_c == -1:
                break
            added.append(best_c)
            cod_set.add(best_c)
            temp.difference_update(nb_sets[best_c])
        return added

    def local_search(cod, cod_set, cov, time_limit):
        """Deletion + 2-for-1 swap local search with greedy repair."""
        rng = random.Random(987654321)
        best_code = cod[:]
        best_set = cod_set.copy()
        best_cov = cov.copy()
        best_size = len(cod)

        def try_apply(removed, added):
            nonlocal cod, cod_set, cov, best_code, best_set, best_cov, best_size
            for c in removed:
                cod.remove(c)
                cod_set.remove(c)
                cov[neighbors_flat[c]] -= 1
            for c in added:
                if c not in cod_set:
                    cod.append(c)
                    cod_set.add(c)
                    cov[neighbors_flat[c]] += 1
            if len(cod) < best_size:
                best_size = len(cod)
                best_code = cod[:]
                best_set = cod_set.copy()
                best_cov = cov.copy()

        # Initial redundancy strip
        cod, cod_set, cov = strip_redundant(cod, cod_set, cov, time_limit)
        if len(cod) < best_size:
            best_size = len(cod)
            best_code = cod[:]
            best_set = cod_set.copy()
            best_cov = cov.copy()

        while time.time() < time_limit:
            # Choose move type
            r = rng.random()
            if r < 0.4:
                # Deletion of 1
                if len(cod) < 1:
                    continue
                c = cod[rng.randint(0, len(cod) - 1)]
                removed = [c]
            elif r < 0.7:
                # Deletion of 2
                if len(cod) < 2:
                    continue
                i, j = rng.sample(range(len(cod)), 2)
                removed = [cod[i], cod[j]]
            else:
                # 2-for-1 swap: remove 2, add 1
                if len(cod) < 2:
                    continue
                i, j = rng.sample(range(len(cod)), 2)
                removed = [cod[i], cod[j]]
                # pick a candidate to add: neighbor of an uncovered word after removal
                # (handled below via repair with max_added=1)

            # simulate removal
            temp_cov = cov.copy()
            temp_set = cod_set.copy()
            for c in removed:
                temp_cov[neighbors_flat[c]] -= 1
                temp_set.remove(c)

            uncovered = np.where(temp_cov == 0)[0]
            if len(uncovered) == 0:
                # pure improvement
                try_apply(removed, [])
                continue

            uncovered_set = set(uncovered.tolist())
            # repair: find up to (1 if swap else 2) codewords to cover
            max_add = 1 if (r >= 0.7) else 2
            added = greedy_repair(uncovered_set, temp_set, time_limit)
            if len(added) > max_add:
                # too expensive, skip
                continue
            new_size = len(cod) - len(removed) + len(added)
            if new_size < len(cod):
                try_apply(removed, added)

        return best_code, best_set, best_cov

    # ---------- Main ----------
    best_code = None
    best_size = float('inf')
    best_set = None
    best_cov = None

    num_starts = 30
    for i in range(num_starts):
        if time.time() > deadline - 20:
            break
        # Greedy with shrinking time budget
        g_time = deadline - 18
        code, code_set, cov = greedy_construct(1000 + i, g_time)

        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            best_set = code_set
            best_cov = cov

        # Refine
        if time.time() < deadline - 8:
            rc, rs, rcov = local_search(code, code_set, cov, deadline - 6)
            if len(rc) < best_size:
                best_size = len(rc)
                best_code = rc
                best_set = rs
                best_cov = rcov

    if best_code is None:
        code, code_set, cov = greedy_construct(0, deadline - 5)
        best_code = code

    return [idx_to_tuple(i) for i in best_code]
