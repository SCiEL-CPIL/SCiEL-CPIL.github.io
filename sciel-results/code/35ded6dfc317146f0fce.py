import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin under 150s wall-clock

    # ---- Generic fallback for non-target parameters ----
    if not (q == 3 and n == 7 and radius == 1):
        return _generic(q, n, radius)

    N = 3 ** n                       # 2187
    ball_size = 1 + n * (q - 1)      # 15

    # ---------- base-3 <-> int helpers ----------
    def int_to_base3(num, length):
        res = [0] * length
        for i in range(length - 1, -1, -1):
            res[i] = num % 3
            num //= 3
        return res

    def base3_to_int(digits):
        res = 0
        for d in digits:
            res = res * 3 + d
        return res

    # ---------- precompute balls (radius-1 neighborhoods) ----------
    balls = []
    for i in range(N):
        digits = int_to_base3(i, n)
        neighbors = [i]
        for pos in range(n):
            original = digits[pos]
            for new_val in range(q):
                if new_val != original:
                    new_digits = digits[:]
                    new_digits[pos] = new_val
                    neighbors.append(base3_to_int(new_digits))
        balls.append(neighbors)

    # cover_map[w] = list of codewords c whose ball contains w
    cover_map = [[] for _ in range(N)]
    for c in range(N):
        for w in balls[c]:
            cover_map[w].append(c)

    # numpy matrix B[c][w] = 1 if w in ball(c)  -> vectorized gain sums
    B = np.zeros((N, N), dtype=np.int32)
    for c in range(N):
        B[c, balls[c]] = 1

    # ---------- greedy set-cover with randomized tie-break ----------
    def greedy_cover(seed):
        rng = random.Random(seed)
        covered = np.zeros(N, dtype=bool)
        cnt = np.array([ball_size] * N, dtype=np.int32)
        code = []
        uncovered_count = N
        while uncovered_count > 0:
            max_val = int(cnt.max())
            cands = np.flatnonzero(cnt == max_val)
            c = int(rng.choice(cands))
            code.append(c)
            ball_c = balls[c]
            newly = [w for w in ball_c if not covered[w]]
            if newly:
                covered[newly] = True
                uncovered_count -= len(newly)
                for w in newly:
                    cnt[cover_map[w]] -= 1
        return code

    # ---------- pure redundancy removal ----------
    def refine(code):
        code_list = list(code)
        if not code_list:
            return code_list
        cover_count = np.zeros(N, dtype=np.int32)
        for c in code_list:
            cover_count[ball_list[c]] += 1
        changed = True
        while changed:
            changed = False
            for i in range(len(code_list)):
                c = code_list[i]
                # check every word in ball(c) is covered by >1 codeword
                removable = True
                for w in ball_list[c]:
                    if cover_count[w] <= 1:
                        removable = False
                        break
                if not removable:
                    continue
                for w in ball_list[c]:
                    cover_count[w] -= 1
                code_list.pop(i)
                changed = True
                break
        return code_list

    # ---------- precompute ball index arrays for fast nonzero lookups ----------
    ball_idx = [np.flatnonzero(B[c]) for c in range(N)]
    # also store as python lists for fast membership / iteration in refine & local_search
    ball_list = [list(balls[c]) for c in range(N)]

    # ---------- local search: remove-one + relocate/swap + kick ----------
    def local_search(code, seed, time_limit):
        rng = random.Random(seed)
        code_list = list(code)
        if not code_list:
            return code_list
        cover_count = np.zeros(N, dtype=np.int32)
        for c in code_list:
            cover_count[ball_list[c]] += 1
        code_set = set(code_list)
        present_mask = np.zeros(N, dtype=bool)
        for c in code_list:
            present_mask[c] = True

        kick_size = 5
        no_improve_count = 0
        max_no_improve = 8

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            indices = list(range(len(code_list)))
            rng.shuffle(indices)
            for i in indices:
                if time.time() > time_limit:
                    break
                c_remove = code_list[i]
                ball_r = ball_list[c_remove]
                cover_count[ball_r] -= 1
                uncovered = [w for w in ball_r if cover_count[w] == 0]
                if not uncovered:
                    # pure removal
                    code_list.pop(i)
                    code_set.discard(c_remove)
                    present_mask[c_remove] = False
                    improved = True
                    no_improve_count = 0
                    break

                # relocation: vectorized scan for a codeword covering all uncovered
                # and not already present.
                u = np.array(uncovered, dtype=np.int64)
                covers_all = np.all(B[:, u] == 1, axis=1)
                # reuse present_mask; temporarily allow c_remove
                pm = present_mask.copy()
                pm[c_remove] = False
                cand = covers_all & (~pm)
                cand[c_remove] = False
                cand_idx = np.flatnonzero(cand)
                if cand_idx.size > 0:
                    found = int(cand_idx[0])
                    # swap: relocate c_remove -> found
                    cover_count[ball_list[found]] += 1
                    code_list[i] = found
                    code_set.discard(c_remove)
                    code_set.add(found)
                    present_mask[c_remove] = False
                    present_mask[found] = True
                    improved = True
                    no_improve_count = 0
                    break
                else:
                    # rollback
                    cover_count[ball_r] += 1

            # Kick: if stuck, try a 2-swap first, else random kick
            if not improved:
                no_improve_count += 1
                if no_improve_count >= max_no_improve and len(code_list) > 1:
                    no_improve_count = 0
                    # 2-swap: remove two codewords, add two new ones
                    pair = rng.sample(range(len(code_list)), 2)
                    a, b = pair
                    ca, cb = code_list[a], code_list[b]
                    ball_a = ball_list[ca]
                    ball_b = ball_list[cb]
                    cover_count[ball_a] -= 1
                    cover_count[ball_b] -= 1
                    unc_a = [w for w in ball_a if cover_count[w] == 0]
                    unc_b = [w for w in ball_b if cover_count[w] == 0]
                    unc_all = set(unc_a) | set(unc_b)
                    if unc_all:
                        u = np.array(sorted(unc_all), dtype=np.int64)
                        covers_all = np.all(B[:, u] == 1, axis=1)
                        pm = present_mask.copy()
                        pm[ca] = False
                        pm[cb] = False
                        cand = covers_all & (~pm)
                        cand[ca] = False
                        cand[cb] = False
                        cand_idx = np.flatnonzero(cand)
                        if cand_idx.size >= 1:
                            picks = []
                            if cand_idx.size >= 2:
                                sel = rng.choice(cand_idx, size=2, replace=False)
                                picks = [int(sel[0]), int(sel[1])]
                            else:
                                picks = [int(cand_idx[0])]
                            for p in picks:
                                cover_count[ball_list[p]] += 1
                                code_set.add(p)
                                present_mask[p] = True
                            code_list[a] = picks[0]
                            if len(picks) > 1:
                                code_list[b] = picks[1]
                            else:
                                code_list.pop(b)
                            code_set.discard(ca)
                            code_set.discard(cb)
                            present_mask[ca] = False
                            present_mask[cb] = False
                            improved = True
                            continue
                    # rollback 2-swap
                    cover_count[ball_a] += 1
                    cover_count[ball_b] += 1
                    # random kick
                    kick_indices = rng.sample(range(len(code_list)), min(kick_size, len(code_list)))
                    for ki in kick_indices:
                        old_c = code_list[ki]
                        cover_count[ball_list[old_c]] -= 1
                        code_set.discard(old_c)
                        present_mask[old_c] = False
                        new_c = rng.randint(0, N - 1)
                        tries = 0
                        while new_c in code_set and tries < 50:
                            new_c = rng.randint(0, N - 1)
                            tries += 1
                        code_list[ki] = new_c
                        cover_count[ball_list[new_c]] += 1
                        code_set.add(new_c)
                        present_mask[new_c] = True
                    improved = True

        return code_list

    # ---------- spectral seeding: power iteration on Hamming graph adjacency ----------
    # The adjacency matrix A has A[i,j]=1 iff i,j differ in exactly one position.
    # Power iteration on A approximates the top eigenvector (constant, useless).
    # Power iteration on (A - I) or on A with the constant mode deflated gives
    # low-frequency modes that identify "spread-out" vertex sets.
    # We use a cheap variant: start with random vector, multiply by A (via balls),
    # subtract mean, repeat. The resulting vector's top entries are good seeds.

    def spectral_seed(k, seed):
        """Return k candidate codewords ranked by spectral centrality."""
        rng = random.Random(seed)
        v = np.random.RandomState(seed).randn(N).astype(np.float64)
        v -= v.mean()
        for _ in range(40):
            # multiply by A: (A v)[c] = sum of v[w] for w in ball(c) minus v[c] (self)
            # Actually A[c][w]=1 iff w in ball(c) and w != c
            # But our balls[c] includes c itself. So A v = sum_{w in ball(c)} v[w] - v[c]
            v_new = np.zeros(N, dtype=np.float64)
            for c in range(N):
                v_new[c] = v[balls[c]].sum() - v[c]
            v_new -= v_new.mean()
            norm = np.linalg.norm(v_new)
            if norm < 1e-12:
                break
            v = v_new / norm
        # top-k entries
        top = np.argsort(v)[-k:]
        return [int(x) for x in top]

    # ---------- driver: spectral seeds + greedy restarts -> best -> local search ----------
    best_code = None
    best_len = float('inf')

    # Spectral seeding: try several spectral vectors to get diverse good starts
    for spec_seed in [42, 137, 256, 999]:
        if time.time() > deadline - 35.0:
            break
        seeds = spectral_seed(60, spec_seed)
        # Build a greedy cover starting from these seeds
        covered = np.zeros(N, dtype=bool)
        cnt = np.array([ball_size] * N, dtype=np.int32)
        code = []
        for s in seeds:
            ball_s = balls[s]
            newly = [w for w in ball_s if not covered[w]]
            if newly:
                covered[newly] = True
                for w in newly:
                    cnt[cover_map[w]] -= 1
            code.append(s)
        # Now finish with greedy
        uncovered_count = int((~covered).sum())
        while uncovered_count > 0:
            max_val = int(cnt.max())
            cands = np.flatnonzero(cnt == max_val)
            c = int(cands[0])
            code.append(c)
            ball_c = balls[c]
            newly = [w for w in ball_c if not covered[w]]
            if newly:
                covered[newly] = True
                uncovered_count -= len(newly)
                for w in newly:
                    cnt[cover_map[w]] -= 1
        refined = refine(code)
        if len(refined) < best_len:
            best_len = len(refined)
            best_code = refined

    # greedy restarts (cheap, vectorized)
    while time.time() < deadline - 30.0:
        seed = int((time.time() - start_time) * 100000) % (10 ** 9)
        code = greedy_cover(seed)
        refined = refine(code)
        if len(refined) < best_len:
            best_len = len(refined)
            best_code = refined

    if best_code is not None and time.time() < deadline - 2.0:
        ls_code = local_search(best_code, 12345, deadline - 2.0)
        if len(ls_code) < best_len:
            best_len = len(ls_code)
            best_code = ls_code
        # Second local search pass with different seed for diversity
        if time.time() < deadline - 1.0:
            ls_code2 = local_search(best_code, 99999, deadline - 1.0)
            if len(ls_code2) < best_len:
                best_len = len(ls_code2)
                best_code = ls_code2
        # Third pass if time remains, with yet another seed
        if time.time() < deadline - 0.5:
            ls_code3 = local_search(best_code, 77777, deadline - 0.5)
            if len(ls_code3) < best_len:
                best_len = len(ls_code3)
                best_code = ls_code3

    if best_code is not None:
        final_code = refine(best_code)
        # Final polish: try to remove one more codeword via local search with tight time
        if time.time() < deadline - 0.5:
            polished = local_search(final_code, int((time.time() - start_time) * 1000) % 10**9, deadline - 0.3)
            if len(polished) < len(final_code):
                final_code = refine(polished)
    else:
        final_code = []

    return [int_to_base3(c, n) for c in final_code]


def _generic(q, n, radius):
    N = q ** n

    def int_to_base(num, length, base):
        res = [0] * length
        for i in range(length - 1, -1, -1):
            res[i] = num % base
            num //= base
        return res

    balls = []
    for i in range(N):
        digits = int_to_base(i, n, q)
        neighbors = set([i])
        for pos in range(n):
            original = digits[pos]
            for new_val in range(q):
                if new_val != original:
                    new_digits = digits[:]
                    new_digits[pos] = new_val
                    val = 0
                    for d in new_digits:
                        val = val * q + d
                    neighbors.add(val)
        balls.append(list(neighbors))

    covered = set()
    code = []
    remaining = set(range(N))
    while remaining:
        best_c = None
        best_count = -1
        for c in remaining:
            count = len(balls[c] & remaining)
            if count > best_count:
                best_count = count
                best_c = c
        code.append(best_c)
        covered.update(balls[best_c])
        remaining -= set(balls[best_c])

    # light refinement
    cover_count = [0] * N
    for c in code:
        for w in balls[c]:
            cover_count[w] += 1
    changed = True
    while changed:
        changed = False
        for i in range(len(code)):
            c = code[i]
            if all(cover_count[w] > 1 for w in balls[c]):
                for w in balls[c]:
                    cover_count[w] -= 1
                code.pop(i)
                changed = True
                break

    return [int_to_base(c, n, q) for c in code]
