"""Fresh design: greedy distance-2 independent-set cover + efficient local search."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque
from heapq import heappush, heappop

import numpy as np


def construct(q, n, radius):
    start = time.time()
    deadline = start + 105.0          # safety margin before the 120 s kill
    rng = random.Random(12345)

    if q <= 0 or n <= 0:
        return []

    # Trivial / small cases
    if radius >= n:
        return [[0] * n]
    if n == 1:
        return [[i] for i in range(q)]
    if q == 2 and radius == 1:
        out = []
        for i in range(1 << (n - 1)):
            bits = [((i >> (n - 1 - k)) & 1) for k in range(n - 1)]
            bits.append(bin(i).count("1") & 1)
            out.append(bits)
        return out

    N = q ** n

    def idx_to_vec(idx):
        v = [0] * n
        x = idx
        for k in range(n):
            v[k] = x % q
            x //= q
        return v

    all_vecs = [idx_to_vec(i) for i in range(N)]

    def hamming_dist_vec(a, b):
        return sum(1 for x, y in zip(a, b) if x != y)

    # Closed balls (radius `radius`)
    balls = []
    for i in range(N):
        vi = all_vecs[i]
        ball = [j for j in range(N) if hamming_dist_vec(vi, all_vecs[j]) <= radius]
        balls.append(ball)

    # Distance-2 graph adjacency (symmetric)
    def distance2_neighbours(idx):
        v = all_vecs[idx]
        nbrs = []
        for i in range(n):
            vi = v[i]
            for j in range(i + 1, n):
                vj = v[j]
                for ai in range(q):
                    if ai == vi:
                        continue
                    for aj in range(q):
                        if aj == vj:
                            continue
                        ni = idx + (ai - vi) * (q ** i) + (aj - vj) * (q ** j)
                        nbrs.append(ni)
        return nbrs

    adj = [[] for _ in range(N)]
    for i in range(N):
        for j in distance2_neighbours(i):
            if j > i:
                adj[i].append(j)
                adj[j].append(i)

    # ---------- Greedy: maximal independent set that is a cover ----------
    covered = np.zeros(N, dtype=bool)
    forbidden = np.zeros(N, dtype=bool)
    chosen = []
    remaining = set(range(N))

    while remaining:
        if time.time() > deadline - 30.0:
            break
        best = -1
        best_score = -1
        for cand in remaining:
            if forbidden[cand]:
                continue
            cnt = 0
            for u in balls[cand]:
                if not covered[u]:
                    cnt += 1
            if cnt > best_score:
                best_score = cnt
                best = cand
        if best < 0:
            best = next(iter(remaining))
        chosen.append(best)
        remaining.discard(best)
        for u in balls[best]:
            covered[u] = True
        for nb in adj[best]:
            forbidden[nb] = True

    # Ensure full coverage
    while not covered.all():
        best = -1
        best_cnt = -1
        for cand in range(N):
            cnt = 0
            for u in balls[cand]:
                if not covered[u]:
                    cnt += 1
            if cnt > best_cnt:
                best_cnt = cnt
                best = cand
        if best < 0:
            break
        chosen.append(best)
        for u in balls[best]:
            covered[u] = True
        if time.time() > deadline - 10.0:
            break

    codewords = [all_vecs[i] for i in chosen]
    cw_idx = [c[0] for c in codewords]

    # ---------- Efficient local search ----------
    # cover_count[u] = # codewords whose ball contains u
    cc = np.zeros(N, dtype=np.int32)
    for idx in cw_idx:
        for u in balls[idx]:
            cc[u] += 1

    # present set for distance-2 checks
    present = set(cw_idx)

    def try_remove(idx):
        # returns True if removable (all in ball have cc>=2)
        for u in balls[idx]:
            if cc[u] <= 1:
                return False
        return True

    def apply_remove(idx):
        for u in balls[idx]:
            cc[u] -= 1
        present.discard(idx)

    def apply_add(idx):
        for u in balls[idx]:
            cc[u] += 1
        present.add(idx)

    improved = True
    while improved and time.time() < deadline - 3.0:
        improved = False

        # Pass 1: single removals
        i = 0
        while i < len(cw_idx):
            idx = cw_idx[i]
            if try_remove(idx):
                apply_remove(idx)
                cw_idx.pop(i)
                improved = True
                continue
            i += 1
        if improved:
            continue

        # Pass 2: 1-swap (remove one, add one)
        swapped = False
        for ri in range(len(cw_idx)):
            if time.time() > deadline - 3.0:
                break
            ridx = cw_idx[ri]
            apply_remove(ridx)
            # Need a candidate that covers every u with cc[u]==0
            # and is not distance-2 to any remaining codeword
            best_add = -1
            for cand in range(N):
                if cand in present:
                    continue
                # distance-2 constraint
                if any(nb in present for nb in adj[cand]):
                    continue
                # must cover all currently-uncovered
                ok = True
                for u in balls[cand]:
                    if cc[u] == 0:
                        # this vertex is uncovered and cand covers it -> good
                        pass
                # check vertices NOT in ball(cand) are covered
                for u in range(N):
                    if cc[u] == 0 and u not in balls[cand]:
                        ok = False
                        break
                if ok:
                    best_add = cand
                    break
            if best_add >= 0:
                apply_add(best_add)
                cw_idx[ri] = best_add
                swapped = True
                improved = True
            else:
                apply_add(ridx)
            if swapped:
                break
        if improved:
            continue

        # Pass 3: 2-swap (remove two, add two) - limited to keep within time
        two_swapped = False
        L = len(cw_idx)
        for ri in range(L):
            if time.time() > deadline - 3.0:
                break
            for rj in range(ri + 1, L):
                ridx = cw_idx[ri]
                ridx2 = cw_idx[rj]
                apply_remove(ridx)
                apply_remove(ridx2)
                found = False
                for c1 in range(N):
                    if c1 in present:
                        continue
                    if any(nb in present for nb in adj[c1]):
                        continue
                    apply_add(c1)
                    for c2 in range(N):
                        if c2 in present or c2 == c1:
                            continue
                        if any(nb in present for nb in adj[c2]):
                            continue
                        if (cc >= 1).all():
                            apply_add(c2)
                            cw_idx[ri] = c1
                            cw_idx[rj] = c2
                            found = True
                            two_swapped = True
                            improved = True
                            break
                    apply_remove(c1)
                    if found:
                        break
                apply_add(ridx)
                apply_add(ridx2)
                if found:
                    break
            if two_swapped:
                break
        if improved:
            continue

    # Final verification for small N
    if N <= 200000:
        final_cc = np.zeros(N, dtype=np.int32)
        for idx in cw_idx:
            for u in balls[idx]:
                final_cc[u] += 1
        if not final_cc.all():
            for u in range(N):
                if final_cc[u] == 0:
                    cw_idx.append(u)
                    for w in balls[u]:
                        final_cc[w] += 1

    seen = set()
    unique = []
    for idx in cw_idx:
        if idx not in seen:
            seen.add(idx)
            unique.append(all_vecs[idx])
    return unique
# EVOLVE-BLOCK-END
