"""Portfolio SEARCH approach for K_3(6,1): greedy IS seed + fast local search + perturbation."""
import itertools
import random
import time

import numpy as np

def _build_space(q, n, radius):
    total = q ** n
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    sphere = []
    for w in words:
        s = [word_to_idx[w]]
        for pos in range(n):
            cur = w[pos]
            for v in range(q):
                if v != cur:
                    w2 = list(w)
                    w2[pos] = v
                    s.append(word_to_idx[tuple(w2)])
        sphere.append(s)
    # NumPy adjacency matrix (rows = center, cols = covered word)
    sphere_np = np.zeros((total, total), dtype=np.int8)
    for i, s in enumerate(sphere):
        for w in s:
            sphere_np[i, w] = 1
    return words, sphere, sphere_np, total

def _verify(code_idx, sphere_np):
    total = sphere_np.shape[0]
    counts = sphere_np[code_idx].sum(axis=0)
    return bool((counts > 0).all())

def _coverage_counts(code_idx, sphere_np):
    return sphere_np[code_idx].sum(axis=0)

def construct(q, n, radius):
    start = time.time()
    DEADLINE = start + 115.0  # safety margin under 120/150s limits

    words, sphere, sphere_np, total = _build_space(q, n, radius)
    word_to_idx = {w: i for i, w in enumerate(words)}

    # --- Seed 1: tetracode direct sum (81 words) as a safety net ---
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    tetracode_idx = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            w = h + list(tail)
            tetracode_idx.append(word_to_idx[tuple(w)])
    tetracode_idx = sorted(set(tetracode_idx))

    # --- Seed 2: greedy independent set on the distance-2 graph ---
    # Two centers at Hamming distance 2 have overlapping spheres; excluding such
    # pairs gives a structurally tighter initial cover.
    # Precompute: for each center, the set of centers at distance exactly 2.
    dist2_adj = []
    for i in range(total):
        neighbors = set()
        for j in range(total):
            if i == j:
                continue
            diff = sum(1 for a, b in zip(words[i], words[j]) if a != b)
            if diff == 2:
                neighbors.add(j)
        dist2_adj.append(neighbors)

    # Greedy IS: repeatedly pick the center with the fewest unremoved d=2 neighbors,
    # add it to the code, and remove all its d=2 neighbors from consideration.
    remaining = set(range(total))
    is_seed = []
    while remaining:
        best_c = -1
        best_deg = total + 1
        for c in remaining:
            deg = len(dist2_adj[c] & remaining)
            if deg < best_deg:
                best_deg = deg
                best_c = c
        is_seed.append(best_c)
        remaining.discard(best_c)
        for nb in dist2_adj[best_c]:
            remaining.discard(nb)
    is_seed = sorted(is_seed)

    # --- Local search: fast redundancy removal using NumPy ---
    def local_search(code_idx):
        code = sorted(set(code_idx))
        counts = _coverage_counts(code, sphere_np)
        changed = True
        while changed:
            changed = False
            new_code = []
            for ci in code:
                # redundant iff every word in sphere[ci] has count >= 2
                if counts[sphere_np[ci] == 1].min() > 1:
                    counts -= sphere_np[ci]
                    changed = True
                else:
                    new_code.append(ci)
            code = new_code
        return code

    def greedy_from_uncovered(uncovered_set):
        chosen = []
        uncov = set(uncovered_set)
        while uncov:
            best_c = -1
            best_cnt = -1
            for c in range(total):
                cnt = 0
                for w in sphere[c]:
                    if w in uncov:
                        cnt += 1
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_c = c
            if best_cnt <= 0:
                break
            chosen.append(best_c)
            for w in sphere[best_c]:
                uncov.discard(w)
        return chosen

    def repair_and_local_search(code_idx, uncovered):
        added = greedy_from_uncovered(uncovered)
        combined = list(code_idx) + added
        return local_search(combined)

    rng = random.Random(20240517)

    # --- Pass 1: local search on both seeds, keep the better ---
    best_code = None
    best_size = total + 1
    for seed in [tetracode_idx, is_seed]:
        cand = local_search(seed)
        if len(cand) < best_size:
            best_size = len(cand)
            best_code = cand

    # --- Pass 2: full greedy from scratch (random tie-breaks) + local search ---
    for trial in range(60):
        if time.time() > DEADLINE - 35:
            break
        centers = list(range(total))
        rng.shuffle(centers)
        uncov = set(range(total))
        code = []
        while uncov and len(code) < 90:
            best_c = -1
            best_cnt = -1
            for c in centers:
                if not uncov:
                    break
                cnt = 0
                for w in sphere[c]:
                    if w in uncov:
                        cnt += 1
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_c = c
            if best_c == -1 or best_cnt <= 0:
                break
            code.append(best_c)
            for w in sphere[best_c]:
                uncov.discard(w)
            centers = [c for c in centers if c != best_c]
            rng.shuffle(centers)
        if uncov:
            continue
        code2 = local_search(code)
        if len(code2) < best_size:
            best_size = len(code2)
            best_code = code2

    # --- Pass 3: perturbation: randomly drop centers, repair, local search ---
    iterations = 0
    while time.time() < DEADLINE - 10:
        iterations += 1
        code = list(best_code)
        k = len(code)
        if k < 2:
            break

        # Drop one center
        drop_idx = rng.randrange(k)
        new_code = [c for j, c in enumerate(code) if j != drop_idx]
        counts = _coverage_counts(new_code, sphere_np)
        uncovered = set(np.where(counts == 0)[0])
        repaired = repair_and_local_search(new_code, uncovered)
        if len(repaired) < best_size:
            best_size = len(repaired)
            best_code = repaired

        # Every 4th iteration, also try dropping two centers
        if iterations % 4 == 0 and k >= 2:
            d1 = rng.randrange(k)
            d2 = rng.randrange(k)
            if d1 == d2:
                d2 = (d2 + 1) % k
            new_code = [c for j, c in enumerate(code) if j != d1 and j != d2]
            counts = _coverage_counts(new_code, sphere_np)
            uncovered = set(np.where(counts == 0)[0])
            repaired = repair_and_local_search(new_code, uncovered)
            if len(repaired) < best_size:
                best_size = len(repaired)
                best_code = repaired

    # Final verification
    if not _verify(best_code, sphere_np):
        best_code = tetracode_idx

    return [list(words[i]) for i in sorted(set(best_code))]
