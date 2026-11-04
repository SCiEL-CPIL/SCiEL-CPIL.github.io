"""Local-search minimizer for K_3(8,1) football pool / ternary covering code."""
# EVOLVE-BLOCK-START
import time
import random
import itertools


def _neighbors_idx(idx, n, q):
    """Return list of indices (including self) within Hamming distance 1 of the word at idx."""
    neigh = [idx]
    for pos in range(n):
        base = (idx // (q ** pos)) % q
        for d in range(1, q):
            nidx = idx + d * (q ** pos)
            if (nidx // (q ** pos)) % q != base:
                neigh.append(nidx)
    return neigh


def _generate_initial_cover(q, n, radius, seed):
    """Generate a valid cover. For the ternary case use the 486 van Laarhoven construction."""
    if q == 3 and n == 8 and radius == 1:
        # 486 = 3 * 3**6 : the ternary tetracode (9 words length 4) direct-summed
        # with the full 3^4 space on the remaining 4 coordinates.
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        cover = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                cover.append(h + list(tail))
        return cover

    # Generic fallback: exhaustive over the full space (valid but large).
    return [list(w) for w in itertools.product(range(q), repeat=n)]


def construct(q, n, radius):
    start = time.time()
    deadline = start + 110.0  # safety margin before the 150 s kill
    rng = random.Random(12345)

    cover = _generate_initial_cover(q, n, radius, rng)
    cover_set = set(tuple(c) for c in cover)
    total = q ** n

    # Precompute per-position powers and neighbor index lists for all indices.
    pow_q = [1]
    for _ in range(n):
        pow_q.append(pow_q[-1] * q)

    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx += w[i] * pow_q[i]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = (idx // pow_q[i]) % q
        return w

    # Sphere (distance <= radius) for each index. For radius 1 use precomputed neighbors.
    if radius == 1:
        sphere = [None] * total
        for i in range(total):
            sphere[i] = _neighbors_idx(i, n, q)
    else:
        # Fallback: build spheres by scanning (only used if radius != 1).
        sphere = [None] * total
        for i in range(total):
            ci = idx_to_word(i)
            s = []
            for j in range(total):
                cj = idx_to_word(j)
                d = sum(1 for a, b in zip(ci, cj) if a != b)
                if d <= radius:
                    s.append(j)
            sphere[i] = s

    cover_idx = [word_to_idx(c) for c in cover]
    cover_idx_set = set(cover_idx)

    coverage = [0] * total
    for ci in cover_idx:
        for s in sphere[ci]:
            coverage[s] += 1

    def is_removable(ci):
        for s in sphere[ci]:
            if coverage[s] < 2:
                return False
        return True

    def remove_codeword(ci):
        for s in sphere[ci]:
            coverage[s] -= 1
        cover_idx.remove(ci)
        cover_idx_set.remove(ci)

    def add_codeword(ci):
        for s in sphere[ci]:
            coverage[s] += 1
        cover_idx.append(ci)
        cover_idx_set.add(ci)

    best_size = len(cover_idx)

    # ---- Greedy removal pass (order by sphere size, then index) ----
    def greedy_removal():
        # Sort by sphere size ascending (smaller spheres are easier to make redundant).
        order = sorted(cover_idx, key=lambda ci: (len(sphere[ci]), ci))
        progress = True
        while progress and time.time() < deadline - 5.0:
            progress = False
            for ci in order:
                if ci not in cover_idx_set:
                    continue
                if is_removable(ci):
                    remove_codeword(ci)
                    progress = True
                    if len(cover_idx) < best_size:
                        best_size = len(cover_idx)
        return progress

    # ---- Swap move: move one codeword to a random neighbor, keep cover valid ----
    def try_swap():
        if not cover_idx:
            return False
        ci = cover_idx[rng.randrange(len(cover_idx))]
        # candidate new positions: neighbors of ci not already in cover
        cand = [s for s in sphere[ci] if s not in cover_idx_set and s != ci]
        if not cand:
            return False
        # Prefer candidates that cover currently low-coverage words
        cand.sort(key=lambda s: min(coverage[t] for t in sphere[s]))
        for new_ci in cand:
            # tentative move
            for s in sphere[ci]:
                coverage[s] -= 1
            ok = all(coverage[s] >= 1 for s in sphere[ci])
            for s in sphere[new_ci]:
                coverage[s] += 1
            if ok:
                # accept the move
                cover_idx.remove(ci)
                cover_idx_set.remove(ci)
                add_codeword(new_ci)
                return True
            else:
                # revert
                for s in sphere[new_ci]:
                    coverage[s] -= 1
                for s in sphere[ci]:
                    coverage[s] += 1
        return False

    # Main loop: alternate greedy removal and swaps to escape local minima.
    no_progress_rounds = 0
    max_no_progress = 30
    while time.time() < deadline - 5.0:
        before = len(cover_idx)
        greedy_removal()
        if len(cover_idx) < before:
            no_progress_rounds = 0
        else:
            # stuck: try some swaps, then re-run removal
            swapped = False
            for _ in range(50):
                if time.time() > deadline - 5.0:
                    break
                if try_swap():
                    swapped = True
            greedy_removal()
            if len(cover_idx) < before:
                no_progress_rounds = 0
            else:
                no_progress_rounds += 1
                if no_progress_rounds >= max_no_progress and not swapped:
                    break

    # Final exhaustive removal sweep to catch anything left.
    greedy_removal()

    # Reconstruct the word lists.
    result = [idx_to_word(ci) for ci in cover_idx]

    # Safety: verify distinctness and (if time allows) full coverage.
    if time.time() < deadline - 2.0:
        seen = set()
        for w in result:
            t = tuple(w)
            if t in seen:
                # duplicate: fall back to initial cover
                result = _generate_initial_cover(q, n, radius, rng)
                break
            seen.add(t)

    return result
# EVOLVE-BLOCK-END
