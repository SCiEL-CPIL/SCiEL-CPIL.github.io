"""
Ternary covering code K_3(6,1) via tetracode construction + local search.
"""
import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin

    if q != 3 or n != 6 or radius != 1:
        return _greedy_construct(q, n, radius, deadline - (time.time() - start_time), start_time)

    return _construct_k3_6_1(deadline - (time.time() - start_time), start_time)


def _greedy_construct(q, n, radius, time_limit, start_time):
    """Simple greedy fallback for non-standard parameters."""
    total_words = q ** n
    if total_words > 50000:
        return []

    def w2i(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx

    # Build neighbor sets
    neighbor_sets = []
    for i in range(total_words):
        w = [(i // (q ** (n - 1 - pos))) % q for pos in range(n)]
        nbrs = {i}
        if radius == 1:
            for pos in range(n):
                for nv in range(q):
                    if nv != w[pos]:
                        nw = w[:]
                        nw[pos] = nv
                        nbrs.add(w2i(nw))
        neighbor_sets.append(nbrs)

    covered = np.zeros(total_words, dtype=bool)
    code = []
    while time.time() < start_time + time_limit - 1.0:
        gains = np.array([len(neighbor_sets[i] - set(np.where(covered)[0])) for i in range(total_words)])
        best = int(np.argmax(gains))
        if gains[best] == 0:
            break
        code.append(best)
        covered[list(neighbor_sets[best])] = True

    result = []
    for idx in code:
        result.append([(idx // (q ** (n - 1 - pos))) % q for pos in range(n)])
    return result


def _construct_k3_6_1(time_limit, start_time):
    n = 6
    q = 3
    total_words = 729

    # Tetracode: perfect 1-error-correcting code of length 4 over GF(3), 9 words
    tetracode = [(a, b, (a + b) % 3, (a + 2 * b) % 3) for a in range(3) for b in range(3)]

    # Direct sum with all 3^2 tails -> 9 * 9 = 81 codewords
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            initial_code.append(h + tail)

    m = len(initial_code)  # 81

    # Precompute word-to-index
    def w2i(w):
        idx = 0
        for v in w:
            idx = idx * 3 + v
        return idx

    # Precompute sphere (13 words each) for all 729 words
    spheres = []
    for i in range(total_words):
        w = [(i // (3 ** (5 - pos))) % 3 for pos in range(6)]
        nbrs = {i}
        for pos in range(6):
            ov = w[pos]
            for nv in range(3):
                if nv != ov:
                    nw = w[:]
                    nw[pos] = nv
                    nbrs.add(w2i(nw))
        spheres.append(sorted(nbrs))

    # Convert to numpy arrays for fast ops
    sphere_arrays = [np.array(s, dtype=np.int32) for s in spheres]

    # Initial code as indices
    code_indices = [w2i(c) for c in initial_code]
    code_set = set(code_indices)

    # Coverage count
    coverage = np.zeros(total_words, dtype=np.int32)
    for ci in code_indices:
        coverage[sphere_arrays[ci]] += 1

    best_size = m
    best_code = code_indices[:]

    # Local search: iterative removals and swaps
    search_deadline = start_time + time_limit - 2.0
    local_iter = 0
    max_iter = 500000

    current_code = code_indices[:]
    current_set = set(current_code)
    current_coverage = coverage.copy()

    while time.time() < search_deadline and local_iter < max_iter:
        local_iter += 1

        # --- Phase 1: Greedy removals (multiple passes) ---
        removed_any = False
        for _pass in range(3):
            changed = False
            # Randomize order each pass
            order = list(range(len(current_code)))
            random.shuffle(order)

            for idx in order:
                if idx >= len(current_code):
                    continue
                ci = current_code[idx]
                # Check if redundant
                redundant = True
                for wi in spheres[ci]:
                    if current_coverage[wi] <= 1:
                        redundant = False
                        break
                if redundant:
                    current_coverage[sphere_arrays[ci]] -= 1
                    current_code.pop(idx)
                    current_set.discard(ci)
                    changed = True
                    removed_any = True

            if not changed:
                break

        if removed_any:
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = current_code[:]

        # --- Phase 2: Swap search (replace one codeword with another) ---
        if len(current_code) > 0:
            # Try to find a beneficial swap
            swap_done = False
            # Pick a random codeword to try removing
            ri = random.randint(0, len(current_code) - 1)
            ci_old = current_code[ri]

            # Words uniquely covered by ci_old
            unique = [wi for wi in spheres[ci_old] if current_coverage[wi] == 1]

            if len(unique) <= 6:
                unique_set = set(unique)
                # Try candidate replacements from all 729 words
                # Sample a subset for speed
                candidates = list(range(total_words))
                random.shuffle(candidates)
                for ci_new in candidates[:200]:
                    if ci_new in current_set:
                        continue
                    # Check if ci_new covers all uniquely-covered words
                    ok = True
                    for wi in unique:
                        if wi not in spheres[ci_new]:
                            ok = False
                            break
                    if ok:
                        # Perform swap
                        current_coverage[sphere_arrays[ci_old]] -= 1
                        current_coverage[sphere_arrays[ci_new]] += 1
                        current_code[ri] = ci_new
                        current_set.discard(ci_old)
                        current_set.add(ci_new)
                        swap_done = True
                        break

            if not swap_done and len(current_code) > best_size - 5:
                # Escape: add a random codeword to try breaking local minimum
                rand_ci = random.randint(0, total_words - 1)
                if rand_ci not in current_set:
                    current_code.append(rand_ci)
                    current_set.add(rand_ci)
                    current_coverage[sphere_arrays[rand_ci]] += 1

    # Final cleanup: remove any remaining redundant codewords
    for _ in range(10):
        changed = False
        order = list(range(len(current_code)))
        random.shuffle(order)
        for idx in order:
            if idx >= len(current_code):
                continue
            ci = current_code[idx]
            redundant = True
            for wi in spheres[ci]:
                if current_coverage[wi] <= 1:
                    redundant = False
                    break
            if redundant:
                current_coverage[sphere_arrays[ci]] -= 1
                current_code.pop(idx)
                current_set.discard(ci)
                changed = True
        if not changed:
            break

    if len(current_code) < best_size:
        best_size = len(current_code)
        best_code = current_code[:]

    # Convert back to words
    result = []
    for idx in best_code:
        w = [(idx // (3 ** (5 - pos))) % 3 for pos in range(6)]
        result.append(w)

    return result
