import time
import random
import numpy as np
import itertools


def construct(q, n, radius):
    """
    Construct a ternary covering code K_3(6,1).
    Returns a list of codewords (lists of integers).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Fallback for non-standard parameters
    if q != 3 or n != 6 or radius != 1:
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    total_words = 3 ** n  # 729
    sphere_size = 1 + n * (q - 1)  # 13

    # --- 1. Precompute Data Structures ---

    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for i in range(n):
            w[n - 1 - i] = temp % q
            temp //= q
        return tuple(w)

    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * q + val
        return idx

    # Precompute spheres as lists and bitmasks
    spheres_list = [[] for _ in range(total_words)]
    spheres_mask = [0] * total_words

    for idx in range(total_words):
        w = list(idx_to_word(idx))
        sphere_set = {idx}
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w[pos] = new_val
                sphere_set.add(word_to_idx(tuple(w)))
                w[pos] = orig_val
        spheres_list[idx] = list(sphere_set)
        mask = 0
        for w_idx in sphere_set:
            mask |= (1 << w_idx)
        spheres_mask[idx] = mask

    # Precompute covered_by: for each word w, which codewords cover it
    covered_by = [[] for _ in range(total_words)]
    covered_by_mask = [0] * total_words
    for c in range(total_words):
        for w in spheres_list[c]:
            covered_by[w].append(c)
    for w in range(total_words):
        mask = 0
        for c in covered_by[w]:
            mask |= (1 << c)
        covered_by_mask[w] = mask

    # Precompute distance-2 neighbors for independent set construction
    # dist2_neighbors[i] = set of j such that Hamming distance between word i and word j is exactly 2
    dist2_neighbors = [set() for _ in range(total_words)]
    for i in range(total_words):
        w1 = idx_to_word(i)
        for j in range(i + 1, total_words):
            w2 = idx_to_word(j)
            d = 0
            for k in range(n):
                if w1[k] != w2[k]:
                    d += 1
                    if d > 2:
                        break
            if d == 2:
                dist2_neighbors[i].add(j)
                dist2_neighbors[j].add(i)

    # --- 2. Initial Code Construction via Greedy Independent Set ---

    def build_initial_code():
        """Build initial cover using greedy independent set on distance-2 graph."""
        rng_init = random.Random(42)
        best_iset = []
        n_restarts = 5

        for restart in range(n_restarts):
            if time.time() > deadline - 60:
                break
            rng_r = random.Random(restart * 12345 + 678)
            remaining = set(range(total_words))
            independent_set = []
            while remaining:
                # Pick vertex with lowest degree among remaining, with random tie-breaking
                sample_size = min(30, len(remaining))
                candidates = rng_r.sample(list(remaining), sample_size)
                best_v = None
                best_deg = float('inf')
                for v in candidates:
                    deg = len(dist2_neighbors[v] & remaining)
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
                if best_v is not None:
                    independent_set.append(best_v)
                    remaining.discard(best_v)
                    remaining -= dist2_neighbors[best_v]
            if len(independent_set) > len(best_iset):
                best_iset = independent_set

        # Check if this forms a valid cover; if not, fall back to tetracode
        counts_check = np.zeros(total_words, dtype=np.int32)
        for idx in best_iset:
            counts_check[spheres_list[idx]] += 1
        if np.any(counts_check == 0):
            # Fall back to tetracode direct sum
            tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
            initial_indices = []
            for h in tetracode:
                for tail in itertools.product(range(q), repeat=n - 4):
                    cw = tuple(h + list(tail))
                    initial_indices.append(word_to_idx(cw))
            return initial_indices
        return best_iset

    initial_indices = build_initial_code()

    # --- 3. Helper Functions ---

    def compute_counts(code_indices):
        counts = np.zeros(total_words, dtype=np.int32)
        for idx in code_indices:
            counts[spheres_list[idx]] += 1
        return counts

    def reduce_code(code_indices, counts=None):
        """Greedily remove redundant codewords."""
        if counts is None:
            counts = compute_counts(code_indices)

        code_set = set(code_indices)
        changed = True
        while changed:
            changed = False
            for idx in list(code_set):
                if np.min(counts[spheres_list[idx]]) > 1:
                    code_set.remove(idx)
                    counts[spheres_list[idx]] -= 1
                    changed = True
        return list(code_set), counts

    def find_valid_swaps(c_remove, current_set, counts, max_candidates=50):
        """
        Find valid c_add values for a given c_remove using CSP-style filtering.
        A swap is valid if every word in Sphere(c_remove) with count==1 is also in Sphere(c_add).
        """
        critical_words = [w for w in spheres_list[c_remove] if counts[w] == 1]
        if not critical_words:
            # No critical words: any c_add not in current_set works, but we want to reduce
            # Actually if no critical words, removing c_remove is safe, so this is handled by reduce
            return []

        # Start with candidates that cover the first critical word
        candidates = set(covered_by[critical_words[0]])
        candidates -= current_set

        # Apply arc consistency: intersect with coverers of remaining critical words
        for w in critical_words[1:]:
            if not candidates:
                break
            filtered = set()
            for c in candidates:
                if (covered_by_mask[w] >> c) & 1:
                    filtered.add(c)
            candidates = filtered

        if max_candidates and len(candidates) > max_candidates:
            candidates = set(random.Random(12345).sample(list(candidates), max_candidates))

        return list(candidates)

    def try_2for1_merge(active_set, counts, rng):
        """
        Try to find two codewords c1, c2 such that some c3 covers Sphere(c1) ∪ Sphere(c2).
        If found, replace {c1, c2} with {c3}.
        Returns (c1, c2, c3) or None.
        """
        active_list = list(active_set)
        if len(active_list) < 2:
            return None

        # Sample a few random pairs
        n_tries = min(20, len(active_list) * (len(active_list) - 1) // 2)
        for _ in range(n_tries):
            c1, c2 = rng.sample(active_list, 2)
            union_sphere = set(spheres_list[c1]) | set(spheres_list[c2])

            # c3 must cover all words in union_sphere
            # Pick a random word from union_sphere to narrow candidates
            sample_word = rng.choice(list(union_sphere))
            candidate_c3s = covered_by[sample_word]

            for c3 in candidate_c3s:
                if c3 == c1 or c3 == c2:
                    continue
                # Check if c3 covers the entire union
                sphere_c3 = spheres_list[c3]
                # Fast check: for each word in union, is it in sphere_c3?
                # Use bitmask: sphere_mask[c3]
                ok = True
                for w in union_sphere:
                    if not (spheres_mask[c3] >> w) & 1:
                        ok = False
                        break
                if ok:
                    return (c1, c2, c3)
        return None

    # --- 4. Main Search Loop ---

    rng = random.Random(42)

    current_code, current_counts = reduce_code(initial_indices)
    best_code = current_code[:]
    best_counts = current_counts.copy()
    best_len = len(best_code)

    no_improve_count = 0
    max_no_improve = 40
    steps = 0
    max_steps = 500000

    while time.time() < deadline and steps < max_steps:
        steps += 1

        # 1. Reduce
        current_code, current_counts = reduce_code(current_code, current_counts)

        if len(current_code) < best_len:
            best_len = len(current_code)
            best_code = current_code[:]
            best_counts = current_counts.copy()
            no_improve_count = 0
            if best_len <= 71:
                break
        else:
            no_improve_count += 1

        if no_improve_count > max_no_improve:
            # Perturb: try random valid swaps
            if current_code:
                current_set = set(current_code)
                found_valid = False
                for _ in range(80):
                    if time.time() > deadline - 3:
                        break
                    c_remove = rng.choice(current_code)
                    if c_remove not in current_set:
                        continue
                    valid_adds = find_valid_swaps(c_remove, current_set, current_counts, max_candidates=30)
                    if valid_adds:
                        c_add = rng.choice(valid_adds)
                        current_set.remove(c_remove)
                        current_set.add(c_add)
                        current_counts[spheres_list[c_remove]] -= 1
                        current_counts[spheres_list[c_add]] += 1
                        current_code = list(current_set)
                        found_valid = True
                        no_improve_count = 0
                        break

                if not found_valid:
                    # Reset to best
                    current_code = best_code[:]
                    current_counts = best_counts.copy()
                    no_improve_count = 0
            continue

        # 2. Try beneficial moves
        improved = False
        current_set = set(current_code)

        # 2a. Try 2-for-1 merge
        if len(current_code) > 2:
            merge_result = try_2for1_merge(current_set, current_counts, rng)
            if merge_result is not None:
                c1, c2, c3 = merge_result
                # Perform merge: remove c1, c2, add c3
                current_set.discard(c1)
                current_set.discard(c2)
                current_set.add(c3)
                current_counts[spheres_list[c1]] -= 1
                current_counts[spheres_list[c2]] -= 1
                current_counts[spheres_list[c3]] += 1
                current_code = list(current_set)
                improved = True

        # 2b. Try single swaps (CSP-guided)
        if not improved:
            candidates = current_code[:]
            rng.shuffle(candidates)

            for c_remove in candidates:
                if c_remove not in current_set:
                    continue

                valid_adds = find_valid_swaps(c_remove, current_set, current_counts, max_candidates=60)
                if not valid_adds:
                    continue

                for c_add in valid_adds:
                    if c_add in current_set:
                        continue

                    # Perform swap
                    current_set.remove(c_remove)
                    current_set.add(c_add)
                    current_counts[spheres_list[c_remove]] -= 1
                    current_counts[spheres_list[c_add]] += 1

                    # Try to reduce
                    new_code, new_counts = reduce_code(list(current_set), current_counts)

                    if len(new_code) < len(current_code):
                        current_code = new_code
                        current_counts = new_counts
                        current_set = set(current_code)
                        improved = True
                        break
                    else:
                        # Revert
                        current_set.remove(c_add)
                        current_set.add(c_remove)
                        current_counts[spheres_list[c_add]] -= 1
                        current_counts[spheres_list[c_remove]] += 1

                if improved:
                    break
                if time.time() > deadline - 5.0:
                    break

        if improved:
            no_improve_count = 0

    # Final verification
    final_counts = np.zeros(total_words, dtype=np.int32)
    for idx in best_code:
        final_counts[spheres_list[idx]] += 1
    if np.any(final_counts == 0):
        # Fallback: return tetracode
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        result = []
        for h in tetracode:
            for tail in itertools.product(range(q), repeat=n - 4):
                result.append(h + list(tail))
        return result

    return [list(idx_to_word(idx)) for idx in best_code]
