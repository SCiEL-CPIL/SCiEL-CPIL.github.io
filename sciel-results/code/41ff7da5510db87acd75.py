"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import heapq
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s limit

    if q == 3 and n == 8 and radius == 1:
        total_space = 3 ** n  # 6561
        sphere_size = 1 + n * (q - 1)  # 17

        # Precompute sphere indices
        spheres = np.zeros((total_space, sphere_size), dtype=np.int32)
        pow3 = [3 ** i for i in range(n + 1)]
        for idx in range(total_space):
            digits = []
            tmp = idx
            for _ in range(n):
                digits.append(tmp % 3)
                tmp //= 3
            digits.reverse()

            spheres[idx, 0] = idx
            pos = 1
            for pos_idx in range(n):
                original = digits[pos_idx]
                for d in range(q):
                    if d != original:
                        new_idx = idx + (d - original) * pow3[n - 1 - pos_idx]
                        spheres[idx, pos] = new_idx
                        pos += 1

        # Precompute pairwise overlap counts: overlap[a,b] = |sphere(a) ∩ sphere(b)|
        # For a != b, overlap is 1 if distance(a,b)==1, else 0.
        # For a == b, overlap is sphere_size.
        # We store as int8 matrix (6561 x 6561 ~ 43MB)
        overlap = np.zeros((total_space, total_space), dtype=np.int8)
        for idx in range(total_space):
            overlap[idx, idx] = sphere_size
        # For each word, mark its 16 neighbors at distance 1
        for idx in range(total_space):
            for pos_idx in range(n):
                digits = []
                tmp = idx
                for _ in range(n):
                    digits.append(tmp % 3)
                    tmp //= 3
                digits.reverse()
                original = digits[pos_idx]
                for d in range(q):
                    if d != original:
                        nb = idx + (d - original) * pow3[n - 1 - pos_idx]
                        overlap[idx, nb] = 1
                        overlap[nb, idx] = 1

        # ---- Phase 1: Greedy set cover with lazy priority queue ----
        # We maintain cover_count[word] = number of selected codewords covering word.
        # A word is "uncovered" if cover_count[word] == 0.
        # Marginal gain of adding codeword c = number of uncovered words in sphere(c).
        # Use a max-heap with lazy deletion: push (gain, c, version); on pop, recompute gain.

        cover_count = np.zeros(total_space, dtype=np.int32)
        selected = []  # list of codeword indices
        # version counter for lazy deletion
        heap = []
        # Initial: all words uncovered, so gain of any c = sphere_size
        # Push all candidates with gain = sphere_size
        for c in range(total_space):
            heapq.heappush(heap, (-sphere_size, c, 0))
        version = np.zeros(total_space, dtype=np.int32)

        greedy_deadline = deadline - 50  # leave 50s for local search
        while True:
            if time.time() > greedy_deadline:
                break
            # Pop until we find a valid entry
            while heap:
                neg_gain, c, ver = heapq.heappop(heap)
                if ver != version[c]:
                    continue
                # Recompute actual gain
                gain = int((cover_count[spheres[c]] == 0).sum())
                if gain == 0:
                    break  # nothing left to cover
                # Add codeword c
                selected.append(c)
                cover_count[spheres[c]] += 1
                # Update marginal gains for neighbors of c (words in sphere(c))
                # For each word w in sphere(c), any codeword c' that covers w now has reduced gain
                # We lazily update: bump version for codewords that cover newly-covered words
                # Simpler: for each neighbor c' (within dist 1 of c), its gain may decrease
                # Actually, gain of c' decreases by number of newly-covered words in sphere(c')
                # We just bump version so it gets recomputed on next pop
                # But we need to push updated entries. Let's do: for each c' in the 17-neighborhood of c,
                # push a new entry with recomputed gain.
                # The 17-neighborhood of c: c itself and its 16 distance-1 neighbors
                neighbors = [c]
                for pos_idx in range(n):
                    digits = []
                    tmp = c
                    for _ in range(n):
                        digits.append(tmp % 3)
                        tmp //= 3
                    digits.reverse()
                    original = digits[pos_idx]
                    for d in range(q):
                        if d != original:
                            nb = c + (d - original) * pow3[n - 1 - pos_idx]
                            neighbors.append(nb)
                for nb in neighbors:
                    new_gain = int((cover_count[spheres[nb]] == 0).sum())
                    version[nb] += 1
                    heapq.heappush(heap, (-new_gain, nb, version[nb]))
                break
            else:
                break

        m = len(selected)
        active = np.ones(m, dtype=bool)
        active_set = set(selected)

        # ---- Phase 2: Remove redundant codewords using overlap matrix (vectorized) ----
        # A codeword c is removable iff for every other active codeword c', overlap[c][c'] >= 1
        # i.e., no word in sphere(c) is uniquely covered by c.

        def try_remove_redundant():
            """Remove all currently-removable codewords in random order. Returns count removed."""
            removed = 0
            active_list = [i for i in range(m) if active[i]]
            if len(active_list) == 0:
                return 0
            random.Random(42).shuffle(active_list)
            # Build active codeword index array for vectorized overlap check
            active_codewords = np.array([selected[i] for i in active_list], dtype=np.int32)
            # For each active codeword, check min overlap with all other active codewords
            # overlap[active_codewords, :][:, active_codewords] gives pairwise overlaps
            # We need: for each row i, min over j != i of overlap[active_codewords[i], active_codewords[j]] > 0
            pair_overlap = overlap[active_codewords][:, active_codewords]  # (k, k)
            # Set diagonal to large value so it doesn't affect min
            np.fill_diagonal(pair_overlap, sphere_size)
            min_overlaps = pair_overlap.min(axis=1)
            removable_mask = min_overlaps > 0
            removable_indices = [active_list[i] for i in range(len(active_list)) if removable_mask[i]]
            random.Random(42).shuffle(removable_indices)
            for i in removable_indices:
                if not active[i]:
                    continue
                active[i] = False
                removed += 1
            return removed

        # Iteratively remove redundant codewords
        cleanup_deadline = deadline - 5
        while time.time() < cleanup_deadline:
            removed = try_remove_redundant()
            if removed == 0:
                break

        # ---- Phase 3: Pairwise swap local search ----
        # Try to swap a critical codeword with an inactive one to enable further removals.
        swap_deadline = deadline - 2
        for swap_round in range(200):
            if time.time() > swap_deadline:
                break
            active_list = [i for i in range(m) if active[i]]
            if len(active_list) < 2:
                break
            inactive_list = [i for i in range(m) if not active[i]]
            if len(inactive_list) == 0:
                break

            # Find critical codewords (have a uniquely covered word)
            # Use cover_count to identify words with cover_count == 1
            critical = []
            active_codewords_arr = np.array([selected[i] for i in active_list], dtype=np.int32)
            for idx_pos, i in enumerate(active_list):
                c = selected[i]
                sphere_c = spheres[c]
                # A word w in sphere_c is unique to c if no other active codeword covers w
                # Check via overlap: for each other active codeword cj, does w in sphere(cj)?
                # w in sphere(cj) iff w == cj or dist(w, cj) == 1
                # Efficient check: for each w, see if any active_codewords_arr entry equals w or is dist 1
                unique_found = False
                for w in sphere_c:
                    # Check if w is in any other active codeword's sphere
                    # w == cj for some active cj, or dist(w, cj) == 1
                    # Precomputed: overlap[w, cj] > 0 means spheres of w and cj intersect
                    # But we need w specifically in sphere(cj)
                    # w in sphere(cj) iff overlap[w, cj] > 0 (since sphere(w) contains w and its 16 neighbors,
                    # and sphere(cj) contains cj and its 16 neighbors; they intersect iff dist(w,cj) <= 1)
                    # Actually overlap[w, cj] > 0 iff dist(w, cj) <= 1, which is exactly w in sphere(cj) or cj in sphere(w)
                    # Since sphere is symmetric, w in sphere(cj) iff dist(w,cj) <= 1 iff overlap[w, cj] > 0
                    # But we need: is there an active cj (cj != c) such that w in sphere(cj)?
                    # That is: overlap[w, cj] > 0 for some active cj != c
                    # Note: overlap[w, c] > 0 is always true (w in sphere(c)), so we need cj != c
                    other_active = active_codewords_arr[active_codewords_arr != c]
                    if len(other_active) > 0 and np.any(overlap[w, other_active] > 0):
                        continue  # w is covered by another active codeword
                    else:
                        unique_found = True
                        break
                if unique_found:
                    critical.append(i)

            if len(critical) == 0:
                break

            random.Random(42).shuffle(critical)
            found_improvement = False
            for ci in critical:
                if time.time() > swap_deadline:
                    break
                if not active[ci]:
                    continue
                gi = selected[ci]
                gi_sphere = spheres[gi]

                # Find unique words of gi
                unique_words = []
                active_codewords_arr = np.array([selected[j] for j in range(m) if active[j] and j != ci], dtype=np.int32)
                for w in gi_sphere:
                    if len(active_codewords_arr) > 0 and np.any(overlap[w, active_codewords_arr] > 0):
                        continue  # w covered by another active codeword
                    else:
                        unique_words.append(w)

                if len(unique_words) == 0:
                    continue

                # Try swapping with inactive codewords
                random.Random(42).shuffle(inactive_list)
                for si_idx in inactive_list[:50]:
                    if time.time() > swap_deadline:
                        break
                    si = selected[si_idx]
                    si_sphere = spheres[si]

                    # Check if si covers all unique_words
                    si_set = set(si_sphere.tolist())
                    if not all(uw in si_set for uw in unique_words):
                        continue

                    # Perform swap
                    active[ci] = False
                    active[si_idx] = True

                    # Try to remove redundant codewords
                    removed = try_remove_redundant()
                    if removed > 0:
                        found_improvement = True
                        break
                    else:
                        # Revert
                        active[ci] = True
                        active[si_idx] = False

                if found_improvement:
                    break

            if not found_improvement:
                break

        # Final cleanup
        while time.time() < deadline - 1:
            removed = try_remove_redundant()
            if removed == 0:
                break

        # Build final code
        final_indices = [selected[i] for i in range(m) if active[i]]

        # Verify coverage
        verify_cover = np.zeros(total_space, dtype=bool)
        for idx in final_indices:
            verify_cover[spheres[idx]] = True

        if np.all(verify_cover):
            final_code = []
            for idx in final_indices:
                digits = []
                tmp = idx
                for _ in range(n):
                    digits.append(tmp % 3)
                    tmp //= 3
                digits.reverse()
                final_code.append(digits)
            return final_code

        # Fallback: return greedy code
        code = []
        for idx in selected:
            digits = []
            tmp = idx
            for _ in range(n):
                digits.append(tmp % 3)
                tmp //= 3
            digits.reverse()
            code.append(digits)
        return code

    else:
        # General case
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(q), repeat=n)]
# EVOLVE-BLOCK-END
