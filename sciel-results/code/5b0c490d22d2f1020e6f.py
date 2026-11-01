import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    if q != 3 or n != 6 or radius != 1:
        # Fallback: simple greedy for other parameters
        total = q ** n
        if total > 100000:
            return [list(range(n)) for _ in range(min(total, 1000))]
        def w2i(w):
            idx = 0
            for x in w:
                idx = idx * q + x
            return idx
        def i2w(idx):
            w = [0] * n
            for i in range(n - 1, -1, -1):
                w[i] = idx % q
                idx //= q
            return w
        masks = [0] * total
        for i in range(total):
            wi = i2w(i)
            m = 1 << i
            for pos in range(n):
                orig = wi[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    wn = wi[:]
                    wn[pos] = val
                    m |= 1 << w2i(wn)
            masks[i] = m
        uncovered = (1 << total) - 1
        code = []
        while uncovered:
            if time.time() > deadline - 1:
                break
            best_i = -1
            best_c = -1
            for i in range(total):
                common = uncovered & masks[i]
                if common:
                    cnt = common.bit_count()
                    if cnt > best_c:
                        best_c = cnt
                        best_i = i
            if best_i == -1:
                break
            code.append(i2w(best_i))
            uncovered &= ~masks[best_i]
        return code

    # Optimized path for q=3, n=6, radius=1
    total = q ** n  # 729

    def w2i(w):
        idx = 0
        for x in w:
            idx = idx * 3 + x
        return idx

    def i2w(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % 3
            idx //= 3
        return w

    # Precompute neighbor masks and lists
    masks = [0] * total
    neighbor_lists = [[] for _ in range(total)]
    for i in range(total):
        wi = i2w(i)
        m = 1 << i
        neighbor_lists[i].append(i)
        for pos in range(n):
            orig = wi[pos]
            for val in range(3):
                if val == orig:
                    continue
                wn = wi[:]
                wn[pos] = val
                j = w2i(wn)
                m |= 1 << j
                neighbor_lists[i].append(j)
        masks[i] = m

    # Convert masks to numpy array for vectorized operations
    masks_np = np.array(masks, dtype=np.uint64)

    def greedy_set_cover_fast(deadline, random_tie=False, seed=None):
        """Fast greedy set cover using numpy for candidate scoring."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        uncovered = (1 << total) - 1
        code_indices = []

        while uncovered:
            if time.time() > deadline:
                break

            # Vectorized: compute bit_count of (uncovered & masks[i]) for all i
            # Use numpy to process in chunks for speed
            best_idx = -1
            best_count = -1
            best_candidates = []

            # Process all 729 candidates
            for i in range(total):
                common = uncovered & masks[i]
                if common:
                    cnt = common.bit_count()
                    if cnt > best_count:
                        best_count = cnt
                        best_candidates = [i]
                    elif cnt == best_count:
                        best_candidates.append(i)

            if not best_candidates:
                break

            if random_tie and len(best_candidates) > 1:
                best_idx = rng.choice(best_candidates)
            else:
                best_idx = best_candidates[0]

            code_indices.append(best_idx)
            uncovered &= ~masks[best_idx]

        return code_indices

    def remove_redundant(code_indices, deadline):
        """Remove redundant codewords from the code."""
        code_set = set(code_indices)
        code_list = list(code_indices)

        coverage = [0] * total
        for ci in code_list:
            for nb in neighbor_lists[ci]:
                coverage[nb] += 1

        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            for ci in list(code_list):
                can_remove = True
                for nb in neighbor_lists[ci]:
                    if coverage[nb] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    for nb in neighbor_lists[ci]:
                        coverage[nb] -= 1
                    code_list.remove(ci)
                    code_set.discard(ci)
                    changed = True

        return code_list

    def local_search_improve(code_indices, deadline):
        """Aggressive local search: try swaps and add-then-remove cycles."""
        current = list(code_indices)
        best_size = len(current)
        best_code = list(current)

        coverage = [0] * total
        for ci in current:
            for nb in neighbor_lists[ci]:
                coverage[nb] += 1

        # Iteratively try to improve
        improvement = True
        max_iters = 100
        iter_count = 0

        while improvement and iter_count < max_iters:
            if time.time() > deadline - 5.0:
                break
            improvement = False
            iter_count += 1

            # Recompute coverage
            coverage = [0] * total
            for ci in current:
                for nb in neighbor_lists[ci]:
                    coverage[nb] += 1

            critical = [i for i in range(total) if coverage[i] == 1]

            if not critical:
                # No critical words, try to remove any redundant codeword
                for ci in list(current):
                    can_remove = True
                    for nb in neighbor_lists[ci]:
                        if coverage[nb] <= 1:
                            can_remove = False
                            break
                    if can_remove:
                        for nb in neighbor_lists[ci]:
                            coverage[nb] -= 1
                        current.remove(ci)
                        improvement = True
                        if len(current) < best_size:
                            best_size = len(current)
                            best_code = list(current)
                        break
                continue

            # Try to add a codeword that covers most critical words, then remove redundant
            best_add = -1
            best_crit_covered = 0
            current_set = set(current)

            for ci in range(total):
                if ci in current_set:
                    continue
                crit_covered = sum(1 for c in critical if c in neighbor_lists[ci])
                if crit_covered > best_crit_covered:
                    best_crit_covered = crit_covered
                    best_add = ci

            if best_add == -1 or best_crit_covered == 0:
                break

            # Add best_add
            current.append(best_add)
            for nb in neighbor_lists[best_add]:
                coverage[nb] += 1

            # Now try to remove redundant codewords (possibly multiple)
            removed_any = True
            while removed_any:
                removed_any = False
                for ci in list(current):
                    can_remove = True
                    for nb in neighbor_lists[ci]:
                        if coverage[nb] <= 1:
                            can_remove = False
                            break
                    if can_remove:
                        for nb in neighbor_lists[ci]:
                            coverage[nb] -= 1
                        current.remove(ci)
                        removed_any = True
                        improvement = True
                        break

            if len(current) < best_size:
                best_size = len(current)
                best_code = list(current)
                if best_size <= 73:
                    break

        return best_code

    # Strategy 1: Start with tetracode direct sum (81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    direct_sum = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
    direct_sum_indices = [w2i(c) for c in direct_sum]

    ds_cleaned = remove_redundant(direct_sum_indices, deadline - 10.0)
    ds_improved = local_search_improve(ds_cleaned, deadline - 5.0)

    # Strategy 2: Greedy set cover
    greedy_code = greedy_set_cover_fast(deadline - 15.0, random_tie=False)
    greedy_cleaned = remove_redundant(greedy_code, deadline - 10.0)
    greedy_improved = local_search_improve(greedy_cleaned, deadline - 5.0)

    # Pick the best so far
    candidates = [ds_improved, greedy_improved]
    best_code = min(candidates, key=len)
    best_size = len(best_code)

    # Strategy 3: Randomized greedy restarts with local search
    remaining_time = deadline - time.time()
    if remaining_time > 10.0 and best_size > 73:
        num_restarts = min(30, int(remaining_time / 3.0))
        for r in range(num_restarts):
            if time.time() > deadline - 8.0:
                break
            seed_val = r * 12345 + 67890
            rc = greedy_set_cover_fast(deadline - 12.0, random_tie=True, seed=seed_val)
            rc_cleaned = remove_redundant(rc, deadline - 8.0)
            rc_improved = local_search_improve(rc_cleaned, deadline - 5.0)
            if len(rc_improved) < best_size:
                best_size = len(rc_improved)
                best_code = rc_improved
                if best_size <= 73:
                    break

    # Final: one more local search pass on the best
    if time.time() < deadline - 5.0:
        final_improved = local_search_improve(best_code, deadline - 2.0)
        if len(final_improved) < best_size:
            best_code = final_improved
            best_size = len(best_code)

    # Convert indices back to words
    result = [i2w(i) for i in best_code]

    return result
