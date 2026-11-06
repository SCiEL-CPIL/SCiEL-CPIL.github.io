import random
import time
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin for safety

    N = 16
    points = list(range(N))

    # Precompute all possible triples
    all_triples = []
    all_triple_masks = []
    for combo in itertools.combinations(points, 3):
        a, b, c = combo
        mask = (1 << a) | (1 << b) | (1 << c)
        all_triples.append(combo)
        all_triple_masks.append(mask)

    # Fast popcount for Python ints (using bin().count is slow, but for 16-bit ints it's okay.
    # Better: use a lookup table for 8-bit or just rely on Python's efficiency for small ints?
    # Actually, bin(x).count('1') is surprisingly slow in a tight loop.
    # Let's use a precomputed table for 16-bit integers.
    popcount_table = [0] * 65536
    for i in range(1, 65536):
        popcount_table[i] = popcount_table[i >> 1] + (i & 1)

    def popcount(x):
        return popcount_table[x & 0xFFFF]

    def is_valid_addition(new_mask, existing_masks):
        for m in existing_masks:
            if popcount(new_mask & m) > 1:
                return False
        return True

    def compute_alpha_exact(triples):
        if not triples:
            return N

        triples_masks = [(1 << a) | (1 << b) | (1 << c) for a, b, c in triples]

        # Minimum Vertex Cover via Branch and Bound
        best_cover_size = N

        # Sort triples by "difficulty" or just keep them.
        # To improve pruning, we can order triples.
        # Let's just use the list as is for simplicity, but pick the first uncovered one.

        def min_vertex_cover(triples_masks, current_cover_mask, depth):
            nonlocal best_cover_size

            # Pruning
            if depth >= best_cover_size:
                return

            # Check if all covered
            # Optimization: maintain a count of uncovered triples?
            # For N=16 and ~40 triples, checking all is fast.
            uncovered_idx = -1
            for i, tm in enumerate(triples_masks):
                if (tm & current_cover_mask) == 0:
                    uncovered_idx = i
                    break

            if uncovered_idx == -1:
                # All covered
                if depth < best_cover_size:
                    best_cover_size = depth
                return

            target_triple = triples_masks[uncovered_idx]

            # Branch on the 3 vertices of target_triple
            # Extract vertices
            # v1, v2, v3
            v1 = (target_triple & -target_triple).bit_length() - 1
            rem1 = target_triple ^ (1 << v1)
            v2 = (rem1 & -rem1).bit_length() - 1
            rem2 = rem1 ^ (1 << v2)
            v3 = rem2.bit_length() - 1

            # Order branches?
            # Heuristic: pick the vertex that appears in the most other uncovered triples?
            # This is expensive to compute on the fly.
            # Let's just branch on v1, v2, v3.

            # Branch 1: Add v1
            min_vertex_cover(triples_masks, current_cover_mask | (1 << v1), depth + 1)
            # Branch 2: Add v2
            min_vertex_cover(triples_masks, current_cover_mask | (1 << v2), depth + 1)
            # Branch 3: Add v3
            min_vertex_cover(triples_masks, current_cover_mask | (1 << v3), depth + 1)

        min_vertex_cover(triples_masks, 0, 0)
        return N - best_cover_size

    best_alpha = N + 1
    best_triples = []

    num_restarts = 100  # Increase restarts if time permits
    restarts_done = 0

    while time.time() < deadline and restarts_done < num_restarts:
        restarts_done += 1

        triples = []
        masks = []

        # Randomized Greedy Construction
        indices = list(range(len(all_triples)))
        random.shuffle(indices)

        for idx in indices:
            t = all_triples[idx]
            m = all_triple_masks[idx]
            if is_valid_addition(m, masks):
                triples.append(t)
                masks.append(m)

        # Calculate initial alpha
        current_alpha = compute_alpha_exact(triples)

        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_triples = triples[:]

        # Local Search: Simulated Annealing / Hill Climbing
        current_triples = triples[:]
        current_masks = masks[:]
        current_alpha = compute_alpha_exact(current_triples)

        # Time slice for local search
        # Allocate remaining time proportionally or fixed chunk?
        # Let's use a fixed chunk per restart, ensuring we finish before deadline.
        # Or just loop until deadline.
        
        local_start = time.time()
        local_time_limit = min(1.5, deadline - time.time())
        if local_time_limit <= 0:
            break

        while time.time() - local_start < local_time_limit:
            if not current_triples:
                break

            # Pick a random triple to remove
            idx_remove = random.randint(0, len(current_triples) - 1)
            removed_t = current_triples[idx_remove]
            removed_m = current_masks[idx_remove]

            # Remove it
            current_triples.pop(idx_remove)
            current_masks.pop(idx_remove)

            found_replacement = False
            new_alpha = current_alpha

            # Sample candidates
            # We want to find a triple that reduces alpha or maintains it.
            # Sampling 50-100 candidates per step is reasonable.
            num_samples = 80
            for _ in range(num_samples):
                new_idx = random.randint(0, len(all_triples) - 1)
                new_t = all_triples[new_idx]
                new_m = all_triple_masks[new_idx]

                if new_t == removed_t:
                    continue

                if is_valid_addition(new_m, current_masks):
                    current_triples.append(new_t)
                    current_masks.append(new_m)

                    new_alpha = compute_alpha_exact(current_triples)

                    if new_alpha < current_alpha:
                        found_replacement = True
                        break
                    elif new_alpha == current_alpha:
                        # Accept with probability to allow exploration
                        # Temperature-like parameter?
                        # For now, accept with 0.3 probability to avoid getting stuck
                        if random.random() < 0.3:
                            found_replacement = True
                            break
                        else:
                            # Revert
                            current_triples.pop()
                            current_masks.pop()
                    else:
                        # Worse, revert
                        current_triples.pop()
                        current_masks.pop()

            if found_replacement:
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = current_triples[:]
            else:
                # Revert the removal
                current_triples.insert(idx_remove, removed_t)
                current_masks.insert(idx_remove, removed_m)

    # Final check
    if best_triples:
        # Verify validity one last time
        valid = True
        masks_final = [(1 << a) | (1 << b) | (1 << c) for a, b, c in best_triples]
        for i in range(len(masks_final)):
            for j in range(i + 1, len(masks_final)):
                if popcount(masks_final[i] & masks_final[j]) > 1:
                    valid = False
                    break
            if not valid:
                break

        if not valid:
            return []

    return best_triples

if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    print(f"Time: {time.time() - t0}, Triples: {len(res)}")
    if res:
        # Verify
        masks_final = [(1 << a) | (1 << b) | (1 << c) for a, b, c in res]
        valid = True
        for i in range(len(masks_final)):
            for j in range(i + 1, len(masks_final)):
                if popcount(masks_final[i] & masks_final[j]) > 1:
                    valid = False
                    break
            if not valid:
                break
        print(f"Valid: {valid}")
        print(f"Result Length: {len(res)}")
