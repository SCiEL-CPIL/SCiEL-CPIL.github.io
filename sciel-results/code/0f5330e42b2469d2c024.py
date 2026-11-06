import random
import time
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin for safety

    N = 16
    points = list(range(N))

    # Precompute all possible triples and their masks
    all_triples = []
    all_triple_masks = []
    for combo in itertools.combinations(points, 3):
        a, b, c = combo
        mask = (1 << a) | (1 << b) | (1 << c)
        all_triples.append(combo)
        all_triple_masks.append(mask)

    # Fast popcount for Python ints using a precomputed table for 16-bit integers
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
        """Compute the independence number exactly using branch and bound on vertex cover."""
        if not triples:
            return N

        triples_masks = [(1 << a) | (1 << b) | (1 << c) for a, b, c in triples]

        # Minimum Vertex Cover via Branch and Bound
        best_cover_size = N

        def min_vertex_cover(triples_masks, current_cover_mask, depth):
            nonlocal best_cover_size

            # Pruning
            if depth >= best_cover_size:
                return

            # Check if all covered
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
            v1 = (target_triple & -target_triple).bit_length() - 1
            rem1 = target_triple ^ (1 << v1)
            v2 = (rem1 & -rem1).bit_length() - 1
            rem2 = rem1 ^ (1 << v2)
            v3 = rem2.bit_length() - 1

            # Order branches: pick vertex appearing in most other uncovered triples first
            # For efficiency, just branch on all 3
            min_vertex_cover(triples_masks, current_cover_mask | (1 << v1), depth + 1)
            min_vertex_cover(triples_masks, current_cover_mask | (1 << v2), depth + 1)
            min_vertex_cover(triples_masks, current_cover_mask | (1 << v3), depth + 1)

        min_vertex_cover(triples_masks, 0, 0)
        return N - best_cover_size

    def compute_alpha_upper_bound(triples):
        """Compute a fast upper bound on independence number using greedy vertex cover."""
        if not triples:
            return N

        triples_masks = [(1 << a) | (1 << b) | (1 << c) for a, b, c in triples]
        uncovered = set(range(len(triples_masks)))
        cover_size = 0
        remaining = set(range(N))

        while uncovered and remaining:
            # Pick the vertex that covers the most uncovered triples
            best_v = -1
            best_count = -1
            for v in remaining:
                v_mask = 1 << v
                count = sum(1 for i in uncovered if triples_masks[i] & v_mask)
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v == -1 or best_count == 0:
                break
            cover_size += 1
            remaining.discard(best_v)
            v_mask = 1 << best_v
            uncovered = {i for i in uncovered if not (triples_masks[i] & v_mask)}

        # Upper bound: N - cover_size is a lower bound on alpha, not upper
        # For upper bound, use: size of largest independent set <= N - min vertex cover
        # But we want a quick check. Let's just return N - cover_size as a lower bound estimate
        # Actually for our purpose we want to know if alpha is small, so let's use exact when needed
        return N - cover_size  # This is a lower bound, not upper

    def count_monochromatic_triples(triples, coloring):
        """Count triples that are monochromatic (all same color)."""
        count = 0
        for a, b, c in triples:
            if coloring[a] == coloring[b] == coloring[c]:
                count += 1
        return count

    def get_independence_from_coloring(triples, coloring):
        """Get upper bound on independence number from a 2-coloring.
        
        If we color points 0/1, any independent set can have at most
        (size of color 0 class) + (size of color 1 class) points, but must avoid
        monochromatic triples. A simple upper bound: for each color class,
        the max independent set within that class is bounded by the number of
        points minus the minimum vertex cover of the induced subgraph.
        
        A simpler approach: the independence number is at most the size of the
        larger color class, since any independent set is contained in one color
        class only if we require no monochromatic triples... that's not right.
        
        Actually, an independent set can contain points from both colors. The
        constraint is just that no triple from the system is fully contained.
        
        Let me think differently: if we have a 2-coloring, the number of
        monochromatic triples gives us information. If there are m monochromatic
        triples, then any independent set must "avoid" at least one point from
        each monochromatic triple. This is a covering argument.
        
        For a practical upper bound: alpha <= max(|color0|, |color1|) is NOT valid.
        
        Let me just use the exact computation but with a faster initial bound.
        """
        # This function is a placeholder for the heuristic; we'll use exact computation
        return None  # Force exact computation

    best_alpha = N + 1
    best_triples = []

    num_restarts = 200
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
        local_start = time.time()
        local_time_limit = min(1.0, deadline - time.time())
        if local_time_limit <= 0:
            break

        # Simulated annealing parameters
        temperature = 0.5
        cooling_rate = 0.999

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
            num_samples = 50
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

                    delta = new_alpha - current_alpha
                    if delta < 0:
                        found_replacement = True
                        break
                    elif delta == 0:
                        if random.random() < 0.3:
                            found_replacement = True
                            break
                        else:
                            current_triples.pop()
                            current_masks.pop()
                    else:
                        # Accept with probability exp(-delta/temperature)
                        if random.random() < (2.718281828 ** (-delta / max(temperature, 0.01))):
                            found_replacement = True
                            break
                        else:
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

            temperature *= cooling_rate
            if temperature < 0.01:
                temperature = 0.01

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
