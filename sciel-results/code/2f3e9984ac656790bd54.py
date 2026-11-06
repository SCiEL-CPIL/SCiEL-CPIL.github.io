import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill

    points = list(range(16))
    N = 16

    # Precompute all possible triples and their masks
    all_triples = list(itertools.combinations(points, 3))
    all_triple_masks = [
        (1 << a) | (1 << b) | (1 << c) for (a, b, c) in all_triples
    ]
    triple_count = len(all_triples)

    # Fast popcount for 16-bit masks
    def popcount(x):
        c = 0
        while x:
            x &= x - 1
            c += 1
        return c

    def is_valid_addition(new_mask, existing_masks):
        for m in existing_masks:
            inter = new_mask & m
            if inter and (inter & (inter - 1)):  # more than one shared point
                return False
        return True

    # --- Exact independence number via minimum vertex cover (branch & bound) ---
    def compute_alpha(triples):
        if not triples:
            return 16

        triples_masks = [(1 << a) | (1 << b) | (1 << c) for a, b, c in triples]
        tcount = len(triples_masks)

        # Precompute vertex bitmasks for each triple (3 bits set)
        triple_verts = []
        for tm in triples_masks:
            verts = []
            for i in range(16):
                if tm & (1 << i):
                    verts.append(i)
            triple_verts.append(verts)

        # Lower bound: max number of pairwise-disjoint uncovered triples
        def disjoint_lb(uncovered_mask):
            # uncovered_mask is a bitmask over triples (bit t set = triple t uncovered)
            # Greedily pick disjoint triples
            count = 0
            used_pts = 0
            # try a few orderings for a better bound
            order = list(range(tcount))
            random.shuffle(order)
            for t in order:
                if not (uncovered_mask >> t) & 1:
                    continue
                if triples_masks[t] & used_pts:
                    continue
                used_pts |= triples_masks[t]
                count += 1
            return count

        best_cover = [16]

        # Branch and bound: pick a vertex to add to cover
        def bnb(cover_mask, depth):
            if depth >= best_cover[0]:
                return

            # Find first uncovered triple
            uncovered = -1
            for t in range(tcount):
                if (triples_masks[t] & cover_mask) == 0:
                    uncovered = t
                    break
            if uncovered == -1:
                best_cover[0] = depth
                return

            # Lower bound pruning
            # Compute which triples are still uncovered
            unc_mask = 0
            for t in range(tcount):
                if (triples_masks[t] & cover_mask) == 0:
                    unc_mask |= (1 << t)
            if depth + disjoint_lb(unc_mask) >= best_cover[0]:
                return

            # Branch on the 3 vertices of the uncovered triple
            for v in triple_verts[uncovered]:
                bnb(cover_mask | (1 << v), depth + 1)

        bnb(0, 0)
        return 16 - best_cover[0]

    # --- 2-coloring balance score (auxiliary heuristic) ---
    def color_balance(triples):
        # Returns (max_class_size - min_cover_needed, total_monochromatic_triples)
        # We just use total monochromatic triples as a proxy for "unbalanced"
        # Lower is better (more balanced).
        # Randomly assign colors and count monochromatic triples
        # This is only used as a soft signal in local search, not the main objective.
        return 0

    best_alpha = 17
    best_triples = []

    # --- Phase 1: Multiple restarts of greedy + local search ---
    num_restarts = 40
    restarts_done = 0

    while time.time() < deadline and restarts_done < num_restarts:
        restarts_done += 1

        # --- Greedy initialization ---
        triples = []
        masks = []

        # Random order of all triples
        indices = list(range(triple_count))
        random.shuffle(indices)

        for idx in indices:
            m = all_triple_masks[idx]
            if is_valid_addition(m, masks):
                triples.append(all_triples[idx])
                masks.append(m)

        current_alpha = compute_alpha(triples)
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_triples = triples[:]

        # --- Local search: swap-based simulated annealing ---
        local_deadline = time.time() + (95.0 / num_restarts)
        temperature = 0.5
        cooling_rate = 0.97

        current_triples = triples[:]
        current_masks = masks[:]
        current_alpha = compute_alpha(current_triples)

        while time.time() < local_deadline:
            if not current_triples:
                break

            # Mutation: remove one triple, try to add a different one
            idx_remove = random.randint(0, len(current_triples) - 1)
            removed_t = current_triples[idx_remove]
            removed_m = current_masks[idx_remove]

            current_triples.pop(idx_remove)
            current_masks.pop(idx_remove)

            added = False
            # Try several random candidates to re-add
            for _ in range(50):
                new_idx = random.randint(0, triple_count - 1)
                new_t = all_triples[new_idx]
                new_m = all_triple_masks[new_idx]
                if new_t == removed_t:
                    continue
                if is_valid_addition(new_m, current_masks):
                    current_triples.append(new_t)
                    current_masks.append(new_m)
                    added = True
                    break

            if not added:
                # Revert
                current_triples.insert(idx_remove, removed_t)
                current_masks.insert(idx_remove, removed_m)
                continue

            new_alpha = compute_alpha(current_triples)

            if new_alpha < current_alpha:
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = current_triples[:]
            elif new_alpha == current_alpha:
                # Keep (neutral move)
                pass
            else:
                # Worse: accept with SA probability
                delta = new_alpha - current_alpha
                if random.random() < pow(2.718281828, -delta / max(temperature, 0.01)):
                    current_alpha = new_alpha
                else:
                    # Revert
                    current_triples.pop()
                    current_masks.pop()
                    current_triples.insert(idx_remove, removed_t)
                    current_masks.insert(idx_remove, removed_m)

            temperature *= cooling_rate

    # --- Final polish: hill-climbing on the best solution ---
    if best_triples and time.time() < deadline - 5:
        final_deadline = time.time() + 5
        current_triples = best_triples[:]
        current_masks = [(1 << t[0]) | (1 << t[1]) | (1 << t[2]) for t in current_triples]
        current_alpha = best_alpha

        while time.time() < final_deadline:
            if not current_triples:
                break
            idx_remove = random.randint(0, len(current_triples) - 1)
            removed_t = current_triples[idx_remove]
            removed_m = current_masks[idx_remove]

            new_idx = random.randint(0, triple_count - 1)
            new_t = all_triples[new_idx]
            new_m = all_triple_masks[new_idx]
            if new_t == removed_t:
                continue

            current_triples.pop(idx_remove)
            current_masks.pop(idx_remove)

            if is_valid_addition(new_m, current_masks):
                current_triples.append(new_t)
                current_masks.append(new_m)
                new_alpha = compute_alpha(current_triples)

                if new_alpha <= current_alpha:
                    current_alpha = new_alpha
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_triples = current_triples[:]
                else:
                    current_triples.pop()
                    current_masks.pop()
                    current_triples.insert(idx_remove, removed_t)
                    current_masks.insert(idx_remove, removed_m)
            else:
                current_triples.insert(idx_remove, removed_t)
                current_masks.insert(idx_remove, removed_m)

    return best_triples


if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    print(f"Time: {time.time() - t0:.2f}s, Triples: {len(res)}")
    # Verify validity
    valid = True
    for i in range(len(res)):
        for j in range(i + 1, len(res)):
            inter = set(res[i]) & set(res[j])
            if len(inter) > 1:
                print(f"Invalid pair: {res[i]} and {res[j]}")
                valid = False
    print(f"Valid: {valid}")
    print(f"Alpha: {compute_alpha(res)}")
