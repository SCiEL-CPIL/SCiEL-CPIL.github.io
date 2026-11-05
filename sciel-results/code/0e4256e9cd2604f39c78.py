import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 148  # 2s safety margin under 150s kill

    # ---------- Precomputation ----------
    # All t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    n_sub = len(t_subsets)
    sub_to_idx = {s: i for i, s in enumerate(t_subsets)}

    # All k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}

    # For each block, the list of t-subset indices it covers
    # block_covers[b_idx] = list of t-subset indices
    block_covers = []
    for b in all_blocks:
        covers = [sub_to_idx[sub] for sub in itertools.combinations(b, t)]
        block_covers.append(covers)

    # For each t-subset, the list of block indices that cover it
    sub_covers = [[] for _ in range(n_sub)]
    for b_idx, covers in enumerate(block_covers):
        for s_idx in covers:
            sub_covers[s_idx].append(b_idx)

    # ---------- Greedy construction (randomized) ----------
    def greedy_construct(rng, aggressive=False):
        """Build a covering using randomized greedy set cover."""
        # uncovered bitmask (use int for speed)
        uncovered = (1 << n_sub) - 1
        counts = [0] * n_sub
        selected = []  # list of block indices

        while uncovered:
            if time.time() > deadline:
                break
            # Pick a random uncovered subset
            pop = uncovered.bit_count()
            if pop == 0:
                break
            rank = rng.randrange(pop)
            temp = uncovered
            idx = -1
            for i in range(n_sub):
                if temp & (1 << i):
                    rank -= 1
                    if rank < 0:
                        idx = i
                        break
            if idx == -1:
                idx = (uncovered & -uncovered).bit_length() - 1

            # Find best block covering this subset
            candidates = sub_covers[idx]
            if not candidates:
                break

            best_block = -1
            best_gain = -1
            if aggressive:
                # Evaluate all candidates
                for b_idx in candidates:
                    gain = 0
                    for s in block_covers[b_idx]:
                        if uncovered & (1 << s):
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_block = b_idx
                # Add some randomness: with prob 0.3 pick random from top-3
                if best_gain > 0 and rng.random() < 0.3:
                    # Get top candidates by gain
                    scored = []
                    for b_idx in candidates:
                        gain = 0
                        for s in block_covers[b_idx]:
                            if uncovered & (1 << s):
                                gain += 1
                        scored.append((gain, b_idx))
                    scored.sort(key=lambda x: -x[0])
                    top_n = min(3, len(scored))
                    if top_n > 0:
                        best_block = scored[rng.randrange(top_n)][1]
            else:
                # Pick random candidate
                best_block = rng.choice(candidates)

            if best_block == -1:
                break

            selected.append(best_block)
            # Update uncovered and counts
            for s in block_covers[best_block]:
                uncovered &= ~(1 << s)
                counts[s] += 1

        return selected, counts

    # ---------- Deletion phase ----------
    def delete_redundant(blocks, counts):
        """Remove all redundant blocks. Returns new blocks list and counts."""
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            to_remove = []
            for b_idx in blocks:
                redundant = True
                for s in block_covers[b_idx]:
                    if counts[s] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(b_idx)
            for b_idx in to_remove:
                blocks.remove(b_idx)
                for s in block_covers[b_idx]:
                    counts[s] -= 1
            if to_remove:
                changed = True
        return blocks, counts

    # ---------- Local search: add-and-swap ----------
    def local_search(blocks, counts, rng, max_iter=500):
        """Try to improve by adding one block and deleting up to two redundant ones."""
        best_len = len(blocks)
        best_blocks = list(blocks)
        best_counts = list(counts)
        no_improve = 0
        max_no_improve = 30

        for it in range(max_iter):
            if time.time() > deadline:
                break
            if no_improve > max_no_improve:
                break

            # Try adding a random block not already in the set
            if not blocks:
                break
            # Sample a few candidate blocks to add
            tried = 0
            max_tries = min(20, n_blocks)
            while tried < max_tries:
                tried += 1
                cand = rng.randrange(n_blocks)
                if cand in blocks:
                    continue

                # Simulate adding cand
                new_counts = list(counts)
                for s in block_covers[cand]:
                    new_counts[s] += 1

                # Find blocks that become redundant
                redundant = []
                for b_idx in blocks:
                    if b_idx == cand:
                        continue
                    is_red = True
                    for s in block_covers[b_idx]:
                        if new_counts[s] == 1:
                            is_red = False
                            break
                    if is_red:
                        redundant.append(b_idx)

                # Try removing 1 or 2 redundant blocks
                best_del = None
                best_del_len = len(blocks) + 1  # after adding
                if len(redundant) >= 1:
                    best_del = [redundant[0]]
                    best_del_len = len(blocks)  # +1 -1
                if len(redundant) >= 2:
                    # Try all pairs? Too expensive. Try random pairs.
                    for _ in range(min(5, len(redundant) * (len(redundant) - 1) // 2)):
                        pair = rng.sample(redundant, 2)
                        # Check if removing both is valid (each has a unique subset)
                        valid = True
                        for s in block_covers[pair[0]]:
                            if new_counts[s] == 1 and s not in block_covers[pair[1]]:
                                valid = False
                                break
                        if not valid:
                            for s in block_covers[pair[1]]:
                                if new_counts[s] == 1 and s not in block_covers[pair[0]]:
                                    valid = False
                                    break
                        if valid:
                            if len(blocks) + 1 - 2 < best_del_len:
                                best_del = pair
                                best_del_len = len(blocks) - 1

                if best_del is not None and best_del_len < best_len:
                    # Apply the move
                    new_blocks = list(blocks)
                    new_blocks.append(cand)
                    for b_idx in best_del:
                        new_blocks.remove(b_idx)
                    # Update counts
                    for s in block_covers[cand]:
                        new_counts[s] += 1  # already done above, but re-verify
                    for b_idx in best_del:
                        for s in block_covers[b_idx]:
                            new_counts[s] -= 1

                    blocks = new_blocks
                    counts = new_counts
                    best_len = len(blocks)
                    best_blocks = list(blocks)
                    best_counts = list(counts)
                    no_improve = 0
                    break  # restart inner loop
                else:
                    no_improve += 1

        return best_blocks, best_counts, best_len

    # ---------- Perturbation for ILS ----------
    def perturb(blocks, counts, rng, strength=3):
        """Remove 'strength' random blocks, then re-cover the exposed subsets."""
        new_blocks = list(blocks)
        new_counts = list(counts)

        # Remove random blocks
        to_remove = rng.sample(new_blocks, min(strength, len(new_blocks)))
        for b_idx in to_remove:
            new_blocks.remove(b_idx)
            for s in block_covers[b_idx]:
                new_counts[s] -= 1

        # Find uncovered subsets and add blocks to cover them
        uncovered = []
        for s in range(n_sub):
            if new_counts[s] == 0:
                uncovered.append(s)

        while uncovered:
            if time.time() > deadline:
                break
            idx = uncovered[0]
            candidates = sub_covers[idx]
            if not candidates:
                break
            # Pick best candidate
            best_block = -1
            best_gain = -1
            for b_idx in candidates:
                if b_idx in new_blocks:
                    continue
                gain = 0
                for s in block_covers[b_idx]:
                    if new_counts[s] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            if best_block == -1:
                # Add any candidate
                best_block = candidates[0]
            new_blocks.append(best_block)
            for s in block_covers[best_block]:
                new_counts[s] += 1
            # Remove newly covered from uncovered
            uncovered = [s for s in uncovered if new_counts[s] > 0]

        return new_blocks, new_counts

    # ---------- Main search loop ----------
    best_blocks = None
    best_counts = None
    best_size = float('inf')

    rng = random.Random(42)

    # Phase 1: Multiple greedy constructions
    for attempt in range(30):
        if time.time() > deadline - 30:
            break
        aggressive = attempt % 2 == 0
        blocks, counts = greedy_construct(rng, aggressive=aggressive)
        blocks, counts = delete_redundant(blocks, counts)
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = list(blocks)
            best_counts = list(counts)

    if best_blocks is None:
        return []

    # Phase 2: Local search on best solution
    if time.time() < deadline - 20:
        blocks, counts, sz = local_search(best_blocks, best_counts, rng, max_iter=200)
        if sz < best_size:
            best_size = sz
            best_blocks = blocks
            best_counts = counts

    # Phase 3: ILS - perturb and local search
    current_blocks = list(best_blocks)
    current_counts = list(best_counts)
    current_size = best_size

    il_iter = 0
    while time.time() < deadline - 5:
        il_iter += 1
        # Perturb
        strength = 2 + (il_iter % 3)  # 2, 3, 4
        pert_blocks, pert_counts = perturb(current_blocks, current_counts, rng, strength)

        # Local search
        ls_blocks, ls_counts, ls_size = local_search(pert_blocks, pert_counts, rng, max_iter=100)

        if ls_size < current_size:
            current_blocks = ls_blocks
            current_counts = ls_counts
            current_size = ls_size
            if current_size < best_size:
                best_size = current_size
                best_blocks = list(current_blocks)
                best_counts = list(current_counts)
        # With some probability, accept non-improving moves (simulated annealing style)
        elif rng.random() < 0.1 and ls_size <= current_size + 2:
            current_blocks = ls_blocks
            current_counts = ls_counts
            current_size = ls_size

    return [list(all_blocks[b]) for b in best_blocks]
