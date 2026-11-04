import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    all_tsubsets_set = set(all_tsubsets)

    best_blocks = None

    # Run multiple random trials with varying strategies
    trial = 0
    while time.time() < deadline:
        trial += 1
        # Use different random seeds and strategies per trial
        strategy = trial % 4  # 0: standard greedy, 1: weighted, 2: diversity, 3: hybrid

        if strategy == 0:
            blocks, valid = _greedy_cover(v, k, t, all_tsubsets_set, deadline, trial, mode='standard')
        elif strategy == 1:
            blocks, valid = _greedy_cover(v, k, t, all_tsubsets_set, deadline, trial, mode='weighted')
        elif strategy == 2:
            blocks, valid = _greedy_cover(v, k, t, all_tsubsets_set, deadline, trial, mode='diversity')
        else:
            blocks, valid = _greedy_cover(v, k, t, all_tsubsets_set, deadline, trial, mode='hybrid')

        if valid:
            # Local improvement: remove redundant blocks
            improved_blocks = _remove_redundant(blocks, v, k, t, deadline)
            if len(improved_blocks) < len(blocks):
                blocks = improved_blocks

            # Try swap-based improvement
            swapped_blocks = _swap_improve(blocks, v, k, t, all_tsubsets_set, deadline, trial)
            if len(swapped_blocks) < len(blocks):
                blocks = swapped_blocks
                # Remove redundant again after swaps
                blocks = _remove_redundant(blocks, v, k, t, deadline)

            # Try additional local search: 2-opt style block exchange
            if time.time() < deadline:
                blocks = _two_opt_improve(blocks, v, k, t, all_tsubsets_set, deadline, trial)

            if best_blocks is None or len(blocks) < len(best_blocks):
                best_blocks = blocks

        # If we find a very good solution, stop early
        if best_blocks and len(best_blocks) <= 76:
            break

    if best_blocks is None:
        # Fallback: run a deterministic greedy to ensure we return something valid
        blocks, valid = _greedy_cover(v, k, t, all_tsubsets_set, deadline, -1, mode='standard')
        if valid:
            best_blocks = _remove_redundant(blocks, v, k, t, deadline)
        else:
            best_blocks = []

    return best_blocks

def _greedy_cover(v, k, t, all_tsubsets_set, deadline, seed_val, mode='standard'):
    """
    Greedy covering with different modes for diversity.
    """
    rng = random.Random(seed_val if seed_val >= 0 else time.time())

    uncovered = set(all_tsubsets_set)
    blocks = []

    points = list(range(v))

    while uncovered:
        if time.time() > deadline:
            return None, False

        # Pick a random uncovered t-subset as seed
        # For diversity mode, bias toward less-covered regions
        if mode == 'diversity':
            # Prefer t-subsets that are "harder" to cover (fewer possible blocks)
            candidates = list(uncovered)
            # Weight by inverse of how many k-blocks could contain them
            # A t-subset can be in C(v-t, k-t) blocks
            weight = 1.0 / (len(candidates) ** 0.5)  # mild bias
            seed = tuple(rng.choices(candidates, weights=[1.0 + weight * (1 - i/len(candidates)) for i in range(len(candidates))])[0])
        else:
            seed = tuple(rng.choice(list(uncovered)))

        block = list(seed)
        block_set = set(block)

        while len(block) < k:
            if time.time() > deadline:
                return None, False

            best_x = -1
            best_score = -1
            best_tiebreakers = []

            t_minus_1_subsets = list(itertools.combinations(block, t-1))

            # Precompute for speed
            for x in points:
                if x in block_set:
                    continue

                # Count how many uncovered t-subsets this point covers
                gain = 0
                for p in t_minus_1_subsets:
                    s = tuple(sorted(p + (x,)))
                    if s in uncovered:
                        gain += 1

                if gain == 0:
                    continue

                if mode == 'weighted':
                    # Weight by rarity: subsets that appear in fewer total k-blocks
                    # are more valuable to cover now
                    # Approximate: prefer points that create more "unique" coverage
                    score = gain * 1.0
                elif mode == 'diversity':
                    # Prefer points that spread out coverage
                    score = gain * 1.0 + 0.1 * (v - x)  # mild positional bias for diversity
                elif mode == 'hybrid':
                    # Mix of weighted and standard
                    score = gain * 1.0 + 0.05 * gain * (1 + x / v)
                else:
                    score = gain * 1.0

                if score > best_score:
                    best_score = score
                    best_x = x
                    best_tiebreakers = [x]
                elif score == best_score:
                    best_tiebreakers.append(x)

            if best_x == -1:
                # No point improves coverage; pick random
                remaining = [x for x in points if x not in block_set]
                if not remaining:
                    return None, False
                chosen_x = rng.choice(remaining)
            else:
                if len(best_tiebreakers) > 1:
                    weights = [1.0 + 0.1 * i for i in range(len(best_tiebreakers))]
                    chosen_x = rng.choices(best_tiebreakers, weights=weights)[0]
                else:
                    chosen_x = best_tiebreakers[0]

            block.append(chosen_x)
            block_set.add(chosen_x)

        block.sort()
        blocks.append(tuple(block))

        for s in itertools.combinations(block, t):
            uncovered.discard(s)

    return blocks, True

def _remove_redundant(blocks, v, k, t, deadline):
    """
    Remove redundant blocks. A block is redundant if removing it
    doesn't leave any t-subset uncovered.
    """
    if not blocks:
        return blocks

    # Build a count for each t-subset: how many blocks cover it
    cover_count = defaultdict(int)
    for block in blocks:
        for s in itertools.combinations(block, t):
            cover_count[s] += 1

    improved = list(blocks)

    # Try removing each block if it's redundant
    # We iterate backwards so that removal doesn't affect indices
    for i in range(len(improved) - 1, -1, -1):
        if time.time() > deadline:
            break

        block = improved[i]
        redundant = True
        for s in itertools.combinations(block, t):
            if cover_count[s] <= 1:
                redundant = False
                break

        if redundant:
            for s in itertools.combinations(block, t):
                cover_count[s] -= 1
            improved.pop(i)

    return improved

def _swap_improve(blocks, v, k, t, all_tsubsets_set, deadline, seed_val):
    """
    Try 1-for-1 block replacement: replace a block with another that covers
    the same critical t-subsets, potentially allowing subsequent removals.
    """
    if not blocks:
        return blocks

    rng = random.Random(seed_val if seed_val >= 0 else time.time())
    improved = list(blocks)
    n_blocks = len(improved)

    # Build cover count
    cover_count = defaultdict(int)
    for block in improved:
        for s in itertools.combinations(block, t):
            cover_count[s] += 1

    # Find which t-subsets are covered exactly once (critical)
    critical = set(s for s, c in cover_count.items() if c == 1)

    # Try multiple passes of replacement
    for _pass in range(3):
        if time.time() > deadline:
            break

        any_change = False
        for i in range(n_blocks):
            if time.time() > deadline:
                break

            old_block = improved[i]
            old_tsubs = list(itertools.combinations(old_block, t))
            old_tsub_set = set(old_tsubs)

            # Count how many of old_block's t-subsets are critical
            old_critical = [s for s in old_tsubs if s in critical]

            if not old_critical:
                continue

            # Generate candidate replacement blocks
            old_set = set(old_block)
            candidates = set()

            # Generate random candidates by modifying old_block
            for _ in range(50):
                if time.time() > deadline:
                    break
                new_block = list(old_block)
                # Remove 1-3 random elements
                remove_count = rng.randint(1, min(3, k))
                to_remove = rng.sample(new_block, remove_count)
                for x in to_remove:
                    new_block.remove(x)
                # Add random new elements
                while len(new_block) < k:
                    x = rng.randint(0, v-1)
                    if x not in new_block:
                        new_block.append(x)
                candidates.add(tuple(sorted(new_block)))

            # Also try candidates that share many elements with other blocks
            for j in range(n_blocks):
                if j == i:
                    continue
                other_block = improved[j]
                other_list = list(other_block)
                for _ in range(5):
                    new_block = list(old_block)
                    replace_count = rng.randint(1, 2)
                    to_replace = rng.sample(new_block, replace_count)
                    for x in to_replace:
                        new_block.remove(x)
                        available = [y for y in other_list if y not in new_block]
                        if available:
                            new_block.append(rng.choice(available))
                        else:
                            new_block.append(x)
                    if len(new_block) == k and len(set(new_block)) == k:
                        candidates.add(tuple(sorted(new_block)))

            # Evaluate candidates
            best_new_block = None
            best_gain = -1

            for cand in candidates:
                if cand == old_block:
                    continue

                cand_tsubs = list(itertools.combinations(cand, t))
                cand_tsub_set = set(cand_tsubs)

                # Check if cand covers all critical t-subsets of old_block
                covers_all_critical = all(s in cand_tsub_set for s in old_critical)
                if not covers_all_critical:
                    continue

                # Calculate gain: how many new t-subsets does cand cover?
                new_subs = cand_tsub_set - old_tsub_set
                gain = len(new_subs)

                # Bonus if this creates redundancy (makes other blocks redundant)
                redundancy_bonus = 0
                for s in new_subs:
                    if cover_count.get(s, 0) >= 1:
                        redundancy_bonus += 1

                total_score = gain + redundancy_bonus * 0.5

                if total_score > best_gain:
                    best_gain = total_score
                    best_new_block = cand

            if best_new_block is not None and best_gain > 0:
                # Replace old_block with best_new_block
                for s in old_tsubs:
                    cover_count[s] -= 1
                    if cover_count[s] == 0:
                        del cover_count[s]
                for s in itertools.combinations(best_new_block, t):
                    cover_count[s] += 1

                improved[i] = best_new_block
                any_change = True
                critical = set(s for s, c in cover_count.items() if c == 1)

        if not any_change:
            break

    return improved

def _two_opt_improve(blocks, v, k, t, all_tsubsets_set, deadline, seed_val):
    """
    Try 2-block exchanges: replace two blocks with two different blocks
    that together cover all the same t-subsets but potentially allow
    further reductions.
    """
    if len(blocks) < 2:
        return blocks

    rng = random.Random(seed_val if seed_val >= 0 else time.time())
    improved = list(blocks)
    n_blocks = len(improved)

    # Build cover count
    cover_count = defaultdict(int)
    for block in improved:
        for s in itertools.combinations(block, t):
            cover_count[s] += 1

    # Try a few random 2-exchange attempts
    max_attempts = 20
    for _attempt in range(max_attempts):
        if time.time() > deadline:
            break

        i = rng.randint(0, n_blocks - 1)
        j = rng.randint(0, n_blocks - 1)
        if i == j:
            continue

        block_i = improved[i]
        block_j = improved[j]

        # Generate candidate replacements for block_i
        cand_i_candidates = set()
        for _ in range(20):
            new_block = list(block_i)
            remove_count = rng.randint(1, min(2, k))
            to_remove = rng.sample(new_block, remove_count)
            for x in to_remove:
                new_block.remove(x)
            while len(new_block) < k:
                x = rng.randint(0, v - 1)
                if x not in new_block:
                    new_block.append(x)
            cand_i_candidates.add(tuple(sorted(new_block)))

        # Generate candidate replacements for block_j
        cand_j_candidates = set()
        for _ in range(20):
            new_block = list(block_j)
            remove_count = rng.randint(1, min(2, k))
            to_remove = rng.sample(new_block, remove_count)
            for x in to_remove:
                new_block.remove(x)
            while len(new_block) < k:
                x = rng.randint(0, v - 1)
                if x not in new_block:
                    new_block.append(x)
            cand_j_candidates.add(tuple(sorted(new_block)))

        # Find a pair that together covers all t-subsets of original pair
        old_tsubs_i = set(itertools.combinations(block_i, t))
        old_tsubs_j = set(itertools.combinations(block_j, t))
        old_all_tsubs = old_tsubs_i | old_tsubs_j

        best_pair = None
        best_score = -1

        for ci in cand_i_candidates:
            if ci == block_i:
                continue
            tsubs_ci = set(itertools.combinations(ci, t))

            for cj in cand_j_candidates:
                if cj == block_j:
                    continue
                if ci == cj:
                    continue
                tsubs_cj = set(itertools.combinations(cj, t))

                new_all_tsubs = tsubs_ci | tsubs_cj

                # Check if new pair covers all old t-subsets
                if not old_all_tsubs.issubset(new_all_tsubs):
                    continue

                # Score: how many new t-subsets does the pair cover?
                gain = len(new_all_tsubs - old_all_tsubs)

                # Bonus for creating redundancy
                redundancy_bonus = 0
                for s in new_all_tsubs - old_all_tsubs:
                    if cover_count.get(s, 0) >= 1:
                        redundancy_bonus += 1

                total_score = gain + redundancy_bonus * 0.5

                if total_score > best_score:
                    best_score = total_score
                    best_pair = (ci, cj)

        if best_pair is not None and best_score > 0:
            ci, cj = best_pair
            # Update cover counts
            for s in old_tsubs_i:
                cover_count[s] -= 1
                if cover_count[s] == 0:
                    del cover_count[s]
            for s in old_tsubs_j:
                cover_count[s] -= 1
                if cover_count[s] == 0:
                    del cover_count[s]
            for s in itertools.combinations(ci, t):
                cover_count[s] += 1
            for s in itertools.combinations(cj, t):
                cover_count[s] += 1

            improved[i] = ci
            improved[j] = cj

    return improved
