import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110

    # Pre-compute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_t_subsets)
    subset_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}

    # Pre-compute all k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_k_blocks)

    # Pre-compute masks and reverse lookup
    block_masks = [0] * n_blocks
    t_sub_to_blocks = [[] for _ in range(n_subsets)]
    t_sub_freq = [0] * n_subsets

    for b_idx, block in enumerate(all_k_blocks):
        mask = 0
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            mask |= (1 << idx)
            t_sub_to_blocks[idx].append(b_idx)
            t_sub_freq[idx] += 1
        block_masks[b_idx] = mask

    if time.time() > deadline:
        return greedy_fallback(v, k, t, deadline)

    best_blocks = None
    best_score = float('inf')

    def bit_count(x):
        return x.bit_count()

    # Information-theoretic (group-testing) precomputation:
    # For each block, the "entropy value" of a triple it covers is
    # log(number of candidate blocks covering it). A block that covers
    # rare triples (few candidate blocks) yields more information per
    # block, so we weight gain by this.
    block_info = [0.0] * n_blocks
    for b_idx in range(n_blocks):
        block = all_k_blocks[b_idx]
        val = 0.0
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            val += t_sub_freq[idx]
        block_info[b_idx] = val
    # Normalize to [0,1]
    max_info = max(block_info) if block_info else 1.0
    if max_info > 0:
        block_info = [c / max_info for c in block_info]

    # Weighted greedy with multiple strategies
    # Strategies: 0=cardinality, 1=rare-triple target, 2=entropy reduction,
    #             3=information-theoretic gain (group testing), 4=mixed
    n_strategies = 5
    for run in range(120):
        if time.time() > deadline:
            break

        uncovered_mask = (1 << n_subsets) - 1
        blocks = []
        used_blocks = set()

        strategy = run % n_strategies

        while uncovered_mask:
            if time.time() > deadline:
                break

            # Choose target triple
            if strategy == 1:
                # Weighted: prioritize rare uncovered triples
                best_candidate = -1
                best_freq = float('inf')
                for _ in range(20):
                    pos = random.randint(0, n_subsets - 1)
                    if (uncovered_mask >> pos) & 1:
                        if t_sub_freq[pos] < best_freq:
                            best_freq = t_sub_freq[pos]
                            best_candidate = pos
                if best_candidate == -1:
                    temp = uncovered_mask
                    target_idx = (temp & -temp).bit_length() - 1
                else:
                    target_idx = best_candidate
            else:
                # Random uncovered triple
                target_idx = -1
                for _ in range(15):
                    pos = random.randint(0, n_subsets - 1)
                    if (uncovered_mask >> pos) & 1:
                        target_idx = pos
                        break
                if target_idx == -1:
                    temp = uncovered_mask
                    target_idx = (temp & -temp).bit_length() - 1

            # Get candidates
            candidates = t_sub_to_blocks[target_idx]
            best_block_idx = -1
            best_gain = -1.0

            for b_idx in candidates:
                if b_idx in used_blocks:
                    continue
                cand_mask = block_masks[b_idx]
                gain = bit_count(uncovered_mask & cand_mask)

                if strategy == 0:
                    # Pure cardinality
                    score = gain
                elif strategy == 1:
                    # Cardinality with rare-triple boost
                    score = gain + 0.2 * (1.0 - t_sub_freq[target_idx] / max(t_sub_freq))
                elif strategy == 2:
                    # Entropy reduction: weight each newly-covered triple by
                    # log(candidate count) - covers rare triples more heavily
                    score = gain + 1.5 * block_info[b_idx] * 10
                elif strategy == 3:
                    # Group-testing information gain: maximize
                    # sum over newly covered triples of log(freq)
                    # (information-theoretic separation)
                    score = gain + 2.0 * block_info[b_idx] * 10
                else:
                    # Mixed: blend cardinality and information
                    score = gain + 1.0 * block_info[b_idx] * 10

                if score > best_gain:
                    best_gain = score
                    best_block_idx = b_idx

            if best_block_idx == -1:
                break

            blocks.append(all_k_blocks[best_block_idx])
            used_blocks.add(best_block_idx)
            uncovered_mask &= ~block_masks[best_block_idx]

        if not uncovered_mask:
            # Post-processing: Remove redundant blocks
            coverage_count = {}
            for b in blocks:
                for sub in itertools.combinations(b, t):
                    coverage_count[sub] = coverage_count.get(sub, 0) + 1

            changed = True
            while changed:
                changed = False
                new_blocks = []
                for b in blocks:
                    is_redundant = True
                    for sub in itertools.combinations(b, t):
                        if coverage_count[sub] == 1:
                            is_redundant = False
                            break
                    if is_redundant:
                        changed = True
                        for sub in itertools.combinations(b, t):
                            coverage_count[sub] -= 1
                    else:
                        new_blocks.append(b)
                blocks = new_blocks
                if time.time() > deadline:
                    break

            if len(blocks) < best_score:
                best_score = len(blocks)
                best_blocks = blocks.copy()

    if best_blocks is None:
        # Fallback: deterministic greedy
        uncovered_mask = (1 << n_subsets) - 1
        blocks = []
        used_blocks = set()
        while uncovered_mask:
            if time.time() > deadline:
                break
            temp = uncovered_mask
            target_idx = (temp & -temp).bit_length() - 1
            candidates = t_sub_to_blocks[target_idx]
            best_block_idx = -1
            best_gain = -1
            for b_idx in candidates:
                if b_idx in used_blocks:
                    continue
                cand_mask = block_masks[b_idx]
                gain = bit_count(uncovered_mask & cand_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            if best_block_idx == -1:
                break
            blocks.append(all_k_blocks[best_block_idx])
            used_blocks.add(best_block_idx)
            uncovered_mask &= ~block_masks[best_block_idx]
        best_blocks = blocks

    return best_blocks

def greedy_fallback(v, k, t, deadline):
    start_time = time.time()
    all_t_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    n_subsets = len(all_t_subsets)

    def block_mask(block):
        mask = 0
        for sub in itertools.combinations(block, t):
            mask |= (1 << subset_to_idx[sub])
        return mask

    uncovered_mask = (1 << n_subsets) - 1
    blocks = []
    while uncovered_mask:
        if time.time() > deadline:
            break
        temp = uncovered_mask
        target_idx = (temp & -temp).bit_length() - 1
        target = all_t_subsets[target_idx]

        remaining_points = [p for p in range(v) if p not in target]
        best_block = None
        best_gain = -1
        for combo in itertools.combinations(remaining_points, k - t):
            candidate_block = tuple(sorted(target + combo))
            cand_mask = block_mask(candidate_block)
            gain = bin(uncovered_mask & cand_mask).count('1')
            if gain > best_gain:
                best_gain = gain
                best_block = candidate_block
        blocks.append(list(best_block))
        best_mask = block_mask(best_block)
        uncovered_mask &= ~best_mask
    return blocks
