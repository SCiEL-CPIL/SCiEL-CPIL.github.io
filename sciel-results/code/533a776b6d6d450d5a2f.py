import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    # 1. Precomputation
    # Generate all t-subsets and assign each a compact index 0..N-1
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_to_idx = {}
    t_idx_to_mask = []
    for i, sub in enumerate(all_t_subsets):
        t_sub_to_idx[sub] = i
        mask = 0
        for p in sub:
            mask |= (1 << p)
        t_idx_to_mask.append(mask)

    # Generate all k-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subsets)

    # Precompute for each k-block:
    #   block_cover_mask: a Python int bitmask over t-subset indices (bit i set iff t-sub i is in this block)
    #   block_t_indices: list of t-subset indices covered by this block
    block_cover_mask = [0] * n_k
    block_t_indices = [[] for _ in range(n_k)]
    for bi, block in enumerate(all_k_subsets):
        cm = 0
        tlist = []
        for t_sub in itertools.combinations(block, t):
            ti = t_sub_to_idx[t_sub]
            cm |= (1 << ti)
            tlist.append(ti)
        block_cover_mask[bi] = cm
        block_t_indices[bi] = tlist

    # For each t-subset index, list of block indices that cover it
    t_idx_to_blocks = [[] for _ in range(n_t)]
    for bi in range(n_k):
        for ti in block_t_indices[bi]:
            t_idx_to_blocks[ti].append(bi)

    # Compute scarcity weight for each t-subset: 1 / (number of blocks covering it)
    # This helps break ties by preferring blocks that cover rarer t-subsets
    t_scarcity = [1.0 / len(t_idx_to_blocks[ti]) for ti in range(n_t)]

    # Precompute weighted gain contribution: for each block, a float array is too heavy.
    # Instead, we'll compute a "scarcity score" on the fly using a small helper.

    # 2. Greedy with restarts and local polish
    best_blocks = None
    best_count = float('inf')

    # full_mask has all n_t bits set
    full_mask = (1 << n_t) - 1

    def local_polish(blocks_list, max_iter=50):
        """Try to improve a complete covering by removing redundant blocks and adding better ones."""
        if not blocks_list:
            return blocks_list, False
        
        # Build current coverage: for each t-subset, how many blocks cover it
        cover_count = [0] * n_t
        block_indices_in_solution = []
        for b in blocks_list:
            bi = None
            # Find index of this block
            for idx in range(len(all_k_subsets)):
                if all_k_subsets[idx] == tuple(b):
                    bi = idx
                    break
            if bi is None:
                continue
            block_indices_in_solution.append(bi)
            for ti in block_t_indices[bi]:
                cover_count[ti] += 1
        
        improved = False
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Try to find a block that can be removed (all its t-subsets are covered by >= 2 blocks)
            removable = []
            for bi in block_indices_in_solution:
                removable_candidate = True
                for ti in block_t_indices[bi]:
                    if cover_count[ti] <= 1:
                        removable_candidate = False
                        break
                if removable_candidate:
                    removable.append(bi)
            
            if not removable:
                break
            
            # Try removing a random removable block
            bi_to_remove = random.choice(removable)
            for ti in block_t_indices[bi_to_remove]:
                cover_count[ti] -= 1
            
            # Check if still covered
            still_covered = True
            for ti in block_t_indices[bi_to_remove]:
                if cover_count[ti] < 1:
                    still_covered = False
                    break
            
            if still_covered:
                block_indices_in_solution.remove(bi_to_remove)
                improved = True
            else:
                # Restore
                for ti in block_t_indices[bi_to_remove]:
                    cover_count[ti] += 1
        
        # Try to add a block that covers the most uncovered (or rarely covered) t-subsets
        # Actually, since we only removed redundant blocks, all are still covered.
        # Let's try a different polish: swap a block for another that allows more removals.
        
        # Simpler polish: try replacing each block with an alternative that has higher scarcity gain
        for idx, bi in enumerate(block_indices_in_solution[:]):
            if time.time() > deadline - 5:
                break
            
            # Find alternative blocks that cover the same t-subsets as bi (or a superset)
            # This is expensive, so skip for now
            pass
        
        return [list(all_k_subsets[bi]) for bi in block_indices_in_solution], improved

    while time.time() < deadline:
        uncovered_mask = full_mask
        blocks = []
        used_blocks = set()

        while uncovered_mask:
            n_uncovered = uncovered_mask.bit_count()
            r = random.randrange(n_uncovered)
            temp = uncovered_mask
            for _ in range(r):
                temp &= temp - 1
            target_bit = temp & (-temp)
            target_idx = target_bit.bit_length() - 1

            candidates = t_idx_to_blocks[target_idx]
            if not candidates:
                break

            # Calculate gain for each candidate: count of uncovered bits in block's cover
            # Also compute scarcity-weighted score for tie-breaking
            max_gain = 0
            best_candidates = []
            best_scarcity = -1

            for bi in candidates:
                if bi in used_blocks:
                    continue
                gain = (uncovered_mask & block_cover_mask[bi]).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_candidates = [bi]
                    # Compute scarcity score for this block
                    sc = 0.0
                    for ti in block_t_indices[bi]:
                        if (uncovered_mask >> ti) & 1:
                            sc += t_scarcity[ti]
                    best_scarcity = sc
                elif gain == max_gain:
                    best_candidates.append(bi)

            if not best_candidates:
                # Allow reusing blocks if necessary
                max_gain = 0
                best_candidates = []
                best_scarcity = -1
                for bi in candidates:
                    gain = (uncovered_mask & block_cover_mask[bi]).bit_count()
                    if gain > max_gain:
                        max_gain = gain
                        best_candidates = [bi]
                        sc = 0.0
                        for ti in block_t_indices[bi]:
                            if (uncovered_mask >> ti) & 1:
                                sc += t_scarcity[ti]
                        best_scarcity = sc
                    elif gain == max_gain:
                        best_candidates.append(bi)

            if not best_candidates:
                break

            # Among best candidates, prefer higher scarcity score
            if len(best_candidates) > 1:
                scored = []
                for bi in best_candidates:
                    sc = 0.0
                    for ti in block_t_indices[bi]:
                        if (uncovered_mask >> ti) & 1:
                            sc += t_scarcity[ti]
                    scored.append((sc, bi))
                scored.sort(key=lambda x: -x[0])
                # Pick from top few with some randomness
                top_n = min(3, len(scored))
                chosen_bi = random.choice([s[1] for s in scored[:top_n]])
            else:
                chosen_bi = best_candidates[0]

            used_blocks.add(chosen_bi)
            blocks.append(all_k_subsets[chosen_bi])
            uncovered_mask &= ~block_cover_mask[chosen_bi]

        if uncovered_mask:
            continue

        # Try local polish
        polished, improved = local_polish(blocks, max_iter=30)
        if len(polished) < len(blocks):
            blocks = polished

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]

    if best_blocks is None:
        # Fallback: deterministic greedy
        uncovered_mask = full_mask
        blocks = []
        while uncovered_mask:
            target_bit = uncovered_mask & (-uncovered_mask)
            target_idx = target_bit.bit_length() - 1
            candidates = t_idx_to_blocks[target_idx]
            max_gain = 0
            best_bi = candidates[0]
            for bi in candidates:
                gain = (uncovered_mask & block_cover_mask[bi]).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_bi = bi
            blocks.append(all_k_subsets[best_bi])
            uncovered_mask &= ~block_cover_mask[best_bi]
        return [list(b) for b in blocks]

    return best_blocks
