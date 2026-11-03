"""Covering design C(v,k,t): vectorized greedy + swap/removal local search."""
import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 128  # safety margin before 150s kill
    t0 = deadline  # placeholder; real start below

    # ---------- Precomputation ----------
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}

    # For each point x, the list of t-subset indices that contain x.
    point_tsubs = [[] for _ in range(v)]
    for idx, ts in enumerate(all_tsubs):
        for x in ts:
            point_tsubs[x].append(idx)
    point_tsubs_arr = [np.array(lst, dtype=np.int32) for lst in point_tsubs]

    # For each k-subset (block), the union of t-subset indices it covers.
    # We build a helper to compute a block's covered indices quickly.
    def block_cover(block):
        """Return a boolean mask (length n_tsubs) of t-subsets covered by block."""
        mask = np.zeros(n_tsubs, dtype=bool)
        for x in block:
            mask[point_tsubs_arr[x]] = True
        return mask

    # All possible k-blocks as tuples (for swap generation we sample random ones).
    # We don't enumerate all C(19,7)=50388 blocks' covers up front (memory/time),
    # but we do precompute covers for the blocks we actually use.

    # ---------- Greedy construction ----------
    def greedy_build(rng, use_soft_bias=True):
        """Randomized greedy covering. Returns list of blocks (sorted tuples)."""
        uncovered = np.ones(n_tsubs, dtype=bool)  # True = uncovered
        blocks = []
        # Soft surrogate: a weight per t-subset that grows as it stays uncovered.
        # We use a smooth "leverage" = log-sum-exp style accumulator.
        # Simpler effective surrogate: weight[w] = 1 + count_of_times_w_was_the_argmax_uncovered
        # We implement a light version: track how many times each t-subset is the
        # "hardest" (fewest covering candidates) via a running potential.
        potential = np.zeros(n_tsubs, dtype=np.float64)

        while uncovered.any():
            # Find the uncovered t-subset with the highest potential (soft bias).
            # Restrict argmax to uncovered entries.
            weighted = potential.copy()
            weighted[~uncovered] = -1e9
            # Add a small random perturbation for diversity.
            weighted += rng.random(n_tsubs) * 0.5
            start_idx = int(np.argmax(weighted))
            start_tsub = all_tsubs[start_idx]

            # Build a block around start_tsub, greedily adding points that
            # maximize newly-covered t-subsets (with soft-bias tie-break).
            block = list(start_tsub)
            in_block = set(block)
            current_cover = block_cover(block)
            newly = current_cover & uncovered

            while len(block) < k:
                best_gain = -1
                best_x = None
                best_newly = None
                for x in range(v):
                    if x in in_block:
                        continue
                    cand_cover = current_cover.copy()
                    cand_cover[point_tsubs_arr[x]] = True
                    cand_newly = cand_cover & uncovered
                    gain = int(cand_newly.sum())
                    # Soft surrogate tie-break: prefer x that also covers
                    # high-potential uncovered t-subsets.
                    if gain == best_gain:
                        # compute a smooth surrogate score
                        score = float(cand_newly @ potential) if use_soft_bias else 0.0
                        if best_x is None:
                            best_x, best_gain, best_newly, best_cover = x, gain, cand_newly, cand_cover
                        else:
                            if score > (best_newly @ potential if use_soft_bias else 0.0) + rng.random()*0.1:
                                best_x, best_gain, best_newly, best_cover = x, gain, cand_newly, cand_cover
                    elif gain > best_gain:
                        best_x, best_gain, best_newly, best_cover = x, gain, cand_newly, cand_cover
                if best_x is None:
                    break
                block.append(best_x)
                in_block.add(best_x)
                current_cover = best_cover
                newly = best_newly
                # Update potential: uncovered t-subsets that are still hard get boosted.
                # Smooth surrogate: potential grows with log of how many blocks have
                # failed to cover them. We approximate by adding a small amount to
                # every still-uncovered t-subset, weighted by coverage scarcity.
                if use_soft_bias:
                    still_uncovered = uncovered & ~current_cover
                    # boost proportional to 1/(1+covered_count) is costly; use a
                    # light log-sum-exp style: potential += softplus-like increment.
                    potential[still_uncovered] += 0.05

            block.sort()
            blocks.append(tuple(block))
            # Mark covered.
            final_cover = block_cover(block)
            uncovered[final_cover] = False
            # Boost potential for newly-covered hard t-subsets is not needed;
            # instead decay potential for now-covered ones (they are resolved).
            if use_soft_bias:
                potential[final_cover] *= 0.5

        return blocks

    # ---------- Local search: removal + swap ----------
    def local_search(blocks, rng):
        """Reduce block count via removal passes and 1-for-1 swaps. In-place-ish."""
        blocks = list(blocks)
        # Build coverage count and per-block cover masks.
        cover_masks = [block_cover(b) for b in blocks]
        cov_count = np.zeros(n_tsubs, dtype=np.int32)
        for m in cover_masks:
            cov_count[m] += 1

        # Removal pass: drop any block whose removal keeps min cov >= 1.
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if cov_count.sum() == 0:
                    break
                test = cov_count.copy()
                test[cover_masks[i]] -= 1
                if test.min() >= 1:
                    # remove block i
                    cov_count = test
                    blocks.pop(i)
                    cover_masks.pop(i)
                    changed = True
                    break

        # Swap pass: for each block i, try replacing it with a random new block
        # that covers more of the t-subsets currently covered only by i.
        # A swap is beneficial if (new_cover & only_by_i).sum() > (old_cover & ~new_cover & uncovered_by_others).sum()
        # i.e. new block covers strictly more "unique-to-i" t-subsets than it loses.
        n_blocks = len(blocks)
        if n_blocks == 0:
            return blocks

        # Precompute, for each t-subset, how many blocks cover it (cov_count) and
        # the set of blocks covering it is implicit. "only_by_i" = cov_count==1 & cover_masks[i].
        improved = True
        while improved and time.time() < deadline - 8:
            improved = False
            for i in range(len(blocks)):
                if time.time() > deadline - 8:
                    break
                only_i = (cov_count == 1) & cover_masks[i]
                if not only_i.any():
                    continue
                only_i_count = int(only_i.sum())
                if only_i_count == 0:
                    continue
                # Try a few random replacement blocks.
                for _ in range(12):
                    # Sample a random k-block.
                    new_block = tuple(sorted(rng.sample(range(v), k)))
                    new_cover = block_cover(new_block)
                    # Gain: t-subsets covered by new_block that are only covered by i.
                    gain = int((new_cover & only_i).sum())
                    # Loss: t-subsets covered by i but NOT by new_block and NOT
                    # covered by any other block (i.e., would become uncovered).
                    # Those are (cover_masks[i] & ~new_cover) & (cov_count == 1).
                    loss = int((cover_masks[i] & ~new_cover & (cov_count == 1)).sum())
                    if gain > loss:
                        # Perform swap.
                        # Update cov_count: remove old, add new.
                        cov_count[cover_masks[i]] -= 1
                        cov_count[new_cover] += 1
                        blocks[i] = new_block
                        cover_masks[i] = new_cover
                        improved = True
                        break  # restart scan from block 0
                # small random perturbation of potential not needed here
        return blocks

    # ---------- Main driver ----------
    start = time.time()
    best_blocks = None
    best_count = float('inf')

    # Deterministic-ish first greedy (low randomness) for a quick baseline.
    rng0 = random.Random(12345)
    b = greedy_build(rng0, use_soft_bias=True)
    b = local_search(b, random.Random(12345))
    best_blocks = b
    best_count = len(b)

    # Randomized restarts.
    seed = 1
    while time.time() < deadline - 20:
        rng = random.Random(seed)
        b = greedy_build(rng, use_soft_bias=True)
        b = local_search(b, rng)
        if len(b) < best_count:
            best_count = len(b)
            best_blocks = b
        seed += 1
        # Occasionally re-run local search on the best with a fresh rng to polish.
        if seed % 3 == 0 and time.time() < deadline - 15:
            polish_rng = random.Random(seed * 7 + 1)
            polished = local_search(list(best_blocks), polish_rng)
            if len(polished) < best_count:
                best_count = len(polished)
                best_blocks = polished

    # Final polish with remaining time.
    if time.time() < deadline - 5:
        polish_rng = random.Random(99999)
        polished = local_search(list(best_blocks), polish_rng)
        if len(polished) < best_count:
            best_blocks = polished

    return [list(b) for b in best_blocks]
