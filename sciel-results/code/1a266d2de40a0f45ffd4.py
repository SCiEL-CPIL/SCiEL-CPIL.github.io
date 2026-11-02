# EVOLVE-BLOCK-START
import itertools
import random
import time

def construct(v, k, t):
    start = time.time()
    deadline = start + 120.0
    margin = 6.0

    def remaining():
        return deadline - margin - time.time()

    # All t-subsets as bitmasks (t small, e.g., 3)
    all_ts = list(itertools.combinations(range(v), t))
    all_ts.sort()
    nT = len(all_ts)
    ts_mask = []
    for c in all_ts:
        m = 0
        for p in c:
            m |= (1 << p)
        ts_mask.append(m)
    ts_set = set(ts_mask)

    def block_to_mask(block):
        m = 0
        for p in block:
            m |= (1 << p)
        return m

    # Precompute: for each point p, the list of t-subset indices containing p
    point_to_ts = [[] for _ in range(v)]
    for i, m in enumerate(ts_mask):
        mm = m
        while mm:
            lb = mm & (-mm)
            p = lb.bit_length() - 1
            point_to_ts[p].append(i)
            mm -= lb

    # Precompute: for each t-subset index, the list of points in it (for quick iteration)
    ts_points = [tuple(c) for c in all_ts]

    def greedy_build(rng=None):
        coverage = [0] * nT
        blocks = []
        uncovered = set(range(nT))
        if rng is None:
            rng = random.Random()
        while uncovered:
            seed_idx = rng.choice(list(uncovered))
            block = list(ts_points[seed_idx])
            while len(block) < k:
                best_p = -1
                best_gain = -1
                in_block = set(block)
                pts = list(range(v))
                rng.shuffle(pts)
                for p in pts:
                    if p in in_block:
                        continue
                    gain = 0
                    for idx in point_to_ts[p]:
                        if coverage[idx] != 0:
                            continue
                        ok = True
                        for q in ts_points[idx]:
                            if q == p:
                                continue
                            if q not in in_block:
                                ok = False
                                break
                        if ok:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_p = p
                if best_p == -1 or best_gain == 0:
                    for p in pts:
                        if p not in in_block:
                            best_p = p
                            break
                block.append(best_p)
            block.sort()
            blocks.append(block)
            bm = block_to_mask(block)
            for i, m in enumerate(ts_mask):
                if (m & bm) == m:
                    if coverage[i] == 0:
                        uncovered.discard(i)
                    coverage[i] += 1
        return blocks, coverage

    def try_remove(blocks):
        n = len(blocks)
        block_ts = []
        for b in blocks:
            bm = block_to_mask(b)
            lst = []
            for i, m in enumerate(ts_mask):
                if (m & bm) == m:
                    lst.append(i)
            block_ts.append(lst)
        cov = [0] * nT
        for lst in block_ts:
            for i in lst:
                cov[i] += 1
        order = list(range(n))
        random.shuffle(order)
        removed = [False] * n
        for bi in order:
            if removed[bi]:
                continue
            ok = True
            for i in block_ts[bi]:
                if cov[i] < 2:
                    ok = False
                    break
            if ok:
                removed[bi] = True
                for i in block_ts[bi]:
                    cov[i] -= 1
        newblocks = [b for idx, b in enumerate(blocks) if not removed[idx]]
        return newblocks

    def swap_improve(blocks, budget):
        """Try to swap blocks to enable further deletions."""
        best = list(blocks)
        best_len = len(best)
        rng = random.Random(42)
        
        while budget > 0 and time.time() < deadline - margin:
            # Try multiple random swap attempts
            improved = False
            for _ in range(50):
                if time.time() > deadline - margin:
                    break
                n = len(best)
                if n < 2:
                    break
                # Pick a random block to try removing
                bi = rng.randint(0, n - 1)
                # Check if this block is "critical" (has t-subsets only it covers)
                bm = block_to_mask(best[bi])
                critical = []
                for i, m in enumerate(ts_mask):
                    if (m & bm) == m:
                        # Count how many blocks cover this t-subset
                        cnt = 0
                        for j, b in enumerate(best):
                            if j == bi:
                                continue
                            bmask = block_to_mask(b)
                            if (m & bmask) == m:
                                cnt += 1
                        if cnt == 0:
                            critical.append(i)
                            break
                if critical:
                    continue  # Can't remove this block without breaking coverage
                
                # Try to find a replacement block that covers all critical t-subsets
                # and allows the original block to be removed
                # For simplicity, try swapping with a random other block
                bj = rng.randint(0, n - 1)
                if bj == bi:
                    continue
                
                # Check if swapping bi and bj helps
                # New set: remove bi, add bj (if not already present)
                if best[bj] in best[:bi] + best[bi+1:]:
                    continue
                
                new_blocks = [b for idx, b in enumerate(best) if idx != bi]
                new_blocks.append(best[bj])
                
                # Try to delete more blocks from new_blocks
                cand = try_remove(new_blocks)
                if len(cand) < best_len:
                    best = cand
                    best_len = len(cand)
                    improved = True
                    break
            
            if not improved:
                break
        
        return best

    # Strategy:
    # 1) Build an initial greedy covering.
    # 2) Improve via removal local search.
    # 3) Try swap-based improvement.
    # 4) If time remains, do additional greedy builds with different randomizations and keep best.

    best_blocks = None
    best_count = None

    # First greedy build (deterministic-ish)
    if remaining() <= 0:
        blocks, _ = greedy_build(random.Random(0))
        return blocks

    blocks, _ = greedy_build(random.Random(0))
    best_blocks = blocks
    best_count = len(blocks)

    # Improve with deletion
    if remaining() > 0:
        improved = try_remove(best_blocks)
        if len(improved) < best_count:
            best_blocks = improved
            best_count = len(improved)

    # Try swap-based improvement
    if remaining() > 1.0:
        swapped = swap_improve(best_blocks, remaining())
        if len(swapped) < best_count:
            best_blocks = swapped
            best_count = len(swapped)

    # Additional randomized greedy builds if time allows
    rng = random.Random(12345)
    while remaining() > 3.0:
        blocks2, _ = greedy_build(rng)
        cand = try_remove(blocks2)
        if len(cand) < best_count:
            best_blocks = cand
            best_count = len(cand)
        # Also try swap improvement on this candidate
        if remaining() > 2.0:
            swapped = swap_improve(cand, remaining())
            if len(swapped) < best_count:
                best_blocks = swapped
                best_count = len(swapped)

    return best_blocks
# EVOLVE-BLOCK-END
