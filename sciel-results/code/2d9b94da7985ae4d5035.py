"""Covering design C(v,k,t) via randomized greedy + local search with deadline guard."""
import itertools
import random
import time
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 115.0  # safety margin before 120s kill
    rng = random.Random(42)

    # Precompute all t-subsets as frozensets for fast membership
    all_tsubs = list(itertools.combinations(range(v), t))
    all_tset = set(all_tsubs)

    # Precompute: for each point, which t-subsets contain it (for fast gain calc)
    # point_tsubs[x] = set of t-subsets containing point x
    point_tsubs = [set() for _ in range(v)]
    for ts in all_tsubs:
        for x in ts:
            point_tsubs[x].add(ts)

    def count_covered(block):
        """Count how many t-subsets of `block` are in uncovered."""
        cnt = 0
        for ts in itertools.combinations(block, t):
            if ts in uncovered:
                cnt += 1
        return cnt

    def greedy_build(uncovered, block, max_iter=500000):
        """Grow block greedily to size k, maximizing uncovered coverage."""
        iter_count = 0
        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # gain = number of uncovered t-subsets that would be covered by adding x
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    if ts in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                if gain == 0:
                    continue
            block.append(best_x)
            iter_count += 1
            if time.time() > deadline:
                return False
        return True

    def full_greedy(uncovered):
        """Standard greedy: pick uncovered t-subset, grow a block around it."""
        blocks = []
        unc = set(uncovered)
        while unc:
            if time.time() > deadline:
                break
            # Pick the uncovered t-subset with fewest possible completions (bottleneck)
            # For simplicity, pick the one with smallest lex order (fast)
            # Better: pick one that appears in fewest candidate k-blocks
            # Approximation: just pick min for speed, then grow
            target = min(unc)
            block = list(target)
            block.sort()
            if not greedy_build(unc, block):
                break
            block.sort()
            blocks.append(block)
            # Remove covered t-subsets
            for ts in itertools.combinations(block, t):
                unc.discard(ts)
        return blocks, unc

    def random_greedy(uncovered, seed):
        """Randomized greedy: pick uncovered t-subset with some randomness."""
        local_rng = random.Random(seed)
        blocks = []
        unc = set(uncovered)
        while unc:
            if time.time() > deadline:
                break
            # Pick a random uncovered t-subset (with slight bias toward "hard" ones)
            # Use random choice for diversity
            target = local_rng.choice(list(unc))
            block = list(target)
            block.sort()
            # Shuffle the order of target points to get different growth
            local_rng.shuffle(block)
            if not greedy_build(unc, block):
                break
            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                unc.discard(ts)
        return blocks, unc

    def is_valid(blocks):
        """Check if blocks cover all t-subsets."""
        covered = set()
        for block in blocks:
            for ts in itertools.combinations(block, t):
                covered.add(ts)
        return covered == all_tset

    def local_search(blocks):
        """Try to remove blocks that are redundant, or replace blocks to reduce count."""
        if time.time() > deadline:
            return blocks
        # Try removing each block if coverage is maintained
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(blocks)):
                if time.time() > deadline:
                    break
                # Temporarily remove block i
                remaining = blocks[:i] + blocks[i+1:]
                # Check if all t-subsets are still covered
                covered = set()
                for block in remaining:
                    for ts in itertools.combinations(block, t):
                        covered.add(ts)
                if covered == all_tset:
                    blocks = remaining
                    improved = True
                    break  # restart scan
        return blocks

    # Phase 1: Get a valid covering with standard greedy
    blocks, remaining = full_greedy(all_tset)
    if remaining:
        # Greedy didn't finish; use random restarts to finish
        for attempt in range(20):
            if time.time() > deadline:
                break
            blocks2, remaining2 = random_greedy(remaining, seed=1000 + attempt)
            if not remaining2:
                blocks = blocks + blocks2
                break
            # Merge: keep blocks that cover more
            # For now, just extend
            blocks = blocks + blocks2
            remaining = remaining2
        if remaining:
            # Last resort: add blocks for each remaining t-subset
            for ts in remaining:
                if time.time() > deadline:
                    break
                block = list(ts)
                block.sort()
                greedy_build(set(), block)
                blocks.append(block)

    # Ensure validity
    if not is_valid(blocks):
        # Add missing coverage
        covered = set()
        for block in blocks:
            for ts in itertools.combinations(block, t):
                covered.add(ts)
        for ts in (all_tset - covered):
            if time.time() > deadline:
                break
            block = list(ts)
            block.sort()
            greedy_build(set(), block)
            blocks.append(block)

    # Phase 2: Local search to reduce block count
    blocks = local_search(blocks)

    # Phase 3: Multiple randomized restarts, keep best
    best_blocks = blocks
    best_count = len(blocks)
    for restart in range(50):
        if time.time() > deadline:
            break
        rb, rrem = random_greedy(all_tset, seed=2000 + restart)
        if rrem:
            # Finish
            for ts in rrem:
                if time.time() > deadline:
                    break
                block = list(ts)
                block.sort()
                greedy_build(set(), block)
                rb.append(block)
        if len(rb) < best_count and is_valid(rb):
            rb = local_search(rb)
            if len(rb) < best_count:
                best_blocks = rb
                best_count = len(rb)

    return best_blocks
