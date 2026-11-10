# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict


def construct(v, k, t):
    deadline = time.time() + 115  # safety margin before 120s kill
    rng = random.Random(12345)

    # Precompute all t-subsets
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    t_index = {ts: i for i, ts in enumerate(all_t)}

    # Precompute for each t-subset, the list of all k-subsets containing it
    # This is C(v-t, k-t) per t-subset; for v=24,k=7,t=3: C(21,4)=5985 per t-subset
    # Total t-subsets: C(24,3)=2024. Total pairs: 2024*5985 ~ 12M - too many to store.
    # Instead, for a given block, compute which t-subsets it covers quickly.

    # For greedy: given a partial block (list of points), compute gain = number of
    # uncovered t-subsets that would be covered by adding point x.
    # A t-subset {a,b,c} is covered by block B if {a,b,c} ⊆ B.
    # For partial block P and candidate x, the newly covered t-subsets are those
    # containing x and (t-1) elements from P, that are currently uncovered.

    # Represent uncovered as a boolean list (or bitset via Python int for speed)
    # Python ints are arbitrary precision - we can use a bitset where bit i is set
    # if t-subset i is uncovered.

    # Build bitset of all t-subsets uncovered
    full_mask = (1 << n_t) - 1

    def block_to_tmask(block):
        """Return bitmask of t-subsets contained in block."""
        mask = 0
        for ts in itertools.combinations(block, t):
            mask |= (1 << t_index[ts])
        return mask

    def greedy_cover(rng, deadline):
        """One greedy run with randomization."""
        uncovered_mask = full_mask
        blocks = []
        # Precompute: for each point, list of t-subsets containing it
        # Actually, for gain computation: for partial block P, candidate x:
        # gain = popcount(uncovered & mask_of_t_subsets_containing_x_and_(t-1)_from_P)
        
        # Precompute for each (t-1)-subset S and point x not in S:
        # the t-subset S ∪ {x}, and its index
        # For gain: sum over (t-1)-subsets of P of the t-subset formed with x
        # That's C(|P|, t-1) lookups per candidate x.
        
        while uncovered_mask and time.time() < deadline:
            # Pick a random uncovered t-subset to seed the block
            # Find an uncovered t-subset
            # Instead of iterating, pick a random bit position and check
            # Better: maintain a list of uncovered indices? Too slow to update.
            # Use: pick random index, if not uncovered, retry. Or just pick first uncovered.
            
            # For efficiency, let's pick a random uncovered t-subset
            # We'll try random indices
            seed_ts = None
            for _ in range(100):
                idx = rng.randrange(n_t)
                if (uncovered_mask >> idx) & 1:
                    seed_ts = all_t[idx]
                    break
            if seed_ts is None:
                # Fallback: scan
                for idx in range(n_t):
                    if (uncovered_mask >> idx) & 1:
                        seed_ts = all_t[idx]
                        break
            if seed_ts is None:
                break

            block = list(seed_ts)
            rng.shuffle(block)  # randomize order within seed
            block_set = set(block)
            
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    # Compute gain: number of uncovered t-subsets that contain x and (t-1) from block
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = t_index[ts]
                        if (uncovered_mask >> idx) & 1:
                            gain += 1
                    if gain > best_gain or (gain == best_gain and gain > 0 and rng.random() < 0.3):
                        best_gain = gain
                        best_x = x
                if best_gain <= 0:
                    # No useful point; add random one
                    candidates = [x for x in range(v) if x not in block_set]
                    if candidates:
                        best_x = rng.choice(candidates)
                    else:
                        break
                block.append(best_x)
                block_set.add(best_x)
            
            block.sort()
            blocks.append(block)
            uncovered_mask &= ~block_to_tmask(block)
            
            if time.time() > deadline:
                break
        
        return blocks

    def local_search(blocks, deadline):
        """Try to remove redundant blocks, then try swaps."""
        # Count coverage of each t-subset
        cover_count = [0] * n_t
        for blk in blocks:
            for ts in itertools.combinations(blk, t):
                cover_count[t_index[ts]] += 1
        
        # Try removing blocks that don't uniquely cover any t-subset
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(blocks)):
                # Check if block i is needed
                needed = False
                for ts in itertools.combinations(blocks[i], t):
                    if cover_count[t_index[ts]] == 1:
                        needed = True
                        break
                if not needed:
                    # Remove block i
                    for ts in itertools.combinations(blocks[i], t):
                        cover_count[t_index[ts]] -= 1
                    blocks.pop(i)
                    changed = True
                    break
        
        # Try block replacement: replace a block with a better one
        # For each block, try to find a replacement that covers at least as much
        # This is expensive; do limited iterations
        if time.time() < deadline:
            for _ in range(20):
                if time.time() > deadline:
                    break
                # Pick a random block
                i = rng.randrange(len(blocks))
                # Try to find a replacement block
                # The block should cover at least the unique t-subsets of block i
                unique_ts = [ts for ts in itertools.combinations(blocks[i], t) 
                           if cover_count[t_index[ts]] == 1]
                if not unique_ts:
                    continue
                # Find a k-subset that covers all unique_ts
                # Start with union of unique_ts points
                must_include = set()
                for ts in unique_ts:
                    must_include.update(ts)
                if len(must_include) > k:
                    continue
                # Build a block containing must_include
                block = list(must_include)
                block_set = set(block)
                # Fill remaining
                while len(block) < k:
                    best_x = -1
                    best_gain = -1
                    for x in range(v):
                        if x in block_set:
                            continue
                        gain = 0
                        for rest in itertools.combinations(block, t - 1):
                            ts2 = tuple(sorted(rest + (x,)))
                            idx = t_index[ts2]
                            if cover_count[idx] > 0:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    if best_x == -1 or best_gain <= 0:
                        break
                    block.append(best_x)
                    block_set.add(best_x)
                block.sort()
                if block == blocks[i]:
                    continue
                # Check if this replacement is valid (covers all unique_ts)
                valid = True
                for ts in unique_ts:
                    if not set(ts).issubset(set(block)):
                        valid = False
                        break
                if valid:
                    # Apply replacement
                    old_block = blocks[i]
                    for ts in itertools.combinations(old_block, t):
                        cover_count[t_index[ts]] -= 1
                    blocks[i] = block
                    for ts in itertools.combinations(block, t):
                        cover_count[t_index[ts]] += 1
                    changed = True
        
        return blocks

    best_blocks = None
    best_len = float('inf')
    
    # Multiple greedy restarts with local search
    n_restarts = 50  # will be limited by deadline
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        blocks = greedy_cover(rng, deadline)
        if time.time() > deadline:
            break
        blocks = local_search(blocks, deadline)
        if time.time() > deadline:
            break
        
        # Verify coverage
        uncovered_mask = full_mask
        for blk in blocks:
            uncovered_mask &= ~block_to_tmask(blk)
        if uncovered_mask != 0:
            # Not a valid covering; skip
            continue
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]
    
    if best_blocks is None:
        # Fallback: do one more greedy without local search
        best_blocks = greedy_cover(rng, deadline)
        # Ensure validity
        uncovered_mask = full_mask
        for blk in best_blocks:
            uncovered_mask &= ~block_to_tmask(blk)
        while uncovered_mask:
            # Find uncovered t-subset and add a block
            for idx in range(n_t):
                if (uncovered_mask >> idx) & 1:
                    seed_ts = all_t[idx]
                    break
            block = list(seed_ts)
            block_set = set(block)
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx2 = t_index[ts]
                        if (uncovered_mask >> idx2) & 1:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    break
                block.append(best_x)
                block_set.add(best_x)
            block.sort()
            best_blocks.append(block)
            uncovered_mask &= ~block_to_tmask(block)
    
    return best_blocks
# EVOLVE-BLOCK-END
