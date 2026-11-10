"""Strategy: Bitset-accelerated greedy set cover with randomized restarts and local search pruning.

Uses numpy bitsets to represent the set of uncovered t-subsets and precomputed block signatures.
The greedy step picks the block (from a candidate pool) that covers the most uncovered subsets.
After construction, we run an aggressive local-search pass to remove redundant blocks.
We perform multiple randomized restarts and keep the best valid covering found.
"""
import itertools
import time
import random
import sys
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118  # 118s out of 120s limit, leaving 2s safety margin

    # Precompute all t-subsets and map to indices
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubsets)
    tsub_to_idx = {tsub: i for i, tsub in enumerate(all_tsubsets)}

    # We will use numpy bitsets for fast coverage tracking.
    # Each bit in a numpy uint64 array represents whether a t-subset is covered.
    # num_tsubs for v=24, t=3 is C(24,3) = 2024. 2024 / 64 = 32 words.
    num_words = (num_tsubs + 63) // 64

    def make_empty_bitset():
        return [0] * num_words

    def set_bit(bs, idx):
        word = idx >> 6
        bit = idx & 63
        bs[word] |= (1 << bit)

    def clear_bit(bs, idx):
        word = idx >> 6
        bit = idx & 63
        bs[word] &= ~(1 << bit)

    def get_bit(bs, idx):
        word = idx >> 6
        bit = idx & 63
        return (bs[word] >> bit) & 1

    def popcount(bs):
        # Count set bits
        cnt = 0
        for w in bs:
            cnt += bin(w).count('1')
        return cnt

    def bitset_and(bs1, bs2):
        return [bs1[i] & bs2[i] for i in range(num_words)]

    def bitset_or(bs1, bs2):
        return [bs1[i] | bs2[i] for i in range(num_words)]

    # Precompute for each possible block (7-subset of 24), the bitset of t-subsets it covers.
    # There are C(24,7) = 346104 possible blocks. That's a lot, but we can compute on the fly
    # or precompute a smaller set. Let's precompute for all blocks to allow fast lookup.
    # Actually, 346104 blocks * 32 words * 8 bytes = ~88 MB. That's fine.
    # But computing them takes time. Let's compute lazily or use a cache.
    
    # Alternative: Precompute the "signature" of each block as a frozenset of t-subset indices,
    # then convert to bitset when needed. Or just precompute all bitsets.
    
    # Let's precompute a dictionary: block_tuple -> bitset (list of ints)
    block_bitsets = {}
    
    # Generate all blocks
    all_blocks = list(itertools.combinations(range(v), k))
    
    # Precompute bitsets for all blocks
    for b in all_blocks:
        bs = make_empty_bitset()
        for tsub in itertools.combinations(b, t):
            idx = tsub_to_idx[tsub]
            set_bit(bs, idx)
        block_bitsets[b] = bs
    
    # For greedy, we want to pick the block that covers the most uncovered t-subsets.
    # Uncovered is represented as a bitset.
    # The gain of a block is popcount(bitset_and(block_bitset, uncovered)).
    
    # To speed up, we can precompute for each block the list of t-subset indices it covers.
    block_tsub_indices = {}
    for b in all_blocks:
        indices = [tsub_to_idx[tsub] for tsub in itertools.combinations(b, t)]
        block_tsub_indices[b] = indices

    best_blocks = None
    best_size = float('inf')
    
    num_restarts = 8
    time_per_restart = (deadline - start_time - 5) / num_restarts

    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # State: uncovered bitset, list of chosen blocks
        uncovered = make_empty_bitset()
        # Initialize uncovered to all 1s
        for i in range(num_tsubs):
            set_bit(uncovered, i)
        
        # Actually, setting all bits one by one is slow. Let's initialize differently.
        # uncovered[i] should be 1 for all i < num_tsubs.
        for w in range(num_words):
            uncovered[w] = 0xFFFFFFFFFFFFFFFF
        # Mask off the last word if needed
        if num_tsubs % 64 != 0:
            last_bits = num_tsubs % 64
            uncovered[-1] &= (1 << last_bits) - 1
        
        blocks = []
        cover_count = [0] * num_tsubs  # To track redundancy for local search
        
        # Greedy Construction
        # We need to find the block with max gain quickly.
        # Naive: iterate over all 346k blocks, compute gain. That's 346k * 32 ops per step.
        # Number of steps is ~76. Total: 76 * 346k * 32 ~ 830M ops. Too slow in Python.
        
        # Optimization: Use a candidate pool or a smarter selection.
        # Idea: Randomly sample a subset of blocks and pick the best among them.
        # Or: Use the structure. If we pick a random uncovered t-subset, the best block
        # must contain it. There are C(v-3, k-3) = C(21, 4) = 5985 blocks containing a given triple.
        # So we can pick a random uncovered t-subset, then check all blocks containing it.
        # 5985 blocks * 32 words = ~190k ops per step. 76 steps * 190k = 14M ops. Feasible.
        
        # Precompute: for each t-subset index, list of block indices that contain it.
        tsub_to_blocks = [[] for _ in range(num_tsubs)]
        for b_idx, b in enumerate(all_blocks):
            for tsub in itertools.combinations(b, t):
                idx = tsub_to_idx[tsub]
                tsub_to_blocks[idx].append(b_idx)
        
        # Greedy loop
        while popcount(uncovered) > 0 and time.time() < start_time + r*time_per_restart + time_per_restart*0.8:
            # Pick a random uncovered t-subset
            # Find a set bit in uncovered
            # Randomly select a word, then a bit
            while True:
                w = random.randint(0, num_words - 1)
                if uncovered[w] != 0:
                    break
            # Pick a random set bit in that word
            bits = [i for i in range(64) if (uncovered[w] >> i) & 1]
            bit = random.choice(bits)
            target_idx = w * 64 + bit
            if target_idx >= num_tsubs:
                continue
            
            # Find the best block among those containing this t-subset
            candidates = tsub_to_blocks[target_idx]
            best_block = None
            best_gain = -1
            
            # Sample if too many candidates? 5985 is small enough to check all.
            for b_idx in candidates:
                b = all_blocks[b_idx]
                bs = block_bitsets[b]
                # Compute gain: popcount(uncovered & bs)
                gain = 0
                for i in range(num_words):
                    gain += bin(uncovered[i] & bs[i]).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
            
            if best_block is None:
                break
            
            # Add the block
            blocks.append(best_block)
            bs = block_bitsets[best_block]
            for i in range(num_words):
                uncovered[i] &= ~bs[i]
            
            # Update cover_count for local search
            for idx in block_tsub_indices[best_block]:
                cover_count[idx] += 1
        
        # Check validity
        if popcount(uncovered) != 0:
            continue  # Invalid covering, skip
        
        # Local Search: Remove redundant blocks
        # A block is redundant if all its t-subsets are covered by other blocks.
        # We can check this by temporarily removing it and seeing if uncovered becomes non-empty.
        # Or, more efficiently, for each t-subset in the block, check if cover_count[idx] > 1.
        
        changed = True
        while changed and time.time() < start_time + r*time_per_restart + time_per_restart*0.95:
            changed = False
            # Try to remove blocks in random order
            indices = list(range(len(blocks)))
            random.shuffle(indices)
            for i in indices:
                if i >= len(blocks):
                    continue
                b = blocks[i]
                redundant = True
                for tsub in itertools.combinations(b, t):
                    idx = tsub_to_idx[tsub]
                    if cover_count[idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block
                    blocks.pop(i)
                    for idx in block_tsub_indices[b]:
                        cover_count[idx] -= 1
                    changed = True
                    break  # Restart the loop
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = [list(b) for b in blocks]
    
    if best_blocks is None:
        # Fallback: return a valid covering if possible, else empty
        return []
    
    return best_blocks
