import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    deadline = time.time() + 148
    
    def time_left():
        return deadline - time.time()
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    n_words = (n_tsubs + 63) // 64
    
    # Precompute point coverage bitsets
    point_bits = []
    for x in range(v):
        bits = [0] * n_words
        for i, ts in enumerate(all_tsubs):
            if x in ts:
                bits[i // 64] |= (1 << (i % 64))
        point_bits.append(bits)
    
    def block_coverage_bitset(block):
        bits = [0] * n_words
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[tuple(ts)]
            bits[idx // 64] |= (1 << (idx % 64))
        return bits
    
    def count_bits(bits):
        return sum(bin(w).count('1') for w in bits)
    
    def bits_or(a, b):
        return [a[i] | b[i] for i in range(n_words)]
    
    def bits_and_not(a, b):
        return [a[i] & ~b[i] for i in range(n_words)]
    
    def bits_popcount_and_not(a, b):
        return sum(bin(a[i] & ~b[i]).count('1') for i in range(n_words))
    
    def is_full(bits):
        for i in range(n_words):
            expected = (1 << 64) - 1
            if i == n_words - 1:
                lb = n_tsubs % 64
                if lb == 0:
                    lb = 64
                expected = (1 << lb) - 1
            if bits[i] != expected:
                return False
        return True
    
    def greedy_cover(seed):
        rng = random.Random(seed)
        
        uncovered = [0] * n_words
        for i in range(n_words):
            uncovered[i] = (1 << 64) - 1
        lb = n_tsubs % 64
        if lb == 0:
            lb = 64
        uncovered[-1] = (1 << lb) - 1
        
        blocks = []
        block_bitsets = []
        
        while time_left() > 8:
            best_block = None
            best_gain = 0
            best_bs = None
            
            # Sample uncovered t-subsets
            sample = []
            for i in range(n_tsubs):
                if (uncovered[i // 64] >> (i % 64)) & 1:
                    sample.append(all_tsubs[i])
                    if len(sample) >= 25:
                        break
            
            if not sample:
                break
            
            rng.shuffle(sample)
            n_try = min(10, len(sample))
            
            for start_ts in sample[:n_try]:
                block = list(start_ts)
                cur_bits = [0] * n_words
                for x in block:
                    cur_bits = bits_or(cur_bits, point_bits[x])
                
                while len(block) < k and time_left() > 4:
                    best_x = -1
                    best_g = -1
                    for x in range(v):
                        if x in block:
                            continue
                        nb = bits_or(cur_bits, point_bits[x])
                        g = bits_popcount_and_not(nb, cur_bits)
                        if g > best_g:
                            best_g = g
                            best_x = x
                    if best_x < 0:
                        break
                    block.append(best_x)
                    cur_bits = bits_or(cur_bits, point_bits[best_x])
                
                block.sort()
                bs = block_coverage_bitset(block)
                gain = bits_popcount_and_not(bs, uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                    best_bs = bs
            
            if best_block is None or best_gain == 0:
                break
            
            blocks.append(best_block)
            block_bitsets.append(best_bs)
            uncovered = bits_and_not(uncovered, best_bs)
            
            if is_full([0] * n_words) or count_bits(uncovered) == 0:
                break
        
        return blocks, block_bitsets
    
    def remove_redundant(blocks, block_bitsets):
        n = len(blocks)
        changed = True
        while changed and time_left() > 5:
            changed = False
            for i in range(n - 1, -1, -1):
                if time_left() < 5:
                    break
                union = [0] * n_words
                for j in range(n):
                    if j != i:
                        union = bits_or(union, block_bitsets[j])
                extra = bits_and_not(block_bitsets[i], union)
                if count_bits(extra) == 0:
                    del blocks[i]
                    del block_bitsets[i]
                    n -= 1
                    changed = True
                    break
        return blocks, block_bitsets
    
    best_blocks = None
    best_count = float('inf')
    
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    
    for seed in seeds:
        if time_left() < 20:
            break
        
        blocks, block_bitsets = greedy_cover(seed)
        
        if not blocks:
            continue
        
        blocks, block_bitsets = remove_redundant(blocks, block_bitsets)
        
        # Verify
        union = [0] * n_words
        for bs in block_bitsets:
            union = bits_or(union, bs)
        
        if count_bits(union) == n_tsubs:
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = list(blocks)
        
        if time_left() < 10:
            break
    
    if best_blocks is None:
        blocks, _ = greedy_cover(42)
        best_blocks = blocks
    
    return best_blocks
