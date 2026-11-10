"""Bitset-accelerated greedy covering for C(24,7,3)."""
import itertools
import numpy as np
from math import comb


def construct(v, k, t):
    # Total number of t-subsets
    total_t_subsets = comb(v, t)
    
    # Precompute index for each t-subset
    # We need a fast way to map a t-subset (tuple of sorted ints) to an index
    # For v=24, t=3, there are C(24,3) = 2024 subsets
    # We can precompute a dictionary or use combinatorial number system
    
    # Using combinatorial number system (combinadic) for fast indexing
    # For a sorted tuple (a, b, c) with 0 <= a < b < c < v
    # The rank is C(c, t) - C(b, t) + C(a, t) ... wait, let's use the standard formula
    # Rank of (a0, a1, ..., a_{t-1}) where a0 < a1 < ... < a_{t-1}
    # rank = C(a_{t-1}, t) - C(a_{t-2}, t) + ... + (-1)^t * C(a_0, t)
    # Actually, the standard combinadic formula for rank in lexicographic order:
    # Let's just precompute all t-subsets and their indices in a dict for simplicity,
    # since 2024 is small. But for speed, let's use numpy bitsets.
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_index = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute coverage mask for each k-subset
    # There are C(24,7) = 346104 k-subsets. This might be too many to precompute all masks.
    # Instead, we compute masks on the fly or use a different strategy.
    
    # Alternative: Use numpy bitset of size total_t_subsets (2024 bits = 253 bytes)
    # For greedy, we need to find the k-subset that covers the most uncovered t-subsets.
    # We can iterate over all k-subsets and compute their intersection with uncovered.
    # But C(24,7) = 346104, and for each we'd need to compute the mask.
    
    # Better approach: Precompute masks for all k-subsets using numpy
    # Each mask is a numpy array of uint64 or use Python's int as a bitset (arbitrary precision)
    
    # Let's use Python integers as bitsets. total_t_subsets = 2024, so we need 2024 bits.
    # Python ints handle this fine.
    
    # Precompute mask for each k-subset
    # This is C(24,7) = 346104 masks, each with up to C(7,3) = 35 bits set.
    # Memory: 346104 * ~256 bytes = ~88 MB. This might be too much.
    
    # Let's use a different approach: greedy with on-the-fly computation
    # But the directive says to use bitset-accelerated greedy with precomputed coverage masks.
    
    # Let's try precomputing. 346104 * 256 bytes ≈ 88 MB. Should be OK.
    
    # Actually, let's use numpy for efficiency.
    # Represent each mask as a numpy array of uint64. 2024 bits = 32 uint64s.
    # 346104 * 32 * 8 bytes = ~88 MB. OK.
    
    num_words = (total_t_subsets + 63) // 64  # 2024 / 64 = 32 words
    
    # Precompute masks for all k-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # masks[i] is a numpy array of num_words uint64 values
    # This will take a while to precompute. Let's see if we can do it faster.
    
    # Alternative: Don't precompute all k-subset masks. Instead, for each iteration,
    # compute the best k-subset by considering candidates.
    
    # Let's try a hybrid: precompute masks for t-subsets to k-subsets mapping.
    # For each t-subset, which k-subsets contain it? C(v-t, k-t) = C(21, 4) = 5985.
    # Total: 2024 * 5985 ≈ 12 million entries. Too many.
    
    # Let's just precompute all k-subset masks. 346104 masks.
    # For each k-subset, iterate over its C(7,3) = 35 t-subsets and set bits.
    
    # Use a list of numpy arrays
    masks = np.zeros((num_k_subsets, num_words), dtype=np.uint64)
    
    for i, ksub in enumerate(all_k_subsets):
        for tsub in itertools.combinations(ksub, t):
            idx = t_subset_index[tsub]
            word_idx = idx // 64
            bit_idx = idx % 64
            masks[i, word_idx] |= (1 << bit_idx)
    
    # Greedy algorithm
    # uncovered: bitset of uncovered t-subsets, initially all 1s
    uncovered = np.ones(num_words, dtype=np.uint64)
    # Mask out bits beyond total_t_subsets
    if total_t_subsets % 64 != 0:
        uncovered[-1] &= (1 << (total_t_subsets % 64)) - 1
    
    blocks = []
    used_k_subsets = [False] * num_k_subsets
    
    while True:
        # Check if all covered
        if not np.any(uncovered):
            break
        
        # Find the k-subset with maximum coverage of uncovered t-subsets
        # Compute intersection: masks[i] & uncovered, then count bits
        # This is vectorized numpy operation
        intersections = masks & uncovered[np.newaxis, :]
        # Count bits in each row
        # Use a lookup table for popcount of uint64
        # Or use np.unpackbits for each word, but that's slow.
        
        # Let's use a simpler approach: for each candidate, compute the popcount
        # Vectorized: unpackbits works on the whole array
        # intersections shape: (346104, 32)
        # unpackbits: (346104, 32*8) = (346104, 256)
        unpacked = np.unpackbits(intersections, axis=1)
        popcounts = unpacked.sum(axis=1)
        
        best_idx = np.argmax(popcounts)
        best_count = popcounts[best_idx]
        
        if best_count == 0:
            break
        
        blocks.append(list(all_k_subsets[best_idx]))
        used_k_subsets[best_idx] = True
        
        # Update uncovered
        uncovered &= ~masks[best_idx]
    
    return blocks
