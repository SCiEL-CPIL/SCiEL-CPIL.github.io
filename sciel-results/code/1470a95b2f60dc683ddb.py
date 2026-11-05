"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
from functools import lru_cache

def construct():
    # We want a strictly increasing sequence a_1 < a_2 < ... < a_k in [1, 256]
    # such that all nonempty consecutive block sums are distinct.
    # 
    # Key insight: Let S = {sum of a_i..a_j for 1 <= i <= j <= k}.
    # We need |S| = k*(k+1)/2 and all elements of S distinct.
    #
    # The greedy approach in the seed program picks the smallest possible x at each step.
    # However, a smarter search might skip some early values to allow longer sequences.
    #
    # Let's implement a DFS with backtracking. To make it efficient, we'll use the fact that
    # the maximum value is 256, so the sequence length is bounded.
    #
    # First, let's estimate the maximum possible length.
    # The sum of all elements is at most 256*255/2 = 32640.
    # The largest consecutive block sum is the sum of all elements.
    # The smallest is a_1 >= 1.
    # All k*(k+1)/2 block sums must be distinct integers in [1, 32640].
    # So k*(k+1)/2 <= 32640 => k^2 + k - 65280 <= 0 => k <= ~255.
    # But also, the elements themselves are distinct and in [1,256], so k <= 256.
    # More tightly: if we pick the smallest k integers 1,2,...,k, the largest block sum is k(k+1)/2.
    # For k=20, that's 210. For k=30, that's 465. 
    # Actually, let's just run the search and see.
    
    # Let's try a DFS that builds the sequence from left to right.
    # State: current sequence, set of all consecutive block sums so far.
    # When we add a new element x at the end:
    #   - New block sums are: x, a_{n-1}+x, a_{n-2}+a_{n-1}+x, ..., sum(a)+x
    #   - We need none of these to be in the existing set of sums.
    
    # To optimize, we can use the fact that the sequence must be strictly increasing.
    # Let's try to find the longest possible sequence.
    
    best = []
    best_len = 0
    
    # Use DFS with backtracking
    def dfs(seq, sums, start_val):
        nonlocal best, best_len
        if len(seq) > best_len:
            best_len = len(seq)
            best = list(seq)
        
        # Try to extend the sequence
        # The next value must be > seq[-1] if seq is non-empty, else >= 1
        # And <= 256
        if seq:
            min_next = seq[-1] + 1
        else:
            min_next = 1
        
        if min_next > 256:
            return
        
        # Compute suffix sums of current seq
        # suffix_sums[i] = sum of seq[-i:] for i=1..len(seq)
        # When we add x, new sums are x, suffix_sums[1]+x, suffix_sums[2]+x, ..., suffix_sums[len(seq)]+x
        
        # To avoid recomputing, let's precompute suffix sums incrementally
        # Actually, let's just iterate over possible x values
        
        # Optimization: we can prune if even with the largest possible remaining values,
        # we can't beat the current best. But let's first see if simple DFS works.
        
        for x in range(min_next, 257):
            # Compute new consecutive block sums ending at x
            new_sums = set()
            total = x
            new_sums.add(total)
            for a in reversed(seq):
                total += a
                new_sums.add(total)
            
            # Check if any new sum conflicts with existing sums
            if new_sums & sums:
                continue
            
            # Also check that new_sums are all distinct (they should be by construction
            # since the sequence is strictly increasing, but let's verify)
            # Actually, if seq is strictly increasing, the suffix sums are strictly increasing,
            # so new_sums should be distinct. But let's double-check.
            if len(new_sums) != len(seq) + 1:
                continue
            
            # Recurse
            seq.append(x)
            sums.update(new_sums)
            dfs(seq, sums, x + 1)
            sums.difference_update(new_sums)
            seq.pop()
    
    dfs([], set(), 1)
    
    return best

# EVOLVE-BLOCK-END
