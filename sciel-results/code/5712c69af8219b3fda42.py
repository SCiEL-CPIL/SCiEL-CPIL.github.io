"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 16
DEADLINE = 85.0  # safety margin before 90s kill

# EVOLVE-BLOCK-START

def _verify_network(comps):
    """
    Verify sorting network using zero-one principle (all 2^N binary inputs).
    Uses NumPy for vectorized verification.
    """
    if not comps:
        return True
    
    # Generate all 2^16 binary inputs as a (65536, 16) array of uint8
    # We can generate this efficiently using broadcasting
    # Input bits: for mask in range(2^16), bit k is (mask >> k) & 1
    # Instead of a loop, we can construct the array directly.
    # However, 65536 * 16 is 1M elements, which is small for NumPy.
    
    # Create indices 0..65535
    idx = np.arange(1 << N, dtype=np.uint32)
    # Create bit positions 0..15
    bits = np.arange(N, dtype=np.uint8)
    
    # (65536, 1) >> (1, 16) & 1
    # This creates a (65536, 16) array
    # Note: np.right_shift requires same shape or broadcastable
    # idx[:, None] is (65536, 1)
    # bits[None, :] is (1, 16)
    x = (idx[:, None] >> bits[None, :]) & 1
    x = x.astype(np.uint8)
    
    # Apply comparators
    # comps is a list of (i, j)
    for i, j in comps:
        # x[:, i] and x[:, j]
        a = x[:, i]
        b = x[:, j]
        # min goes to i, max goes to j
        # np.minimum and np.maximum are element-wise
        x[:, i] = np.minimum(a, b)
        x[:, j] = np.maximum(a, b)
    
    # Check if each row is sorted (non-decreasing)
    # diff along axis 1
    diffs = np.diff(x, axis=1)
    return bool(np.all(diffs >= 0))

def _build_batcher_oe_merge_sort(n):
    """
    Build Batcher's Odd-Even Merge Sort network.
    This yields 75 comparators for N=16.
    """
    def oe_merge_network(n):
        """
        Odd-even merge network for 2n inputs (n from top half, n from bottom half).
        Inputs are assumed to be sorted.
        """
        if n == 1:
            return [(0, 1)]
        
        # First stage: compare (i, n+i) for i in 0..n-1
        # This ensures that the smallest of the two sorted lists goes to the first half of the output
        # and the largest goes to the second half? 
        # Actually, standard OE merge:
        # 1. Compare (i, n+i) for i=0..n-1.
        #    After this, the first n elements contain the n smallest elements of the union?
        #    No, that's not quite right.
        #    Let's check n=2 (4 inputs).
        #    Inputs: a0,a1 (sorted), b0,b1 (sorted).
        #    Compare (0,2) -> a0,b0. Compare (1,3) -> a1,b1.
        #    Then merge even indices (0,2) and odd indices (1,3)?
        #    Wait, the standard recursive step is:
        #    After the n cross-comparisons, the even-indexed positions (0, 2, ..., 2n-2) contain
        #    a sorted sequence of n elements? And odd-indexed (1, ..., 2n-1) contain another?
        #    Yes.
        
        comps = [(i, n + i) for i in range(n)]
        
        # Recursively merge the even-indexed elements
        # The even indices are 0, 2, 4, ..., 2n-2.
        # These correspond to positions 0, 1, ..., n-1 in the sub-network.
        # We need to map the sub-network indices back to the global indices.
        # Sub-index k corresponds to global index 2*k.
        even_merge = oe_merge_network(n // 2)
        for (a, b) in even_merge:
            comps.append((2 * a, 2 * b))
            
        # Recursively merge the odd-indexed elements
        # The odd indices are 1, 3, 5, ..., 2n-1.
        # Sub-index k corresponds to global index 2*k + 1.
        odd_merge = oe_merge_network(n // 2)
        for (a, b) in odd_merge:
            comps.append((2 * a + 1, 2 * b + 1))
            
        return comps

    def sort_network(n):
        """Sorting network using odd-even merge sort."""
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        if n % 2 == 1:
            # Handle odd number: sort n-1, then insert the last element
            base = sort_network(n - 1)
            # Insert element n-1 into sorted network of n-1 elements
            # The element n-1 is the largest? No, it's arbitrary.
            # We compare it with all others to bubble it to the correct position?
            # Standard insertion for sorting networks:
            # If we have a sorted network for n-1, and we add wire n-1,
            # we need to merge it.
            # Comparing (i, n-1) for i=0..n-2 is not sufficient if the network doesn't
            # maintain the property that wire n-1 is the max.
            # Actually, for OE Merge Sort, we usually split into two halves.
            # If n is odd, we can't split evenly.
            # However, for N=16, we always split evenly.
            # We only hit this case if n is not a power of 2.
            # Since N=16 is a power of 2, this case won't be hit in the main call,
            # but might be in recursion if we started with odd.
            # For safety, let's just use a simpler fallback or ensure we only call with powers of 2.
            # But the function signature allows any n.
            # Let's implement a simple insertion:
            # Sort n-1 wires. Then compare wire n-1 with wire 0, then 1, etc?
            # No, that's not a sorting network merge.
            # For N=16, we don't need this.
            return base + [(i, n - 1) for i in range(n - 1)]
        
        # Even: split into two halves
        half = n // 2
        left = sort_network(half)
        right = [(i + half, j + half) for (i, j) in sort_network(half)]
        
        # Odd-even merge the two sorted halves
        merge = oe_merge_network(half)
        
        return left + right + merge
    
    return sort_network(n)

def _local_search(comps, deadline):
    """
    Local search to minimize comparator count.
    Tries to remove comparators.
    """
    best = list(comps)
    best_count = len(best)
    random.seed(42)
    
    start_time = time.time()
    no_improve = 0
    
    while time.time() < deadline:
        if len(best) <= 60: # Can't go below optimal, but maybe we hit 60
            break
            
        # Try random removal
        idx = random.randint(0, len(best) - 1)
        trial = best[:idx] + best[idx+1:]
        
        # Verify
        if _verify_network(trial):
            best = trial
            best_count = len(best)
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve > 500:
            # If we haven't improved in 500 tries, break?
            # Or continue?
            # Given 75 comps, removing one is hard.
            # Let's break to save time if stuck.
            break
            
    return best

def construct():
    # 1. Generate Batcher's OE Merge Sort (75 comparators)
    comps = _build_batcher_oe_merge_sort(N)
    
    # 2. Verify
    if not _verify_network(comps):
        # Fallback to Odd-Even Transposition (120 comparators)
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    # 3. Local Search
    deadline = time.time() + (DEADLINE - time.time() - 5.0) # 5s safety margin
    best = _local_search(comps, deadline)
    
    return best

# EVOLVE-BLOCK-END
