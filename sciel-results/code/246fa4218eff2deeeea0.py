import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
DEADLINE = 55.0  # Safety margin before 60s kill

# Precompute full matrix of all 2^N boolean inputs
# Each row is a distinct input vector
all_masks = np.arange(1 << N, dtype=np.int32)
W0_full = np.zeros((1 << N, N), dtype=np.uint8)
for i in range(N):
    W0_full[:, i] = (all_masks >> i) & 1

# Precompute a subset for fast rejection
# Use a mix of low, high, and random inputs to catch errors early
subset_indices = np.unique(np.concatenate([
    np.arange(0, 200),       # Low weights
    np.arange((1 << N) - 200, 1 << N), # High weights
    np.random.choice(1 << N, 2000, replace=False)
]))
W0_subset = W0_full[subset_indices]

def _batchers_merge_sort(n):
    """
    Generate a Batcher's Merge Sort network for n wires.
    Returns a list of (i, j) comparators.
    """
    if n <= 1:
        return []
    
    # Recursive step: split into two halves
    half = n // 2
    left = _batchers_merge_sort(half)
    right = _batchers_merge_sort(n - half)
    
    # Shift right comparators
    right_shifted = [(i + half, j + half) for i, j in right]
    
    # Merge step
    # The merge step for Batcher's odd-even mergesort
    # It performs log(n) stages of odd-even transposition
    # But specifically for merge, we interleave the two sorted lists
    # Standard Batcher merge:
    # Compare (0, n), (1, n-1), ..., then (0, n-1), (1, n-2), etc.
    
    # Actually, let's use the standard recursive definition of Batcher's Merge Sort
    # 1. Sort first half
    # 2. Sort second half
    # 3. Merge the two sorted halves
    
    # The merge function for Batcher's odd-even merge
    def merge(left_size, right_size, offset=0):
        """
        Generate comparators to merge two sorted sequences of sizes left_size and right_size.
        offset is the starting index in the global array.
        """
        if left_size == 0 or right_size == 0:
            return []
        
        comparators = []
        # The merge algorithm compares (i, i + left_size + right_size - 1 - i) ?
        # No, Batcher's merge is:
        # For k in range(log2(left_size + right_size)):
        #   For i in range(left_size + right_size):
        #     if i % 2 == (k % 2):
        #       compare (i, i + 1) -- wait, this is odd-even transposition.
        
        # Batcher's Merge Sort uses a specific merge pattern.
        # Let's use the standard recursive construction:
        # merge(A, B) where A and B are sorted.
        # If len(A) < len(B), swap.
        # If len(A) == 1:
        #   Compare (A[0], B[0]), (A[0], B[1]), ...
        # Else:
        #   Split A into A1, A2
        #   merge(A1, B)
        #   merge(A2, B)
        #   merge(A1, B) -- wait, this is getting complicated.
        
        # Simpler: Use the known structure.
        # For N=18, we can just use a generic sorting network generator.
        # However, implementing Batcher correctly is tricky.
        # Let's use a simpler, known efficient network: Shell's network or just use a standard library if available? No.
        
        # Let's implement a simple Batcher's Merge Sort correctly.
        # Reference: 
        # def batcher(n):
        #   if n == 1: return []
        #   m = n // 2
        #   left = batcher(m)
        #   right = batcher(n - m)
        #   return left + [(i+m, j+m) for i,j in right] + merge(m, n-m)
        
        # The merge step for Batcher's odd-even merge:
        # It is equivalent to log2(n) stages of odd-even transposition.
        # But we only need to merge two sorted lists.
        
        # Let's use a simpler approach: Odd-Even Transposition is O(N^2).
        # Batcher is O(N log^2 N).
        # For N=18, OET is 153. Batcher is ~90.
        
        # Let's implement Batcher's Merge Sort properly.
        pass

    # Since implementing Batcher from scratch in this short time is error-prone,
    # and the current OET seed is 153, let's try a different known good seed.
    # A common approach is to use a "seed" network that is known to be good.
    # However, the prompt says "seed odd-even transposition network uses 153".
    # The inspiration used OET and got 1.26.
    # Let's stick with OET but optimize the search heavily.
    # OR, let's try to generate a Batcher network correctly.
    
    # Correct Batcher's Merge Sort:
    def _batchers(n):
        if n <= 1:
            return []
        m = n // 2
        left = _batchers(m)
        right = _batchers(n - m)
        right_shifted = [(i + m, j + m) for i, j in right]
        
        # Merge step
        # The merge step in Batcher's algorithm is an odd-even merge.
        # It consists of log2(n) stages.
        # Stage k (0-indexed):
        #   Compare pairs (i, i + 2^k) for i in range(n - 2^k)
        #   But only if the parity matches?
        # Actually, Batcher's merge is:
        # For k = 0 to log2(n) - 1:
        #   For i = 0 to n - 1:
        #     if (i % 2) == (k % 2):
        #       compare (i, i + 1) -- No, this is OET.
        
        # Let's use the standard Batcher merge pattern:
        # It is a sequence of comparators that merges two sorted lists.
        # The number of comparators in the merge is roughly n log n.
        
        # Let's just use OET as the seed, as it is guaranteed to be correct and simple.
        # The key is the optimization.
        return left + right_shifted + _merge(m, n - m)
        
    def _merge(left_size, right_size, offset=0):
        """
        Generate comparators to merge two sorted sequences of sizes left_size and right_size.
        """
        if left_size == 0 or right_size == 0:
            return []
        
        comparators = []
        # Batcher's merge is recursive:
        # 1. If left_size < right_size, swap them.
        # 2. If left_size == 1:
        #    Compare (offset, offset + right_size - 1), (offset, offset + right_size - 2), ...
        #    Actually, if left is size 1, we just compare it with all of right in reverse?
        #    No, Batcher's merge for size 1 and m is:
        #    Compare (0, m-1), (0, m-2), ..., (0, 0)? No.
        
        # Let's use a simpler, robust method:
        # We can just use the OET network for the merge step? No, that's O(N^2).
        
        # Given the time constraint, I will use OET for the initial network.
        # It is 153 comparators.
        # The optimization will reduce it significantly.
        return []

    # Fallback to OET if Batcher implementation is too complex
    return [(i, i + 1) for phase in range(n) for i in range(phase % 2, n - 1, 2)]

def _odd_even_transposition(n):
    """Generate the odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(n) for i in range(phase % 2, n - 1, 2)]

def _verify_subset(net):
    """
    Verify network on the subset of inputs.
    Returns True if sorted, False otherwise.
    """
    if not net:
        return True
        
    W = W0_subset.copy()
    for i, j in net:
        col_i = W[:, i]
        col_j = W[:, j]
        # min = AND, max = OR for bits
        tmp = col_i | col_j
        col_i[:] = col_i & col_j
        col_j[:] = tmp
        
    # Check if sorted: W[:, k] <= W[:, k+1] for all k
    # i.e., no 1 before 0
    # W[:, :-1] > W[:, 1:] should be all False
    if np.any(W[:, :-1] > W[:, 1:]):
        return False
    return True

def _verify_full(net):
    """
    Verify network on all inputs.
    Returns True if sorted, False otherwise.
    """
    if not net:
        return True
        
    W = W0_full.copy()
    for i, j in net:
        col_i = W[:, i]
        col_j = W[:, j]
        tmp = col_i | col_j
        col_i[:] = col_i & col_j
        col_j[:] = tmp
        
    if np.any(W[:, :-1] > W[:, 1:]):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Start with Odd-Even Transposition network
    net = _odd_even_transposition(N)
    best_net = net[:]
    best_count = len(net)
    
    # Phase 1: Greedy Removal
    changed = True
    while changed:
        if time.time() > deadline - 5:
            break
        changed = False
        
        # Try removing from end to start
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline - 5:
                return best_net
            
            new_net = net[:idx] + net[idx+1:]
            
            # Quick subset check
            if _verify_subset(new_net):
                # Full check
                if _verify_full(new_net):
                    net = new_net
                    best_net = net[:]
                    best_count = len(net)
                    changed = True
                    break
        
        # If no change from end-to-start, try start-to-end
        if not changed:
            for idx in range(len(net)):
                if time.time() > deadline - 5:
                    return best_net
                
                new_net = net[:idx] + net[idx+1:]
                
                if _verify_subset(new_net):
                    if _verify_full(new_net):
                        net = new_net
                        best_net = net[:]
                        best_count = len(net)
                        changed = True
                        break
    
    # Phase 2: Simulated Annealing
    if time.time() < deadline - 3:
        T = 1.0
        min_T = 0.05
        cooling = 0.9995
        current_count = len(net)
        
        while time.time() < deadline - 2:
            if T < min_T:
                # Restart from best
                net = best_net[:]
                current_count = best_count
                T = 1.0
                
            # Generate move
            r = random.random()
            if r < 0.4:
                # Remove
                if len(net) == 0:
                    break
                idx = random.randint(0, len(net) - 1)
                new_net = net[:idx] + net[idx+1:]
            elif r < 0.7:
                # Insert
                idx = random.randint(0, len(net))
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                new_net = net[:idx] + [(i, j)] + net[idx:]
            else:
                # Mutate
                if len(net) == 0:
                    continue
                idx = random.randint(0, len(net) - 1)
                i, j = net[idx]
                ni = random.randint(0, N - 2)
                nj = random.randint(ni + 1, N - 1)
                if (ni, nj) == (i, j):
                    continue
                new_net = net[:]
                new_net[idx] = (ni, nj)
            
            # Verify
            if not _verify_subset(new_net):
                continue
            if not _verify_full(new_net):
                continue
                
            new_count = len(new_net)
            delta = new_count - current_count
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                net = new_net
                current_count = new_count
                if current_count < best_count:
                    best_net = net[:]
                    best_count = current_count
                    
            T *= cooling
    
    # Final validation
    if not _verify_full(best_net):
        best_net = _odd_even_transposition(N)
        
    return best_net
