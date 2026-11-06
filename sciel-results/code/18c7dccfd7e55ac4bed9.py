import time
import numpy as np

N = 16
MAX_TIME = 55.0

def _generate_batcher_network():
    """
    Generates Batcher's Odd-Even Merge Sort network for N=16.
    Returns a flat list of comparators.
    """
    def merge(n, offset):
        if n == 1:
            return [(offset, offset + 1)]
        
        h = n // 2
        # The first step of merging two sorted lists of size n
        # compares element i of the first list with element i of the second list
        # Wait, standard OES merge:
        # 1. Compare (offset+i, offset+i+n) for i in 0..n-1
        # 2. Recursively merge the even indices and odd indices.
        
        step1 = [(offset + i, offset + i + n) for i in range(n)]
        
        def strided_merge(n, offset, stride):
            if n == 1:
                return [(offset, offset + stride)]
            
            h = n // 2
            # Compare strided elements
            step1 = [(offset + i * stride, offset + (n + i) * stride) for i in range(n)]
            
            # Recurse on even and odd positions
            even = strided_merge(h, offset, 2 * stride)
            odd = strided_merge(h, offset + stride, 2 * stride)
            
            return step1 + even + odd
        
        even_merge = strided_merge(h, offset, 2)
        odd_merge = strided_merge(h, offset + 1, 2)
        
        return step1 + even_merge + odd_merge

    def sort(n, offset):
        if n <= 1:
            return []
        h = n // 2
        left = sort(h, offset)
        right = sort(h, offset + h)
        m = merge(h, offset)
        return left + right + m

    return sort(N, 0)

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    # Pre-compute all boolean inputs as a matrix of 0s and 1s
    # Shape: (65536, 16)
    # Input k is represented by its binary digits.
    n_inputs = 1 << N
    indices = np.arange(n_inputs)
    # We want bit i of k to be at wire i.
    # If we use standard bit order, wire 0 is LSB.
    # The problem doesn't specify wire order vs bit order, just that inputs are all 2^16 boolean sequences.
    # So we can map input k to vector v where v[i] = (k >> i) & 1.
    bits = (indices[:, None] >> np.arange(N)) & 1
    
    # State: For each input, we track the current state of the 16 wires.
    # Since we process all inputs in parallel, state is a (65536, 16) matrix of 0/1.
    # However, for the "unsorted" check, we need to know if the output is sorted.
    # A vector is sorted if v[0] <= v[1] <= ... <= v[15].
    # For boolean vectors, this means all 0s come before all 1s.
    # So v is sorted if there is no pair (i, j) with i < j such that v[i] == 1 and v[j] == 0.
    # Equivalently, if we define the "inversion" mask for a vector, it's sorted if inversions == 0.
    
    # To speed up the "is sorted" check and the "effect of comparator" calculation:
    # We can track, for each input, the set of "bad" pairs (i, j) where i < j and v[i] > v[j].
    # If a comparator (a, b) with a < b is applied:
    # - If v[a] == 0 and v[b] == 1: No change.
    # - If v[a] == 1 and v[b] == 0: Swap to 0, 1. This fixes the pair (a, b) and might fix other pairs?
    #   Actually, swapping 1 and 0 to 0 and 1:
    #   - Pairs involving a: (i, a) for i < a. v[i] > v[a]=0 is always false (since v[i] is 0 or 1, and if v[i]=1, it was bad before, now v[a]=0 so v[i]>v[a] is true -> wait.
    #   Let's trace carefully.
    #   Before: v[a]=1, v[b]=0.
    #   After: v'[a]=0, v'[b]=1.
    #   Consider pair (i, a) with i < a.
    #   Before: v[i] > 1? Impossible. So (i, a) was not bad.
    #   After: v[i] > 0? If v[i]=1, then (i, a) becomes bad.
    #   So fixing (a, b) might create new inversions with (i, a) and (b, j).
    #   
    # This makes tracking "unsorted inputs" via a simple count difficult because one move can break others.
    # However, we can simply re-evaluate the sortedness of all inputs after each step.
    # With 65536 inputs and ~100 comparators, re-evaluating is 65536 * 16 * 100 ops ~ 100M ops.
    # In NumPy, this is very fast (milliseconds).
    # So we can afford to re-check the whole state after each comparator addition.
    
    # Strategy:
    # 1. Start with empty network. State = Identity (unsorted for most inputs).
    # 2. Greedily add comparators.
    #    At each step, for every possible comparator (i, j), calculate the new state if applied.
    #    Count how many inputs become sorted (or reduce the total inversion count).
    #    Pick the best comparator.
    #    Stop when all inputs are sorted or time limit.
    # 3. This is a set-cover-like greedy. It might get stuck, but usually finds a valid network quickly.
    #    To ensure we find a *small* network, we can try to remove comparators afterwards.
    
    # Actually, a better metric than "number of sorted inputs" is "total inversions across all inputs".
    # Let's track the total inversions.
    # Inversions for input k: sum_{i<j} [v_k[i] > v_k[j]].
    # We want to minimize sum_k Inversions(k).
    
    # Pre-compute all possible comparators
    all_comps = [(i, j) for i in range(N) for j in range(i + 1, N)]
    # We can vectorize the inversion calculation.
    
    # Initial state: Identity
    # v_k[i] = bit i of k.
    # Current state matrix: (65536, 16)
    state = bits.copy()
    
    # Function to compute total inversions for current state
    def total_inversions(state):
        # state: (M, N)
        # Inversion (i, j) exists if state[:, i] > state[:, j]
        # Since state is 0/1, this is state[:, i] & ~state[:, j]
        # Sum over all i < j
        invs = 0
        for i in range(N):
            for j in range(i + 1, N):
                # Count how many inputs have 1 at i and 0 at j
                invs += np.sum(state[:, i] & ~state[:, j])
        return invs
    
    # Greedy Construction
    current_comps = []
    current_inv = total_inversions(state)
    
    # We will run a greedy loop.
    # To make it faster, we can approximate the gain of a comparator without applying it to the whole state?
    # Gain of (i, j):
    # It fixes inversions where i is 1 and j is 0.
    # It creates inversions where k < i and state[k]=1 (now state[i]=0, so 1>0 is inversion) 
    #   Wait, if state[i] becomes 0, then for any k < i with state[k]=1, (k, i) becomes an inversion.
    #   Previously (k, i) was not an inversion because state[i]=1 (1>1 is false).
    # Similarly for j: if state[j] becomes 1, then for any l > j with state[l]=0, (j, l) becomes an inversion.
    #   Previously (j, l) was not an inversion because state[j]=0 (0>0 is false).
    
    # Let's compute the gain for all comparators in one go?
    # For each (i, j):
    #   Fixes: count of inputs where state[i]=1, state[j]=0.
    #   Breaks_left: count of inputs where state[i]=1, state[j]=0 AND (exists k < i with state[k]=1).
    #   This is getting complicated to vectorize perfectly without iterating i, j.
    # Given N=16, iterating 120 comparators * 65536 inputs is fine if vectorized per comparator.
    
    while time.time() < deadline - 5.0:
        best_gain = -1
        best_comp = None
        
        # Calculate gain for each possible comparator
        # We only consider comparators not already in the network? 
        # Actually, adding a duplicate comparator is useless.
        # We can just check all 120.
        
        for i, j in all_comps:
            if (i, j) in current_comps:
                continue
            
            # Count fixes: state[i]=1, state[j]=0
            mask_fix = (state[:, i] == 1) & (state[:, j] == 0)
            n_fixes = np.sum(mask_fix)
            
            if n_fixes == 0:
                continue
            
            # Count breaks
            # Breaks at i: inputs where mask_fix is true, AND any k < i has state[k]=1
            # Breaks at j: inputs where mask_fix is true, AND any l > j has state[l]=0
            
            # Vectorized check for "any k < i has state[k]=1"
            # We can precompute prefix sums of 1s? Or just sum.
            # Since N is small, let's just sum.
            
            # For each input in mask_fix, check if it has a 1 in [0, i-1]
            # And check if it has a 0 in [j+1, 15]
            
            # This is O(M) per comparator. 120 * 65536 ~ 8M ops. Very fast.
            
            # Let's do it efficiently.
            # left_ones: (M, i) boolean matrix? No, just sum.
            # For mask_fix inputs:
            #   has_left_one = np.any(state[:, :i][mask_fix] == 1, axis=1)
            #   has_right_zero = np.any(state[:, j+1:][mask_fix] == 0, axis=1)
            
            # Note: If i=0, no left ones. If j=15, no right zeros.
            
            if i > 0:
                left_ones = np.any(state[:, :i][mask_fix] == 1, axis=1)
            else:
                left_ones = np.zeros(np.sum(mask_fix), dtype=bool)
                
            if j < N - 1:
                right_zeros = np.any(state[:, j+1:][mask_fix] == 0, axis=1)
            else:
                right_zeros = np.zeros(np.sum(mask_fix), dtype=bool)
                
            n_breaks = np.sum(left_ones | right_zeros)
            
            gain = n_fixes - n_breaks
            
            if gain > best_gain:
                best_gain = gain
                best_comp = (i, j)
                
        if best_comp is None or best_gain <= 0:
            break
            
        # Apply best_comp
        i, j = best_comp
        # Swap 1,0 to 0,1
        mask_swap = (state[:, i] == 1) & (state[:, j] == 0)
        state[:, i] = np.where(mask_swap, 0, state[:, i])
        state[:, j] = np.where(mask_swap, 1, state[:, j])
        
        current_comps.append(best_comp)
        current_inv -= best_gain
        
        if current_inv == 0:
            break
            
    # Now we have a valid network (if current_inv == 0) or a partial one.
    # If not valid, we might need to fall back to Batcher or continue with local search.
    # Let's check validity.
    # For boolean inputs, inv=0 means sorted.
    
    # To be safe, if we didn't reach 0 inversions, we should ensure we return a valid network.
    # The greedy might get stuck. 
    # Strategy: If stuck, append a Batcher network? No, that's huge.
    # Better: If stuck, use the Batcher network as a fallback.
    
    # But usually, greedy set cover for sorting networks works well.
    # Let's verify.
    # If not valid, we can try to finish with a local search on the current partial network?
    # Or just return Batcher if this fails.
    
    # Let's check if current_comps is valid.
    # We can just check current_inv == 0.
    
    if current_inv > 0:
        # Fallback: Use Batcher
        best_comps = _generate_batcher_network()
    else:
        best_comps = current_comps
        
    # Now, perform a reduction phase (local search for removals)
    # This is where we can improve on the greedy result.
    # The greedy result might have redundant comparators.
    
    # We need a fast validity check for the reduction phase.
    # We can use the vectorized check on the original bits.
    
    def _is_valid(comps):
        # Apply network to all inputs
        s = bits.copy()
        for i, j in comps:
            min_val = np.minimum(s[:, i], s[:, j])
            max_val = np.maximum(s[:, i], s[:, j])
            s[:, i] = min_val
            s[:, j] = max_val
        # Check sorted
        # diffs >= 0
        diffs = np.diff(s, axis=1)
        return np.all(diffs >= 0)
        
    # Verify best_comps
    if not _is_valid(best_comps):
        # Something went wrong, return a known valid one (Batcher)
        return _generate_batcher_network()
        
    # Local Search Reduction
    rng = np.random.default_rng(42)
    
    while time.time() < deadline - 1.0:
        improved = False
        # Try to remove a random comparator
        if len(best_comps) > 1:
            idx = rng.integers(0, len(best_comps))
            removed = best_comps.pop(idx)
            if _is_valid(best_comps):
                improved = True
            else:
                best_comps.insert(idx, removed)
                
        # If stuck, try a swap?
        # For now, just deletion.
        if not improved and time.time() > deadline - 2.0:
             break
             
    return best_comps

# EVOLVE-BLOCK-END
