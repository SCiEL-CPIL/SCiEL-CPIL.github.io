"""
Strategy:
1. Constructive Generation: Generate Batcher's Odd-Even Merge Sort network for N=16. 
   This is known to be efficient (63 comparators for N=16, often reducible).
2. Deterministic Structural Reduction: 
   - The zero-one principle allows us to verify sorting.
   - We exploit the recursive structure of Batcher's network.
   - Instead of random removal, we systematically test for redundancy in a specific order:
     a. Comparators that are "far apart" in the initial stages (often redundant if subsequent stages handle the sorting).
     b. Comparators in the "merge" stages that are duplicated by the structure of the merge itself.
   - Specifically, we use a "sweep" approach: 
     For each comparator in the network (from end to start), check if removing it preserves validity.
     To speed this up, we don't check all 65536 inputs immediately. We use a "critical set" of inputs 
     derived from the specific comparator's position. However, for N=16, full vectorized check is fast enough 
     if done sparingly.
3. Targeted Local Search: 
   - If the deterministic sweep yields no more improvements, we switch to a local search.
   - But unlike the previous random SA, this local search is "informed":
     - It prioritizes removing comparators that have the highest "redundancy score" (e.g., those that 
       compare wires that are already likely sorted due to previous operations).
     - It uses a "batch" evaluation: try removing a set of k comparators at once (k=1,2) to escape 
       local minima that single removals get stuck in.

Key Difference from Previous:
- Previous: Random SA/Local Search from Batcher.
- Current: Deterministic Structural Pruning (exploiting Batcher's recursion) + Informed Local Search.
- The deterministic phase is expected to quickly strip the network from 63 to ~55-58 without random noise.
"""

import time
import numpy as np

N = 16
MAX_TIME = 58.0 # Safety margin for 60s limit

def _generate_batcher_network():
    """
    Generates Batcher's Odd-Even Merge Sort network for N=16.
    """
    def merge(n, offset):
        if n == 1:
            return [(offset, offset + 1)]
        
        h = n // 2
        # Step 1: Compare elements i and i+n
        step1 = [(offset + i, offset + i + n) for i in range(n)]
        
        def strided_merge(n, offset, stride):
            if n == 1:
                return [(offset, offset + stride)]
            
            h = n // 2
            # Compare elements i*stride and (n+i)*stride
            step1 = [(offset + i * stride, offset + (n + i) * stride) for i in range(n)]
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

def _apply_network_vectorized(comps, inputs):
    """
    Apply the network to a batch of inputs.
    inputs: shape (M, N)
    comps: list of (i, j)
    """
    vec = inputs.copy()
    for i, j in comps:
        min_val = np.minimum(vec[:, i], vec[:, j])
        max_val = np.maximum(vec[:, i], vec[:, j])
        vec[:, i] = min_val
        vec[:, j] = max_val
    return vec

def _is_sorted_network_vectorized(comps, bits):
    """
    Check if the network sorts all 2^N boolean inputs.
    bits: shape (2^N, N)
    """
    if not comps:
        return np.all(np.diff(bits, axis=1) >= 0) # Trivial check for empty
    out = _apply_network_vectorized(comps, bits)
    diffs = np.diff(out, axis=1)
    return np.all(diffs >= 0)

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    # Pre-compute all boolean inputs
    n_inputs = 1 << N
    indices = np.arange(n_inputs)
    # bits[i, j] is the j-th bit of i. 
    # We want wire 0 to be the LSB or MSB? 
    # Standard sorting network: input x_0 ... x_{N-1}.
    # Let's say wire k corresponds to bit k of the input index.
    # If we sort the bits, the output should be sorted.
    # bits = (indices[:, None] >> np.arange(N)) & 1
    # This makes wire 0 the LSB, wire N-1 the MSB.
    # Sorting the vector [b_0, ..., b_{N-1}] in ascending order means 0s then 1s.
    bits = (indices[:, None] >> np.arange(N)) & 1
    
    # 1. Generate Initial Network
    current_comps = _generate_batcher_network()
    
    # Verify initial network
    if not _is_sorted_network_vectorized(current_comps, bits):
        # Fallback to Odd-Even Transposition if Batcher gen is broken (shouldn't happen)
        current_comps = []
        for i in range(N):
            if i % 2 == 0:
                for j in range(0, N - 1, 2):
                    current_comps.append((j, j + 1))
            else:
                for j in range(1, N - 1, 2):
                    current_comps.append((j, j + 1))
    
    best_comps = current_comps[:]
    best_len = len(best_comps)
    
    rng = np.random.default_rng(42)
    all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Phase 1: Deterministic Structural Reduction
    # We iterate through the network and try to remove comparators.
    # Order: From the "end" of the network backwards, as later comparators are more likely 
    # to be redundant if the earlier ones have done most of the work.
    # Also, we prioritize comparators that are "far apart" in the first few layers.
    
    # Create a list of indices to try, sorted by a heuristic:
    # Heuristic: Comparators appearing later in the network are tried first for removal?
    # Actually, removing a comparator from the *beginning* is harder because it affects everything.
    # Removing a comparator from the *end* is easier.
    # Let's try removing from the end of the list first.
    
    max_passes = 10
    for _ in range(max_passes):
        if time.time() > deadline - 1.0:
            break
            
        removed_any = False
        # Iterate backwards
        for idx in range(len(current_comps) - 1, -1, -1):
            if time.time() > deadline - 1.0:
                break
                
            # Try to remove current_comps[idx]
            candidate = current_comps[:idx] + current_comps[idx+1:]
            
            # Fast check
            if _is_sorted_network_vectorized(candidate, bits):
                current_comps = candidate
                best_comps = current_comps[:]
                best_len = len(current_comps)
                removed_any = True
                # Since we removed one, the indices shift. 
                # We can continue iterating backwards, but the list is shorter.
                # To be safe and systematic, we restart the pass if we find a removal.
                # However, restarting is expensive. 
                # Let's just break and restart the outer loop to re-evaluate the new structure.
                break
        
        if not removed_any:
            # No more single removals possible in this pass
            break
            
    # Phase 2: Informed Local Search
    # If we are stuck, we try "swaps" or "double removals" to escape local minima.
    # A "double removal" tries to remove two comparators at once.
    
    # Pre-compute a "redundancy score" for each comparator?
    # For now, we use random sampling but biased towards "safe" moves.
    
    while time.time() < deadline - 1.0:
        improved = False
        
        # Move 1: Random Single Removal (with restart)
        if len(current_comps) > 1:
            idx = rng.integers(0, len(current_comps))
            candidate = current_comps[:idx] + current_comps[idx+1:]
            if _is_sorted_network_vectorized(candidate, bits):
                current_comps = candidate
                best_comps = current_comps[:]
                best_len = len(current_comps)
                improved = True
        
        # Move 2: Random Replacement (Swap)
        if not improved and len(current_comps) > 1:
            idx = rng.integers(0, len(current_comps))
            old_comp = current_comps[idx]
            # Pick a random new comparator
            new_comp = all_comps[rng.integers(0, len(all_comps))]
            
            if new_comp != old_comp:
                candidate = current_comps[:]
                candidate[idx] = new_comp
                if _is_sorted_network_vectorized(candidate, bits):
                    current_comps = candidate
                    best_comps = current_comps[:]
                    # best_len unchanged
                    improved = True
        
        # Move 3: Double Removal (Try to remove 2 at once)
        # This helps if two comparators are jointly redundant but individually critical.
        if not improved and len(current_comps) > 2:
            # Pick two distinct indices
            idxs = rng.choice(len(current_comps), size=2, replace=False)
            i1, i2 = sorted(idxs)
            
            candidate = current_comps[:]
            # Remove i2 first (higher index) to keep i1 valid
            del candidate[i2]
            del candidate[i1]
            
            if _is_sorted_network_vectorized(candidate, bits):
                current_comps = candidate
                best_comps = current_comps[:]
                best_len = len(current_comps)
                improved = True
        
        # If no improvement for a long time, we might insert a comparator to jump out?
        # Given the time limit, we stick to reduction/swap.
        
        # To prevent infinite loops on the same random choices, we rely on the RNG.
        # If we want to be more aggressive, we could add a "restart" from best_comps 
        # with a perturbation.
        
        if not improved and rng.random() < 0.01:
            # Occasional full restart from best with random perturbation
            current_comps = best_comps[:]
            # Perturb: swap a random comparator with another
            if len(current_comps) > 1:
                idx = rng.integers(0, len(current_comps))
                new_comp = all_comps[rng.integers(0, len(all_comps))]
                if new_comp != current_comps[idx]:
                    current_comps[idx] = new_comp

    return best_comps

# EVOLVE-BLOCK-END
