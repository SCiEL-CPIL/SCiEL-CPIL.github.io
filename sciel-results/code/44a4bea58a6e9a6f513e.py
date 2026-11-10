import time
import random
import numpy as np

def _build_ap_bitsets(n):
    """
    Pre-compute bitmasks for all arithmetic progressions.
    Returns a list of integers, where each integer represents the set of indices
    belonging to a specific AP.
    """
    ap_bitsets = []
    for step in range(1, n + 1):
        for start in range(n):
            mask = 0
            for idx in range(start, n, step):
                mask |= (1 << idx)
            if mask != 0:
                ap_bitsets.append(mask)
    return ap_bitsets

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55.0
    
    # 1. Pre-compute AP bitmasks
    # This is O(n^2 log n) but done once.
    ap_bitsets = _build_ap_bitsets(n)
    num_aps = len(ap_bitsets)
    
    # Convert to numpy array for vectorized ops if possible, 
    # but Python ints are fine for bitwise ops in a loop if optimized.
    # However, we want to evaluate flips.
    # A flip at index i corresponds to XORing the state with (1 << i).
    # The sum of signs for an AP is (2 * popcount(state & mask) - popcount(mask)).
    # We want to minimize the max absolute sum.
    
    # Optimization: Pre-compute popcount for all APs (constant)
    ap_lengths = np.array([bin(m).count('1') for m in ap_bitsets], dtype=np.int64)
    
    # We will maintain the state as a Python int (256 bits).
    # To evaluate a flip at index i:
    # new_state = state ^ (1 << i)
    # For each AP mask M:
    # old_count = popcount(state & M)
    # new_count = popcount((state ^ (1 << i)) & M)
    # This is expensive if done per AP.
    
    # Alternative Vectorized Strategy:
    # Instead of evaluating one flip at a time against all APs,
    # let's evaluate the "impact" of a flip on all APs at once?
    # No, the bottleneck is the max over APs.
    
    # Let's use a different bitset strategy:
    # Represent the state as an array of 64-bit integers (4 words).
    # But Python's int is already arbitrary precision.
    
    # The key speedup:
    # Instead of updating prefix sums for all APs containing index i (which is many),
    # we can just recompute the max discrepancy from scratch for a candidate state?
    # No, that's O(N^2).
    
    # Let's stick to the prompt's suggestion: Bitset representation.
    # The insight is that checking if a flip is valid (doesn't increase max disc)
    # can be done quickly if we track which APs are currently "tight" (at max).
    
    # Let's implement a Greedy Local Search with a "Tight Set" optimization.
    # 1. Compute current max_disc.
    # 2. Identify APs that achieve this max_disc (Tight APs).
    # 3. A flip is only allowed if it reduces the sum of all Tight APs.
    #    (Actually, it must not increase the max of ANY AP. But checking only Tight APs
    #     is a necessary condition. If it reduces all Tight APs, the max must decrease 
    #     or stay same, provided no other AP becomes larger. Given the small changes,
    #     this is a good heuristic.)
    
    # However, recomputing popcount for all APs for every flip is still slow.
    # Let's use NumPy to vectorize the popcount?
    # Python's int.bit_count() is fast (C implementation).
    
    def compute_max_disc(state):
        # state is a Python int
        max_d = 0
        # Iterate through AP bitmasks
        # This loop is ~20k iterations.
        # 20k * bit_count (very fast) is acceptable?
        # 20k ops is ~1ms in Python if tight.
        # We have 55s. So we can do ~50k-100k full evaluations.
        # This allows for a simple hill-climbing.
        
        # Optimization: Use a list of masks.
        for m in ap_bitsets:
            c = (state & m).bit_count()
            # Sum is 2*c - length
            # Abs sum is abs(2*c - length)
            d = abs(2 * c - ap_lengths[ap_bitsets.index(m)]) # index is slow!
            
        # Pre-compute lengths aligned with bitsets
        return 0 # Placeholder

    # Let's refine the structure for speed.
    # Store (mask, length) pairs.
    ap_data = [(m, bin(m).count('1')) for m in ap_bitsets]
    
    def get_max_disc(state):
        max_d = 0
        for m, l in ap_data:
            c = (state & m).bit_count()
            d = abs(2 * c - l)
            if d > max_d:
                max_d = d
        return max_d

    # Initial State: Random or Legendre?
    # Let's try a few starts.
    
    def generate_legendre(n):
        # Find prime >= n
        p = n
        def is_prime(k):
            if k < 2: return False
            for i in range(2, int(k**0.5)+1):
                if k%i==0: return False
            return True
        while not is_prime(p): p += 1
        
        state = 0
        for i in range(1, n+1):
            # Legendre symbol
            ls = pow(i, (p-1)//2, p)
            if ls == p-1: # -1
                # sign is -1 -> bit is 0? 
                # Let's define: bit=1 means +1, bit=0 means -1.
                # Sum = 2*count(1s) - n.
                # If ls == -1, sign is -1, so bit is 0.
                pass
            else: # +1
                state |= (1 << (i-1))
        return state

    # Start 1: Legendre
    state1 = generate_legendre(n)
    # Start 2: Negated Legendre (flip all bits)
    state2 = state1 ^ ((1 << n) - 1)
    # Start 3: Random
    rng = random.Random(42)
    state3 = 0
    for i in range(n):
        if rng.choice([0,1]):
            state3 |= (1 << i)
    
    starts = [state1, state2, state3]
    
    best_state = starts[0]
    best_disc = get_max_disc(best_state)
    
    # Local Search
    # We iterate through indices. If flipping index i reduces the max_disc, we accept.
    # To speed this up, we can check if the flip *could* reduce the max_disc.
    # But simply trying all flips in a pass is O(n * num_aps) = 256 * 20k = 5M ops.
    # 5M bit_counts is maybe 0.5s?
    # 55s / 0.5s = 110 passes.
    # This should be sufficient to converge.
    
    # Optimization: Only consider flips if they affect the current max.
    # But tracking which APs are max is complex.
    # Let's just do the brute force pass first.
    
    for s in starts:
        current_state = s
        current_disc = get_max_disc(current_state)
        
        if current_disc < best_disc:
            best_disc = current_disc
            best_state = current_state
            
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Shuffle indices for variety? Or fixed order?
            # Fixed order is deterministic and fine.
            indices = list(range(n))
            rng.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline - 2:
                    break
                
                flip_mask = 1 << i
                new_state = current_state ^ flip_mask
                new_disc = get_max_disc(new_state)
                
                if new_disc < current_disc:
                    current_state = new_state
                    current_disc = new_disc
                    improved = True
                    if current_disc < best_disc:
                        best_disc = current_disc
                        best_state = current_state
                        
    # Convert best_state back to list of signs
    # bit i is 1 -> +1, bit i is 0 -> -1
    result = []
    for i in range(n):
        if (best_state >> i) & 1:
            result.append(1)
        else:
            result.append(-1)
            
    return result

# Test/Check
if __name__ == "__main__":
    signs = construct()
    print(len(signs))
    print(signs[:10])
