"""Size-efficient sorting network on 17 wires - improved seed with optimization."""
import time
import numpy as np

N = 17
TIME_LIMIT = 80.0  # seconds (safety margin from 90s)


def _apply_comparator(vec, i, j):
    """Apply a single compare-exchange to a vector."""
    a, b = vec[i], vec[j]
    if a > b:
        vec[i] = b
        vec[j] = a
    return vec


def _apply_network(vec, network):
    """Apply an entire network to a vector."""
    for i, j in network:
        a, b = vec[i], vec[j]
        if a > b:
            vec[i] = b
            vec[j] = a
    return vec


def _is_valid_network(network):
    """Check if network sorts all 2^N Boolean inputs using zero-one principle."""
    for mask in range(1 << N):
        vec = np.array([(mask >> i) & 1 for i in range(N)], dtype=np.int8)
        _apply_network(vec, network)
        # Check if sorted (non-decreasing)
        if np.any(np.diff(vec) < 0):
            return False
    return True


def _verify_network_fast(network):
    """Faster verification using bit operations on packed integers."""
    # Each Boolean input is an N-bit integer.
    # After applying the network, the output must be non-decreasing.
    # For Boolean values, non-decreasing means all 0s come before all 1s.
    # So the output must be of the form: 0...0 1...1
    
    # Precompute for each comparator the bit positions
    # Use a lookup table approach for speed
    
    # Actually, let's use a different approach:
    # For each input mask (0 to 2^N - 1), simulate the network.
    # Represent the wire values as bits in an integer.
    
    for mask in range(1 << N):
        state = mask
        for i, j in network:
            # Extract bits at positions i and j
            bit_i = (state >> i) & 1
            bit_j = (state >> j) & 1
            if bit_i > bit_j:
                # Swap: set bit_i to 0, bit_j to 1
                state = (state & ~(1 << i)) | (1 << j)
            # else: already in order (bit_i=0, bit_j=1) or equal
        # Check if sorted: all 0s before all 1s
        # This means the state should be of the form: lower k bits are 1, upper are 0
        # Wait no: sorted non-decreasing means 0,0,...,0,1,1,...,1
        # So lower bits are 0, upper bits are 1
        # Actually wire 0 is the "first" position, wire N-1 is the "last"
        # Sorted non-decreasing: wire[0] <= wire[1] <= ... <= wire[N-1]
        # For Boolean: 0,0,...,0,1,1,...,1
        # So there exists k such that bits 0..k-1 are 0 and bits k..N-1 are 1
        # OR all 0s (k=0) or all 1s (k=N)
        
        # Check: the state must be a suffix of 1s
        # i.e., for some k, state == ((1 << (N-k)) - 1) << k  ... no wait
        # If k ones at the end: bits k through N-1 are 1, bits 0 through k-1 are 0
        # That's: state has zeros in lower k positions and ones in upper N-k positions
        # = ~((1 << k) - 1) & ((1 << N) - 1)  for k zeros, or equivalently
        # = ((1 << N) - 1) ^ ((1 << k) - 1)
        
        # Simpler check: if we look at the bits from high to low, 
        # once we see a 0, all lower bits must be 0.
        # Equivalently: the state, when viewed as a bitmask, must have the form
        # where it's a "suffix" of 1s.
        
        # Fast check: for the state to be sorted (0...01...1),
        # there should be no pattern "10" in the bit sequence from bit 0 upward.
        # i.e., no i < j such that bit i = 1 and bit j = 0.
        # This is equivalent to: state & (state >> 1) == 0 is NOT right...
        
        # Let me think again. Sorted means: if bit i is 1, then bit j must be 1 for all j > i.
        # Equivalently: if bit j is 0, then bit i must be 0 for all i < j.
        # This means: the set of 1-bits forms a suffix (consecutive ones at the top).
        
        # Check: state must equal ((1<<N)-1) & (-(1<<k)) for some k in 0..N
        # i.e., state + (1<<k) must be divisible by (1<<(k+1)) for appropriate k... 
        # This is getting complicated. Let's just check directly.
        
        # Direct check: find the lowest 1-bit and highest 0-bit below it
        # If there's a 1 below a 0, it's not sorted.
        # 
        # Alternative: state is sorted iff state == 0 or state == ((1<<N)-1) or
        # state has the form: for some 1 <= k <= N-1, 
        # state = ((1<<N) - 1) - ((1<<k) - 1) = ((1<<N)-1) ^ ((1<<k)-1)
        # 
        # Even simpler: iterate through bits and verify monotonicity.
        # But for 17 bits, that's fine.
        
        # Actually the simplest fast check:
        # state is sorted (non-decreasing from bit 0 to bit N-1) iff
        # there is no i < j with bit_i = 1 and bit_j = 0
        # iff: for all i, if bit_i = 1 then all bits j > i are also 1
        # iff: the number of trailing zeros of (~state & ((1<<N)-1)) 
        #       equals N - popcount(state)
        # Hmm, let's just do it simply.
        
        # Simple check: 
        # Find leftmost 1. All bits to its right must be 1.
        # Find rightmost 0. All bits to its left must be 0.
        
        if state != 0 and state != (1 << N) - 1:
            # Find position of lowest set bit
            # and position of highest clear bit
            # If lowest_set_bit_pos > highest_clear_bit_pos, then not sorted
            # Wait, that's not right either.
            
            # Let's just check: is state of the form 0...01...1?
            # Count trailing zeros (zeros at low end)
            tz = 0
            tmp = state
            while tmp == 0 or (tmp & 1) == 0:
                tz += 1
                tmp >>= 1
                if tz >= N:
                    break
            # Now bits 0..tz-1 should be 0, bits tz..N-1 should be 1
            expected = ((1 << N) - 1) ^ ((1 << tz) - 1)
            if state != expected:
                return False
    
    return True


def construct():
    """Construct a sorting network for N=17 wires."""
    start_time = time.time()
    
    # Start with odd-even transposition network as seed
    # This has N*(N-1)/2 = 17*16/2 = 136 comparators
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
    
    # Verify the seed is valid
    assert _verify_network_fast(network), "Seed network is invalid!"
    
    # Try to optimize by removing redundant comparators
    # Strategy: try removing each comparator and check if network remains valid
    # Work from the end backwards (later comparators are more likely redundant)
    
    optimized = network[:]
    changed = True
    while changed and (time.time() - start_time) < TIME_LIMIT:
        changed = False
        # Try removing from end to start
        for idx in range(len(optimized) - 1, -1, -1):
            if time.time() - start_time > TIME_LIMIT:
                return optimized
            
            # Try removing comparator at idx
            candidate = optimized[:idx] + optimized[idx+1:]
            
            if _verify_network_fast(candidate):
                optimized = candidate
                changed = True
                break  # Restart from the end
    
    return optimized
