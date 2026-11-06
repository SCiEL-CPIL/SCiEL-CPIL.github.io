"""
Size-efficient sorting networks on 17 wires.
Strategy: Bitmask vector representation for 0-1 principle verification.
- Pre-compute comparator masks for all 2^17 inputs.
- Represent the state of all inputs as a single large integer (131072 bits).
- Verification is a single bitwise AND.
- Search: Local search (removal) + Random insertion (annealing).
"""
import time
import random

N = 17
NUM_INPUTS = 1 << N  # 131072

# Pre-compute the "sorted" mask.
# An input 'x' is sorted if it is of the form 00...011...1.
# There are 18 such inputs (0 ones, 1 one, ..., 17 ones).
# We create a bitmask 'sorted_mask' where bit 'x' is 1 if input 'x' is sorted.
_sorted_mask = 0
for num_ones in range(N + 1):
    # The sorted value for 'num_ones' ones is (1 << num_ones) - 1
    val = (1 << num_ones) - 1
    _sorted_mask |= (1 << val)

# Pre-compute masks for every possible comparator (i, j).
# mask[i][j] is an integer of 2^N bits.
# Bit 'x' is 1 if input 'x' has bit i = 1 and bit j = 0.
# This indicates that a swap is required for this input.
_comp_masks = {}
for i in range(N):
    for j in range(i + 1, N):
        m = 0
        # Iterate through all inputs? No, we can construct this more cleverly or just loop.
        # Looping 131072 * 136 times is ~17M ops, which is fast in Python if optimized, 
        # but let's do it once at startup.
        # Actually, we can generate these masks efficiently.
        # For a fixed i, j:
        # We need to set bit 'x' in m if ((x >> i) & 1) == 1 and ((x >> j) & 1) == 0.
        
        # A faster way:
        # The pattern repeats every 2^(i+1) for bit i.
        # But since N is small, a simple loop is acceptable for initialization (runs once).
        for x in range(NUM_INPUTS):
            if ((x >> i) & 1) and not ((x >> j) & 1):
                m |= (1 << x)
        _comp_masks[(i, j)] = m

# Map (i, j) to index for faster lookup in lists if needed, but dict is fine.

def _odd_even_transposition_network(n):
    """Generate odd-even transposition sorting network."""
    network = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            network.append((i, i + 1))
    return network

def _batcher_network(n):
    """
    Generate Batcher's odd-even merge sort network.
    This is a standard constructive algorithm.
    """
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    # Split into two halves
    half = (n + 1) // 2
    left = _batcher_network(half)
    right = _batcher_network(n - half)
    
    # Shift right half
    right_shifted = [(i + half, j + half) for i, j in right]
    
    # Merge
    merge = _odd_even_merge(half, n - half)
    
    return left + right_shifted + merge

def _odd_even_merge(n1, n2):
    """
    Generate the comparators for merging two sorted sequences of length n1 and n2.
    This implements the recursive structure of odd-even merge.
    """
    if n1 == 0 or n2 == 0:
        return []
    if n1 == 1 and n2 == 1:
        return [(0, 1)]
    
    # Split the combined sequence? No, standard odd-even merge definition:
    # Merge A (len n1) and B (len n2).
    # 1. Split A into A0 (even indices) and A1 (odd indices).
    # 2. Split B into B0 (even indices) and B1 (odd indices).
    # 3. Recursively merge A0 and B0.
    # 4. Recursively merge A1 and B1.
    # 5. Compare specific pairs.
    
    # This recursive definition is tricky to implement correctly for arbitrary n1, n2 
    # in a flat list without careful index tracking.
    # For N=17, Batcher is ~200 comparators. OET is 136.
    # We will stick to OET as the primary seed as it is smaller and valid.
    # We can add a "shuffle" or "bubble sort" based seed if needed, but OET is robust.
    
    # Since implementing a perfect Batcher generator is error-prone and OET is better (136 < ~200),
    # we will rely on OET and the search to improve it.
    return []

def _simulate_network(network):
    """
    Apply the network to ALL inputs simultaneously using bitmasks.
    Returns a bitmask of 2^N bits where bit x is 1 if input x is sorted after the network.
    """
    # Initial state: all inputs are "unsorted" in the sense that we haven't applied anything.
    # We represent the state of each input 'x' as a 17-bit integer.
    # But we want to check if the *output* is sorted.
    # 
    # Alternative View:
    # We track the set of inputs that are currently "sorted".
    # Initially, only the 18 sorted inputs are sorted.
    # state_mask: bit x is 1 if input x is currently sorted.
    
    # However, the network permutes the *values*.
    # If input x is sorted, it stays sorted? No.
    # A sorting network takes an input vector v, applies comparators, and outputs sorted vector.
    # The 0-1 principle says: if it sorts all 0-1 inputs, it sorts all inputs.
    # For a specific 0-1 input v, the output is a permutation of v.
    # The output is sorted iff the output vector is of the form 0..01..1.
    
    # Let's use the "State Evolution" approach on the 0-1 inputs directly.
    # state[x] is the current 17-bit value of input x.
    # We can pack all 17 bits of all 131072 inputs into 17 large integers (bitplanes).
    # plane[k] is an integer where bit x is the k-th bit of input x.
    
    # Initialize planes
    planes = [0] * N
    for x in range(NUM_INPUTS):
        for k in range(N):
            if (x >> k) & 1:
                planes[k] |= (1 << x)
                
    # Apply network
    for i, j in network:
        # Comparator (i, j): min goes to i, max goes to j.
        # For each input x:
        # if bit_i(x) > bit_j(x), swap.
        # bit_i(x) is bit x in planes[i].
        # bit_j(x) is bit x in planes[j].
        
        # We need to update planes[i] and planes[j].
        # new_i[x] = min(planes[i][x], planes[j][x])
        # new_j[x] = max(planes[i][x], planes[j][x])
        
        # Logic:
        # swap_mask = planes[i] & ~planes[j]  (inputs where i=1, j=0 -> i becomes 0, j becomes 1)
        # Actually:
        # If i=1, j=0: i becomes 0, j becomes 1.
        # If i=0, j=1: i becomes 0, j becomes 1. (No change? Wait. min(0,1)=0, max(0,1)=1. Yes.)
        # If i=1, j=1: i becomes 1, j becomes 1.
        # If i=0, j=0: i becomes 0, j becomes 0.
        
        # So:
        # new_i = planes[i] & ~planes[j]  is WRONG.
        # Let's check:
        # i=1, j=0. min=0. So new_i should be 0.
        # i & ~j = 1 & ~0 = 1. This sets new_i to 1. Incorrect.
        
        # Correct logic:
        # new_i[x] = 1 iff (planes[i][x] == 1 and planes[j][x] == 1).
        #           OR (planes[i][x] == 0 and planes[j][x] == 1) -> min is 0.
        #           OR (planes[i][x] == 1 and planes[j][x] == 0) -> min is 0.
        #           OR (planes[i][x] == 0 and planes[j][x] == 0) -> min is 0.
        # So new_i[x] = 1 iff planes[i][x] == 1 AND planes[j][x] == 1.
        # new_i = planes[i] & planes[j]
        
        # new_j[x] = 1 iff (planes[i][x] == 1 and planes[j][x] == 1)
        #           OR (planes[i][x] == 0 and planes[j][x] == 1) -> max is 1.
        #           OR (planes[i][x] == 1 and planes[j][x] == 0) -> max is 1.
        #           OR (planes[i][x] == 0 and planes[j][x] == 0) -> max is 0.
        # So new_j[x] = 1 iff planes[i][x] == 1 OR planes[j][x] == 1.
        # new_j = planes[i] | planes[j]
        
        new_i = planes[i] & planes[j]
        new_j = planes[i] | planes[j]
        
        planes[i] = new_i
        planes[j] = new_j
        
    # Check if all inputs are sorted.
    # An input x is sorted if its bits are 0..01..1.
    # This means for any k < l, if bit k is 1 then bit l must be 1.
    # Equivalently, the vector must be one of the 18 sorted patterns.
    
    # We can check this by verifying that for each k, 
    # the set of inputs with bit k = 1 is a subset of the set of inputs with bit k-1 = 1?
    # No, that's not right.
    # Input x is sorted if:
    # There exists m such that bits 0..m-1 are 0 and bits m..N-1 are 1.
    
    # Let's just check against the pre-computed sorted inputs.
    # We have 18 sorted values.
    # For each sorted value v, check if the current planes match v for all x?
    # No, each x is a different input.
    # We need to check if the output for input x is sorted.
    
    # For a specific input x, the output bits are given by the bits of x in the final planes.
    # out_x = sum( (planes[k] >> x) & 1 * 2^k )
    
    # We want to check if out_x is in {0, 1, 3, 7, ..., 2^17-1}.
    
    # Efficient check:
    # Construct a mask 'valid_output_mask' where bit x is 1 if output x is sorted.
    # We can compute this by iterating through the 18 sorted values?
    # Or we can use the property:
    # out_x is sorted iff for all k, (out_x >> k) & 1 implies (out_x >> (k+1)) & 1 is 0? No, 1s are at the end.
    # Bits: b0 b1 ... b16. Sorted means b0=0, b1=0... bm=1, b(m+1)=1...
    # So b_k <= b_{k+1}.
    
    # Condition: For all k from 0 to N-2, if bit k is 1, bit k+1 must be 1.
    # (planes[k] >> x) & 1 <= (planes[k+1] >> x) & 1
    # This is equivalent to: ~(planes[k] & ~planes[k+1]) has bit x set?
    # If planes[k] has 1 and planes[k+1] has 0, it's invalid.
    # invalid_mask_k = planes[k] & ~planes[k+1]
    
    total_invalid = 0
    for k in range(N - 1):
        total_invalid |= (planes[k] & ~planes[k+1])
        
    # If total_invalid is 0, all inputs are sorted.
    # We return the count of valid inputs or a boolean.
    # For the search, we just need to know if it's fully valid.
    return total_invalid == 0

def _is_valid(network):
    """Check if the network sorts all inputs."""
    return _simulate_network(network)

def construct():
    """
    Construct a sorting network for N wires.
    """
    start_time = time.time()
    deadline = start_time + 85  # 85s limit, 5s safety margin
    
    # Start with Odd-Even Transposition (136 comparators)
    best_network = _odd_even_transposition_network(N)
    best_count = len(best_network)
    
    # Verify seed
    if not _is_valid(best_network):
        # Fallback: Bubble sort network (N*(N-1)/2 = 136 for N=17? No, N(N-1)/2 is for comparators in bubble sort network? 
        # Bubble sort network has N(N-1)/2 comparators? 
        # Actually, simple bubble sort network:
        # Pass 1: (0,1), (2,3)...
        # Pass 2: (1,2), (3,4)...
        # ...
        # Total comparators is N(N-1)/2? No.
        # For N=17, OET is 17*17 = 289? No.
        # OET: N phases. Each phase has N/2 comparators. Total N^2/2.
        # 17^2/2 = 289/2 = 144.5 -> 144 or 145?
        # Let's check OET count.
        # Phase 0: 0,1; 2,3... 16? No, 0 to 15. 8 comparators.
        # Phase 1: 1,2; 3,4... 15,16. 8 comparators.
        # ...
        # 17 phases. 8 or 9 per phase.
        # Total is around 136.
        
        # If OET fails (it shouldn't), we need a backup.
        # But OET is a known valid sorting network.
        pass

    rng = random.Random(42)
    
    # Local Search: Remove redundant comparators
    def local_search(network, max_time):
        improved = True
        current_network = list(network)
        
        while improved and time.time() < max_time:
            improved = False
            # Shuffle the order of removal attempts to avoid bias
            indices = list(range(len(current_network)))
            rng.shuffle(indices)
            
            for idx in indices:
                if time.time() >= max_time:
                    break
                
                # Try removing comparator at idx
                candidate = current_network[:idx] + current_network[idx+1:]
                
                if _is_valid(candidate):
                    current_network = candidate
                    improved = True
                    break # Restart the loop with the new shorter network
        
        return current_network

    # Phase 1: Aggressive Local Search on the seed
    best_network = local_search(best_network, deadline - 10)
    best_count = len(best_network)
    
    # Phase 2: Simulated Annealing / Random Insertion
    # Try to insert random comparators to escape local minima, then remove.
    # We have about 75s left.
    
    current_best = best_network
    current_best_count = best_count
    
    # We can try multiple restarts or a continuous process.
    # Let's do a continuous loop:
    # 1. Take current best.
    # 2. Insert K random comparators.
    # 3. Run local search.
    # 4. If better, update best.
    # 5. Repeat until time runs out.
    
    # To make it faster, we can limit the local search depth or use a timeout per restart.
    # Given the speed of _is_valid, we can do many restarts.
    
    restart_count = 0
    while time.time() < deadline - 5:
        # Create a perturbed version
        perturbed = list(current_best)
        
        # Insert 1 to 5 random comparators
        num_inserts = rng.randint(1, 5)
        for _ in range(num_inserts):
            i = rng.randint(0, N - 2)
            j = rng.randint(i + 1, N - 1)
            pos = rng.randint(0, len(perturbed))
            perturbed.insert(pos, (i, j))
        
        # Run local search on the perturbed network
        # Limit time for this specific restart to ensure we finish
        restart_deadline = min(deadline - 5, time.time() + 5) # Max 5s per restart
        optimized = local_search(perturbed, restart_deadline)
        
        if len(optimized) < current_best_count:
            current_best = optimized
            current_best_count = len(optimized)
            # Reset restart counter or just continue
            restart_count = 0
        else:
            restart_count += 1
            # If we've tried many restarts without improvement, we might change strategy
            # But for now, just keep going.
            
    return current_best

# EVOLVE-BLOCK-END
