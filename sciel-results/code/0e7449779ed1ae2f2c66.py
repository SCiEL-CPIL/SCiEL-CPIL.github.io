import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 85.0  # Safety margin from 90s limit

# Pre-compute initial masks for bit-parallel verification
# Each mask is a 2^N bit integer, but we store them in a uint64 array for vectorization.
# Since N=18, 2^18 = 262,144 bits = 4096 uint64s. 
# Wait, the state is not 2^N bits long. 
# The state represents the values of the N wires across ALL 2^N inputs.
# Wire i's state is a bitstring of length 2^N.
# We can represent this as a NumPy array of uint64s.
# Length of array = 2^N / 64 = 262144 / 64 = 4096.

def _init_masks():
    """
    Initialize the state array where state[i] holds the bitvector for wire i 
    across all 2^N boolean inputs.
    Input x is in [0, 2^N). 
    Wire i has value (x >> i) & 1.
    """
    n_bits = 1 << N
    n_uint64s = n_bits // 64
    
    # We need to construct the bitvector for each wire i.
    # For wire i, the bit at position k is 1 if (k >> i) & 1 is 1.
    # This pattern repeats every 2^(i+1) bits.
    # In each period of 2^(i+1) bits, the first 2^i bits are 0, the next 2^i are 1.
    
    masks = np.zeros((N, n_uint64s), dtype=np.uint64)
    
    for i in range(N):
        # Pattern for wire i
        # Block size is 2^i.
        # Period is 2^(i+1).
        # In uint64 terms, we need to fill the array with this pattern.
        
        # If 2^i is small, we can build a 64-bit mask and tile it.
        # If 2^i >= 64, the pattern spans multiple uint64s.
        
        block_size = 1 << i
        period = 1 << (i + 1)
        
        # Number of periods in the total 2^N bits
        num_periods = (1 << N) // period
        
        # We can construct the pattern for one period of 2^(i+1) bits.
        # Since 2^N is a multiple of 2^(i+1) for i < N, this is clean.
        
        # Let's generate the bitvector for one period (size 2^(i+1) bits)
        # and tile it.
        
        # Actually, simpler: 
        # For each uint64 index u in 0..4095, the bits correspond to inputs u*64 ... u*64+63.
        # This is complex to vectorize easily without loops over the 4096 chunks.
        
        # Alternative: Generate the full 2^N bit integer for each wire using Python big ints, 
        # then convert to numpy array.
        
        # Let's stick to Python big ints for generation, then convert.
        # 2^18 bits is ~32KB per wire. 18 wires is ~576KB. Very fast.
        
        mask_int = 0
        # The pattern for wire i is: 0...0 (2^i times) 1...1 (2^i times)
        # repeated.
        # We can build this by iterating over periods.
        # But 2^N / 2^(i+1) periods can be large for small i.
        # For i=0, period=2, 131072 periods. Too slow in Python loop.
        
        # Use bitwise trick:
        # mask for i=0: 010101...
        # mask for i=1: 00110011...
        
        # Let's use the property that the bitvector for wire i is periodic.
        # We can construct it by repeating a base pattern.
        
        # Base pattern for one period (2^(i+1) bits):
        # Lower 2^i bits are 0, upper 2^i bits are 1.
        # base_pattern = ((1 << (2^i)) - 1) << (2^i)
        
        # Then we need to tile this base_pattern across 2^N bits.
        # Tiling a bit pattern of length L across M bits (where M is multiple of L):
        # This can be done with shifts and ORs, but M is 2^18.
        
        # Let's just use the previous _build_mask logic but optimized.
        
        block = 1 << i
        pattern = (1 << block) - 1
        pattern <<= block
        
        num_periods = (1 << N) // (2 * block)
        m = 0
        shift = 0
        for _ in range(num_periods):
            m |= (pattern << shift)
            shift += 2 * block
        
        # Convert Python int to numpy uint64 array
        # Python int is little-endian in memory if we use to_bytes?
        # np.frombuffer requires bytes.
        # m is 2^18 bits = 32768 bytes.
        m_bytes = m.to_bytes(32768, byteorder='little')
        m_arr = np.frombuffer(m_bytes, dtype=np.uint64)
        masks[i] = m_arr
        
    return masks

_INITIAL_MASKS = _init_masks()

def _verify(comparators):
    """
    Verify if the comparator list sorts all 2**N boolean inputs.
    Uses NumPy vectorized bitwise operations.
    """
    if not comparators:
        return True
        
    state = _INITIAL_MASKS.copy()
    
    for i, j in comparators:
        a = state[i]
        b = state[j]
        state[i] = a & b
        state[j] = a | b
        
    # Check if sorted: state[i] <= state[i+1] for all i
    # This is equivalent to (state[i] & ~state[i+1]) == 0
    # We can check this vectorized over the wires? No, the bits are across inputs.
    # We need to check for any bit position k where state[i][k] == 1 and state[i+1][k] == 0.
    
    # We can sum the violations.
    # violation = state[i] & ~state[i+1]
    # If any bit is set, it's a violation.
    
    # To check if any bit is set in a uint64 array:
    # np.any(violation != 0)
    
    for i in range(N - 1):
        if np.any((state[i] & ~state[i+1]) != 0):
            return False
            
    return True

def _odd_even_transposition():
    """Return the odd-even transposition sorting network for N wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _batcher_odd_even_merge(n):
    """
    Merge two sorted halves of length n//2.
    """
    if n <= 2:
        return [(0, 1)] if n == 2 else []
    
    half = n // 2
    result = []
    
    # Step 1: Compare (1, 2), (3, 4), ..., (2m-3, 2m-2)
    step1 = [(i, i+1) for i in range(1, 2*half-1, 2)]
    result.extend(step1)
    
    # Step 2: Recursively merge even wires
    even_merge = _batcher_odd_even_merge(half)
    even_mapped = [(2*i, 2*j) for i, j in even_merge]
    result.extend(even_mapped)
    
    # Step 3: Recursively merge odd wires
    odd_merge = _batcher_odd_even_merge(half)
    odd_mapped = [(2*i+1, 2*j+1) for i, j in odd_merge]
    result.extend(odd_mapped)
    
    return result

def _batcher_odd_even_mergesort(n):
    """
    Generate a sorting network for n wires using Batcher's Odd-Even Mergesort.
    """
    if n <= 1:
        return []
    
    # Sort even indices: 0, 2, 4, ...
    even_sort = _batcher_odd_even_mergesort((n + 1) // 2)
    even_mapped = [(2*i, 2*j) for i, j in even_sort]
    
    # Sort odd indices: 1, 3, 5, ...
    odd_sort = _batcher_odd_even_mergesort(n // 2)
    odd_mapped = [(2*i+1, 2*j+1) for i, j in odd_sort]
    
    # Merge the two sorted halves
    merge = _batcher_odd_even_merge(n)
    
    return even_mapped + odd_mapped + merge

def _build_initial_network():
    """
    Build an initial network. We'll use Batcher's odd-even mergesort for N=18.
    """
    try:
        net = _batcher_odd_even_mergesort(N)
        if _verify(net):
            return net
    except:
        pass
    
    return _odd_even_transposition()

def _greedy_remove_pass(network, deadline):
    """
    Greedily remove redundant comparators.
    """
    current = network[:]
    changed = True
    while changed:
        changed = False
        for i in range(len(current) - 1, -1, -1):
            if time.time() > deadline:
                return current
            neighbor = current[:i] + current[i+1:]
            if _verify(neighbor):
                current = neighbor
                changed = True
    return current

def _local_search(network, deadline, rng):
    """
    Local search to minimize comparator count.
    Moves: Remove, Add, Swap, Move.
    Accepts if valid and length decreases.
    Occasionally accepts neutral/length-increasing moves to escape local minima?
    No, let's stick to strict improvement or random restarts if stuck.
    """
    current = network[:]
    best = current[:]
    best_len = len(best)
    
    all_comparators = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Simple Hill Climbing with random restarts on stagnation
    stagnation_counter = 0
    max_stagnation = 50
    
    while time.time() < deadline:
        if stagnation_counter > max_stagnation:
            # Restart from best
            current = best[:]
            stagnation_counter = 0
            # Maybe perturb best slightly?
            # Let's just continue from best, but try different moves.
            
        move_type = rng.choices(['remove', 'add', 'swap', 'move'], weights=[5, 2, 1, 1])[0]
        neighbor = current[:]
        
        if move_type == 'remove' and len(neighbor) > 0:
            idx = rng.randrange(len(neighbor))
            neighbor.pop(idx)
            
        elif move_type == 'add':
            comp = rng.choice(all_comparators)
            pos = rng.randrange(len(neighbor) + 1)
            neighbor.insert(pos, comp)
            
        elif move_type == 'swap' and len(neighbor) >= 2:
            i = rng.randrange(len(neighbor))
            j = rng.randrange(len(neighbor))
            if i != j:
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                
        elif move_type == 'move' and len(neighbor) >= 2:
            i = rng.randrange(len(neighbor))
            j = rng.randrange(len(neighbor))
            if i != j:
                comp = neighbor.pop(i)
                if i < j:
                    j -= 1
                neighbor.insert(j, comp)

        if _verify(neighbor):
            if len(neighbor) < len(current):
                current = neighbor
                stagnation_counter = 0
                if len(current) < best_len:
                    best = current[:]
                    best_len = len(current)
            else:
                stagnation_counter += 1
        else:
            stagnation_counter += 1

    return best

def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE_SECONDS
    
    # Start with the seed network
    current = _build_initial_network()
    best = current[:]
    best_len = len(best)
    
    # Initial verification
    if not _verify(best):
        return _odd_even_transposition()
    
    # Phase 1: Greedy removal
    # This is crucial to get down from ~110 to ~90 quickly.
    current = _greedy_remove_pass(current, deadline)
    if len(current) < best_len:
        best = current[:]
        best_len = len(current)
    
    # Phase 2: Local Search
    # Use random restarts within the local search if it gets stuck.
    rng = random.Random(42)
    
    # Allocate remaining time to local search
    # We might want to run multiple local searches with different seeds if time permits.
    # But local search itself has restarts.
    
    current = _local_search(current, deadline, rng)
    if len(current) < best_len:
        best = current[:]
        best_len = len(current)
        
    # Phase 3: Try a few more greedy passes on the result of local search
    # Sometimes local search ends up in a state where greedy removal can still find improvements
    # if the order is different.
    if time.time() < deadline:
        current = _greedy_remove_pass(best, deadline)
        if len(current) < best_len:
            best = current[:]
            best_len = len(current)
            
    # Final check
    if not _verify(best):
        return _odd_even_transposition()
        
    return best

if __name__ == "__main__":
    # Quick test
    net = construct()
    print(f"Constructors found: {len(net)}")
    print(f"Valid: {_verify(net)}")
