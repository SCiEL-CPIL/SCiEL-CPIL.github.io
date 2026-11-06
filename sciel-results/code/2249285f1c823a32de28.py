import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 85.0  # Safety margin from 90s limit

def _build_mask(w, n_bits):
    block = 1 << w
    pattern = (1 << block) - 1
    pattern <<= block
    num_periods = n_bits // (2 * block)
    m = 0
    shift = 0
    for _ in range(num_periods):
        m |= (pattern << shift)
        shift += 2 * block
    return m

# Pre-compute masks at module load time
_INITIAL_MASKS = [0] * N
for _w in range(N):
    _INITIAL_MASKS[_w] = _build_mask(_w, 1 << N)

def _verify(comparators):
    """
    Verify if the comparator list sorts all 2**N boolean inputs.
    Uses bitwise operations on Python integers representing the state of each wire.
    """
    state = _INITIAL_MASKS[:]
    
    for i, j in comparators:
        a = state[i]
        b = state[j]
        state[i] = a & b
        state[j] = a | b
        
    for i in range(N - 1):
        if (state[i] & ~state[i+1]) != 0:
            return False
            
    return True

def _odd_even_transposition():
    """Return the odd-even transposition sorting network for N wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _batcher_odd_even_merge(n):
    """
    Merge two sorted halves of length n//2.
    Returns comparators for the merge step only.
    """
    if n <= 2:
        return [(0, 1)] if n == 2 else []
    
    half = n // 2
    result = []
    
    # Compare (0,1), (2,3), ... in the interleaved view
    # Actually, standard odd-even merge of two sorted lists of size m:
    # 1. Compare (1,2), (3,4), ..., (2m-3, 2m-2)
    # 2. Recursively merge even positions and odd positions
    # Wait, the standard recursive definition is:
    # Merge(2m):
    #   Compare (0, 1), (2, 3), ..., (2m-2, 2m-1)  [This is NOT standard, let me check]
    #   
    # Correct Odd-Even Merge (Batcher):
    # To merge two sorted lists of length m, interleaved on 2m wires:
    # 1. Compare (1, 2), (3, 4), ..., (2m-3, 2m-2)
    # 2. Recursively merge the even-indexed wires (0, 2, ..., 2m-2) -> m elements
    # 3. Recursively merge the odd-indexed wires (1, 3, ..., 2m-1) -> m elements
    #
    # But this assumes the inputs are already sorted in their respective halves?
    # No, the inputs to Merge are the outputs of the recursive sorts in Odd-Even Mergesort.
    #
    # Let's use a known correct implementation for small n or a simpler approach.
    # For n=4: [(0,2), (1,3), (0,1), (2,3), (1,2)]
    # For n=8:
    #   It's complex.
    #
    # Given the time, I will use a simpler, verified network for the initial seed if Batcher is too complex to get right quickly.
    # However, the inspiration suggests Batcher is useful.
    # Let's try to implement it carefully.
    
    # Actually, for the purpose of this task, the exact structure of Batcher's merge is:
    # Merge(n) where n is even:
    #   Let m = n/2
    #   1. Compare (0, 1), (2, 3), ..., (n-2, n-1)  -- NO, this is not right.
    #
    # Let's look at the standard Batcher Odd-Even Mergesort algorithm:
    # Sort(n):
    #   if n == 1: return
    #   Sort(even indices)  # wires 0, 2, 4...
    #   Sort(odd indices)   # wires 1, 3, 5...
    #   Merge(even, odd)    # merge the two sorted lists
    #
    # Merge(2m) where the even-indexed wires contain one sorted list and odd-indexed contain another:
    #   1. Compare (1, 2), (3, 4), ..., (2m-3, 2m-2)
    #   2. Recursively merge the even-indexed wires (0, 2, ..., 2m-2)
    #   3. Recursively merge the odd-indexed wires (1, 3, ..., 2m-1)
    #
    # So, for Merge(2m):
    #   comparators = []
    #   for i in range(1, 2m-1, 2):
    #       comparators.append((i, i+1))
    #   # Now recursively merge even and odd
    #   # The even wires are 0, 2, 4, ..., 2m-2. There are m of them.
    #   # The odd wires are 1, 3, 5, ..., 2m-1. There are m of them.
    #   # We need to map these to 0..m-1 for the recursive call.
    #   # But the recursive call expects a contiguous block?
    #   # No, the recursive call is on the *values* at those positions.
    #   # So we generate the network for m wires and then map the indices.
    
    if n == 2:
        return [(0, 1)]
    
    m = n // 2
    # Step 1: Compare (1, 2), (3, 4), ..., (2m-3, 2m-2)
    step1 = [(i, i+1) for i in range(1, 2*m-1, 2)]
    
    # Step 2: Recursively merge even wires
    # Even wires: 0, 2, 4, ..., 2m-2
    # Map wire 2*k to k
    even_merge = _batcher_odd_even_merge(m)
    even_mapped = [(2*i, 2*j) for i, j in even_merge]
    
    # Step 3: Recursively merge odd wires
    # Odd wires: 1, 3, 5, ..., 2m-1
    # Map wire 2*k+1 to k
    odd_merge = _batcher_odd_even_merge(m)
    odd_mapped = [(2*i+1, 2*j+1) for i, j in odd_merge]
    
    return step1 + even_mapped + odd_mapped

def _batcher_odd_even_mergesort(n):
    """
    Generate a sorting network for n wires using Batcher's Odd-Even Mergesort.
    """
    if n <= 1:
        return []
    
    # Sort even indices: 0, 2, 4, ...
    # Map wire 2*k to k
    even_sort = _batcher_odd_even_mergesort((n + 1) // 2)
    even_mapped = [(2*i, 2*j) for i, j in even_sort]
    
    # Sort odd indices: 1, 3, 5, ...
    # Map wire 2*k+1 to k
    odd_sort = _batcher_odd_even_mergesort(n // 2)
    odd_mapped = [(2*i+1, 2*j+1) for i, j in odd_sort]
    
    # Merge the two sorted halves
    # The merge network expects 2*m wires, where m is the size of each half.
    # Here, the even half has ceil(n/2) elements, and the odd half has floor(n/2) elements.
    # Batcher's odd-even merge assumes equal sizes?
    # If n is odd, the halves are unequal.
    # For n=18, halves are 9 and 9. So it's fine.
    # For general n, we might need to handle unequal halves.
    # But for n=18, it's fine.
    
    merge = _batcher_odd_even_merge(n)
    
    return even_mapped + odd_mapped + merge

def _build_initial_network():
    """
    Build an initial network. We'll use Batcher's odd-even mergesort for N=18.
    If it fails verification, we fall back to odd-even transposition.
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
                break
            neighbor = current[:i] + current[i+1:]
            if _verify(neighbor):
                current = neighbor
                changed = True
    return current

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
    current = _greedy_remove_pass(current, deadline)
    if len(current) < best_len:
        best = current[:]
        best_len = len(current)
    
    # Phase 2: Simulated Annealing
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.99
    
    rng = random.Random(42)
    
    all_comparators = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    while time.time() < deadline and T > T_min:
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
            delta = len(neighbor) - len(current)
            if delta <= 0 or rng.random() < np.exp(-delta / T):
                current = neighbor
                if len(current) < best_len:
                    best = current[:]
                    best_len = len(current)

        T *= cooling_rate
        
    # Final check
    if not _verify(best):
        return _odd_even_transposition()
        
    return best
