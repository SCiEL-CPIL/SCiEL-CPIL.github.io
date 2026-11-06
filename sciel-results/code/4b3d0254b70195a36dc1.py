"""Size-efficient sorting networks on 17 wires."""
import time
import random
import numpy as np

N = 17
NUM_CASES = 1 << N  # 131072

# EVOLVE-BLOCK-START
# Precompute bitmasks for all boolean inputs
_CASES = np.array([(1 << k) - 1 for k in range(N + 1)], dtype=np.uint64)
# cases[m] = bitmask where lower m bits are 1 (i.e., first m wires = 1, rest = 0)
# We need all 2^17 patterns. Build via: for mask in range(2^17), the value is the mask itself
# Actually simpler: just use np.arange(2**17, dtype=np.uint64) — each uint64 IS the bitmask
_CASES_ALL = np.arange(NUM_CASES, dtype=np.uint64)

# Precompute: for each wire i, the mask that has bit i set
_WIRE_MASKS = np.array([1 << i for i in range(N)], dtype=np.uint64)

# Precompute swap masks: swap_mask[i][j] = array over all cases of (bit i set AND bit j not set)
# swap_mask[i,j] = cases & (1<<i) & ~(1<<j)  ... but we need this as a uint64 array? 
# No — each case is a uint64 representing 17 bits. We want to know, for a given case, whether wire i > wire j.
# For boolean: wire i > wire j iff bit_i=1 and bit_j=0.
# We can compute this per-case using numpy bitwise ops on the _CASES_ALL array.

# For speed, precompute for each pair (i,j) the mask of cases where swap is needed:
# swap_needed[i,j][k] = 1 if in case k, bit i=1 and bit j=0
# We store these as uint8 arrays of shape (NUM_CASES,) for each (i,j).
# But that's 136 * 131072 bytes = ~18MB, acceptable.

# Actually, for the verification loop, we can compute on-the-fly using numpy:
# Given current state S (uint64 array of shape NUM_CASES), and comparator (i,j):
#   swap_mask = (S & (1<<i)) & ~(S & (1<<j))
#   S = S ^ (swap_mask * ((1<<i) | (1<<j)))  -- but this needs care
# Let's think: if bit i=1 and bit j=0, we want to swap: set bit i=0, bit j=1.
#   S_new = S ^ (swap_mask & ((1<<i) | (1<<j)))
#   where swap_mask = (S & (1<<i)) & ~(S & (1<<j))

def _apply_comparator(S, i, j):
    """Apply compare-exchange (i,j) to state array S. Returns new S."""
    # swap_mask: cases where wire i has 1 and wire j has 0
    mask_i = S & _WIRE_MASKS[i]
    mask_j = S & _WIRE_MASKS[j]
    swap_mask = mask_i & ~mask_j
    # Flip bits i and j where swap is needed
    flip = swap_mask & (_WIRE_MASKS[i] | _WIRE_MASKS[j])
    return S ^ flip

def verify_network(net):
    """Verify that network sorts all 2^17 boolean inputs. Returns True if valid."""
    S = _CASES_ALL.copy()
    for (i, j) in net:
        S = _apply_comparator(S, i, j)
    # Check sorted: for each adjacent pair (k, k+1), wire k should be <= wire k+1
    # Unsorted if bit k=1 and bit k+1=0
    for k in range(N - 1):
        bad = (S & _WIRE_MASKS[k]) & ~(S & _WIRE_MASKS[k + 1])
        if bad.any():
            return False
    return True

def _batcher_network(n):
    """Return Batcher odd-even-merge sorting network for n wires (0-indexed)."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    m = n // 2
    left = _batcher_network(m)
    right = _batcher_network(n - m)
    right_shifted = [(i + m, j + m) for (i, j) in right]
    merged = _odd_even_merge_network(m, n - m)
    return left + right_shifted + merged

def _odd_even_merge_network(a, b):
    """Odd-even merge two sorted lists of length a and b (a <= b), total a+b wires.
    Wires 0..a-1 are the first sorted list, wires a..a+b-1 are the second.
    Returns comparators to merge them."""
    if a == 0 or b == 0:
        return []
    if a == 1 and b == 1:
        return [(0, 1)]
    
    # Standard odd-even merge:
    # Split first list into even-indexed (positions 0,2,4,...) and odd-indexed (1,3,5,...)
    # Split second list into even-indexed and odd-indexed
    # Recursively merge evens, recursively merge odds
    # Then compare-exchange adjacent pairs
    
    a_even = (a + 1) // 2  # number of even-indexed elements in first list
    a_odd = a // 2
    b_even = (b + 1) // 2
    b_odd = b // 2
    
    result = []
    
    # The even-indexed elements of the first list are at positions 0, 2, 4, ...
    # The even-indexed elements of the second list are at positions a, a+2, a+4, ...
    # After merging, they should be at the even positions of the merged list.
    # This is complex with position tracking. Let me use a different approach.
    
    # Alternative: use the recursive definition directly on indices.
    # For odd-even merge of two sorted arrays A[0..a-1] and B[0..b-1]:
    # 1. Split A into A_even (indices 0,2,4,...) and A_odd (indices 1,3,5,...)
    # 2. Split B into B_even and B_odd
    # 3. Recursively merge A_even and B_even
    # 4. Recursively merge A_odd and B_odd
    # 5. Compare-exchange adjacent pairs in the result
    
    # The merged result has a+b elements. The even positions get the merged evens,
    # odd positions get the merged odds.
    
    # Positions in the merged output:
    # Even positions: 0, 2, 4, ..., up to a+b-1
    # Odd positions: 1, 3, 5, ..., up to a+b-1
    
    # Number of even positions in output of length a+b: (a+b+1)//2
    # Number of odd positions: (a+b)//2
    
    # A_even has a_even elements, B_even has b_even elements.
    # a_even + b_even should equal (a+b+1)//2 or (a+b)//2 depending on parity.
    
    # This is getting complicated. Let me use a known correct implementation.
    
    # Simpler approach: just use a "brick wall" merge for small sizes, 
    # and for the top level use a decent heuristic.
    # For n=17, the merge is of 8 and 9. Let's just do a full pairwise merge
    # which is O(a*b) = 72 comparators for the merge, plus the recursive sorts.
    # Total will be around 100-120, which is fine as a seed.
    
    # Actually, let me implement proper odd-even merge.
    # Key insight: we can track which physical wire corresponds to which logical position.
    
    # Let me use a different strategy: implement the merge by explicitly 
    # constructing the comparator list using the recursive structure.
    
    def _merge_with_tracking(a, b, offset=0):
        """Merge two sorted lists at physical positions [offset, offset+a) and [offset+a, offset+a+b).
        Returns list of (i,j) comparators."""
        if a == 0 or b == 0:
            return []
        if a == 1 and b == 1:
            return [(offset, offset + 1)]
        
        a_even = (a + 1) // 2
        a_odd = a // 2
        b_even = (b + 1) // 2
        b_odd = b // 2
        
        # Physical positions of even-indexed elements of first list:
        # offset, offset+2, offset+4, ... (a_even elements)
        # Physical positions of odd-indexed elements of first list:
        # offset+1, offset+3, offset+5, ... (a_odd elements)
        # Similarly for second list starting at offset+a.
        
        # After recursive merge, the evens occupy the even positions of the merged output,
        # and the odds occupy the odd positions.
        
        # Merged output positions: offset, offset+1, ..., offset+a+b-1
        # Even positions in output: offset, offset+2, offset+4, ...
        # Odd positions in output: offset+1, offset+3, offset+5, ...
        
        # The merged evens (a_even + b_even elements) go to even output positions.
        # The merged odds (a_odd + b_odd elements) go to odd output positions.
        
        # We need to map logical positions to physical positions.
        # This requires tracking which physical wire holds which logical element.
        
        # This is getting too complex for an inline implementation.
        # Let me just use a simple but correct merge: compare all cross pairs.
        
        result = []
        for i in range(a):
            for j in range(a, a + b):
                result.append((offset + i, offset + j))
        return result
    
    return _merge_with_tracking(a, b)

def make_seed():
    """Create a valid sorting network seed."""
    net = _batcher_network(N)
    # Remove duplicates
    seen = set()
    unique = []
    for c in net:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique

def greedy_remove(net, deadline):
    """Try to remove each comparator; keep removal if network stays valid."""
    net = list(net)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in range(len(net) - 1, -1, -1):
            if time.time() >= deadline:
                break
            trial = net[:idx] + net[idx+1:]
            if verify_network(trial):
                net = trial
                changed = True
    return net

def simulated_annealing(net, deadline):
    """Simulated annealing to minimize comparator count."""
    net = list(net)
    best_net = list(net)
    best_len = len(net)
    
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    T = 1.0
    cooling = 0.999
    min_T = 0.001
    
    random.seed(42)
    
    while time.time() < deadline and T > min_T:
        # Choose a move
        move_type = random.random()
        trial = list(net)
        
        if move_type < 0.5 and len(net) > 0:
            # Remove a random comparator
            idx = random.randint(0, len(net) - 1)
            trial.pop(idx)
        elif move_type < 0.75:
            # Add a random comparator
            c = random.choice(all_pairs)
            if c not in trial:
                trial.append(c)
            else:
                continue
        else:
            # Swap two comparators
            if len(net) >= 2:
                i, j = random.sample(range(len(net)), 2)
                trial[i], trial[j] = trial[j], trial[i]
            else:
                continue
        
        if len(trial) == len(net):
            # Same length: always accept if valid (it's a rearrangement)
            if verify_network(trial):
                net = trial
                if len(net) < best_len:
                    best_net = list(net)
                    best_len = len(net)
        else:
            # Different length: check validity
            if verify_network(trial):
                if len(trial) < len(net):
                    net = trial
                    if len(net) < best_len:
                        best_net = list(net)
                        best_len = len(net)
                else:
                    # Accept with probability exp(-delta/T)
                    delta = len(trial) - len(net)
                    if random.random() < np.exp(-delta / T):
                        net = trial
            # If invalid, reject
        
        T *= cooling
    
    # Final greedy removal on best
    best_net = greedy_remove(best_net, deadline)
    return best_net

def construct():
    """Construct a sorting network for N=17 wires. Returns list of (i,j) pairs."""
    start_time = time.time()
    deadline = start_time + 85.0  # 5 second safety margin before 90s kill
    
    # Phase 1: Create seed
    net = make_seed()
    
    # Phase 2: Greedy removal
    net = greedy_remove(net, deadline)
    
    # Phase 3: Simulated annealing if time permits
    if time.time() < deadline - 5.0:
        net = simulated_annealing(net, deadline)
    
    # Phase 4: Final greedy removal
    net = greedy_remove(net, deadline)
    
    # Final verification
    if not verify_network(net):
        # Fallback: odd-even transposition
        net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    return net
# EVOLVE-BLOCK-END
