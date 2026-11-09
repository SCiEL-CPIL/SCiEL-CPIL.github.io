import time
import numpy as np

N = 18
NUM = 1 << N          # 262144
DEADLINE = 55.0       # safety margin before 90s kill

# Precompute the bitmask of unsorted inputs for each wire pair.
# For input b (0..2^N-1), wire w holds bit w of b.
# Input b is unsorted iff exists k: bit k of b = 1 and bit k+1 of b = 0.
# We precompute for each (i,j) the bitmask of inputs where wire i > wire j.
# Then a comparator (i,j) is "needed" for an input iff that input is in mask[i,j]
# AND the input is currently unsorted.

# Precompute masks[i][j] for all i<j: bitmask of inputs where bit_i=1 and bit_j=0
def _precompute_masks():
    masks = {}
    # For each pair (i,j), compute the set of inputs where bit i = 1 and bit j = 0.
    # This is a periodic pattern. We can compute it as a 2^N-bit integer.
    # For efficiency, build via block structure.
    # bit i has period 2^(i+1): blocks of 2^i ones then 2^i zeros.
    # We want inputs where bit i = 1 AND bit j = 0.
    # Represent as a Python int (arbitrary precision) of NUM bits.
    for i in range(N):
        for j in range(i + 1, N):
            # Build the mask
            # bit i = 1 in ranges [2^i, 2^(i+1)), [2^(i+2), 2^(i+3)), ...
            # bit j = 0 in ranges [0, 2^j), [2^(j+1), 2^(j+2)), ...
            # Intersection is complex; just iterate over the pattern via numpy for speed.
            # Actually, let's build it directly using bit operations on a large int.
            # We'll build it as: for each block of size 2^max(i,j+1), compute the pattern.
            # Simpler: use numpy to create the array then pack into int.
            # But 2^18 = 262144, and we have C(18,2)=153 pairs.
            # Let's build each mask as a Python int from a bytes representation.
            # Create the boolean array: bit_i & ~bit_j
            # bit_i: pattern of 2^i ones, 2^i zeros, repeated.
            # We can build it as: np.tile(np.concatenate([np.ones(2**i), np.zeros(2**i)]), 2**(N-i-1))
            # But that's 2^18 elements per mask, and 153 masks -> 40M elements total. Doable.
            # Better: build all masks at once using broadcasting? Hard.
            # Let's just build each mask as a Python int via a loop over the periodic structure.
            # Actually, the fastest: build the bit_i array once, then for each j, mask = bit_i & ~bit_j.
            pass
    return masks

# Precompute per-wire bit patterns as Python ints (NUM bits each)
def _wire_bit_pattern(w):
    """Return a Python int of NUM bits where bit b is 1 iff wire w of input b is 1."""
    # bit w of input b is 1 iff (b >> w) & 1 == 1
    # Pattern: blocks of 2^w ones, 2^w zeros, repeated 2^(N-w-1) times.
    # Build as Python int.
    # We can build it via: sum over blocks.
    # Block k (0-indexed) covers inputs [k*2^(w+1), (k+1)*2^(w+1)).
    # Within it, the second half [k*2^(w+1)+2^w, (k+1)*2^(w+1)) has bit w = 1.
    # So the mask is: for each k, set bits in range [k*2^(w+1)+2^w, (k+1)*2^(w+1)).
    # We can build this as a big int by OR-ing shifted masks.
    # A simpler way: build a bytes array.
    # Let's build it as a Python int by constructing the bit string.
    # For NUM=262144, we need a 262144-bit int.
    # Build via numpy: create uint8 array of length NUM, then pack.
    # Actually, Python int from a bytes object is efficient.
    # Create the boolean pattern as a numpy array of uint8, then pack into bytes, then int.
    # But we need to be careful about bit ordering.
    # Let's use a different approach: build the int directly.
    # The pattern for bit w: 
    #   period = 2^(w+1)
    #   In each period, bits [2^w, 2^(w+1)) are 1.
    # We can build it as:
    #   base = (2^(2^w) - 1) << 2^w   # one period's worth of the "ones" region
    #   Then repeat: for k in range(2^(N-w-1)): base << (k*period)
    #   OR them all together.
    period = 1 << (w + 1)
    half = 1 << w
    # ones in positions [half, period) within each period
    # As a bitmask of 'period' bits: ((2^half - 1) << half)
    period_mask = ((1 << half) - 1) << half
    # Number of periods
    num_periods = 1 << (N - w - 1)
    # Build the full mask
    result = 0
    for k in range(num_periods):
        result |= period_mask << (k * period)
    return result

# Precompute all wire patterns
_WIRE_PATTERNS = []
for _w in range(N):
    _WIRE_PATTERNS.append(_wire_bit_pattern(_w))

# Precompute unsorted mask: inputs that are not sorted.
# An input b is sorted iff for all k, bit k <= bit k+1.
# Unsorted iff exists k: bit k = 1 and bit k+1 = 0.
# Unsorted mask = OR over k of (wire_k & ~wire_{k+1})
_UNSORTED_MASK = 0
for _k in range(N - 1):
    _UNSORTED_MASK |= (_WIRE_PATTERNS[_k] & ~_WIRE_PATTERNS[_k + 1]) & ((1 << NUM) - 1)

# Precompute comparator masks: for pair (i,j), the mask of inputs where wire i > wire j.
# That is wire_i & ~wire_j.
_COMP_MASKS = {}
for _i in range(N):
    for _j in range(_i + 1, N):
        _COMP_MASKS[(_i, _j)] = (_WIRE_PATTERNS[_i] & ~_WIRE_PATTERNS[_j]) & ((1 << NUM) - 1)

def _apply_comparator(state, i, j):
    """Apply comparator (i,j) to the state (list of NUM-bit ints, one per wire).
    Returns new state. min to i, max to j."""
    # new_i = state[i] & state[j]  (min)
    # new_j = state[i] | state[j]  (max)
    # For Boolean inputs, min(a,b) = a & b, max(a,b) = a | b.
    new_i = state[i] & state[j]
    new_j = state[i] | state[j]
    new_state = state[:]
    new_state[i] = new_i
    new_state[j] = new_j
    return new_state

def _initial_state():
    """Initial state: wire w holds the bit pattern of wire w."""
    return _WIRE_PATTERNS[:]

def _is_sorted(state):
    """Check if state is sorted: for all k, state[k] <= state[k+1] (bitwise).
    For Boolean, state[k] <= state[k+1] iff (state[k] & ~state[k+1]) == 0."""
    for k in range(N - 1):
        if (state[k] & ~state[k + 1]) != 0:
            return False
    return True

def _unsorted_count(state):
    """Return the number of unsorted inputs in the state."""
    unsorted = 0
    for k in range(N - 1):
        unsorted |= (state[k] & ~state[k + 1])
    return bin(unsorted).count('1')

def _greedy_build():
    """Build a sorting network greedily using the 0-1 principle.
    At each step, pick the comparator that most reduces the unsorted count."""
    state = _initial_state()
    network = []
    
    # Precompute for each (i,j) the mask of inputs where wire i > wire j.
    # We already have _COMP_MASKS.
    
    # Greedy: repeatedly pick the (i,j) that minimizes the unsorted count after application.
    # To speed up, we can precompute the effect of each comparator on the current unsorted set.
    
    max_iters = 200
    for _ in range(max_iters):
        if time.time() > DEADLINE:
            break
        if _is_sorted(state):
            break
        
        # Find the best comparator
        best_gain = -1
        best_pair = None
        # For each (i,j), compute the new unsorted count if we apply (i,j).
        # To do this fast, we need to compute the new state's unsorted mask.
        # new_state[i] = state[i] & state[j]
        # new_state[j] = state[i] | state[j]
        # Other wires unchanged.
        # The unsorted mask changes only in comparisons involving i or j.
        # Specifically, pairs (i,k) for k != j, and (k,j) for k != i, and (i,j) itself.
        # This is complex. Let's just compute the full unsorted count for candidate pairs.
        # But that's O(N^2 * N) per step, which is fine for a few hundred steps.
        
        # Actually, let's just try all (i,j) and compute the new unsorted count.
        for i in range(N):
            for j in range(i + 1, N):
                # Apply (i,j) to get new state
                new_i = state[i] & state[j]
                new_j = state[i] | state[j]
                # Compute new unsorted count
                # We need to check all adjacent pairs in the new state.
                # New state: same as state except wire i = new_i, wire j = new_j.
                unsorted = 0
                for k in range(N - 1):
                    a = state[k] if k != i and k != j else (new_i if k == i else new_j)
                    b = state[k + 1] if k + 1 != i and k + 1 != j else (new_i if k + 1 == i else new_j)
                    unsorted |= (a & ~b)
                cnt = bin(unsorted).count('1')
                gain = _unsorted_count(state) - cnt
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (i, j)
        
        if best_pair is None or best_gain <= 0:
            break
        
        i, j = best_pair
        network.append((i, j))
        state = _apply_comparator(state, i, j)
    
    return network

def _prune(network):
    """Prune redundant comparators from the network."""
    if not network:
        return network
    
    # Verify the network is valid first
    state = _initial_state()
    for i, j in network:
        state = _apply_comparator(state, i, j)
    if not _is_sorted(state):
        return network  # Don't prune if not valid
    
    changed = True
    while changed and time.time() < DEADLINE:
        changed = False
        for idx in range(len(network) - 1, -1, -1):
            if time.time() > DEADLINE:
                break
            # Remove comparator at idx
            comp = network.pop(idx)
            # Check validity
            state = _initial_state()
            for i, j in network:
                state = _apply_comparator(state, i, j)
            if _is_sorted(state):
                changed = True
            else:
                network.insert(idx, comp)
    return network

def construct():
    start = time.time()
    
    # Build network greedily
    network = _greedy_build()
    
    # Prune
    network = _prune(network)
    
    # Final safety check
    if time.time() > DEADLINE - 5:
        # Return what we have
        return network
    
    # Verify
    state = _initial_state()
    for i, j in network:
        state = _apply_comparator(state, i, j)
    if not _is_sorted(state):
        # Fallback: return a known valid network (odd-even transposition)
        network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    return network

# Precompute at module load time (should be fast)
# _WIRE_PATTERNS and _UNSORTED_MASK are already computed above.
