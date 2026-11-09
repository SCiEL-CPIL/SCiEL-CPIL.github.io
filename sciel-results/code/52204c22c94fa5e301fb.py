"""Size-efficient sorting networks on 17 wires.
Uses fast big-integer verification + greedy removal with deadline guard.
"""
import time
import random

N = 17
MAX_COMPARATORS = 4096
DEADLINE_MARGIN = 2.0  # seconds before the 90s kill
START_TIME = time.time()
TIME_LIMIT = 90.0 - DEADLINE_MARGIN


def _seed_network():
    """Odd-even transposition sorting network for N wires."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps


def _verify(comps):
    """Verify that the compare-exchange network sorts all 2^N boolean inputs.
    Uses the zero-one principle and fast bitwise operations.
    Returns True if valid, False otherwise.
    """
    n = len(comps)
    # For each bit position b (0..N-1), we track which wire holds the bit.
    # Represent state as: for each of N bits, which wire it's on.
    # Use an integer where bit (w*N + b) indicates bit b is on wire w.
    # Actually, simpler: represent the mapping from bits to wires.
    # State: list of length N where state[w] = which bit is on wire w.
    # But for speed with big ints: encode as N*N bits.
    
    # Alternative faster approach: 
    # For a given boolean input x (N bits), simulate the network.
    # But 2^17 = 131072 inputs, and each simulation is O(n) comparisons.
    # With n ~ 100-136, that's ~17M operations per full check. Too slow in Python.
    
    # Better: use the "trace" method.
    # For each output wire j, determine which input wire's value ends up there
    # for a given input pattern. But this depends on the input.
    
    # Fastest known method for zero-one principle:
    # For each subset S of wires (represented as bitmask), check that after
    # applying the network, the wires holding 1s are the top |S| wires.
    # There are 2^N = 131072 subsets. For each, simulate the network.
    # Each simulation: for each comparator (i,j), if wire i has 1 and wire j has 0, swap.
    # Represent the set of wires with 1s as a bitmask (N bits).
    # Comparator (i,j): if bit i is set and bit j is not, swap them.
    # This is: if (mask >> i) & 1 and not (mask >> j) & 1: mask ^= (1<<i) | (1<<j)
    
    # Let's use this. 131072 * n operations. For n=100, that's ~13M simple ops.
    # In Python this might take ~5-10 seconds. Need to be careful.
    
    # Even faster: precompute for each comparator the effect on a mask.
    # But the effect depends on the mask state.
    
    # Let's just do it directly but efficiently.
    
    for mask in range(1 << N):
        m = mask
        for (i, j) in comps:
            # If bit i is 1 and bit j is 0, swap
            if (m >> i) & 1:
                if not (m >> j) & 1:
                    m ^= (1 << i) | (1 << j)
        # Check: the top popcount(mask) wires should be 1
        popcount = bin(mask).count('1')
        if popcount == 0:
            if m != 0:
                return False
        else:
            # Top popcount wires: bits N-popcount to N-1
            expected = ((1 << N) - 1) ^ ((1 << (N - popcount)) - 1)
            if m != expected:
                return False
    return True


def _verify_fast(comps):
    """Faster verification using precomputed comparator effects.
    For each comparator (i,j), precompute:
    - For each possible state of bits i and j (4 states), the transformation.
    But since we're working with masks, let's try a different approach.
    
    Use the fact that we only need to check 2^N masks.
    Optimize the inner loop.
    """
    n_comps = len(comps)
    # Pre-extract i, j as separate lists for faster iteration
    is_list = [c[0] for c in comps]
    js_list = [c[1] for c in comps]
    
    for mask in range(1 << N):
        m = mask
        for k in range(n_comps):
            i = is_list[k]
            j = js_list[k]
            # Fast swap if needed
            bi = (m >> i) & 1
            bj = (m >> j) & 1
            if bi and not bj:
                m ^= (1 << i) | (1 << j)
        popcount = bin(mask).count('1')
        if popcount == 0:
            if m != 0:
                return False
        else:
            expected = ((1 << N) - 1) ^ ((1 << (N - popcount)) - 1)
            if m != expected:
                return False
    return True


def _verify_fastest(comps):
    """Even faster: use lookup tables for pairs of bits.
    Actually, let's try a different strategy: instead of iterating over all 2^N masks,
    use the structure of the network.
    
    Alternative: use the "column" method or just accept the 2^N iteration
    but optimize the inner loop heavily.
    
    Let's try: for each mask, the simulation can be done by tracking
    only the relevant bits. But that's complex.
    
    Let's just make the inner loop as fast as possible.
    """
    n_comps = len(comps)
    is_list = [c[0] for c in comps]
    js_list = [c[1] for c in comps]
    # Precompute 1<<i and 1<<j for all comparators
    masks_i = [1 << i for i in is_list]
    masks_j = [1 << j for j in js_list]
    # Combined mask for swap
    swap_masks = [mi | mj for mi, mj in zip(masks_i, masks_j)]
    
    for mask in range(1 << N):
        m = mask
        for k in range(n_comps):
            mi = masks_i[k]
            mj = masks_j[k]
            # Check if bit i is set and bit j is not
            if (m & mi) and not (m & mj):
                m ^= swap_masks[k]
        popcount = bin(mask).count('1')
        if popcount == 0:
            if m != 0:
                return False
        else:
            expected = ((1 << N) - 1) ^ ((1 << (N - popcount)) - 1)
            if m != expected:
                return False
    return True


def _verify_superfast(comps):
    """Super fast verification using precomputed results for sub-masks.
    
    Key insight: for a comparator (i,j), the operation on mask m is:
    if (m has bit i set) and (m has bit j clear): swap bits i and j.
    
    We can precompute for each comparator and each of the 4 possible states
    of (bit_i, bit_j) what the resulting change is. But the state depends on m.
    
    Alternative: use numpy? No, let's stick with Python but be smarter.
    
    Actually, let's use a different verification strategy:
    Instead of checking all 2^N inputs, we can use the fact that the network
    is a sorting network if and only if it sorts all 2^N boolean inputs.
    But we can speed this up by early termination and by using the structure.
    
    Let's try: for each mask, simulate and check. But use a tighter loop.
    """
    n_comps = len(comps)
    if n_comps == 0:
        return N <= 1
    
    is_list = [c[0] for c in comps]
    js_list = [c[1] for c in comps]
    masks_i = [1 << i for i in is_list]
    masks_j = [1 << j for j in js_list]
    swap_masks = [mi | mj for mi, mj in zip(masks_i, masks_j)]
    
    # Precompute expected values for each popcount
    expected_for_popcount = {}
    for pc in range(N + 1):
        if pc == 0:
            expected_for_popcount[pc] = 0
        else:
            expected_for_popcount[pc] = ((1 << N) - 1) ^ ((1 << (N - pc)) - 1)
    
    for mask in range(1 << N):
        m = mask
        for k in range(n_comps):
            mi = masks_i[k]
            mj = masks_j[k]
            if (m & mi) and not (m & mj):
                m ^= swap_masks[k]
        pc = bin(mask).count('1')
        if m != expected_for_popcount[pc]:
            return False
    return True


def _time_remaining():
    return TIME_LIMIT - (time.time() - START_TIME)


def construct():
    """Construct a sorting network for N=17 wires.
    Returns a list of (i, j) pairs.
    """
    global START_TIME
    START_TIME = time.time()
    
    # Start with the seed network
    comps = _seed_network()
    best_comps = comps[:]
    best_count = len(comps)
    
    # Verify the seed
    if not _verify_superfast(comps):
        # This shouldn't happen, but just in case
        return comps
    
    # Greedy removal: try to remove comparators one at a time
    # For each comparator, try removing it and check if the network still sorts.
    # To speed up, we can use a heuristic: try removing comparators that seem redundant.
    
    # Strategy: iterate through comparators in reverse order (later ones might be redundant)
    # and try to remove each one.
    
    i = 0
    while i < len(comps) and _time_remaining() > 5.0:
        # Try removing comparator at position i
        trial = comps[:i] + comps[i+1:]
        if _verify_superfast(trial):
            comps = trial
            best_comps = comps[:]
            best_count = len(comps)
            # Don't increment i, try to remove the next one (which is now at position i)
        else:
            i += 1
    
    # If we have time, do multiple passes
    passes = 0
    while _time_remaining() > 10.0:
        improved = False
        i = 0
        while i < len(comps) and _time_remaining() > 5.0:
            trial = comps[:i] + comps[i+1:]
            if _verify_superfast(trial):
                comps = trial
                best_comps = comps[:]
                best_count = len(comps)
                improved = True
            else:
                i += 1
        passes += 1
        if not improved:
            break
    
    return best_comps
