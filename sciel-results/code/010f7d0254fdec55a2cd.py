"""Size-efficient sorting networks on 17 wires.
Uses fast big-integer based verification and a greedy removal + mutation search
to minimize comparator count within the 90s wall-clock limit.
"""
import time
import random

N = 17
MASK = (1 << (1 << N)) - 1  # 2^17 - 1 = 131071

# Precompute: for each comparator (i,j), the bit masks needed to check correctness
# Actually, for verification we use the zero-one principle:
# A sorting network sorts all 2^N binary inputs iff it sorts all inputs where
# the output is sorted (all 0s then all 1s, or equivalently the largest input
# goes to the last wire).

# More efficient: simulate all 2^N inputs through the network using bit-parallelism.
# Each "state" across all 2^N inputs can be represented as N integers, one per wire,
# where bit k of wire w's integer tells us the value on wire w for input k.

def _verify(network):
    """Verify the sorting network using the zero-one principle with bit-parallelism.
    Returns True if the network sorts all 2^N binary inputs.
    """
    # Initialize: for input k (0 to 2^N - 1), wire w has bit (k >> w) & 1
    # So wire w's integer has bits: for k in 0..2^N-1, bit k = (k >> w) & 1
    # This is: bits w, w+1, ..., w+(N-1-w) are set in a pattern...
    # Actually: bit k of wire w = (k >> w) & 1
    # So wire w's integer = sum over k of ((k >> w) & 1) << k
    # This equals: for each position where bit w of k is 1, set bit k.
    # k ranges 0 to 2^N-1. Bit w of k is 1 for k in ranges:
    # [2^w, 2^(w+1)-1], [2^(w+2), 2^(w+2)+2^(w+1)-1], ...
    
    wires = [0] * N
    for w in range(N):
        # Build the integer for wire w
        val = 0
        for k in range(1 << N):
            if (k >> w) & 1:
                val |= (1 << k)
        wires[w] = val
    
    # Apply each comparator
    for (i, j) in network:
        a = wires[i]
        b = wires[j]
        # min: a AND b, max: a OR b
        wires[i] = a & b
        wires[j] = a | b
    
    # Check: after sorting, the output should be sorted.
    # For all binary inputs, after sorting, wire 0 should have the minimum (all 0s for inputs with 0 ones, etc.)
    # Actually, the sorted output for input with 's' ones is: wires 0..N-s-1 = 0, wires N-s..N-1 = 1
    # So wire w should be 1 for input k iff k has more than N-1-w ones... no.
    # Wire w = 1 iff the w-th smallest element is 1 iff input has more than N-1-w ones... 
    # Wait: sorted in non-decreasing order. Wire 0 = min, wire N-1 = max.
    # Wire w = 1 iff there are more than N-1-w zeros... 
    # Let me think: if input has 's' ones, then sorted output is (N-s) zeros followed by s ones.
    # So wire w = 1 iff w >= N - s, i.e., s >= N - w, i.e., the number of ones in input k is at least N - w.
    # So wire w should be 1 for input k iff popcount(k) >= N - w.
    
    for w in range(N):
        expected = 0
        for k in range(1 << N):
            if bin(k).count('1') >= N - w:
                expected |= (1 << k)
        if wires[w] != expected:
            return False
    return True


def _verify_fast(network):
    """Faster verification using precomputed expected values."""
    # Precompute expected wire values
    expected_wires = [0] * N
    for w in range(N):
        val = 0
        for k in range(1 << N):
            if bin(k).count('1') >= N - w:
                val |= (1 << k)
        expected_wires[w] = val
    
    # Initial wire values
    wires = [0] * N
    for w in range(N):
        val = 0
        for k in range(1 << N):
            if (k >> w) & 1:
                val |= (1 << k)
        wires[w] = val
    
    for (i, j) in network:
        a = wires[i]
        b = wires[j]
        wires[i] = a & b
        wires[j] = a | b
    
    for w in range(N):
        if wires[w] != expected_wires[w]:
            return False
    return True


# Even faster: use a different representation.
# Represent all 2^N inputs as N big integers where bit k of wire w = value on wire w for input k.
# But we can precompute the initial values more efficiently.

def _precompute_initial_wires():
    """Precompute initial wire values for all 2^N binary inputs."""
    wires = [0] * N
    for w in range(N):
        val = 0
        # Bit k of wire w = (k >> w) & 1
        # Set bit k in val for all k where bit w of k is 1
        block = 1 << w
        for start in range(0, 1 << N, block * 2):
            for k in range(start + block, start + 2 * block):
                val |= (1 << k)
        wires[w] = val
    return wires


def _precompute_expected_wires():
    """Precompute expected sorted wire values."""
    wires = [0] * N
    for w in range(N):
        val = 0
        for k in range(1 << N):
            if bin(k).count('1') >= N - w:
                val |= (1 << k)
        wires[w] = val
    return wires


_INITIAL_WIRES = _precompute_initial_wires()
_EXPECTED_WIRES = _precompute_expected_wires()


def _verify_quick(network):
    """Quick verification using precomputed values."""
    wires = _INITIAL_WIRES[:]
    for (i, j) in network:
        a = wires[i]
        b = wires[j]
        wires[i] = a & b
        wires[j] = a | b
    for w in range(N):
        if wires[w] != _EXPECTED_WIRES[w]:
            return False
    return True


def _odd_even_transposition_network():
    """Generate the odd-even transposition sorting network for N wires."""
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net


def _bitonic_network(n):
    """Generate a bitonic sorting network for n wires (must be power of 2)."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    left = _bitonic_network(half)
    right = [(i + half, j + half) for (i, j) in left]
    
    # Merge phase
    merge = []
    for k in range(1, half + 1):
        for i in range(n - k):
            if i < half and i + k < n:
                # Compare (i, i+k) and (n-1-i-k, n-1-i)
                merge.append((i, i + k))
                merge.append((n - 1 - i - k, n - 1 - i))
    
    return left + right + merge


def construct():
    """Construct a sorting network for 17 wires, minimizing comparator count."""
    start_time = time.time()
    deadline = 85.0  # Leave 5s safety margin
    
    # Start with odd-even transposition (136 comparators)
    best = _odd_even_transposition_network()
    best_size = len(best)
    
    if not _verify_quick(best):
        # Should not happen, but fallback
        return best
    
    # Try to reduce via greedy removal
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        # Try removing each comparator
        for idx in range(len(best)):
            if (time.time() - start_time) > deadline:
                break
            trial = best[:idx] + best[idx+1:]
            if _verify_quick(trial):
                best = trial
                best_size -= 1
                improved = True
                break  # Restart from beginning after a removal
        if (time.time() - start_time) > deadline:
            break
    
    # Now try mutation-based search: swap, insert, delete, etc.
    random.seed(42)
    stuck_count = 0
    
    while (time.time() - start_time) < deadline:
        # Try various mutations
        mutated = False
        
        # 1. Try removing a random comparator
        if len(best) > 0:
            idx = random.randint(0, len(best) - 1)
            trial = best[:idx] + best[idx+1:]
            if _verify_quick(trial):
                best = trial
                best_size -= 1
                mutated = True
                stuck_count = 0
            else:
                stuck_count += 1
        
        # 2. Try swapping two adjacent comparators
        if not mutated and len(best) > 1:
            idx = random.randint(0, len(best) - 2)
            trial = best[:]
            trial[idx], trial[idx+1] = trial[idx+1], trial[idx]
            if _verify_quick(trial):
                best = trial
                mutated = True
                stuck_count = 0
            else:
                stuck_count += 1
        
        # 3. Try replacing a comparator with a different one
        if not mutated and len(best) > 0:
            idx = random.randint(0, len(best) - 1)
            i, j = best[idx]
            # Try a random different comparator
            ni = random.randint(0, N - 1)
            nj = random.randint(0, N - 1)
            if ni == nj:
                nj = (nj + 1) % N
            if ni > nj:
                ni, nj = nj, ni
            if (ni, nj) == (i, j):
                continue
            trial = best[:]
            trial[idx] = (ni, nj)
            if _verify_quick(trial):
                best = trial
                mutated = True
                stuck_count = 0
            else:
                stuck_count += 1
        
        # 4. Try inserting a comparator at a random position
        if not mutated and best_size < 4096:
            pos = random.randint(0, len(best))
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i == j:
                j = (j + 1) % N
            if i > j:
                i, j = j, i
            trial = best[:]
            trial.insert(pos, (i, j))
            if _verify_quick(trial):
                best = trial
                best_size += 1
                mutated = True
                stuck_count = 0
            else:
                stuck_count += 1
        
        # 5. Try reordering: move a comparator to a different position
        if not mutated and len(best) > 1:
            src = random.randint(0, len(best) - 1)
            dst = random.randint(0, len(best) - 1)
            if src == dst:
                dst = (dst + 1) % len(best)
            trial = best[:]
            comp = trial.pop(src)
            trial.insert(dst, comp)
            if _verify_quick(trial):
                best = trial
                mutated = True
                stuck_count = 0
            else:
                stuck_count += 1
        
        # Break if stuck too many times
        if stuck_count > 200:
            # Try a more aggressive reset: go back to a slightly larger network
            # and try again
            if best_size < len(_odd_even_transposition_network()):
                # We've made progress, so keep trying but reset stuck counter
                stuck_count = 0
            else:
                break
    
    # Final greedy removal pass
    for idx in range(len(best)):
        if (time.time() - start_time) > deadline:
            break
        trial = best[:idx] + best[idx+1:]
        if _verify_quick(trial):
            best = trial
    
    return best
