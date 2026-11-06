"""Size-efficient sorting network synthesis for 18 wires."""
import time
import random
import math

N = 18
_TOTAL_BITS = 1 << N  # 262144

_INITIAL_BITSETS = None

def _init_bitsets():
    """Initialize 18 bitsets, each of length 2^N."""
    total_bits = _TOTAL_BITS
    bitsets = [0] * N
    
    for k in range(N):
        period = 1 << (k + 1)
        half_period = 1 << k
        pattern = (1 << half_period) - 1
        pattern <<= half_period
        
        current_val = pattern
        current_len = period
        
        while current_len < total_bits:
            current_val |= (current_val << current_len)
            current_len *= 2
            
        bitsets[k] = current_val

    return bitsets

def _get_initial_bitsets():
    global _INITIAL_BITSETS
    if _INITIAL_BITSETS is None:
        _INITIAL_BITSETS = _init_bitsets()
    return _INITIAL_BITSETS

def _verify(comparators):
    """Check if network sorts all 2^N Boolean inputs using bitset simulation."""
    if not comparators:
        return N <= 1
        
    bitsets = list(_get_initial_bitsets())
    
    for i, j in comparators:
        bi = bitsets[i]
        bj = bitsets[j]
        bitsets[i] = bi & bj
        bitsets[j] = bi | bj
        
    for k in range(N - 1):
        if bitsets[k] & ~bitsets[k + 1]:
            return False
            
    return True

def _odd_even_transposition():
    """Odd-even transposition sorting network for N wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _batcher_odd_even_merge(n):
    """Odd-even merge of n already-sorted halves. Returns (i,j) list."""
    if n <= 1:
        return []
    result = []

    def _oem(k, offset):
        if k <= 1:
            return
        half_k = k // 2
        for i in range(offset, offset + k, 2):
            _oem(half_k, i)
        for i in range(offset + 1, offset + k, 2):
            _oem(half_k, i)
        for i in range(offset, offset + k - 1):
            result.append((i, i + 1))

    _oem(n, 0)
    return result

def _batcher_odd_even_mergesort(n):
    """Batcher's odd-even mergesort for n (power-of-2) wires."""
    if n <= 1:
        return []
    half = n // 2
    left = _batcher_odd_even_mergesort(half)
    right = _batcher_odd_even_mergesort(half)
    right = [(i + half, j + half) for i, j in right]
    merge = _batcher_odd_even_merge(n)
    return left + right + merge

def _build_batcher_network():
    """Build Batcher network on next power of 2 >= N, restricted to N wires."""
    p = 1
    while p < N:
        p *= 2
    network = _batcher_odd_even_mergesort(p)
    network = [(i, j) for i, j in network if i < N and j < N]
    seen = set()
    unique = []
    for comp in network:
        if comp not in seen:
            seen.add(comp)
            unique.append(comp)
    return unique

def _greedy_reduce(current, deadline):
    """Greedy reduction: scan from end, remove if still valid."""
    current = list(current)
    i = len(current) - 1
    while i >= 0:
        if time.time() > deadline:
            break
        
        saved = current[i]
        current.pop(i)
        
        if _verify(current):
            # Keep removed, move to previous
            i -= 1
        else:
            # Restore
            current.insert(i, saved)
            i -= 1
            
    return current

def _simulated_annealing(start_network, deadline):
    """Simulated Annealing with diverse moves."""
    current = list(start_network)
    best = list(current)
    best_size = len(current)
    
    T = 3.0
    cooling_rate = 0.998
    min_T = 0.01
    rng = random.Random(42)
    
    all_comparators = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    steps_since_improvement = 0
    max_stall = 500
    
    while time.time() < deadline and T > min_T:
        steps_since_improvement += 1
        
        if steps_since_improvement > max_stall:
            # Restart with perturbation
            current = list(best)
            for _ in range(rng.randint(2, 5)):
                comp = rng.choice(all_comparators)
                pos = rng.randrange(len(current) + 1)
                current.insert(pos, comp)
            current = _greedy_reduce(current, min(deadline, time.time() + 1))
            steps_since_improvement = 0
        
        r = rng.random()
        neighbor = list(current)
        valid_move = True
        delta = 0
        
        if r < 0.35:
            # Remove
            if len(neighbor) > 0:
                idx = rng.randrange(len(neighbor))
                neighbor.pop(idx)
                delta = -1
            else:
                continue
        elif r < 0.55:
            # Add
            comp = rng.choice(all_comparators)
            pos = rng.randrange(len(neighbor) + 1)
            neighbor.insert(pos, comp)
            delta = 1
        elif r < 0.70:
            # Swap
            if len(neighbor) >= 2:
                i = rng.randrange(len(neighbor))
                j = rng.randrange(len(neighbor))
                if i != j:
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                    delta = 0
                else:
                    continue
            else:
                continue
        elif r < 0.85:
            # Move
            if len(neighbor) >= 2:
                i = rng.randrange(len(neighbor))
                j = rng.randrange(len(neighbor))
                if i != j:
                    comp = neighbor.pop(i)
                    if i < j:
                        j -= 1
                    neighbor.insert(j, comp)
                    delta = 0
                else:
                    continue
            else:
                continue
        else:
            # Replace
            if len(neighbor) == 0:
                continue
            idx = rng.randrange(len(neighbor))
            dist = rng.randint(1, N - 1)
            if rng.random() < 0.5:
                dist = rng.randint(1, 3)
            a = rng.randint(0, N - 1 - dist)
            b = a + dist
            neighbor[idx] = (a, b)
            delta = 0
        
        if not valid_move:
            continue
        
        if _verify(neighbor):
            new_size = len(neighbor)
            
            if new_size < best_size:
                best = neighbor
                best_size = new_size
                steps_since_improvement = 0
            
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                current = neighbor
            else:
                steps_since_improvement += 1
        else:
            steps_since_improvement += 1
        
        T *= cooling_rate
    
    return best

def construct():
    """Construct a compare-exchange network that sorts all 18-input sequences."""
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Start with Batcher network
    best = _build_batcher_network()
    
    if not _verify(best):
        best = _odd_even_transposition()
    
    # Phase 1: Greedy reduction
    best = _greedy_reduce(best, deadline)
    
    # Phase 2: Simulated Annealing
    while time.time() < deadline - 2:
        if random.random() < 0.7:
            start_net = list(best)
        else:
            # Perturbed Batcher
            start_net = _build_batcher_network()
            for _ in range(random.randint(3, 8)):
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                start_net.append((i, j))
            reduce_deadline = min(deadline, time.time() + 3)
            start_net = _greedy_reduce(start_net, reduce_deadline)
            
            if time.time() > deadline:
                break
        
        candidate = _simulated_annealing(start_net, min(deadline, time.time() + 20))
        
        if len(candidate) < len(best):
            best = candidate
            if time.time() < deadline - 5:
                quick_deadline = min(deadline, time.time() + 3)
                best = _greedy_reduce(best, quick_deadline)
                
    # Final cleanup
    if time.time() < deadline - 1:
        best = _greedy_reduce(best, deadline)
        
    return best
