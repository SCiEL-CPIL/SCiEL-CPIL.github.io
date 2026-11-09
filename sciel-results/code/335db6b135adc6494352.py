import time
import numpy as np
import random

N = 16
TOTAL_ZERO_ONE = 1 << N

T_START = time.time()
DEADLINE = T_START + 85.0

# Precompute targets and initial wires using numpy for speed
def _precompute():
    # For each input mask m (0..65535), sorted output: wire k=1 iff popcount(m) > 15-k
    # i.e., the top popcount(m) wires are 1
    inputs = np.arange(TOTAL_ZERO_ONE, dtype=np.uint32)
    
    # popcount for all 65536 values
    popcounts = np.zeros(TOTAL_ZERO_ONE, dtype=np.uint8)
    for i in range(TOTAL_ZERO_ONE):
        popcounts[i] = bin(i).count('1')
    
    # TARGETS[k]: a 65536-bit array (as uint64 chunks) where bit m = 1 iff wire k should be 1 after sorting
    # wire k is 1 iff popcount(m) > 15-k, i.e., popcount(m) >= 16-k
    # So for wire k, it's 1 for inputs with popcount >= 16-k
    # We'll store as list of uint64: 65536 bits = 1024 uint64s
    TARGETS = [np.zeros(1024, dtype=np.uint64) for _ in range(N)]
    for k in range(N):
        threshold = 16 - k  # wire k is 1 if popcount >= threshold
        # For each input m, if popcounts[m] >= threshold, set bit m
        for m in range(TOTAL_ZERO_ONE):
            if popcounts[m] >= threshold:
                TARGETS[k][m // 64] |= (1 << (m % 64))
    
    # _INIT_WIRES[k]: wire k holds bit k of input m
    INIT_WIRES = [np.zeros(1024, dtype=np.uint64) for _ in range(N)]
    for k in range(N):
        for m in range(TOTAL_ZERO_ONE):
            if (m >> k) & 1:
                INIT_WIRES[k][m // 64] |= (1 << (m % 64))
    
    return TARGETS, INIT_WIRES

TARGETS, INIT_WIRES = _precompute()

def _verify(comp_list):
    """Verify sorting network using numpy bit operations."""
    if not comp_list:
        return False
    wires = [w.copy() for w in INIT_WIRES]
    for i, j in comp_list:
        a = wires[i]
        b = wires[j]
        lo = np.bitwise_and(a, b)
        hi = np.bitwise_or(a, b)
        wires[i] = lo
        wires[j] = hi
    for k in range(N):
        if not np.array_equal(wires[k], TARGETS[k]):
            return False
    return True

def _verify_partial(comp_list, start_idx, end_idx):
    """Verify only a subset of comparators' effect (for local search speedup)."""
    wires = [w.copy() for w in INIT_WIRES]
    for i, j in comp_list[start_idx:end_idx]:
        a = wires[i]
        b = wires[j]
        lo = np.bitwise_and(a, b)
        hi = np.bitwise_or(a, b)
        wires[i] = lo
        wires[j] = hi
    for k in range(N):
        if not np.array_equal(wires[k], TARGETS[k]):
            return False
    return True

def _random_network(n_comps=80):
    """Generate a random sorting network candidate."""
    comps = []
    for _ in range(n_comps):
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        comps.append((i, j))
    return comps

def _local_search(network, max_iters=2000):
    """Local search: try deleting, swapping, inserting, and replacing comparators."""
    best = list(network)
    best_len = len(best)
    improved = True
    iters = 0
    
    while improved and iters < max_iters and time.time() < DEADLINE - 5.0:
        improved = False
        iters += 1
        
        # Try deletion
        for idx in range(len(best)):
            trial = best[:idx] + best[idx+1:]
            if _verify(trial):
                best = trial
                best_len = len(best)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swapping adjacent comparators
        for idx in range(len(best) - 1):
            trial = best[:]
            trial[idx], trial[idx+1] = trial[idx+1], trial[idx]
            if _verify(trial):
                best = trial
                improved = True
                break
        
        if improved:
            continue
        
        # Try replacing a comparator with a different one
        for idx in range(len(best)):
            old_i, old_j = best[idx]
            for _ in range(5):
                new_i = random.randint(0, N - 2)
                new_j = random.randint(new_i + 1, N - 1)
                if (new_i, new_j) == (old_i, old_j):
                    continue
                trial = best[:]
                trial[idx] = (new_i, new_j)
                if _verify(trial):
                    best = trial
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Try inserting a new comparator
        if len(best) < 120:
            for _ in range(3):
                pos = random.randint(0, len(best))
                new_i = random.randint(0, N - 2)
                new_j = random.randint(new_i + 1, N - 1)
                trial = best[:pos] + [(new_i, new_j)] + best[pos:]
                if _verify(trial):
                    best = trial
                    improved = True
                    break
    
    return best

def _batcher_oe_merge_sort(n):
    def oe_merge(left, right):
        comps = []
        h = len(left)
        if h == 1:
            comps.append((left[0], right[0]))
            return comps
        le = [left[k] for k in range(0, h, 2)]
        lo = [left[k] for k in range(1, h, 2)]
        re = [right[k] for k in range(0, h, 2)]
        ro = [right[k] for k in range(1, h, 2)]
        comps += oe_merge(le, re)
        comps += oe_merge(lo, ro)
        for k in range(1, h, 2):
            comps.append((left[k], right[k - 1]))
        return comps

    def sort_block(wires):
        m = len(wires)
        if m == 1:
            return []
        h = m // 2
        left = wires[:h]
        right = wires[h:]
        comps = []
        comps += sort_block(left)
        comps += sort_block(right)
        comps += oe_merge(left, right)
        return comps

    return sort_block(list(range(n)))

def _odd_even_transposition(n):
    """Odd-even transposition sort network."""
    comps = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comps.append((i, i + 1))
    return comps

def construct():
    best = None
    best_len = float('inf')
    
    # Start with known good constructions
    candidates = []
    
    # Batcher odd-even merge sort
    batcher_net = _batcher_oe_merge_sort(N)
    if _verify(batcher_net):
        candidates.append(batcher_net)
    
    # Odd-even transposition
    oet_net = _odd_even_transposition(N)
    if _verify(oet_net):
        candidates.append(oet_net)
    
    # Try random constructions
    random.seed(42)
    for attempt in range(30):
        if time.time() > DEADLINE - 10.0:
            break
        for n_comps in [70, 80, 90, 100]:
            if time.time() > DEADLINE - 10.0:
                break
            net = _random_network(n_comps)
            if _verify(net):
                candidates.append(net)
    
    # Local search on each candidate
    for cand in candidates:
        if time.time() > DEADLINE - 5.0:
            break
        result = _local_search(cand, max_iters=500)
        if len(result) < best_len:
            best = result
            best_len = len(result)
    
    # If no verified network found, use odd-even transposition as fallback
    if best is None:
        best = oet_net
    
    # Final verification
    if not _verify(best):
        best = oet_net
    
    return [(int(i), int(j)) for (i, j) in best]
