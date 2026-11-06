import numpy as np
import time
import random

N = 16
SORTED_SET = set((1 << k) - 1 for k in range(N + 1))
ALL_MASKS = np.arange(1 << N, dtype=np.int32)
SORTED_ARRAY = np.zeros(1 << N, dtype=bool)
for k in range(N + 1):
    SORTED_ARRAY[(1 << k) - 1] = True

def _merge(k):
    """
    Generates comparators to merge two sorted lists of length k.
    The lists are assumed to be at indices 0..k-1 and k..2k-1.
    """
    if k == 1:
        return [(0, 1)]
    
    k_half = k // 2
    
    # 1. Merge even-indexed elements
    even_comps = []
    for i, j in _merge(k_half):
        mi = 2 * i if i < k_half else 2 * (i - k_half) + k
        mj = 2 * j if j < k_half else 2 * (j - k_half) + k
        even_comps.append((mi, mj))
        
    # 2. Merge odd-indexed elements
    odd_comps = []
    for i, j in _merge(k_half):
        mi = 2 * i + 1 if i < k_half else 2 * (i - k_half) + k + 1
        mj = 2 * j + 1 if j < k_half else 2 * (j - k_half) + k + 1
        odd_comps.append((mi, mj))
        
    # 3. Final comparisons between adjacent pairs (2i+1, 2i+2)
    final_comps = [(2 * i + 1, 2 * i + 2) for i in range(k - 1)]
    
    return even_comps + odd_comps + final_comps

def _batcher(n):
    """
    Generates Batcher's Odd-Even Merge Sort network for n wires.
    """
    if n == 1:
        return []
    
    k = n // 2
    left = _batcher(k)
    right = [(i + k, j + k) for i, j in _batcher(k)]
    merge = _merge(k)
    
    return left + right + merge

def _simulate_network(network, masks):
    """Vectorized simulation of network on all masks."""
    m = masks.copy()
    for i, j in network:
        bit_i = (m >> i) & 1
        bit_j = (m >> j) & 1
        # swap where bit_i=1 and bit_j=0
        swap_mask = bit_i & (~bit_j)
        if np.any(swap_mask):
            m = np.where(swap_mask, (m & ~(1 << i)) | (1 << j), m)
    return m

def _full_check_fast(network):
    """Fast full check using vectorized simulation."""
    results = _simulate_network(network, ALL_MASKS)
    return np.all(SORTED_ARRAY[results])

def _fast_check_sample(network, sample):
    """Check against a sample of masks."""
    results = _simulate_network(network, sample)
    return np.all(SORTED_ARRAY[results])

def construct():
    deadline = time.time() + 55
    rng = random.Random(42)
    
    # Start with Batcher's Odd-Even Merge Sort
    network = _batcher(N)
    
    # Create samples for fast verification
    all_possible = list(range(1 << N))
    rng.shuffle(all_possible)
    fast_sample = np.array(all_possible[:1000], dtype=np.int32)
    med_sample = np.array(all_possible[:5000], dtype=np.int32)
    
    def try_remove(idx):
        """Try removing comparator at index idx. Returns True if successful."""
        nonlocal network
        new_network = network[:idx] + network[idx+1:]
        if _fast_check_sample(new_network, fast_sample):
            if _fast_check_sample(new_network, med_sample):
                if _full_check_fast(new_network):
                    network = new_network
                    return True
        return False
    
    def try_remove_pair(i, j):
        """Try removing comparators at indices i and j. Returns True if successful."""
        nonlocal network
        if i > j:
            i, j = j, i
        new_network = network[:i] + network[i+1:j] + network[j+1:]
        if _fast_check_sample(new_network, fast_sample):
            if _fast_check_sample(new_network, med_sample):
                if _full_check_fast(new_network):
                    network = new_network
                    return True
        return False
    
    # Phase 1: Aggressive single-removal, multiple passes
    # Try from the end first (comparators near the end are often redundant)
    for _pass in range(50):
        if time.time() > deadline - 5:
            break
        removed = False
        n_comps = len(network)
        
        if n_comps == 0:
            break
        
        # Try from the end, then from the beginning
        for idx in range(n_comps - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            if try_remove(idx):
                removed = True
                break
        
        if not removed:
            # Try from the beginning too
            for idx in range(n_comps):
                if time.time() > deadline - 5:
                    break
                if try_remove(idx):
                    removed = True
                    break
        
        if not removed:
            break
    
    # Phase 2: Try removing pairs of adjacent comparators
    for _pass in range(20):
        if time.time() > deadline - 5:
            break
        removed = False
        n_comps = len(network)
        
        if n_comps < 2:
            break
        
        for i in range(n_comps - 1):
            if time.time() > deadline - 5:
                break
            if try_remove_pair(i, i + 1):
                removed = True
                break
        
        if not removed:
            break
    
    # Phase 3: Final single-removal pass
    for _pass in range(20):
        if time.time() > deadline - 5:
            break
        removed = False
        n_comps = len(network)
        
        if n_comps == 0:
            break
        
        for idx in range(n_comps - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            if try_remove(idx):
                removed = True
                break
        
        if not removed:
            break
    
    return network
