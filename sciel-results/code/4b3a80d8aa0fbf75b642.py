import time
import numpy as np
import random
from collections import defaultdict

N = 16
DEADLINE = time.time() + 85  # safety margin before 90s kill

def _bitonic_sort_network(n):
    """Generate bitonic sort network for n wires."""
    if n <= 1:
        return []
    
    def generate(start, length, direction):
        if length <= 1:
            return []
        
        half = length // 2
        comps = []
        
        # Sort halves in opposite directions
        comps.extend(generate(start, half, 1))
        comps.extend(generate(start + half, half, -1))
        
        # Bitonic Merge
        for i in range(half):
            a = start + i
            b = start + half + i
            if direction == 1:
                comps.append((a, b))
            else:
                comps.append((b, a))
        
        # Recursively merge
        comps.extend(generate(start, half, direction))
        comps.extend(generate(start + half, half, direction))
        
        return comps
    
    return generate(0, n, 1)

def _verify_full(network):
    """Full verification using numpy over all 65536 states."""
    states = np.arange(65536, dtype=np.int64)
    for i, j in network:
        bi = (states >> i) & 1
        bj = (states >> j) & 1
        swap = bi > bj
        states = np.where(swap, states ^ ((1 << i) | (1 << j)), states)
    return bool(np.all(((states + 1) & states) == 0))

def _verify_sample(network, num_samples=1000, seed=123):
    """Fast verification using random samples."""
    rng = np.random.default_rng(seed)
    states = rng.integers(0, 65536, size=num_samples, dtype=np.int64)
    for i, j in network:
        bi = (states >> i) & 1
        bj = (states >> j) & 1
        swap = bi > bj
        states = np.where(swap, states ^ ((1 << i) | (1 << j)), states)
    return bool(np.all(((states + 1) & states) == 0))

def _compute_wire_influence(network):
    """Compute how many times each wire appears in the network."""
    influence = [0] * N
    for i, j in network:
        influence[i] += 1
        influence[j] += 1
    return influence

def _greedy_remove_with_influence(best, rng, deadline):
    """Greedy removal prioritizing low-influence comparators."""
    current = list(best)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        influence = _compute_wire_influence(current)
        
        # Score each comparator by sum of wire influences (lower = more likely removable)
        scored = []
        for idx, (i, j) in enumerate(current):
            score = influence[i] + influence[j]
            scored.append((score, idx))
        
        # Sort by score (ascending - try removing low-influence first)
        scored.sort()
        
        for score, idx in scored:
            if time.time() > deadline:
                break
            candidate = current[:idx] + current[idx+1:]
            
            # Fast check
            if _verify_sample(candidate, num_samples=300, seed=idx):
                # Full check for safety before accepting
                if _verify_full(candidate):
                    current = candidate
                    improved = True
                    break
    
    return current

def _local_search_swap(best, rng, deadline):
    """Try swapping adjacent comparators to find better orderings."""
    current = list(best)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(current) - 1):
            if time.time() > deadline:
                break
            # Swap comparators at positions i and i+1
            candidate = current.copy()
            candidate[i], candidate[i+1] = candidate[i+1], candidate[i]
            
            if _verify_sample(candidate, num_samples=200, seed=i):
                if _verify_full(candidate):
                    current = candidate
                    improved = True
                    break
    
    return current

def construct():
    start = time.time()
    
    # Start with Bitonic Sort (80 comparators)
    net = _bitonic_sort_network(N)
    best = list(net)
    
    # Verify initial network
    if not _verify_full(best):
        # Fallback to odd-even transposition (120 comparators)
        best = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    rng = random.Random(12345)
    
    # Phase 1: Greedy removal with influence-based prioritization
    best = _greedy_remove_with_influence(best, rng, DEADLINE - 20)
    
    # Phase 2: Local search to improve ordering
    if time.time() < DEADLINE - 10:
        best = _local_search_swap(best, rng, DEADLINE - 10)
    
    # Phase 3: Additional greedy removal passes
    improved = True
    while improved and time.time() < DEADLINE - 5:
        improved = False
        order = list(range(len(best)))
        rng.shuffle(order)
        for idx in order:
            if time.time() > DEADLINE - 5:
                break
            candidate = best[:idx] + best[idx+1:]
            
            if _verify_sample(candidate, num_samples=300, seed=idx):
                if _verify_full(candidate):
                    best = candidate
                    improved = True
    
    # Final full verification
    if _verify_full(best):
        return best
    else:
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
