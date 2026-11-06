import time
import random
import numpy as np

N = 16
N_BOOL = 1 << N
TARGET = N_BOOL - 1

def _batcher_network(n):
    """
    Generates Batcher's Bitonic Sort network for n wires.
    Returns a list of (i, j) comparators.
    """
    comparators = []
    
    def sort(n, offset, step):
        if n == 2:
            comparators.append((offset, offset + step))
        else:
            h = n // 2
            sort(h, offset, step)
            sort(h, offset + h * step, step)
            merge(n, offset, step)

    def merge(n, offset, step):
        if n == 2:
            comparators.append((offset, offset + step))
        else:
            h = n // 2
            for i in range(h):
                comparators.append((offset + i * step, offset + i * step + h * step))
            merge(h, offset, 2 * step)
            merge(h, offset + h * step, 2 * step)

    sort(n, 0, 1)
    return comparators

def _verify(comparators):
    """
    Vectorized verification of the sorting network.
    Returns True if the network sorts all 2^16 inputs.
    """
    states = np.arange(N_BOOL, dtype=np.uint32)
    
    for (i, j) in comparators:
        bi = (states >> i) & 1
        bj = (states >> j) & 1
        mask = bi > bj
        if np.any(mask):
            states[mask] = (states[mask] & ~(1 << i)) | (1 << j)
            
    return np.all(states == TARGET)

def _perturb(comparators):
    """
    Randomly modify one comparator to a different valid comparator.
    """
    if not comparators:
        return comparators
    
    idx = random.randint(0, len(comparators) - 1)
    i, j = comparators[idx]
    
    new_i = random.randint(0, N - 2)
    new_j = random.randint(new_i + 1, N - 1)
    
    attempts = 0
    while (new_i, new_j) == (i, j) and attempts < 10:
        new_i = random.randint(0, N - 2)
        new_j = random.randint(new_i + 1, N - 1)
        attempts += 1
        
    if (new_i, new_j) == (i, j):
        return comparators
        
    new_comparators = comparators[:]
    new_comparators[idx] = (new_i, new_j)
    return new_comparators

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # Initial network: Batcher's Bitonic Sort (63 comparators)
    best = _batcher_network(N)
    
    if not _verify(best):
        # Fallback
        best = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        if not _verify(best):
            return best

    best_len = len(best)
    stuck_count = 0
    
    while time.time() < deadline:
        current = best[:]
        random.shuffle(current)
        
        removed_any = False
        for idx in range(len(current) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            trial = current[:idx] + current[idx+1:]
            
            if _verify(trial):
                current = trial
                removed_any = True
                best_len -= 1
                best = current[:]
        
        if not removed_any:
            stuck_count += 1
            if stuck_count > 5:
                perturbed = _perturb(current)
                if _verify(perturbed):
                    current = perturbed
                    best = current[:]
                    stuck_count = 0
        else:
            stuck_count = 0

    return best
