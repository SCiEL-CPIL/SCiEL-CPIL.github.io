"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
TIME_LIMIT = 85.0  # Safety margin before 90s kill


def _bitonic_sort(n):
    """Generate bitonic sort network for n wires."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    comps = []
    _bitonic_merge(comps, 0, n, 1)
    return comps


def _bitonic_merge(comps, offset, size, direction):
    """Bitonic merge for size elements at offset, direction 1=ascending."""
    if size <= 1:
        return
    
    half = size // 2
    _bitonic_merge(comps, offset, half, direction)
    _bitonic_merge(comps, offset + half, size - half, -direction)
    
    for i in range(min(half, size - half)):
        a = offset + i
        b = offset + half + i
        comps.append((a, b))


def _verify_network_fast(comparators):
    """Verify that a comparator network sorts all 2^N Boolean inputs using bit-parallel."""
    if len(comparators) > MAX_COMPARATORS:
        return False
    
    num_inputs = 1 << N
    state = np.zeros((N, num_inputs), dtype=np.uint8)
    for i in range(N):
        state[i] = (np.arange(num_inputs, dtype=np.uint32) >> i) & 1
    
    for (i, j) in comparators:
        min_val = state[i] & state[j]
        max_val = state[i] | state[j]
        state[i] = min_val
        state[j] = max_val
    
    for i in range(N - 1):
        if np.any(state[i] > state[i + 1]):
            return False
    
    return True


def _greedy_remove(comparators, time_limit):
    """Greedy removal of redundant comparators."""
    start_time = time.time()
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for idx in range(len(comparators) - 1, -1, -1):
            if (time.time() - start_time) >= time_limit:
                break
            new_comps = comparators[:idx] + comparators[idx+1:]
            if _verify_network_fast(new_comps):
                comparators = new_comps
                improved = True
                break
    return comparators


def _local_search(comparators, time_limit):
    """Local search: try removals and swaps."""
    start_time = time.time()
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for idx in range(len(comparators)):
            if (time.time() - start_time) >= time_limit:
                break
            new_comps = comparators[:idx] + comparators[idx+1:]
            if _verify_network_fast(new_comps):
                comparators = new_comps
                improved = True
                break
        if improved:
            continue
        for i in range(len(comparators)):
            for j in range(i + 1, len(comparators)):
                if (time.time() - start_time) >= time_limit:
                    break
                candidate = comparators[:]
                candidate[i], candidate[j] = candidate[j], candidate[i]
                if _verify_network_fast(candidate):
                    comparators = candidate
                    improved = True
                    break
            if improved:
                break
    return comparators


def _simulated_annealing(comparators, time_limit):
    """Simulated annealing to find a smaller valid network."""
    start_time = time.time()
    best = comparators[:]
    best_len = len(best)
    current = comparators[:]
    current_len = len(current)
    temp = 1.0
    
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    while (time.time() - start_time) < time_limit:
        move_type = random.random()
        
        if move_type < 0.3 and len(current) > 0:
            idx = random.randint(0, len(current) - 1)
            candidate = current[:idx] + current[idx+1:]
            candidate_len = current_len - 1
        elif move_type < 0.6 and len(current) < MAX_COMPARATORS:
            current_set = set(current)
            available = [p for p in all_pairs if p not in current_set]
            if not available:
                continue
            new_comp = random.choice(available)
            pos = random.randint(0, len(current))
            candidate = current[:pos] + [new_comp] + current[pos:]
            candidate_len = current_len + 1
        else:
            if len(current) < 2:
                continue
            if random.random() < 0.5:
                i, j = random.sample(range(len(current)), 2)
                candidate = current[:]
                candidate[i], candidate[j] = candidate[j], candidate[i]
            else:
                i = random.randint(0, len(current) - 1)
                j = random.randint(0, len(current))
                if j > i:
                    j -= 1
                if i == j:
                    continue
                candidate = current[:]
                comp = candidate.pop(i)
                candidate.insert(j, comp)
            candidate_len = current_len
        
        if not _verify_network_fast(candidate):
            continue
        
        delta = candidate_len - current_len
        if delta < 0:
            current = candidate
            current_len = candidate_len
            if current_len < best_len:
                best = current[:]
                best_len = current_len
        else:
            if random.random() < np.exp(-delta / max(temp, 1e-10)):
                current = candidate
                current_len = candidate_len
        
        temp *= 0.999
        if temp < 0.01:
            temp = 0.01
    
    return best


def construct():
    """Construct a sorting network on N wires with minimal comparators."""
    start_time = time.time()
    
    # Step 1: Generate dense seed (bitonic sort)
    seed = _bitonic_sort(N)
    
    # Verify seed
    if not _verify_network_fast(seed):
        seed = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    # Step 2: Greedy removal
    elapsed = time.time() - start_time
    remaining = TIME_LIMIT - elapsed - 10.0
    if remaining > 5.0:
        seed = _greedy_remove(seed, remaining)
    
    # Step 3: Local search
    elapsed = time.time() - start_time
    remaining = TIME_LIMIT - elapsed - 5.0
    if remaining > 3.0:
        seed = _local_search(seed, remaining)
    
    # Step 4: Simulated annealing
    elapsed = time.time() - start_time
    remaining = TIME_LIMIT - elapsed - 2.0
    if remaining > 5.0:
        seed = _simulated_annealing(seed, remaining)
    
    # Final verification
    if not _verify_network_fast(seed):
        seed = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    return seed

# EVOLVE-BLOCK-START
# (construct and helpers are defined above)
# EVOLVE-BLOCK-END
