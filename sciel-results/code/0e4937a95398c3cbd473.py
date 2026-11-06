import time
import random
import numpy as np

N = 16
MAX_COMPARATORS = 4096
SEED_DEADLINE = 45.0

_N_INPUTS = 1 << N
_INDICES = np.arange(_N_INPUTS, dtype=np.uint32)
_INPUT_MATRIX = np.zeros((_N_INPUTS, N), dtype=np.uint8)
for _i in range(N):
    _INPUT_MATRIX[:, _i] = (_INDICES >> _i) & 1

def _verify_network_fastest(network):
    x = _INPUT_MATRIX.copy()
    for i, j in network:
        mask = x[:, i] > x[:, j]
        if np.any(mask):
            x[mask, i] = 0
            x[mask, j] = 1
    for i in range(N - 1):
        if np.any(x[:, i] > x[:, i + 1]):
            return False
    return True

def _compute_depth(network):
    if not network:
        return 0
    n = len(network)
    wire_to_last_comp = {}
    depths = [0] * n
    for idx, (i, j) in enumerate(network):
        max_dep = 0
        for wire in (i, j):
            if wire in wire_to_last_comp:
                prev_idx = wire_to_last_comp[wire]
                if depths[prev_idx] + 1 > max_dep:
                    max_dep = depths[prev_idx] + 1
        depths[idx] = max_dep
        wire_to_last_comp[i] = idx
        wire_to_last_comp[j] = idx
    return max(depths) if depths else 0

def construct():
    start_time = time.time()
    deadline = start_time + SEED_DEADLINE
    
    def bitonic_sort_network(n):
        def bitonic_merge(n, offset=0, ascending=True):
            if n == 1: return []
            half = n // 2
            comparators = []
            for i in range(half):
                a = i + offset
                b = i + half + offset
                if not ascending: a, b = b, a
                comparators.append((a, b))
            comparators.extend(bitonic_merge(half, offset, ascending))
            comparators.extend(bitonic_merge(half, half + offset, not ascending))
            return comparators
        
        def bitonic_sort(n, offset=0, ascending=True):
            if n == 1: return []
            half = n // 2
            comparators = []
            comparators.extend(bitonic_sort(half, offset, True))
            comparators.extend(bitonic_sort(half, half + offset, False))
            comparators.extend(bitonic_merge(n, offset, ascending))
            return comparators
        
        return bitonic_sort(n)

    best_network = bitonic_sort_network(N)
    best_count = len(best_network)
    
    if not _verify_network_fastest(best_network):
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        
    # Simulated Annealing
    current_network = list(best_network)
    current_count = len(current_network)
    temp = 10.0
    cooling_rate = 0.95
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() < deadline - 5:
        if no_improve_count > max_no_improve:
            # Restart with a small perturbation of the best network
            current_network = list(best_network)
            # Randomly swap a few comparators to escape local minima
            for _ in range(3):
                if len(current_network) > 1:
                    i = random.randint(0, len(current_network) - 1)
                    j = random.randint(0, len(current_network) - 1)
                    current_network[i], current_network[j] = current_network[j], current_network[i]
            no_improve_count = 0
            temp = 10.0
            continue

        # Choose a move: Remove a comparator or Swap two comparators
        if random.random() < 0.7 and len(current_network) > 60: # Bias towards removal if > 60
            idx = random.randint(0, len(current_network) - 1)
            candidate = current_network[:idx] + current_network[idx+1:]
        else:
            if len(current_network) < 2:
                continue
            i = random.randint(0, len(current_network) - 1)
            j = random.randint(0, len(current_network) - 1)
            candidate = current_network[:]
            candidate[i], candidate[j] = candidate[j], candidate[i]
        
        if len(candidate) < current_count:
            # If we removed a comparator, it's automatically better if valid
            is_valid = _verify_network_fastest(candidate)
            if is_valid:
                current_network = candidate
                current_count = len(current_network)
                if current_count < best_count:
                    best_network = list(current_network)
                    best_count = current_count
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        else:
            # Same count (swap), only accept if it enables future removals (hard to know)
            # Or just accept with probability exp(-delta/T) where delta is 0
            # For swaps, we just accept to explore
            if random.random() < 0.5:
                current_network = candidate
                no_improve_count += 1
            else:
                no_improve_count += 1
                
        temp *= cooling_rate
        if temp < 0.1:
            temp = 0.1

    return best_network
