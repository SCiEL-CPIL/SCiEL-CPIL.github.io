import numpy as np
import time
import random

N = 16

# EVOLVE-BLOCK-START
def _batcher_16():
    """Generate Batcher's bitonic sort network for 16 wires."""
    def bitonic_merge(n):
        comps = []
        k = n // 2
        while k >= 1:
            for i in range(n - k):
                comps.append((i, i + k))
            k //= 2
        return comps
    
    def bitonic_sort(n):
        if n == 1:
            return []
        half = n // 2
        left = bitonic_sort(half)
        
        right_asc = bitonic_sort(half)
        right_desc = []
        for i, j in right_asc:
            i_new = half - 1 - j
            j_new = half - 1 - i
            right_desc.append((half + i_new, half + j_new))
        
        merge = bitonic_merge(n)
        return left + right_desc + merge
    
    return bitonic_sort(16)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Precompute all 2^16 Boolean inputs as uint8 matrix for fast verification
    # Shape: (16, 65536)
    inputs = (np.arange(1 << N, dtype=np.int32)[:, None] >> np.arange(N, dtype=np.int32)[None, :]) & 1
    
    # Start with Batcher's bitonic sort (88 comparators)
    best_net = _batcher_16()
    best_count = len(best_net)
    
    # Fast verification using vectorized NumPy - check ALL inputs
    def is_valid_full(network):
        wires = inputs.copy()
        for i, j in network:
            min_val = np.minimum(wires[i], wires[j])
            max_val = np.maximum(wires[i], wires[j])
            wires[i] = min_val
            wires[j] = max_val
        diffs = np.diff(wires, axis=0)
        return np.all(diffs >= 0)
    
    # Fast approximate verification - check random subset of inputs
    def is_valid_approx(network, n_samples=1024):
        # Sample random inputs
        sample_indices = np.random.randint(0, 1 << N, size=n_samples)
        wires = inputs[:, sample_indices].copy()
        for i, j in network:
            min_val = np.minimum(wires[i], wires[j])
            max_val = np.maximum(wires[i], wires[j])
            wires[i] = min_val
            wires[j] = max_val
        diffs = np.diff(wires, axis=0)
        return np.all(diffs >= 0)
    
    # Phase 1: Greedy single removal (fast, deterministic)
    while time.time() < deadline:
        improved = False
        n = len(best_net)
        indices = list(range(n))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            if idx >= len(best_net):
                continue
            
            candidate = best_net[:idx] + best_net[idx+1:]
            if is_valid_full(candidate):
                best_net = candidate
                best_count = len(best_net)
                improved = True
                break
        
        if not improved:
            break
    
    # Phase 2: Simulated annealing with multiple move types
    # Use approximate verification for speed, full verification for acceptance
    current_net = best_net[:]
    current_count = best_count
    best_sa_net = best_net[:]
    best_sa_count = best_count
    
    T = 10.0  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.995
    
    # Precompute a set of random input samples for fast verification
    n_samples = 2048
    sample_indices = np.random.randint(0, 1 << N, size=n_samples)
    
    def apply_network(network, wires):
        """Apply network to wires in-place."""
        for i, j in network:
            min_val = np.minimum(wires[i], wires[j])
            max_val = np.maximum(wires[i], wires[j])
            wires[i] = min_val
            wires[j] = max_val
        return wires
    
    def check_sample(network):
        """Check network on precomputed sample inputs."""
        wires = inputs[:, sample_indices].copy()
        apply_network(network, wires)
        diffs = np.diff(wires, axis=0)
        return np.all(diffs >= 0)
    
    while time.time() < deadline and T > T_min:
        T *= cooling_rate
        n = len(current_net)
        
        # Choose a random move
        move_type = random.random()
        
        if move_type < 0.4 and n > 0:
            # Remove a random comparator
            idx = random.randint(0, n - 1)
            candidate = current_net[:idx] + current_net[idx+1:]
        elif move_type < 0.7 and n > 1:
            # Swap two comparators
            i, j = random.sample(range(n), 2)
            candidate = current_net[:]
            candidate[i], candidate[j] = candidate[j], candidate[i]
        elif move_type < 0.85:
            # Insert a random comparator
            i, j = random.sample(range(N), 2)
            if i > j:
                i, j = j, i
            idx = random.randint(0, n)
            candidate = current_net[:idx] + [(i, j)] + current_net[idx:]
        else:
            # Replace a comparator with a different one
            if n == 0:
                continue
            idx = random.randint(0, n - 1)
            i, j = random.sample(range(N), 2)
            if i > j:
                i, j = j, i
            candidate = current_net[:]
            candidate[idx] = (i, j)
        
        # Check candidate
        if not check_sample(candidate):
            continue
        
        # Full verification for promising candidates
        if not is_valid_full(candidate):
            continue
        
        new_count = len(candidate)
        
        # Acceptance criterion
        if new_count < current_count:
            current_net = candidate
            current_count = new_count
            if new_count < best_sa_count:
                best_sa_net = candidate[:]
                best_sa_count = new_count
        else:
            # Accept worse solutions with probability exp(-delta/T)
            delta = new_count - current_count
            if delta == 0:
                current_net = candidate
            elif random.random() < np.exp(-delta / T):
                current_net = candidate
    
    # Update best
    if best_sa_count < best_count:
        best_net = best_sa_net
        best_count = best_sa_count
    
    # Phase 3: Final greedy removal pass
    while time.time() < deadline:
        improved = False
        n = len(best_net)
        indices = list(range(n))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            if idx >= len(best_net):
                continue
            
            candidate = best_net[:idx] + best_net[idx+1:]
            if is_valid_full(candidate):
                best_net = candidate
                best_count = len(best_net)
                improved = True
                break
        
        if not improved:
            break
    
    return best_net
# EVOLVE-BLOCK-END
