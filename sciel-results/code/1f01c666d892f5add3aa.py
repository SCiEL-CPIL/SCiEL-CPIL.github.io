import numpy as np
import time

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
    timeout = 85.0  # Safety margin before 90s limit
    
    # Precompute all 2^16 Boolean inputs as uint8 matrix for fast verification
    # Shape: (16, 65536)
    inputs = (np.arange(1 << N, dtype=np.int32)[:, None] >> np.arange(N, dtype=np.int32)[None, :]) & 1
    
    # Start with Batcher's bitonic sort (88 comparators)
    best_net = _batcher_16()
    best_count = len(best_net)
    
    # Fast verification using vectorized NumPy
    def is_valid(network):
        wires = inputs.copy()
        for i, j in network:
            min_val = np.minimum(wires[i], wires[j])
            max_val = np.maximum(wires[i], wires[j])
            wires[i] = min_val
            wires[j] = max_val
        diffs = np.diff(wires, axis=0)
        return np.all(diffs >= 0)
    
    # Local search: aggressive single removals with restart
    while time.time() - start_time < timeout:
        improved = False
        n = len(best_net)
        indices = list(range(n))
        np.random.shuffle(indices)
        
        for idx in indices:
            if time.time() - start_time > timeout:
                break
            if idx >= len(best_net):
                continue
            
            candidate = best_net[:idx] + best_net[idx+1:]
            if is_valid(candidate):
                best_net = candidate
                best_count = len(best_net)
                improved = True
                break
        
        if not improved:
            break
    
    # Phase 2: Try removing pairs of comparators
    while time.time() - start_time < timeout:
        improved = False
        n = len(best_net)
        
        # Generate random pairs to try
        for _ in range(200):
            if time.time() - start_time > timeout:
                break
            
            i, j = np.random.choice(n, size=2, replace=False)
            i, j = int(i), int(j)
            
            remove_set = {i, j}
            candidate = [net for k, net in enumerate(best_net) if k not in remove_set]
            if is_valid(candidate):
                best_net = candidate
                best_count = len(best_net)
                improved = True
                break
        
        if not improved:
            break
    
    # Phase 3: Try removing triples
    while time.time() - start_time < timeout:
        improved = False
        n = len(best_net)
        
        for _ in range(100):
            if time.time() - start_time > timeout:
                break
            
            i, j, k = np.random.choice(n, size=3, replace=False)
            i, j, k = int(i), int(j), int(k)
            
            remove_set = {i, j, k}
            candidate = [net for idx, net in enumerate(best_net) if idx not in remove_set]
            if is_valid(candidate):
                best_net = candidate
                best_count = len(best_net)
                improved = True
                break
        
        if not improved:
            break
    
    # Phase 4: Swap-based local search to escape local minima
    while time.time() - start_time < timeout:
        improved = False
        n = len(best_net)
        
        # Try random swaps
        for _ in range(100):
            if time.time() - start_time > timeout:
                break
            if n < 2:
                break
                
            i, j = np.random.choice(n, size=2, replace=False)
            i, j = int(i), int(j)
            
            candidate = best_net[:]
            candidate[i], candidate[j] = candidate[j], candidate[i]
            
            if is_valid(candidate):
                # Accept swap and try to remove again
                best_net = candidate
                improved = True
                break
        
        if not improved:
            break
    
    # Final phase: More aggressive removal attempts after swaps
    while time.time() - start_time < timeout:
        improved = False
        n = len(best_net)
        indices = list(range(n))
        np.random.shuffle(indices)
        
        for idx in indices:
            if time.time() - start_time > timeout:
                break
            if idx >= len(best_net):
                continue
            
            candidate = best_net[:idx] + best_net[idx+1:]
            if is_valid(candidate):
                best_net = candidate
                best_count = len(best_net)
                improved = True
                break
        
        if not improved:
            break
    
    return best_net
# EVOLVE-BLOCK-END
