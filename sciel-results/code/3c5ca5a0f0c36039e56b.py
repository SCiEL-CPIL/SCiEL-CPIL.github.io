"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 17

# EVOLVE-BLOCK-START
def _check_network(network, full_check=True, sample_size=1000):
    """
    Check if the network sorts all inputs.
    Uses 0-1 principle.
    """
    if full_check:
        # Generate all 2^N boolean inputs
        indices = np.arange(2**N)
        inputs = ((indices[:, None] >> np.arange(N)[None, :]) & 1).astype(np.int8)
        
        # Apply network
        for i, j in network:
            min_val = np.minimum(inputs[:, i], inputs[:, j])
            max_val = np.maximum(inputs[:, i], inputs[:, j])
            inputs[:, i] = min_val
            inputs[:, j] = max_val
            
        # Check if sorted: each row should be non-decreasing
        diffs = np.diff(inputs, axis=1)
        if np.any(diffs < 0):
            return False
        return True
    else:
        # Sample check
        samples = np.random.randint(0, 2, size=(sample_size, N), dtype=np.int8)
        for i, j in network:
            min_val = np.minimum(samples[:, i], samples[:, j])
            max_val = np.maximum(samples[:, i], samples[:, j])
            samples[:, i] = min_val
            samples[:, j] = max_val
        diffs = np.diff(samples, axis=1)
        if np.any(diffs < 0):
            return False
        return True

def _generate_oet_network(n):
    """Generate Odd-Even Transposition Network."""
    comps = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comps.append((i, i + 1))
    return comps

def _generate_aksn_network(n):
    """
    Generate a sorting network based on the AKS approach or a known good construction.
    For now, use a batcher odd-even merge sort as a better seed.
    """
    def _batcher_odd_even_merge_sort(n):
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        half = n // 2
        # Recursively sort each half
        left = _batcher_odd_even_merge_sort(half)
        right = _batcher_odd_even_merge_sort(n - half)
        
        # Shift right comparators
        right = [(i + half, j + half) for i, j in right]
        
        # Odd-even merge
        merged = _odd_even_merge(half, n - half)
        
        return left + right + merged
    
    def _odd_even_merge(n, m):
        """Odd-even merge of two sorted sequences of length n and m."""
        if n == 0:
            return []
        if m == 0:
            return []
        if n == 1 and m == 1:
            return [(0, 1)]
        
        # Compare and swap at positions (0,1), (2,3), etc. and (1,2), (3,4), etc.
        comps = []
        
        # First phase: compare (0,1), (2,3), ...
        for i in range(0, n + m - 1, 2):
            comps.append((i, i + 1))
        
        # Recursively merge odd and even subsequences
        # Odd subsequence: positions 1, 3, 5, ...
        # Even subsequence: positions 0, 2, 4, ...
        
        # This is complex; for simplicity, let's use a known network for small n
        # For n=17, we can use a precomputed good network or fall back to OET
        
        # Actually, let's just use a simpler approach: use the OET but with some optimization
        # For now, return OET for simplicity
        return []
    
    # For simplicity and reliability, let's use OET as the base
    return _generate_oet_network(n)

def _try_remove_comparator(network, idx):
    """Try removing comparator at index idx and check if network still sorts."""
    new_network = network[:idx] + network[idx+1:]
    return new_network

def _try_move_comparator(network, idx, new_idx):
    """Try moving comparator at idx to new_idx."""
    comp = network[idx]
    new_network = network[:idx] + network[idx+1:]
    new_network = new_network[:new_idx] + [comp] + new_network[new_idx:]
    return new_network

def construct():
    """
    Construct a sorting network for 17 wires using local search optimization.
    Starts with Odd-Even Transposition and tries to remove/move comparators.
    Uses simulated annealing for better exploration.
    """
    start_time = time.time()
    time_limit = 55.0  # Leave 5s buffer for overhead
    
    # Start with OET
    network = _generate_oet_network(N)
    current_size = len(network)
    
    # Verify initial network
    if not _check_network(network, full_check=True):
        return network

    best_network = network.copy()
    best_size = current_size
    
    # Simulated annealing parameters
    T = 1.0
    cooling = 0.999
    T_min = 0.01
    
    attempts = 0
    while time.time() - start_time < time_limit and T > T_min:
        attempts += 1
        if not network:
            break
            
        # Randomly choose a move type
        # Weight towards removal when network is large
        if len(network) > 10:
            move_type = random.choice(['remove', 'remove', 'remove', 'move', 'move'])
        else:
            move_type = random.choice(['remove', 'move'])
            
        original_network = network.copy()
        
        if move_type == 'remove':
            idx = random.randint(0, len(network) - 1)
            new_network = network[:idx] + network[idx+1:]
            new_size = len(new_network)
            
            # Quick check with samples
            if _check_network(new_network, full_check=False):
                # Full check
                if _check_network(new_network, full_check=True):
                    # Accept (it's an improvement)
                    network = new_network
                    current_size = new_size
                    if current_size < best_size:
                        best_size = current_size
                        best_network = network.copy()
                else:
                    # Reject
                    network = original_network
        elif move_type == 'move':
            idx = random.randint(0, len(network) - 1)
            # Move to a random position
            new_idx = random.randint(0, len(network) - 1)
            if new_idx == idx:
                continue
            comp = network[idx]
            new_network = network[:idx] + network[idx+1:]
            new_network = new_network[:new_idx] + [comp] + new_network[new_idx:]
            
            # Quick check
            if _check_network(new_network, full_check=False):
                # Full check
                if _check_network(new_network, full_check=True):
                    # Accept (size unchanged, but structure changed)
                    network = new_network
                else:
                    # With some probability, accept to escape local minima
                    if random.random() < T:
                        network = new_network
                    else:
                        network = original_network
        
        # Cool down
        T *= cooling
        
    return best_network
# EVOLVE-BLOCK-END
