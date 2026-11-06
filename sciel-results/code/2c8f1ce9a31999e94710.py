import time
import random
import numpy as np

N = 18
TOTAL_INPUTS = 1 << N
DEADLINE = 80.0  # seconds, safety margin before 90s kill

def _precompute_inputs():
    """
    Precompute all 2^N Boolean inputs as a (N, 2^N) uint8 array.
    Column k represents the binary representation of k, with bit 0 as the first wire.
    """
    all_inputs = np.zeros((N, TOTAL_INPUTS), dtype=np.uint8)
    for i in range(N):
        period = 1 << (i + 1)
        half = 1 << i
        pattern = np.zeros(period, dtype=np.uint8)
        pattern[half:] = 1
        all_inputs[i] = np.tile(pattern, TOTAL_INPUTS // period)
    return all_inputs

# Precompute at module level to save time in construct
_ALL_INPUTS = _precompute_inputs()

def _verify_network(network):
    """
    Vectorized verification of the sorting network.
    Returns True if the network sorts all 0-1 inputs.
    """
    x = _ALL_INPUTS.copy()
    
    for i, j in network:
        mask = (x[i] == 1) & (x[j] == 0)
        if np.any(mask):
            x[i][mask] = 0
            x[j][mask] = 1
            
    diffs = x[1:].astype(np.int8) - x[:-1].astype(np.int8)
    if np.any(diffs == -1):
        return False
    return True

def _verify_network_subset(network, subset_size=4096):
    """
    Fast verification on a random subset of inputs.
    Returns True if the network sorts the subset.
    """
    x = _ALL_INPUTS[:, :subset_size].copy()
    
    for i, j in network:
        mask = (x[i] == 1) & (x[j] == 0)
        if np.any(mask):
            x[i][mask] = 0
            x[j][mask] = 1
            
    diffs = x[1:].astype(np.int8) - x[:-1].astype(np.int8)
    if np.any(diffs == -1):
        return False
    return True

def construct():
    start_time = time.time()
    
    # Start with odd-even transposition network
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # Verify initial network
    if not _verify_network(network):
        return network
    
    best_network = list(network)
    best_count = len(network)
    
    rng = random.Random(42)
    
    # Phase 1: Greedy removal with subset pre-filtering
    changed = True
    while changed:
        changed = False
        for idx in range(len(network) - 1, -1, -1):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            candidate = network[:idx] + network[idx+1:]
            if len(candidate) < len(network):
                # Quick check on a subset first
                if _verify_network_subset(candidate, 4096):
                    # Full check
                    if _verify_network(candidate):
                        network = candidate
                        changed = True
                        best_network = list(network)
                        best_count = len(network)
        
        if time.time() - start_time > DEADLINE - 5:
            break
    
    # Phase 2: Add-Remove cycles
    existing_pairs = set(network)
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    max_rounds = 30
    for _round in range(max_rounds):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        found_improvement = False
        # Shuffle pairs to explore different regions
        shuffled_pairs = all_pairs[:]
        rng.shuffle(shuffled_pairs)
        
        for add_pair in shuffled_pairs:
            if time.time() - start_time > DEADLINE - 10:
                break
                
            if add_pair in existing_pairs:
                continue
                
            test_network = network + [add_pair]
            
            for idx in range(len(test_network) - 2, -1, -1):
                if time.time() - start_time > DEADLINE - 10:
                    break
                    
                removed_network = test_network[:idx] + test_network[idx+1:]
                
                if len(removed_network) < len(network):
                    if _verify_network_subset(removed_network, 4096):
                        if _verify_network(removed_network):
                            network = removed_network
                            existing_pairs = set(network)
                            best_network = list(network)
                            best_count = len(network)
                            found_improvement = True
                            break
                        
            if found_improvement:
                break
            
        if not found_improvement:
            break
    
    # Phase 3: Local search with swaps
    for iteration in range(20):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        improved = False
        for idx in range(len(network) - 1):
            if time.time() - start_time > DEADLINE - 5:
                break
                
            candidate = network[:]
            candidate[idx], candidate[idx+1] = candidate[idx+1], candidate[idx]
            
            if _verify_network_subset(candidate, 4096):
                if _verify_network(candidate):
                    # Try to remove comparators from this swapped network
                    temp_network = list(candidate)
                    temp_changed = True
                    while temp_changed:
                        temp_changed = False
                        for tidx in range(len(temp_network) - 1, -1, -1):
                            if time.time() - start_time > DEADLINE - 5:
                                break
                            t_candidate = temp_network[:tidx] + temp_network[tidx+1:]
                            if len(t_candidate) < len(temp_network):
                                if _verify_network_subset(t_candidate, 4096):
                                    if _verify_network(t_candidate):
                                        temp_network = t_candidate
                                        temp_changed = True
                        
                        if time.time() - start_time > DEADLINE - 5:
                            break
                    
                    if len(temp_network) < len(network):
                        network = temp_network
                        best_network = list(network)
                        best_count = len(network)
                        improved = True
        
        if not improved:
            break
    
    # Phase 4: Random restarts with perturbation if time remains
    if time.time() - start_time < DEADLINE - 10:
        for restart in range(5):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            perturbed = list(best_network)
            for _ in range(3):
                if len(perturbed) > 1:
                    i = rng.randint(0, len(perturbed) - 2)
                    perturbed[i], perturbed[i+1] = perturbed[i+1], perturbed[i]
            
            if _verify_network_subset(perturbed, 4096):
                if _verify_network(perturbed):
                    temp_network = list(perturbed)
                    temp_changed = True
                    while temp_changed:
                        temp_changed = False
                        for tidx in range(len(temp_network) - 1, -1, -1):
                            if time.time() - start_time > DEADLINE - 5:
                                break
                            t_candidate = temp_network[:tidx] + temp_network[tidx+1:]
                            if len(t_candidate) < len(temp_network):
                                if _verify_network_subset(t_candidate, 4096):
                                    if _verify_network(t_candidate):
                                        temp_network = t_candidate
                                        temp_changed = True
                        
                        if time.time() - start_time > DEADLINE - 5:
                            break
                    
                    if len(temp_network) < best_count:
                        best_network = list(temp_network)
                        best_count = len(temp_network)
    
    return best_network

def _apply_network(network, x):
    x = list(x)
    for i, j in network:
        if x[i] > x[j]:
            x[i], x[j] = x[j], x[i]
    return x

def _is_sorted_network(network):
    for mask in range(1 << N):
        x = [(mask >> i) & 1 for i in range(N)]
        result = _apply_network(network, x)
        if result != sorted(result):
            return False
    return True

def _depth(network):
    return len(network)
