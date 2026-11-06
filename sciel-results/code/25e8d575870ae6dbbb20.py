"""Improved program for size-efficient sorting networks on 17 wires."""
import random
import time
import numpy as np

N = 17
MAX_COMPARATORS = 4096
TIME_LIMIT = 55.0  # Leave some buffer for 60s limit

# EVOLVE-BLOCK-START

def _verify_network_vectorized(layers):
    """
    Verify the network sorts all 2^N binary inputs using numpy vectorization.
    layers: list of lists of (i, j) comparators, each layer has disjoint comparators.
    Returns True if all inputs are sorted.
    """
    n = N
    total = 1 << n
    
    # Create array of all 2^n inputs, each row is one input (n bits)
    # Use uint8 for memory efficiency
    inputs = np.zeros((total, n), dtype=np.uint8)
    for i in range(n):
        # Column i: bit i of each mask
        # Pattern: 0 for first 2^i, 1 for next 2^i, repeat
        pattern = np.zeros(total, dtype=np.uint8)
        for start in range(0, total, 2 ** (i + 1)):
            pattern[start + 2**i : start + 2**(i+1)] = 1
        inputs[:, i] = pattern
    
    # Apply each layer
    for layer in layers:
        if not layer:
            continue
        # For each comparator in the layer, apply to all rows simultaneously
        # Since comparators in a layer are disjoint, we can process them in any order
        # but we must be careful: we need to use the values BEFORE this layer for all comparators
        # So we create a copy of the relevant columns
        
        # Collect all wire indices involved
        wires_involved = set()
        for i, j in layer:
            wires_involved.add(i)
            wires_involved.add(j)
        
        # Get current values for all involved wires
        involved_list = sorted(wires_involved)
        current_vals = inputs[:, involved_list].copy()
        
        # For each comparator, determine swap mask
        for i, j in layer:
            idx_i = involved_list.index(i)
            idx_j = involved_list.index(j)
            # Swap if current_vals[:, idx_i] > current_vals[:, idx_j]
            swap_mask = current_vals[:, idx_i] > current_vals[:, idx_j]
            if np.any(swap_mask):
                inputs[swap_mask, i] = current_vals[swap_mask, idx_j]
                inputs[swap_mask, j] = current_vals[swap_mask, idx_i]
    
    # Check if all rows are sorted
    # A row is sorted if inputs[:, k] <= inputs[:, k+1] for all k
    for k in range(n - 1):
        if np.any(inputs[:, k] > inputs[:, k + 1]):
            return False
    return True

def _count_comparators(layers):
    return sum(len(layer) for layer in layers)

def _generate_seed_network():
    """Generate the odd-even transposition network as layers."""
    layers = []
    for phase in range(N):
        layer = []
        for i in range(phase % 2, N - 1, 2):
            layer.append((i, i + 1))
        layers.append(layer)
    return layers

def _generate_batcher_network(n):
    """
    Generate a Batcher odd-even merge sort network.
    Returns list of layers (each layer is a list of disjoint comparators).
    """
    if n <= 1:
        return []
    
    # For odd-even merge sort, we recursively build the network
    # This is a simplified version - we'll use a known good structure
    
    # Actually, let's use a simpler approach: generate a bitonic sort network
    return _generate_bitonic_network(n)

def _generate_bitonic_network(n):
    """
    Generate a bitonic sort network for n wires.
    Returns list of layers.
    """
    if n <= 1:
        return []
    
    layers = []
    
    def bitonic_merge(lo, hi, direction):
        """Generate comparators for bitonic merge from lo to hi."""
        if hi - lo <= 1:
            return []
        mid = (lo + hi) // 2
        # First, recursively merge the two halves
        l1 = bitonic_merge(lo, mid, direction)
        l2 = bitonic_merge(mid, hi, direction)
        layers.extend(l1)
        layers.extend(l2)
        # Then, compare elements that are (hi-lo)/2 apart
        step = (hi - lo) // 2
        for i in range(lo, mid):
            j = i + step
            if direction:
                # We want ascending, so compare (i, j) with i < j
                layers.append([(i, j)])
            else:
                layers.append([(i, j)])
        return []
    
    # For a full bitonic sort, we need to create bitonic sequences first
    # Simplified: just use the merge network assuming input is already bitonic
    
    # Actually, let's use a simpler known-good network structure
    # Use the odd-even transposition as a baseline and try to improve
    
    return _generate_seed_network()

def _local_search(layers, time_limit=30.0):
    """
    Perform local search to minimize comparator count.
    Try removing comparators and see if the network is still valid.
    """
    start_time = time.time()
    best_layers = [list(layer) for layer in layers]
    best_count = _count_comparators(best_layers)
    
    # First, verify the starting network
    if not _verify_network_vectorized(best_layers):
        return layers
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        # Try removing each comparator
        for li in range(len(best_layers)):
            if time.time() - start_time >= time_limit:
                break
            for ci in range(len(best_layers[li])):
                if time.time() - start_time >= time_limit:
                    break
                # Try removing this comparator
                trial = [list(layer) for layer in best_layers]
                trial[li].pop(ci)
                
                # Skip empty layers
                new_layers = [layer for layer in trial if layer]
                
                # Verify
                if _verify_network_vectorized(new_layers):
                    count = _count_comparators(new_layers)
                    if count < best_count:
                        best_layers = [list(layer) for layer in new_layers]
                        best_count = count
                        improved = True
                        break
            if improved:
                break
    
    return best_layers

def _try_add_comparator(layers, time_limit=10.0):
    """
    Try adding a comparator to potentially enable removal of others later.
    This is a heuristic - usually we want to minimize, so this is less useful.
    """
    pass

def _optimize_network(initial_layers, time_limit=50.0):
    """
    Optimize the network using local search and other heuristics.
    """
    start_time = time.time()
    
    # Start with the seed network
    best_layers = [list(layer) for layer in initial_layers]
    best_count = _count_comparators(best_layers)
    
    # Verify the seed network
    if not _verify_network_vectorized(best_layers):
        return initial_layers
    
    # Phase 1: Local search - remove redundant comparators
    best_layers = _local_search(best_layers, time_limit=min(30.0, time_limit * 0.6))
    best_count = _count_comparators(best_layers)
    
    # Phase 2: Try different initial structures and local search
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 10.0:
        # Try some alternative initial networks
        alternatives = []
        
        # Alternative 1: Different ordering of odd-even transposition
        alt1 = _generate_seed_network()
        # Reverse the wire order
        alt1_reversed = []
        for layer in alt1:
            new_layer = [tuple(sorted((N-1-i, N-1-j))) for i, j in layer]
            alt1_reversed.append(new_layer)
        alternatives.append(alt1_reversed)
        
        for alt in alternatives:
            if time.time() - start_time >= time_limit:
                break
            alt_layers = [list(layer) for layer in alt]
            if _verify_network_vectorized(alt_layers):
                alt_count = _count_comparators(alt_layers)
                if alt_count < best_count:
                    best_layers = alt_layers
                    best_count = alt_count
                # Local search on this alternative
                remaining = time_limit - (time.time() - start_time)
                if remaining > 5.0:
                    improved = _local_search(best_layers, time_limit=min(remaining * 0.5, 15.0))
                    improved_count = _count_comparators(improved)
                    if improved_count < best_count:
                        best_layers = improved
                        best_count = improved_count
    
    return best_layers

def construct():
    # Generate the seed network
    seed_layers = _generate_seed_network()
    
    # Verify it's correct
    if not _verify_network_vectorized(seed_layers):
        # Should not happen
        return [(i, j) for layer in seed_layers for i, j in layer]
    
    # Try to optimize
    try:
        optimized_layers = _optimize_network(seed_layers, time_limit=TIME_LIMIT)
        
        # Verify the optimized network
        if _verify_network_vectorized(optimized_layers):
            # Convert layers to flat list of comparators
            comparators = []
            for layer in optimized_layers:
                for comp in layer:
                    comparators.append(comp)
            
            # Check constraint
            if len(comparators) <= MAX_COMPARATORS:
                return comparators
    except Exception:
        pass
    
    # Fallback to seed network
    comparators = []
    for layer in seed_layers:
        for comp in layer:
            comparators.append(comp)
    return comparators

# EVOLVE-BLOCK-END
