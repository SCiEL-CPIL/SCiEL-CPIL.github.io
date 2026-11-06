"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import numpy as np

N = 17
MAX_COMPARATORS = 4096
TIME_LIMIT = 55.0  # Leave some buffer for 60s limit

# EVOLVE-BLOCK-START

def _verify_network_numpy(layers):
    """
    Verify the network sorts all 2^N binary inputs using NumPy vectorization.
    layers: list of lists of (i, j) tuples.
    Returns True if valid, False otherwise.
    """
    n = N
    total = 1 << n
    
    # Initialize X with all binary inputs
    # X[k, i] is the value of wire i for input k
    # We can generate this using bit manipulation or arange
    # X = np.array([[(k >> i) & 1 for i in range(n)] for k in range(total)]) is too slow
    # Use bit tricks
    bits = np.arange(total, dtype=np.uint8)
    X = np.zeros((total, n), dtype=np.uint8)
    for i in range(n):
        X[:, i] = (bits >> i) & 1
        
    # Apply layers
    for layer in layers:
        # We must apply comparators in the layer simultaneously or sequentially?
        # In a sorting network, a "layer" consists of disjoint comparators.
        # Since they are disjoint, they can be applied in any order or simultaneously.
        # With NumPy, we can update X in place for disjoint pairs safely?
        # If (i,j) and (k,l) are disjoint, updating X[:, i], X[:, j] doesn't affect X[:, k], X[:, l].
        # So we can iterate through the layer and update X in place.
        for i, j in layer:
            # min to i, max to j
            min_val = np.minimum(X[:, i], X[:, j])
            max_val = np.maximum(X[:, i], X[:, j])
            X[:, i] = min_val
            X[:, j] = max_val
            
    # Check if all outputs are sorted
    # A binary string is sorted if it is of the form 0...01...1
    # This is equivalent to X[:, k] <= X[:, k+1] for all k
    # Or simply: X is sorted if for every row, there is no 1 followed by 0.
    # X[:, :-1] > X[:, 1:] should be all False.
    if np.any(X[:, :-1] > X[:, 1:]):
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

def _remove_comparator(layers, layer_idx, comp_idx):
    """Remove a specific comparator from a specific layer."""
    new_layers = [list(layer) for layer in layers]
    if layer_idx < len(new_layers) and comp_idx < len(new_layers[layer_idx]):
        new_layers[layer_idx].pop(comp_idx)
    return [tuple(layer) for layer in new_layers]

def _add_comparator_random(layers):
    """Add a random comparator to a random layer (or create new layer)."""
    new_layers = [list(layer) for layer in layers]
    
    # Choose a layer
    if new_layers and random.random() < 0.8:
        layer_idx = random.randint(0, len(new_layers) - 1)
    else:
        layer_idx = len(new_layers)
        new_layers.append([])
        
    # Choose wires
    i = random.randint(0, N - 2)
    j = random.randint(i + 1, N - 1)
    
    # Check if (i,j) or (j,i) already in this layer? 
    # For simplicity, just add. Duplicates in a layer are redundant but harmless for validity check (just wasted time).
    # However, disjointness is required for "layer" definition in parallel depth, but for sequential execution (which the verifier effectively does by iterating layer by layer), disjointness within a layer allows in-place update.
    # If we add a comparator that conflicts with existing ones in the same layer, the in-place update logic in _verify_network_numpy is still correct?
    # NO. If (0,1) and (1,2) are in the same layer, they are NOT disjoint.
    # _verify_network_numpy updates X[:, 0], X[:, 1] then X[:, 1], X[:, 2]. This is sequential, not simultaneous.
    # The task defines a network as an ordered list of comparators. The "layer" representation is just a grouping for potential parallelism or structure.
    # The verifier applies comparators in the order they appear in the list (flattened from layers).
    # So, if I put conflicting comparators in the same "layer", they will be applied sequentially in the order they appear in that layer's list.
    # This is valid for the "ordered list of integer pairs" interface.
    # So I can just append (i,j) to the layer list.
    
    new_layers[layer_idx].append((i, j))
    return [tuple(layer) for layer in new_layers]

def _swap_comparators(layers, layer_idx1, comp_idx1, layer_idx2, comp_idx2):
    """Swap two comparators in the network."""
    new_layers = [list(layer) for layer in layers]
    if layer_idx1 >= len(new_layers) or comp_idx1 >= len(new_layers[layer_idx1]):
        return new_layers
    if layer_idx2 >= len(new_layers) or comp_idx2 >= len(new_layers[layer_idx2]):
        return new_layers
        
    c1 = new_layers[layer_idx1][comp_idx1]
    c2 = new_layers[layer_idx2][comp_idx2]
    
    new_layers[layer_idx1][comp_idx1] = c2
    new_layers[layer_idx2][comp_idx2] = c1
    return [tuple(layer) for layer in new_layers]

def _optimize_network(initial_layers, time_limit=50.0):
    """
    Optimize the network using Greedy Removal and Random Perturbation.
    """
    start_time = time.time()
    
    best_network = [list(layer) for layer in initial_layers]
    best_count = _count_comparators(best_network)
    
    # Verify seed
    if not _verify_network_numpy(best_network):
        return initial_layers
        
    # Greedy Removal Pass 1
    # Try removing each comparator. If valid, keep removed.
    # We iterate backwards to avoid index shifting issues, or just rebuild.
    current = [list(layer) for layer in best_network]
    
    # Flatten to list of (layer_idx, comp_idx) to iterate
    # But removing changes indices.
    # Strategy: Iterate through all possible removals repeatedly until no more can be removed?
    # Or just one pass? One pass is fast.
    
    # Let's do a randomized greedy removal
    while time.time() - start_time < time_limit * 0.7: # Use 70% of time for removal
        # Pick a random comparator to try to remove
        # Get all indices
        all_indices = []
        for li, layer in enumerate(current):
            for ci, _ in enumerate(layer):
                all_indices.append((li, ci))
                
        if not all_indices:
            break
            
        random.shuffle(all_indices)
        
        removed_any = False
        for li, ci in all_indices:
            # Check if still exists (might have been removed by previous step in this batch? No, we pick one at a time)
            # Actually, we pick one, test, if valid, remove it and restart the shuffle?
            # That's expensive.
            # Let's just pick one, test, if valid, remove and continue to next random pick.
            
            # Re-fetch current state
            if li >= len(current) or ci >= len(current[li]):
                continue
                
            # Try removing
            candidate = _remove_comparator(current, li, ci)
            
            # Verify
            if _verify_network_numpy(candidate):
                current = candidate
                best_count = _count_comparators(current)
                if best_count < best_network and _count_comparators(best_network) > best_count:
                    best_network = [list(layer) for layer in current]
                removed_any = True
                # Break to re-shuffle? Or continue?
                # If we continue, indices are invalid.
                # So we break the inner loop and restart the outer loop logic?
                # Simpler: Just pick one random comparator, test, if valid remove. Repeat.
                break
        
        if not removed_any:
            # No more simple removals possible.
            # Try some random swaps or additions?
            # For now, just stop removal phase.
            break
            
    # Random Perturbation Phase (if time remains)
    while time.time() - start_time < time_limit:
        # Try a random move: Remove a random comp, Add a random comp, or Swap
        move_type = random.random()
        
        candidate = [list(layer) for layer in current]
        
        if move_type < 0.5:
            # Try to remove a random comparator
            all_indices = []
            for li, layer in enumerate(candidate):
                for ci, _ in enumerate(layer):
                    all_indices.append((li, ci))
            if all_indices:
                li, ci = random.choice(all_indices)
                candidate = _remove_comparator(candidate, li, ci)
                if _verify_network_numpy(candidate):
                    current = candidate
                    best_count = _count_comparators(current)
                    if best_count < _count_comparators(best_network):
                        best_network = [list(layer) for layer in current]
                        
        elif move_type < 0.8:
            # Try to add a random comparator
            candidate = _add_comparator_random(candidate)
            if _verify_network_numpy(candidate):
                # Only accept if it doesn't increase size? Or if it allows future removals?
                # Usually we want to minimize size. Adding increases size.
                # So only accept if it enables a removal? That's complex.
                # Skip adding for now, focus on removal.
                pass
        else:
            # Swap two random comparators
            all_indices = []
            for li, layer in enumerate(candidate):
                for ci, _ in enumerate(layer):
                    all_indices.append((li, ci))
            if len(all_indices) >= 2:
                (li1, ci1), (li2, ci2) = random.sample(all_indices, 2)
                candidate = _swap_comparators(candidate, li1, ci1, li2, ci2)
                if _verify_network_numpy(candidate):
                    # Swapping doesn't change count. Accept if valid?
                    # It might help future removals.
                    current = candidate

    return best_network

def construct():
    # Generate the seed network
    seed_layers = _generate_seed_network()
    
    # Verify it's correct
    if not _verify_network_numpy(seed_layers):
        # Should not happen
        return [(i, j) for layer in seed_layers for i, j in layer]
    
    # Try to optimize
    try:
        optimized_layers = _optimize_network(seed_layers, time_limit=TIME_LIMIT)
        
        # Verify the optimized network
        if _verify_network_numpy(optimized_layers):
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
