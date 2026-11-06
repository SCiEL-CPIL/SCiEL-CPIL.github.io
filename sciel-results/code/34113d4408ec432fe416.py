import numpy as np
import time
import random

N = 17
MAX_COMPARATORS = 4096

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit

    # Pre-compute all 2^N boolean inputs
    inputs = np.zeros((2**N, N), dtype=np.uint8)
    for i in range(N):
        inputs[:, i] = (np.arange(2**N) >> (N - 1 - i)) & 1

    def simulate_layer(state, layer):
        """Apply a single layer of disjoint comparators to state."""
        for i, j in layer:
            si = state[:, i]
            sj = state[:, j]
            min_val = np.minimum(si, sj)
            max_val = np.maximum(si, sj)
            state[:, i] = min_val
            state[:, j] = max_val
        return state

    def simulate_network(state, layers):
        """Simulate the full network (list of layers)."""
        for layer in layers:
            simulate_layer(state, layer)
        return state

    def is_sorted(state):
        """Check if all rows are sorted (non-decreasing)."""
        return np.all(state[:, 1:] >= state[:, :-1])

    def is_valid(layers):
        """Check if the network sorts all 2^N inputs."""
        state = inputs.copy()
        state = simulate_network(state, layers)
        return is_sorted(state)

    def count_comparators(layers):
        """Total number of comparators in the network."""
        return sum(len(layer) for layer in layers)

    # Generate initial network as layers: Odd-Even Transposition Network
    def generate_oetn_layers():
        layers = []
        for phase in range(N):
            layer = []
            for i in range(phase % 2, N - 1, 2):
                layer.append((i, i + 1))
            layers.append(layer)
        return layers

    # Verify a layer has no conflicting comparators (disjoint wires)
    def is_valid_layer(layer):
        """Check that no wire appears in more than one comparator in the layer."""
        used = set()
        for i, j in layer:
            if i in used or j in used:
                return False
            used.add(i)
            used.add(j)
        return True

    # Start with OETN
    layers = generate_oetn_layers()
    
    if not is_valid(layers):
        # Fallback: return flat list
        return [(i, j) for layer in layers for i, j in layer]

    best_layers = layers[:]
    best_count = count_comparators(layers)

    # Strategy 1: Greedy removal - try to remove each comparator from its layer
    def try_removals_layered(layers, max_time):
        """Try to remove redundant comparators from layers."""
        improved = True
        total_removed = 0
        while improved and time.time() < max_time:
            improved = False
            for li in range(len(layers)):
                if time.time() >= max_time:
                    break
                for ci in range(len(layers[li])):
                    if time.time() >= max_time:
                        break
                    # Remove this comparator
                    candidate_layers = [l[:] for l in layers]
                    candidate_layers[li] = [c for k, c in enumerate(candidate_layers[li]) if k != ci]
                    
                    # Remove empty layers
                    candidate_layers = [l for l in candidate_layers if l]
                    
                    if candidate_layers and is_valid(candidate_layers):
                        layers = candidate_layers
                        total_removed += 1
                        improved = True
                        break
                if improved:
                    break
        return layers, total_removed

    # Strategy 2: Move comparator to different layer
    def try_moves(layers, max_time, max_moves=500):
        """Try moving comparators to different layers."""
        moves_done = 0
        for _ in range(max_moves):
            if time.time() >= max_time:
                break
            
            # Pick a random comparator
            li = random.randint(0, len(layers) - 1)
            if not layers[li]:
                continue
            ci = random.randint(0, len(layers[li]) - 1)
            comp = layers[li][ci]
            
            # Try moving to a random other layer
            target_li = random.randint(0, len(layers) - 1)
            if target_li == li:
                continue
            
            # Create candidate
            candidate_layers = [l[:] for l in layers]
            candidate_layers[li].remove(comp)
            candidate_layers[target_li].append(comp)
            
            # Check if target layer is still valid (disjoint)
            if not is_valid_layer(candidate_layers[target_li]):
                continue
            
            # Remove empty layers
            candidate_layers = [l for l in candidate_layers if l]
            
            if candidate_layers and is_valid(candidate_layers):
                layers = candidate_layers
                moves_done += 1
        
        return layers, moves_done

    # Strategy 3: Add random comparator and try to remove
    def try_add_remove(layers, max_time):
        """Add a random comparator, then try greedy removal."""
        if len(layers) == 0:
            layers = [[]]
        
        # Add a random comparator to a random layer
        li = random.randint(0, len(layers) - 1)
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        
        candidate_layers = [l[:] for l in layers]
        candidate_layers[li].append((i, j))
        
        # Check layer validity
        if not is_valid_layer(candidate_layers[li]):
            return layers, False
        
        # Try to remove it back (should be valid since we added it)
        # Actually, let's just do greedy removal on the augmented network
        result_layers, _ = try_removals_layered(candidate_layers, max_time)
        
        if count_comparators(result_layers) < count_comparators(layers):
            return result_layers, True
        return layers, False

    # Phase 1: Greedy removal
    layers, removed = try_removals_layered(layers, deadline - 20.0)
    if count_comparators(layers) < best_count:
        best_layers = layers[:]
        best_count = count_comparators(layers)

    # Phase 2: Try moves
    if time.time() < deadline - 10.0:
        layers2, _ = try_moves(layers, deadline - 10.0, max_moves=200)
        if count_comparators(layers2) < best_count:
            best_layers = layers2[:]
            best_count = count_comparators(layers2)

    # Phase 3: More aggressive local search with add-remove
    if time.time() < deadline - 5.0:
        for attempt in range(10):
            if time.time() >= deadline - 5.0:
                break
            
            start_layers = best_layers[:]
            result_layers, improved = try_add_remove(start_layers, deadline - 5.0)
            if improved and count_comparators(result_layers) < best_count:
                best_layers = result_layers[:]
                best_count = count_comparators(result_layers)
            
            # Also try moves from best
            if time.time() < deadline - 5.0:
                moved_layers, _ = try_moves(best_layers[:], deadline - 5.0, max_moves=100)
                if count_comparators(moved_layers) < best_count:
                    best_layers = moved_layers[:]
                    best_count = count_comparators(moved_layers)

    # Phase 4: Final greedy removal pass
    if time.time() < deadline - 2.0:
        final_layers, _ = try_removals_layered(best_layers[:], deadline - 2.0)
        if count_comparators(final_layers) < best_count:
            best_layers = final_layers[:]
            best_count = count_comparators(final_layers)

    # Final verification
    if not is_valid(best_layers):
        # Fallback to original OETN
        oetn = generate_oetn_layers()
        return [(i, j) for layer in oetn for i, j in layer]

    # Convert layers to flat list of comparators
    result = []
    for layer in best_layers:
        for i, j in layer:
            result.append((i, j))
    
    return result
