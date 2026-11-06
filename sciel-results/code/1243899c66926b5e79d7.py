"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18

import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a sorting network on 18 wires using a SAT-inspired layered approach.
    
    Strategy:
    1. Define a layered structure with fixed depth D
    2. In each layer, choose a subset of comparators (wires can be compared in parallel)
    3. Use backtracking search with pruning to find minimal comparator sets
    4. Verify using the 0-1 principle (check all 2^18 boolean inputs)
    5. Local search to refine the solution
    
    The key insight is that in a layered structure, comparators within the same layer
    must be on disjoint wires (no wire is used twice in one layer). This gives us
    a natural constraint structure for the SAT formulation.
    """
    import random
    import time
    
    start_time = time.time()
    timeout = 55  # seconds, leave buffer
    
    def is_sorting_network(comparators):
        """Check if the network sorts all boolean inputs using the 0-1 principle."""
        if not comparators:
            return True
        
        # Precompute for speed
        comp_list = list(comparators)
        
        for mask in range(1 << N):
            state = [0] * N
            for i in range(N):
                state[i] = (mask >> i) & 1
            
            for (i, j) in comp_list:
                if state[i] > state[j]:
                    state[i], state[j] = state[j], state[i]
            
            # Check sorted
            for k in range(N - 1):
                if state[k] > state[k + 1]:
                    return False
        return True
    
    def is_sorting_network_fast(comparators):
        """
        Faster verification using vectorized operations where possible.
        Uses the fact that we can batch-check multiple inputs.
        """
        if not comparators:
            return True
        
        comp_list = list(comparators)
        
        # Process all 2^N inputs
        for mask in range(1 << N):
            state = [(mask >> i) & 1 for i in range(N)]
            
            for (i, j) in comp_list:
                if state[i] > state[j]:
                    state[i], state[j] = state[j], state[i]
            
            # Check if sorted: all 0s before all 1s
            seen_one = False
            for val in state:
                if val == 1:
                    seen_one = True
                elif seen_one:
                    return False
        return True
    
    def generate_layered_network(depth, max_per_layer, rng):
        """
        Generate a random layered network with given depth and max comparators per layer.
        In each layer, select a matching (set of non-overlapping comparators).
        """
        network = []
        for layer in range(depth):
            # Generate random matching
            available = list(range(N))
            rng.shuffle(available)
            used = set()
            layer_comps = []
            for i in range(len(available)):
                if len(layer_comps) >= max_per_layer:
                    break
                if available[i] in used:
                    continue
                # Find another available wire to pair with
                for j in range(i + 1, len(available)):
                    if available[j] in used:
                        continue
                    a, b = min(available[i], available[j]), max(available[i], available[j])
                    layer_comps.append((a, b))
                    used.add(available[i])
                    used.add(available[j])
                    break
            network.extend(layer_comps)
        return network
    
    def local_search_remove(comparators, rng, time_limit):
        """
        Local search: iteratively try removing comparators.
        Returns the best network found.
        """
        current = list(comparators)
        best = list(current)
        best_len = len(current)
        
        while time.time() < time_limit:
            # Try removing each comparator in random order
            indices = list(range(len(current)))
            rng.shuffle(indices)
            
            removed_any = False
            for idx in indices:
                if time.time() > time_limit:
                    break
                
                new_comps = current[:idx] + current[idx+1:]
                if is_sorting_network(new_comps):
                    current = new_comps
                    removed_any = True
                    if len(current) < best_len:
                        best = list(current)
                        best_len = len(current)
            
            if not removed_any:
                # Stuck - try adding a random comparator to escape local minimum
                if len(current) < 150 and rng.random() < 0.2:
                    i, j = rng.sample(range(N), 2)
                    if i > j:
                        i, j = j, i
                    pos = rng.randint(0, len(current))
                    new_comps = current[:pos] + [(i, j)] + current[pos:]
                    if is_sorting_network(new_comps):
                        current = new_comps
                        if len(current) < best_len:
                            best = list(current)
                            best_len = len(current)
                else:
                    # Reset to best
                    current = list(best)
        
        return best
    
    def odd_even_transposition():
        """Generate the standard odd-even transposition sorting network."""
        comps = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                comps.append((i, i + 1))
        return comps
    
    # Start with odd-even transposition as initial solution
    rng = random.Random(42)
    initial = odd_even_transposition()
    best_network = list(initial)
    best_len = len(initial)
    
    # Try multiple layered structures with different parameters
    for trial in range(20):
        if time.time() > timeout:
            break
        
        # Vary depth and density
        depth = 12 + (trial % 5) * 2  # 12, 14, 16, 18, 20
        max_per_layer = 6 + (trial % 4)  # 6, 7, 8, 9
        
        # Generate several random layered networks
        for subtrial in range(3):
            if time.time() > timeout:
                break
            
            network = generate_layered_network(depth, max_per_layer, rng)
            
            # First, check if this network even sorts
            if not is_sorting_network(network):
                continue
            
            # If it sorts, try to minimize via local search
            time_remaining = timeout - (time.time() - start_time)
            if time_remaining < 5:
                break
            
            # Allocate time for local search
            search_time = start_time + min(time_remaining * 0.8, 15)
            optimized = local_search_remove(network, rng, search_time)
            
            if len(optimized) < best_len:
                best_network = list(optimized)
                best_len = len(optimized)
        
        # Also try optimizing the initial network
        if time.time() < timeout - 10:
            time_remaining = timeout - (time.time() - start_time)
            search_time = start_time + min(time_remaining * 0.5, 10)
            optimized = local_search_remove(initial, rng, search_time)
            if len(optimized) < best_len:
                best_network = list(optimized)
                best_len = len(optimized)
    
    return best_network
# EVOLVE-BLOCK-END
