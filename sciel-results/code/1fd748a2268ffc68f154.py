import numpy as np
import time
import random
import sys

N = 18

def construct():
    """
    Constructs a sorting network using a layer-based local search strategy.
    
    1. Seed: Odd-Even Transposition Network (153 comparators).
    2. Structure the network into layers of disjoint comparators.
    3. Optimization:
       a. Greedy removal (linear order, backward pass).
       b. Layer-based Local Search:
          - Swap adjacent layers.
          - Move a comparator from one layer to another (if disjointness allows).
          - Swap comparators between layers.
          - Random reordering within layers (to break symmetry).
       c. Verification is bit-parallel over all 2^18 inputs.
    
    The key structural change is explicitly maintaining and manipulating the 
    "layer" structure, which allows for more intelligent moves that preserve 
    parallelism and explore the space of equivalent networks more effectively.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    # --- Helpers ---

    def odd_even_transposition(n):
        """
        Standard Odd-Even Transposition Sort network.
        """
        net = []
        for phase in range(n):
            for i in range(phase % 2, n - 1, 2):
                net.append((i, i + 1))
        return net

    def build_layers(net):
        """
        Convert a linear list of comparators into a list of layers.
        A layer is a list of comparators that are disjoint (no shared wires).
        We use a greedy assignment: each comparator goes to the first layer
        where it is disjoint with all existing comparators.
        """
        layers = []
        for comp in net:
            i, j = comp
            placed = False
            for k, layer in enumerate(layers):
                # Check if comp is disjoint with all in layer
                disjoint = True
                for c_i, c_j in layer:
                    if i == c_i or i == c_j or j == c_i or j == c_j:
                        disjoint = False
                        break
                if disjoint:
                    layer.append(comp)
                    placed = True
                    break
            if not placed:
                layers.append([comp])
        return layers

    def flatten_layers(layers):
        """Flatten layers back to a linear list of comparators."""
        net = []
        for layer in layers:
            net.extend(layer)
        return net

    # Setup Verification
    size = 1 << N
    k_range = np.arange(size, dtype=np.uint64)
    states = np.empty((N, size), dtype=np.uint8)
    for j in range(N):
        states[j] = ((k_range >> j) & 1).astype(np.uint8)

    def is_sorted():
        return not np.any(states[:-1] > states[1:])

    def apply_comparator(net_states, i, j):
        si = net_states[i]
        sj = net_states[j]
        net_states[i] = si & sj
        net_states[j] = si | sj

    def evaluate_network(net):
        s = states.copy()
        for i, j in net:
            apply_comparator(s, i, j)
        return s

    def is_valid_net(net):
        s = evaluate_network(net)
        return is_sorted()

    # Initialize Seed
    initial_net = odd_even_transposition(N)
    current_net = initial_net[:]
    best_net = initial_net[:]
    best_score = len(best_net)

    # --- Phase 1: Greedy Removal (Linear) ---
    # We do a few passes of greedy removal to get a baseline.
    def greedy_remove_pass(net):
        improved = True
        while improved and time.time() < deadline - 20.0:
            improved = False
            for idx in range(len(net) - 1, -1, -1):
                saved = net[idx]
                del net[idx]
                if is_valid_net(net):
                    improved = True
                else:
                    net.insert(idx, saved)
        return net

    current_net = greedy_remove_pass(current_net)
    if len(current_net) < best_score:
        best_score = len(current_net)
        best_net = current_net[:]

    # --- Phase 2: Layer-Based Local Search ---
    # Convert current net to layers
    current_layers = build_layers(current_net)
    
    T = 1.0
    cooling = 0.99
    sa_deadline = time.time() + (deadline - time.time()) * 0.8
    
    # Precompute all possible comparators for random addition
    all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    while time.time() < sa_deadline:
        T *= cooling
        
        # Choose a move type
        # 0: Swap adjacent layers
        # 1: Move comparator from layer A to layer B
        # 2: Swap comparator in layer A with comparator in layer B
        # 3: Add random comparator (insert into appropriate layer)
        # 4: Remove random comparator
        # 5: Shuffle within a layer
        
        move_type = random.randint(0, 5)
        
        # Deep copy layers for trial
        trial_layers = [layer[:] for layer in current_layers]
        
        valid_move = False
        
        if move_type == 0: # Swap adjacent layers
            if len(trial_layers) > 1:
                idx = random.randint(0, len(trial_layers) - 2)
                trial_layers[idx], trial_layers[idx+1] = trial_layers[idx+1], trial_layers[idx]
                valid_move = True
                
        elif move_type == 1: # Move comparator from layer A to layer B
            if len(trial_layers) > 1:
                src_idx = random.randint(0, len(trial_layers) - 1)
                dst_idx = random.randint(0, len(trial_layers) - 1)
                if src_idx != dst_idx and len(trial_layers[src_idx]) > 0:
                    # Pick a random comparator from src
                    comp_idx = random.randint(0, len(trial_layers[src_idx]) - 1)
                    comp = trial_layers[src_idx][comp_idx]
                    del trial_layers[src_idx][comp_idx]
                    
                    # Check if comp is disjoint with all in dst
                    i, j = comp
                    disjoint = True
                    for c_i, c_j in trial_layers[dst_idx]:
                        if i == c_i or i == c_j or j == c_i or j == c_j:
                            disjoint = False
                            break
                    
                    if disjoint:
                        trial_layers[dst_idx].append(comp)
                        valid_move = True
                    else:
                        # Revert
                        trial_layers[src_idx].insert(comp_idx, comp)
                        
        elif move_type == 2: # Swap comparators between layers
            if len(trial_layers) > 1:
                l1 = random.randint(0, len(trial_layers) - 1)
                l2 = random.randint(0, len(trial_layers) - 1)
                if l1 != l2 and len(trial_layers[l1]) > 0 and len(trial_layers[l2]) > 0:
                    i1 = random.randint(0, len(trial_layers[l1]) - 1)
                    i2 = random.randint(0, len(trial_layers[l2]) - 1)
                    comp1 = trial_layers[l1][i1]
                    comp2 = trial_layers[l2][i2]
                    
                    # Check if comp1 is disjoint with rest of l2 (excluding comp2)
                    # and comp2 is disjoint with rest of l1 (excluding comp1)
                    i1a, i1b = comp1
                    i2a, i2b = comp2
                    
                    disjoint1 = True
                    for c_i, c_j in trial_layers[l2]:
                        if c_i == i2a and c_j == i2b: continue # Skip comp2
                        if i1a == c_i or i1a == c_j or i1b == c_i or i1b == c_j:
                            disjoint1 = False
                            break
                            
                    disjoint2 = True
                    for c_i, c_j in trial_layers[l1]:
                        if c_i == i1a and c_j == i1b: continue # Skip comp1
                        if i2a == c_i or i2a == c_j or i2b == c_i or i2b == c_j:
                            disjoint2 = False
                            break
                            
                    if disjoint1 and disjoint2:
                        trial_layers[l1][i1], trial_layers[l2][i2] = comp2, comp1
                        valid_move = True
                        
        elif move_type == 3: # Add random comparator
            comp = random.choice(all_comps)
            i, j = comp
            # Find a layer where it fits
            placed = False
            for k, layer in enumerate(trial_layers):
                disjoint = True
                for c_i, c_j in layer:
                    if i == c_i or i == c_j or j == c_i or j == c_j:
                        disjoint = False
                        break
                if disjoint:
                    layer.append(comp)
                    placed = True
                    break
            if not placed:
                trial_layers.append([comp])
            valid_move = True
            
        elif move_type == 4: # Remove random comparator
            total_comps = sum(len(layer) for layer in trial_layers)
            if total_comps > 0:
                # Pick a random comparator
                target = random.randint(0, total_comps - 1)
                count = 0
                for k, layer in enumerate(trial_layers):
                    if count + len(layer) > target:
                        idx_in_layer = target - count
                        del layer[idx_in_layer]
                        break
                    count += len(layer)
                # Remove empty layers
                trial_layers = [layer for layer in trial_layers if layer]
                valid_move = True
                
        else: # Shuffle within a layer
            if len(trial_layers) > 0:
                l_idx = random.randint(0, len(trial_layers) - 1)
                if len(trial_layers[l_idx]) > 1:
                    random.shuffle(trial_layers[l_idx])
                    valid_move = True
        
        if not valid_move:
            continue
            
        # Evaluate trial
        trial_net = flatten_layers(trial_layers)
        if is_valid_net(trial_net):
            new_score = len(trial_net)
            old_score = len(current_net)
            delta = new_score - old_score
            
            if delta < 0:
                current_net = trial_net
                current_layers = trial_layers
                if new_score < best_score:
                    best_score = new_score
                    best_net = current_net[:]
            else:
                if delta == 0:
                    if random.random() < 0.5:
                        current_net = trial_net
                        current_layers = trial_layers
                else:
                    prob = np.exp(-delta / T)
                    if random.random() < prob:
                        current_net = trial_net
                        current_layers = trial_layers

    return best_net

def construct_safe():
    try:
        return construct()
    except Exception as e:
        print(f"Error in construct: {e}", file=sys.stderr)
        net = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                net.append((i, i + 1))
        return net

if __name__ == "__main__":
    res = construct()
    print(f"Comparators: {len(res)}")
