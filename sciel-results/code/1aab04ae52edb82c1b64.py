import time
import numpy as np

N = 18
DEADLINE = 85.0

def construct():
    start_time = time.time()
    
    # 1. Generate Inputs
    # Shape: (262144, 18)
    all_inputs = np.zeros((1 << N, N), dtype=np.uint8)
    for i in range(N):
        all_inputs[:, i] = (np.arange(1 << N) >> i) & 1
        
    def is_valid(network):
        state = all_inputs.copy()
        for i, j in network:
            state[:, i], state[:, j] = np.minimum(state[:, i], state[:, j]), np.maximum(state[:, i], state[:, j])
        # Check sorted
        return np.all(state[:, :-1] <= state[:, 1:])
        
    # 2. Initial Network: Odd-Even Transposition
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
            
    # Verify initial
    if not is_valid(network):
        return network # Should not happen
        
    # 3. Greedy Removal
    # Try to remove comparators
    changed = True
    while changed and time.time() < DEADLINE - 5:
        changed = False
        # Random order
        indices = np.random.permutation(len(network))
        for idx in indices:
            if time.time() > DEADLINE - 5:
                break
            # Try removing network[idx]
            new_network = network[:idx] + network[idx+1:]
            if is_valid(new_network):
                network = new_network
                changed = True
                # Restart permutation? Or just continue?
                # Continuing is faster, but might miss removals enabled by previous ones.
                # Let's restart the loop if we found a removal.
                break
                
    # 4. Simulated Annealing / Local Search
    # We want to escape local minima.
    # Moves: Swap, Replace, Remove.
    best_network = network
    best_size = len(network)
    
    T = 1.0
    T_min = 0.01
    alpha = 0.99
    
    current_network = network
    current_size = len(network)
    
    no_improve_count = 0
    
    while time.time() < DEADLINE:
        if no_improve_count > 500:
            # Perturb: Add a random comparator
            i = np.random.randint(0, N-1)
            j = np.random.randint(i+1, N)
            new_net = current_network + [(i, j)]
            # Try to prune it
            # ... (Simplified: just accept if it allows removals later?)
            # For now, just restart or accept perturbation
            current_network = new_net
            current_size = len(new_net)
            no_improve_count = 0
            T = max(T * 0.5, T_min) # Cool down? Or heat up?
            # Actually, if we are stuck, we should explore more.
            # Let's just try a random swap sequence.
            
        # Choose a move
        move_type = np.random.choice(['remove', 'swap', 'replace'], p=[0.5, 0.3, 0.2])
        
        if move_type == 'remove':
            if len(current_network) == 0:
                continue
            idx = np.random.randint(0, len(current_network))
            new_network = current_network[:idx] + current_network[idx+1:]
            new_size = len(new_network)
            
            if is_valid(new_network):
                current_network = new_network
                current_size = new_size
                no_improve_count = 0
                if current_size < best_size:
                    best_network = current_network
                    best_size = current_size
            else:
                no_improve_count += 1
                
        elif move_type == 'swap':
            if len(current_network) < 2:
                continue
            i1, i2 = np.random.choice(len(current_network), 2, replace=False)
            # Swap
            temp = current_network[i1]
            current_network[i1] = current_network[i2]
            current_network[i2] = temp
            
            if is_valid(current_network):
                no_improve_count = 0
                # Size is same, so no update to best_size
            else:
                # Revert
                current_network[i2] = current_network[i1]
                current_network[i1] = temp
                no_improve_count += 1
                
        elif move_type == 'replace':
            if len(current_network) == 0:
                continue
            idx = np.random.randint(0, len(current_network))
            # Replace with random comparator
            i = np.random.randint(0, N-1)
            j = np.random.randint(i+1, N)
            new_network = current_network.copy()
            new_network[idx] = (i, j)
            
            if is_valid(new_network):
                current_network = new_network
                no_improve_count = 0
            else:
                no_improve_count += 1
                
        # Cool down
        T *= alpha
        
    return best_network
