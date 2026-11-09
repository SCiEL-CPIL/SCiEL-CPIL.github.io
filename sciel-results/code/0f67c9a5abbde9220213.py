# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 16

def construct():
    start_time = time.time()
    deadline = 88.0  # Safety margin

    # 1. Generate Batcher's Bitonic Sort Network (80 comparators)
    def generate_batcher(n):
        if n == 2:
            return [(0, 1)]
        
        half = n // 2
        left_net = generate_batcher(half)
        right_net = generate_batcher(half)
        right_net_remapped = [(i + half, j + half) for i, j in right_net]
        
        merge_net = []
        d = n // 2
        while d > 0:
            for i in range(0, n, 2 * d):
                for j in range(i, i + d):
                    merge_net.append((j, j + d))
            d //= 2
            
        return left_net + right_net + merge_net

    network = generate_batcher(N)
    
    # 2. Prepare Inputs for Verification
    indices = np.arange(1 << N, dtype=np.uint32)
    inputs = np.zeros((1 << N, N), dtype=np.uint8)
    for j in range(N):
        inputs[:, j] = (indices >> j) & 1
        
    # Pre-allocate buffer for state to avoid repeated allocations
    state_buf = np.empty_like(inputs)

    def is_sorting(net):
        # Copy inputs to buffer
        np.copyto(state_buf, inputs)
        for i, j in net:
            min_val = np.minimum(state_buf[:, i], state_buf[:, j])
            max_val = np.maximum(state_buf[:, i], state_buf[:, j])
            state_buf[:, i] = min_val
            state_buf[:, j] = max_val
        # Check sorted
        return not np.any(state_buf[:, :-1] > state_buf[:, 1:])

    # Verify initial
    if not is_sorting(network):
        network = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                network.append((i, i + 1))
        if not is_sorting(network):
            return network

    # Greedy Removal Phase 1
    def greedy_remove(net):
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            # Shuffle indices to avoid bias towards earlier comparators
            indices = list(range(len(net)))
            random.shuffle(indices)
            for idx in indices:
                candidate = net[:idx] + net[idx+1:]
                if is_sorting(candidate):
                    net = candidate
                    changed = True
                    break # Restart loop with new length
        return net

    best_network = greedy_remove(network)
    best_len = len(best_network)
    
    current_network = best_network[:]
    current_len = best_len
    current_set = set(current_network)

    # Simulated Annealing
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    steps = 0
    while time.time() < deadline and T > min_T:
        steps += 1
        
        # Choose a move
        # Bias towards removal if possible, or swap/replace
        move_type = random.choice(['remove', 'add', 'swap', 'replace'])
        
        new_net = current_network[:]
        new_set = current_set.copy()
        
        if move_type == 'remove':
            if len(new_net) > 1:
                idx = random.randint(0, len(new_net) - 1)
                comp_to_remove = new_net[idx]
                new_net.pop(idx)
                new_set.discard(comp_to_remove)
            else:
                continue
        elif move_type == 'add':
            added = False
            # Try to add a comparator not in current set
            for _ in range(10):
                c = random.choice(all_comps)
                if c not in new_set:
                    new_net.append(c)
                    new_set.add(c)
                    added = True
                    break
            if not added:
                continue
        elif move_type == 'swap':
            if len(new_net) > 1:
                i = random.randint(0, len(new_net) - 1)
                j = random.randint(0, len(new_net) - 1)
                if i != j:
                    new_net[i], new_net[j] = new_net[j], new_net[i]
                    # Set remains same
                else:
                    continue
            else:
                continue
        elif move_type == 'replace':
            if len(new_net) > 0:
                idx = random.randint(0, len(new_net) - 1)
                old_comp = new_net[idx]
                replaced = False
                for _ in range(10):
                    c = random.choice(all_comps)
                    if c not in new_set:
                        new_net[idx] = c
                        new_set.discard(old_comp)
                        new_set.add(c)
                        replaced = True
                        break
                if not replaced:
                    continue

        # Check validity
        if is_sorting(new_net):
            new_len = len(new_net)
            if new_len < current_len:
                current_network = new_net
                current_len = new_len
                current_set = new_set
                if current_len < best_len:
                    best_len = current_len
                    best_network = current_network[:]
                    # Run greedy removal on the new best to tighten it up
                    if time.time() < deadline - 5:
                        best_network = greedy_remove(best_network)
                        best_len = len(best_network)
                        # Update current to best to continue from improved state
                        current_network = best_network[:]
                        current_len = best_len
                        current_set = set(current_network)
            else:
                delta_cost = new_len - current_len
                if delta_cost == 0:
                    current_network = new_net
                    current_len = new_len
                    current_set = new_set
                else:
                    if random.random() < np.exp(-delta_cost / T):
                        current_network = new_net
                        current_len = new_len
                        current_set = new_set
        else:
            # Invalid, reject
            pass
            
        T *= cooling

    # Final greedy removal pass on best_network
    if time.time() < deadline - 1:
        best_network = greedy_remove(best_network)

    return best_network

# EVOLVE-BLOCK-END
