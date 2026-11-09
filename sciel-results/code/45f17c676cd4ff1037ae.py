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
    # Use uint8 for 0/1 values
    # 65536 inputs
    indices = np.arange(1 << N, dtype=np.uint32)
    inputs = np.zeros((1 << N, N), dtype=np.uint8)
    for j in range(N):
        inputs[:, j] = (indices >> j) & 1

    def is_sorting(net):
        state = inputs.copy()
        for i, j in net:
            min_val = np.minimum(state[:, i], state[:, j])
            max_val = np.maximum(state[:, i], state[:, j])
            state[:, i] = min_val
            state[:, j] = max_val
        # Check sorted
        return not np.any(state[:, :-1] > state[:, 1:])

    # Verify initial
    if not is_sorting(network):
        # Fallback to Odd-Even if Batcher fails (should not)
        network = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                network.append((i, i + 1))
        if not is_sorting(network):
            return network

    best_network = network[:]
    best_len = len(network)
    current_network = network[:]
    current_len = len(network)
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    # Precompute all possible comparators
    all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    steps = 0
    while time.time() < deadline and T > min_T:
        steps += 1
        
        # Choose a move
        move_type = random.choice(['remove', 'add', 'swap', 'replace'])
        
        new_net = current_network[:]
        
        if move_type == 'remove':
            if len(new_net) > 1:
                idx = random.randint(0, len(new_net) - 1)
                new_net.pop(idx)
                delta = -1
            else:
                continue
        elif move_type == 'add':
            # Add a random comparator
            # To avoid duplicates, we could check, but for SA it's okay to have duplicates?
            # No, duplicates are useless.
            # We pick a random pair not in the list?
            # Checking "not in list" is O(N).
            # For now, just pick random, if duplicate, it's a no-op effectively but costs time.
            # Better: pick from all_comps, if already in new_net, try a few times or skip.
            added = False
            for _ in range(10):
                c = random.choice(all_comps)
                if c not in new_net:
                    new_net.append(c)
                    added = True
                    break
            if not added:
                continue
            delta = 1
        elif move_type == 'swap':
            if len(new_net) > 1:
                i = random.randint(0, len(new_net) - 1)
                j = random.randint(0, len(new_net) - 1)
                if i != j:
                    new_net[i], new_net[j] = new_net[j], new_net[i]
                    delta = 0
                else:
                    continue
            else:
                continue
        elif move_type == 'replace':
            if len(new_net) > 0:
                idx = random.randint(0, len(new_net) - 1)
                for _ in range(10):
                    c = random.choice(all_comps)
                    if c not in new_net:
                        new_net[idx] = c
                        break
                else:
                    continue
                delta = 0

        # Check validity
        if is_sorting(new_net):
            # Valid
            new_len = len(new_net)
            # Accept if better or with probability
            if new_len < current_len:
                current_network = new_net
                current_len = new_len
                if current_len < best_len:
                    best_len = current_len
                    best_network = current_network[:]
            else:
                # Worse or same
                # Delta cost = new_len - current_len
                delta_cost = new_len - current_len
                # Prob = exp(-delta_cost / T)
                # If delta_cost is 0, prob=1
                # If delta_cost > 0, prob < 1
                if delta_cost == 0:
                    current_network = new_net
                    current_len = new_len
                else:
                    if random.random() < np.exp(-delta_cost / T):
                        current_network = new_net
                        current_len = new_len
        else:
            # Invalid, reject (or high penalty)
            pass
            
        T *= cooling

    # Final greedy removal pass on best_network
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for idx in range(len(best_network) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            candidate = best_network[:idx] + best_network[idx+1:]
            if is_sorting(candidate):
                best_network = candidate
                changed = True
                break # Restart loop to try next removal

    return best_network

# EVOLVE-BLOCK-END
