import time
import random

N = 17

def apply_network(state, network):
    for i, j in network:
        vi = (state >> i) & 1
        vj = (state >> j) & 1
        min_v = vi & vj
        max_v = vi | vj
        state &= ~(1 << i)
        state &= ~(1 << j)
        state |= (min_v << i)
        state |= (max_v << j)
    return state

def is_valid_network(network):
    for mask in range(1 << N):
        result = apply_network(mask, network)
        k = bin(mask).count('1')
        if k == 0:
            sorted_mask = 0
        elif k == N:
            sorted_mask = (1 << N) - 1
        else:
            sorted_mask = ((1 << k) - 1) << (N - k)
        if result != sorted_mask:
            return False
    return True

def construct():
    # Start with Odd-Even Transposition Network
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # Local Search to minimize comparators
    start_time = time.time()
    timeout = 80.0 # 10s margin
    
    # To avoid getting stuck, we can try to remove comparators in random order.
    # We can also try to "swap" comparators if removal fails, but removal is the primary goal.
    
    while time.time() - start_time < timeout:
        improved = False
        # Randomize the order of comparators to try different removals
        indices = list(range(len(network)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() - start_time > timeout:
                break
            
            comp = network[idx]
            new_network = network[:idx] + network[idx+1:]
            
            if is_valid_network(new_network):
                network = new_network
                improved = True
                break # Restart with the new network
        
        if not improved:
            # No more single removals possible.
            # We can try to replace a comparator with another?
            # For now, we stop.
            break
            
    return network
