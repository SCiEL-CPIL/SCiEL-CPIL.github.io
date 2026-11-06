import time
import random
import numpy as np

N = 17
MAX_TIME = 55.0  # Safety margin for 60s limit

# Precompute valid sorted states for quick lookup
_VALID_MASK = np.zeros(1 << N, dtype=bool)
for k in range(N + 1):
    if k == 0:
        val = 0
    else:
        val = (1 << N) - (1 << (N - k))
    _VALID_MASK[val] = True

def seed_network():
    """Odd-even transposition network (136 comparators)."""
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net

def verify_network(net):
    """
    Vectorized verification using NumPy.
    Checks if the network sorts all 2^17 boolean inputs.
    """
    if not net:
        return N <= 1

    states = np.arange(1 << N, dtype=np.uint32)
    
    for i, j in net:
        bi = (states >> i) & 1
        bj = (states >> j) & 1
        mask = (bi == 1) & (bj == 0)
        if np.any(mask):
            states[mask] ^= (1 << i) | (1 << j)
            
    return np.all(_VALID_MASK[states])

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    rng = random.Random(42)
    
    # Start with seed
    best_net = seed_network()
    best_count = len(best_net)
    
    # Simulated annealing
    current = list(best_net)
    current_count = len(current)
    
    # Temperature schedule
    initial_temp = 10.0
    final_temp = 0.01
    cooling_rate = 0.98
    
    temp = initial_temp
    
    # Precompute all possible comparators for quick access
    all_comparators = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    def minimize(net, time_limit):
        """Greedy removal of redundant comparators."""
        current = list(net)
        progress = True
        while progress and time.time() < time_limit:
            progress = False
            indices = list(range(len(current)))
            rng.shuffle(indices)
            for idx in indices:
                if time.time() > time_limit:
                    break
                new_net = current[:idx] + current[idx+1:]
                if verify_network(new_net):
                    current = new_net
                    progress = True
                    break
        return current
    
    # Initial minimization
    current = minimize(current, deadline)
    if len(current) < best_count:
        best_net = current
        best_count = len(current)
    
    # Simulated annealing loop
    while time.time() < deadline and temp > final_temp:
        # Choose a move
        move_type = rng.random()
        
        if move_type < 0.5:
            # Try to remove a comparator (most likely to improve)
            if len(current) <= 1:
                continue
            idx = rng.randint(0, len(current) - 1)
            new_net = current[:idx] + current[idx+1:]
            new_count = len(new_net)
            
        elif move_type < 0.8:
            # Replace a comparator
            idx = rng.randint(0, len(current) - 1)
            new_comp = rng.choice(all_comparators)
            new_net = list(current)
            new_net[idx] = new_comp
            new_count = len(new_net)
            
        else:
            # Swap two comparators
            if len(current) < 2:
                continue
            i, j = rng.sample(range(len(current)), 2)
            new_net = list(current)
            new_net[i], new_net[j] = new_net[j], new_net[i]
            new_count = len(new_net)
        
        # Verify
        if not verify_network(new_net):
            temp *= cooling_rate
            continue
        
        # Accept or reject based on temperature
        delta = new_count - current_count
        if delta < 0:
            # Always accept improvements
            current = new_net
            current_count = new_count
            if current_count < best_count:
                best_net = current
                best_count = current_count
                # Try to minimize after improvement
                if time.time() < deadline - 0.1:
                    minimized = minimize(current, deadline)
                    if len(minimized) < best_count:
                        current = minimized
                        current_count = len(current)
                        best_net = current
                        best_count = current_count
        else:
            # Accept worse solutions with probability exp(-delta/temp)
            if rng.random() < np.exp(-delta / temp):
                current = new_net
                current_count = new_count
        
        temp *= cooling_rate
    
    # Final minimization
    if time.time() < deadline - 0.1:
        minimized = minimize(current, deadline)
        if len(minimized) < best_count:
            best_net = minimized
            best_count = len(minimized)
    
    # Safety check
    if not verify_network(best_net):
        best_net = seed_network()
    
    return best_net

if __name__ == "__main__":
    net = construct()
    print(f"Result size: {len(net)}")
    print(f"Valid: {verify_network(net)}")
