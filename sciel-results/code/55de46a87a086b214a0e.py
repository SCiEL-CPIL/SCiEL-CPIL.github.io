import numpy as np
import random
import time

N = 17
NUM_MASKS = 1 << N

def generate_initial_network():
    """Generate an Odd-Even Transposition network."""
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
    return network

def verify_network(network):
    """
    Verify if the network sorts all 2^N boolean inputs.
    Uses vectorized NumPy simulation.
    """
    if not network:
        return False # Empty network doesn't sort N>1
        
    # Create all masks
    masks = np.arange(NUM_MASKS, dtype=np.uint32)
    # Unpack to bits: shape (NUM_MASKS, N)
    # masks[:, None] >> np.arange(N)
    bits = (masks[:, None] >> np.arange(N, dtype=np.uint8)) & 1
    
    for (i, j) in network:
        bi = bits[:, i]
        bj = bits[:, j]
        # If bi=1 and bj=0, swap
        swap_mask = (bi == 1) & (bj == 0)
        if np.any(swap_mask):
            bits[swap_mask, i] = 0
            bits[swap_mask, j] = 1
            
    # Check if sorted: 0...01...1
    # If any row has a 1 followed by a 0, it's not sorted.
    # Check diffs: bits[:, 1:] - bits[:, :-1] should not be -1.
    diffs = bits[:, 1:].astype(np.int8) - bits[:, :-1].astype(np.int8)
    if np.any(diffs == -1):
        return False
    return True

def construct():
    # Start with a known valid network
    network = generate_initial_network()
    
    # Simulated Annealing parameters
    start_time = time.time()
    max_time = 55.0 # Leave 5s buffer
    current_temp = 1.0
    cooling_rate = 0.99
    min_temp = 0.01
    
    current_cost = len(network)
    best_network = list(network)
    best_cost = current_cost
    
    # Pre-compute some random moves? No, generate on the fly.
    
    while time.time() - start_time < max_time:
        if current_temp < min_temp:
            break
            
        # Generate a move
        move_type = random.random()
        new_network = list(network)
        
        if move_type < 0.3:
            # Remove a comparator
            if len(new_network) > 1:
                idx = random.randint(0, len(new_network) - 1)
                new_network.pop(idx)
        elif move_type < 0.6:
            # Move a comparator to a different position
            if len(new_network) > 2:
                idx1 = random.randint(0, len(new_network) - 1)
                idx2 = random.randint(0, len(new_network) - 1)
                if idx1 != idx2:
                    comp = new_network.pop(idx1)
                    new_network.insert(idx2, comp)
        elif move_type < 0.8:
            # Swap two comparators
            if len(new_network) > 1:
                idx1 = random.randint(0, len(new_network) - 1)
                idx2 = random.randint(0, len(new_network) - 1)
                if idx1 != idx2:
                    new_network[idx1], new_network[idx2] = new_network[idx2], new_network[idx1]
        else:
            # Insert a random comparator (to explore structure, might increase cost temporarily)
            # This helps escape local minima where removal is stuck but a different structure is better
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            idx = random.randint(0, len(new_network))
            new_network.insert(idx, (i, j))
            
        # Evaluate new network
        # Optimization: If we added a comparator, cost increases. 
        # In SA, we usually accept worse states with some probability.
        # But here "cost" is size. We want to minimize size.
        # However, validity is a hard constraint.
        # So we only consider valid networks.
        # If the new network is invalid, we reject it (or penalize it).
        # For simplicity and speed, we only accept valid networks.
        # This makes it a "Valid Local Search" rather than pure SA with penalties.
        
        is_valid = verify_network(new_network)
        
        if is_valid:
            new_cost = len(new_network)
            delta_cost = new_cost - current_cost
            
            # Acceptance probability
            if delta_cost < 0:
                # Better (smaller)
                accept = True
            else:
                # Worse (larger or same)
                # Only accept if we want to explore, but since we minimize size,
                # accepting larger networks is rarely useful unless it leads to a smaller one later.
                # Given the time limit, let's be conservative: only accept if significantly better or same.
                # Actually, pure hill climbing (only accept better) is safer for time.
                # But SA allows escaping. Let's allow accepting worse with probability exp(-delta/T).
                if delta_cost == 0:
                    accept = True
                else:
                    accept_prob = np.exp(-delta_cost / current_temp)
                    accept = random.random() < accept_prob
                    
            if accept:
                network = new_network
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_network = list(network)
        else:
            # Reject invalid
            pass
            
        # Cool down
        current_temp *= cooling_rate
        
    return best_network
