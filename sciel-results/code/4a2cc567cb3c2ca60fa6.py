"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import itertools
import numpy as np
import time

N = 16

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a size-efficient sorting network for 16 wires.
    Uses vectorized NumPy simulation to verify networks rapidly, enabling
    extensive local search and greedy pruning to minimize comparator count.
    """
    
    # Pre-compute sorted values and lookup table
    # Sorted binary output: 0...01...1
    # Values: 0, 1, 3, 7, ..., 2^16 - 1
    sorted_values = np.array([2**k - 1 for k in range(N + 1)], dtype=np.uint16)
    lookup = np.zeros(1 << N, dtype=np.uint8)
    lookup[sorted_values] = 1
    
    # All possible inputs
    inputs = np.arange(1 << N, dtype=np.uint16)
    
    def apply_comparator(x, i, j):
        """Apply a single comparator to an array of states."""
        mask_i = np.uint16(1 << i)
        mask_j = np.uint16(1 << j)
        # Condition: bit i is 1 and bit j is 0
        # We want min at i, max at j.
        # If x has 1 at i and 0 at j, swap.
        needs_swap = (x & mask_i) != 0
        # AND with NOT (x & mask_j)
        # (x & mask_j) == 0
        needs_swap &= (x & mask_j) == 0
        swap_mask = mask_i | mask_j
        # Use np.where for vectorized update
        return np.where(needs_swap, x ^ swap_mask, x)

    def verify(comparators):
        """
        Verify if the network sorts all inputs.
        Returns True if valid, False otherwise.
        """
        x = inputs.copy()
        for i, j in comparators:
            x = apply_comparator(x, i, j)
        # Check if all outputs are in the sorted set
        # lookup[x] gives 1 for sorted, 0 for unsorted
        # If any are 0, it's invalid
        return np.all(lookup[x] == 1)

    # Initialize with Odd-Even Transposition Network (120 comparators)
    # This is a known correct network.
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
    
    current_size = len(network)
    
    # Phase 1: Greedy Removal
    # We repeatedly try to remove comparators.
    # To avoid getting stuck, we shuffle the order of attempts.
    # We use vectorized verification which is fast.
    
    max_time = 50  # Leave 10s buffer
    start_time = time.time()
    
    # Greedy Pruning Loop
    while time.time() - start_time < max_time:
        removed_any = False
        indices = list(range(len(network)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() - start_time > max_time:
                break
            
            if idx >= len(network):
                continue
                
            # Try removing network[idx]
            candidate = network[:idx] + network[idx+1:]
            
            if verify(candidate):
                network = candidate
                current_size = len(network)
                removed_any = True
        
        if not removed_any:
            break
            
    # Phase 2: Local Search / Simulated Annealing-like exploration
    # Try to find a better configuration by swapping, moving, or changing comparators.
    # Goal: Find a valid network with fewer comparators.
    # Since we are stuck in a local minimum for removal, we might need to "reshuffle"
    # the network to allow different removals.
    
    best_network = network
    best_size = current_size
    
    # We will perform random moves.
    # Move types:
    # 1. Swap two comparators.
    # 2. Move a comparator to a new position.
    # 3. Change a comparator's indices.
    # 4. Remove a comparator (if valid).
    # 5. Insert a new comparator (if it helps? No, we want to minimize. 
    #    But sometimes inserting a redundant comparator allows removing two others?
    #    This is "exchanging". Let's stick to simple moves first.)
    
    # Actually, a more powerful move is "Exchange":
    # Replace comparator A with comparator B.
    
    # Let's do a simple random walk on valid networks.
    # Accept a move if it is valid and size is <= current size.
    # If size is smaller, we found a better network.
    
    for _ in range(10000): # Max iterations
        if time.time() - start_time > max_time:
            break
        
        if not network:
            break
            
        move_type = random.random()
        
        candidate = network[:]
        
        if move_type < 0.4:
            # Remove a random comparator
            idx = random.randint(0, len(candidate) - 1)
            candidate.pop(idx)
        elif move_type < 0.7:
            # Swap two comparators
            if len(candidate) < 2:
                continue
            i1, i2 = random.sample(range(len(candidate)), 2)
            candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
        elif move_type < 0.9:
            # Change a comparator's indices
            idx = random.randint(0, len(candidate) - 1)
            old_i, old_j = candidate[idx]
            # Pick new i, j
            new_i = random.randint(0, N - 2)
            new_j = random.randint(new_i + 1, N - 1)
            candidate[idx] = (new_i, new_j)
        else:
            # Insert a random comparator at a random position
            # This increases size, but might enable future removals?
            # Risky for "minimize" goal, but good for escaping local minima.
            # Let's only allow this if we are stuck and haven't improved in a while.
            # For now, let's skip insertion to keep size down.
            continue
            
        if len(candidate) < best_size:
            # If it's smaller, we must verify it strictly.
            if verify(candidate):
                network = candidate
                best_network = network[:]
                best_size = len(network)
        elif len(candidate) == best_size:
            # Same size, verify if valid.
            # If valid, accept to explore.
            if verify(candidate):
                network = candidate
        # If len(candidate) > best_size, reject (we want to minimize)
                
    return best_network
# EVOLVE-BLOCK-END
