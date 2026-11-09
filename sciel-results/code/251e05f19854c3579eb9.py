import time
import random
import numpy as np

N = 17
MAX_TIME = 90.0
SAFETY_MARGIN = 5.0

# Pre-compute all boolean masks for vectorized verification
# Shape: (2^N, N)
_ALL_MASKS = np.array([(mask >> k) & 1 for mask in range(1 << N) for k in range(N)], dtype=np.int8).reshape(-1, N)
# Pre-compute sorted check indices
# We want to check if state[i] <= state[i+1] for all i.
# If any violation, it's not sorted.
# Actually, for boolean inputs, sorting means all 0s come before all 1s.
# So for a mask with k ones, the sorted state is [0]*(N-k) + [1]*k.
# We can pre-compute the target sorted states for all masks?
# Or just check if the final state is sorted.
# Checking if sorted: np.any(state[1:] < state[:-1]) is O(N).
# Vectorized: 
# states: (2^N, N)
# diffs = states[:, 1:] - states[:, :-1]
# violations = np.any(diffs < 0, axis=1)
# valid = ~violations

def _verify_network(network):
    """
    Vectorized verification of the sorting network against all 2^N boolean inputs.
    Returns True if the network sorts all inputs.
    """
    if not network:
        return True # Empty network only sorts if N=1 or inputs are already sorted (not generally true)
    
    # Copy initial states
    states = _ALL_MASKS.copy()
    
    for i, j in network:
        # Compare and swap
        # state_i = states[:, i], state_j = states[:, j]
        # min_i = min(state_i, state_j), max_j = max(state_i, state_j)
        # Since values are 0/1:
        # min = state_i * (1 - state_j) + state_j * 0? No.
        # min(0,0)=0, min(0,1)=0, min(1,0)=0, min(1,1)=1.
        # min = state_i & state_j? No, that's AND.
        # min(0,1)=0. 0&1=0. min(1,0)=0. 1&0=0. min(1,1)=1. 1&1=1.
        # So min is AND for 0/1? Wait.
        # If state_i=1, state_j=0 -> min=0. 1&0=0.
        # If state_i=0, state_j=1 -> min=0. 0&1=0.
        # If state_i=1, state_j=1 -> min=1. 1&1=1.
        # If state_i=0, state_j=0 -> min=0. 0&0=0.
        # Yes, for 0/1, min is AND.
        
        # max is OR.
        # min_i = states[:, i] & states[:, j]
        # max_j = states[:, i] | states[:, j]
        
        # However, we must be careful. The comparator writes min to i and max to j.
        # So new_i = min(old_i, old_j), new_j = max(old_i, old_j).
        
        i_vals = states[:, i]
        j_vals = states[:, j]
        
        new_i = i_vals & j_vals
        new_j = i_vals | j_vals
        
        states[:, i] = new_i
        states[:, j] = new_j

    # Check if sorted
    # Sorted means state[k] <= state[k+1] for all k.
    # Violation if state[k+1] < state[k].
    diffs = states[:, 1:] - states[:, :-1]
    violations = np.any(diffs < 0, axis=1)
    
    return not np.any(violations)

def _odd_even_transposition():
    """Standard odd-even transposition sorting network for N wires."""
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
    return network

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME - SAFETY_MARGIN
    
    # Initialize with seed network
    best_network = _odd_even_transposition()
    best_score = len(best_network)
    
    # Verify seed
    if not _verify_network(best_network):
        print("Seed network invalid!")
        return best_network
        
    print(f"Start: {best_score} comparators")
    
    current_network = list(best_network)
    current_score = best_score
    
    # Simulated Annealing Parameters
    # We want to minimize comparator count.
    # Score = len(network). Lower is better.
    # Energy = len(network).
    
    T = 10.0  # Initial temperature
    cooling = 0.995
    min_T = 0.1
    
    # We need to be able to insert comparators to escape local minima.
    # Max comparators allowed by task is 4096, but we should keep it reasonable, say < 200.
    
    num_steps = 0
    last_improvement = start_time
    
    while time.time() < deadline:
        num_steps += 1
        
        # Generate a neighbor
        move_type = random.choice(['remove', 'swap', 'insert', 'move'])
        candidate = list(current_network)
        
        if move_type == 'remove':
            if len(candidate) == 0:
                continue
            idx = random.randint(0, len(candidate) - 1)
            candidate.pop(idx)
            
        elif move_type == 'swap':
            if len(candidate) < 2:
                continue
            idx1 = random.randint(0, len(candidate) - 1)
            idx2 = random.randint(0, len(candidate) - 1)
            if idx1 == idx2:
                continue
            # Only swap if they don't conflict (touch same wires)? 
            # Actually, swapping order of comparators on disjoint wires is always valid.
            # If they touch, it might change the network semantics.
            # For simplicity, allow any swap, verification will check.
            candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
            
        elif move_type == 'insert':
            if len(candidate) > 200: # Cap to avoid explosion
                continue
            # Random wire pair
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            # Insert at random position
            idx = random.randint(0, len(candidate))
            candidate.insert(idx, (i, j))
            
        elif move_type == 'move':
            if len(candidate) < 2:
                continue
            idx = random.randint(0, len(candidate) - 1)
            new_idx = random.randint(0, len(candidate) - 1)
            if idx == new_idx:
                continue
            comp = candidate.pop(idx)
            candidate.insert(new_idx, comp)

        # Verify candidate
        if _verify_network(candidate):
            cand_score = len(candidate)
            
            # Calculate energy difference
            # We want to minimize score.
            # delta_E = new_E - old_E
            delta_E = cand_score - current_score
            
            # Accept if better, or with probability exp(-delta_E / T) if worse
            if delta_E < 0:
                # Better
                current_network = candidate
                current_score = cand_score
                
                if cand_score < best_score:
                    best_network = candidate
                    best_score = cand_score
                    last_improvement = time.time()
                    print(f"Step {num_steps}: Improved to {best_score} (T={T:.2f})")
            else:
                # Worse
                # Probability of accepting worse solution
                # If T is high, we accept worse to explore.
                if random.random() < np.exp(-delta_E / T):
                    current_network = candidate
                    current_score = cand_score
        
        # Cool down
        T = max(min_T, T * cooling)
        
        # Optional: If stuck for too long, reheat? 
        # For now, simple cooling.
        
    # Final check
    if not _verify_network(best_network):
        # Fallback to seed if something went wrong
        return _odd_even_transposition()
        
    print(f"Final: {best_score} comparators")
    return best_network
