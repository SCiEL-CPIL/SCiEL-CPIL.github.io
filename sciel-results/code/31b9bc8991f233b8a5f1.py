import time
import random
from collections import deque

N = 12
DEADLINE_MARGIN = 10.0  # seconds

def compute_threshold(trans_a, trans_b):
    """
    Computes the exact shortest reset word length using BFS on the subset automaton
    using bitmasks for efficiency.
    Returns the length, or -1 if not synchronizing.
    """
    # Start with the full set of states (all bits set)
    start_state = (1 << N) - 1
    visited = {start_state: 0}
    queue = deque([(start_state, 0)])
    
    # Precompute transitions for single states? 
    # Or just compute on the fly.
    # For each state mask, we need to compute next_a and next_b.
    # next_a = OR of (1 << trans_a[s]) for s in mask
    
    while queue:
        state, dist = queue.popleft()
        if state & (state - 1) == 0: # Power of 2 (single state)
            return dist
        
        # Apply letter 0
        next_state_0 = 0
        # Iterate through set bits
        temp = state
        while temp:
            lsb = temp & -temp
            s = lsb.bit_length() - 1
            next_state_0 |= (1 << trans_a[s])
            temp ^= lsb
            
        if next_state_0 not in visited:
            visited[next_state_0] = dist + 1
            queue.append((next_state_0, dist + 1))
            
        # Apply letter 1
        next_state_1 = 0
        temp = state
        while temp:
            lsb = temp & -temp
            s = lsb.bit_length() - 1
            next_state_1 |= (1 << trans_b[s])
            temp ^= lsb
            
        if next_state_1 not in visited:
            visited[next_state_1] = dist + 1
            queue.append((next_state_1, dist + 1))
            
    return -1

def construct():
    """
    Returns the transition tables for an optimal 12-state synchronizing automaton.
    Uses bitmask-based BFS and Iterated Local Search.
    """
    start_time = time.time()
    
    # Initial construction: Černý automaton
    trans_a = [(i + 1) % N for i in range(N)]
    trans_b = [0 if i == 0 else i - 1 for i in range(N)]
    
    best_threshold = compute_threshold(trans_a, trans_b)
    best_a = trans_a[:]
    best_b = trans_b[:]
    
    # If we already hit the theoretical max, return
    if best_threshold >= 121:
        return [best_a, best_b]
        
    random.seed(42)
    
    # Iterated Local Search
    # We will run multiple rounds of local search from different starting points.
    # Starting points: The current best, and random perturbations of it.
    
    current_a = best_a[:]
    current_b = best_b[:]
    current_threshold = best_threshold
    
    max_rounds = 50
    
    for round_num in range(max_rounds):
        if time.time() - start_time > DEADLINE_MARGIN:
            break
            
        # Local Search Phase
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > DEADLINE_MARGIN:
                break
                
            # Try all single-step mutations
            for letter in range(2):
                for state in range(N):
                    old_dest = (current_a if letter == 0 else current_b)[state]
                    
                    for new_dest in range(N):
                        if new_dest == old_dest:
                            continue
                        
                        # Apply change
                        if letter == 0:
                            current_a[state] = new_dest
                        else:
                            current_b[state] = new_dest
                        
                        new_threshold = compute_threshold(current_a, current_b)
                        
                        if new_threshold > current_threshold:
                            current_threshold = new_threshold
                            improved = True
                            # If we found a better global best, update it
                            if current_threshold > best_threshold:
                                best_threshold = current_threshold
                                best_a = current_a[:]
                                best_b = current_b[:]
                                if best_threshold >= 121:
                                    return [best_a, best_b]
                        else:
                            # Revert
                            if letter == 0:
                                current_a[state] = old_dest
                            else:
                                current_b[state] = old_dest
            
            # Optional: If no improvement in a full pass, we could stop local search
            # But we might want to try a different order or random restart within local search?
            # Standard hill climbing stops when no single move improves.
        
        # Perturbation Phase (for next round)
        # Randomly change a few transitions to escape local optimum
        if time.time() - start_time > DEADLINE_MARGIN:
            break
            
        num_changes = random.randint(2, 5)
        for _ in range(num_changes):
            letter = random.randint(0, 1)
            state = random.randint(0, N - 1)
            new_dest = random.randint(0, N - 1)
            
            if letter == 0:
                current_a[state] = new_dest
            else:
                current_b[state] = new_dest
        
        # Update current threshold after perturbation
        current_threshold = compute_threshold(current_a, current_b)
        
        # If perturbation improved global best, update it
        if current_threshold > best_threshold:
            best_threshold = current_threshold
            best_a = current_a[:]
            best_b = current_b[:]
            if best_threshold >= 121:
                return [best_a, best_b]

    return [best_a, best_b]
