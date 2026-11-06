"""Simulated Annealing search for a 12-state binary automaton with long reset threshold."""
import random
import time
import numpy as np
from collections import deque

N = 12
MAX_TIME = 55.0  # Leave 5s margin for overhead
START_TEMP = 10.0
COOLING = 0.9999
MIN_TEMP = 0.1
RANDOM_SEED = 42

def _get_score(transitions):
    """
    Computes the exact shortest reset word length for the automaton.
    Returns -1 if the automaton is not synchronizing.
    Uses BFS on the subset automaton starting from the full set of states.
    """
    t0 = transitions[0]
    t1 = transitions[1]
    
    # Initial subset is all states: 0..N-1
    # Represent subsets as integers (bitmasks) for speed
    start_mask = (1 << N) - 1
    
    # We want to reach any singleton state (mask with exactly one bit set)
    
    visited = {start_mask: 0}
    queue = deque([start_mask])
    
    while queue:
        mask = queue.popleft()
        dist = visited[mask]
        
        # Check if this is a singleton
        # A singleton mask has exactly one bit set.
        # mask & (mask - 1) == 0 and mask != 0
        if mask & (mask - 1) == 0:
            return dist
            
        # Generate neighbors
        # Apply letter 0
        next_mask_0 = 0
        for i in range(N):
            if mask & (1 << i):
                next_mask_0 |= (1 << t0[i])
                
        if next_mask_0 not in visited:
            visited[next_mask_0] = dist + 1
            queue.append(next_mask_0)
            
        # Apply letter 1
        next_mask_1 = 0
        for i in range(N):
            if mask & (1 << i):
                next_mask_1 |= (1 << t1[i])
                
        if next_mask_1 not in visited:
            visited[next_mask_1] = dist + 1
            queue.append(next_mask_1)
            
    # If queue is empty and we didn't return, it's not synchronizing
    return -1

def construct():
    """
    Use simulated annealing to find a 12-state binary automaton with
    a long shortest reset word.
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    start_time = time.time()
    
    # Initialize with a random automaton
    current = [[random.randint(0, N - 1) for _ in range(N)] for _ in range(2)]
    
    current_score = _get_score(current)
    
    # If not synchronizing, mutate until it is
    while current_score < 0:
        letter = random.randint(0, 1)
        state = random.randint(0, N - 1)
        current[letter][state] = random.randint(0, N - 1)
        current_score = _get_score(current)
        
    best = [row[:] for row in current]
    best_score = current_score
    
    temp = START_TEMP
    
    while temp > MIN_TEMP:
        if time.time() - start_time > MAX_TIME:
            break
            
        # Create a neighbor
        neighbor = [row[:] for row in current]
        letter = random.randint(0, 1)
        state = random.randint(0, N - 1)
        old_val = neighbor[letter][state]
        new_val = random.randint(0, N - 1)
        if new_val == old_val:
            temp *= COOLING
            continue
        neighbor[letter][state] = new_val
        
        neighbor_score = _get_score(neighbor)
        
        # Accept or reject
        if neighbor_score >= current_score:
            current = neighbor
            current_score = neighbor_score
            if current_score > best_score:
                best = [row[:] for row in current]
                best_score = current_score
        else:
            delta = current_score - neighbor_score
            # If neighbor is non-synchronizing (negative score), delta is large positive
            # exp(-large/temp) is 0, so reject.
            if delta < 0:
                accept = True
            else:
                # Avoid overflow
                if delta / temp > 100:
                    accept_prob = 0.0
                else:
                    accept_prob = np.exp(-delta / temp)
                accept = (random.random() < accept_prob)
            
            if accept:
                current = neighbor
                current_score = neighbor_score
                if current_score > best_score:
                    best = [row[:] for row in current]
                    best_score = current_score
        
        temp *= COOLING
        
    return best

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
