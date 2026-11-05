import time
import random
from collections import deque

N = 12

def calculate_threshold(a_trans, b_trans):
    """
    Computes the exact shortest reset word length (threshold) for a binary automaton.
    Uses BFS on the subset automaton with bitmasks for efficiency.
    Returns -1 if not synchronizing.
    """
    start = (1 << N) - 1
    
    # If start is a single state, threshold is 0
    if start & (start - 1) == 0:
        return 0
    
    visited = {start: 0}
    q = deque([start])
    
    # Precompute transition masks for each letter and each state
    # a_masks[i] is the bitmask of the state reached from state i by 'a'
    a_masks = [1 << a_trans[i] for i in range(N)]
    b_masks = [1 << b_trans[i] for i in range(N)]
    
    while q:
        curr = q.popleft()
        curr_dist = visited[curr]
        
        # Check if we have a single state (reset achieved)
        if curr & (start - 1) == 0 and curr != 0:
            # More efficient check: if curr is a power of 2
            if curr & (curr - 1) == 0:
                return curr_dist
        
        # Apply letter 'a'
        next_a = 0
        temp = curr
        while temp:
            lsb = temp & (-temp)
            s = lsb.bit_length() - 1
            next_a |= a_masks[s]
            temp ^= lsb
            
        if next_a not in visited:
            visited[next_a] = curr_dist + 1
            q.append(next_a)
            
        # Apply letter 'b'
        next_b = 0
        temp = curr
        while temp:
            lsb = temp & (-temp)
            s = lsb.bit_length() - 1
            next_b |= b_masks[s]
            temp ^= lsb
            
        if next_b not in visited:
            visited[next_b] = curr_dist + 1
            q.append(next_b)
            
    return -1  # Not synchronizing

def construct():
    """
    Construct a 12-state binary automaton that maximizes the shortest reset word length.
    Uses hill climbing starting from the Černý automaton.
    """
    # Initial Černý automaton
    a = [(i + 1) % N for i in range(N)]
    b = [0] * N
    for i in range(1, N):
        b[i] = i - 1
        
    best_a = a[:]
    best_b = b[:]
    best_score = calculate_threshold(best_a, best_b)
    
    # Deadline: 80 seconds to be safe (limit is 90s)
    start_time = time.time()
    deadline = start_time + 80.0
    
    # Hill Climbing
    # We mutate one transition at a time
    # To speed up, we can precompute the effect of mutations? No, just re-evaluate.
    # Evaluation is fast enough for a few thousand iterations.
    
    # Random seed for reproducibility if needed, but random is fine
    random.seed(42)
    
    while time.time() < deadline:
        # Pick a random letter and state to mutate
        letter_idx = random.randint(0, 1) # 0 for a, 1 for b
        state_idx = random.randint(0, N - 1)
        
        if letter_idx == 0:
            # Mutate a[state_idx]
            old_val = a[state_idx]
            new_val = random.randint(0, N - 1)
            if new_val == old_val:
                continue
            a[state_idx] = new_val
            
            score = calculate_threshold(a, b)
            
            if score > best_score:
                best_score = score
                best_a = a[:]
                best_b = b[:]
            else:
                # Revert
                a[state_idx] = old_val
        else:
            # Mutate b[state_idx]
            old_val = b[state_idx]
            new_val = random.randint(0, N - 1)
            if new_val == old_val:
                continue
            b[state_idx] = new_val
            
            score = calculate_threshold(a, b)
            
            if score > best_score:
                best_score = score
                best_a = a[:]
                best_b = b[:]
            else:
                # Revert
                b[state_idx] = old_val

    return [best_a, best_b]

# EVOLVE-BLOCK-START
# The function construct() is defined above.
# EVOLVE-BLOCK-END
