import time
from collections import deque
import random

N = 12

def calculate_threshold(a_trans, b_trans):
    """
    Computes the exact shortest reset word length (threshold) for a binary automaton.
    Uses BFS on the subset automaton with bitmasks for efficiency.
    """
    start = (1 << N) - 1
    
    # If start is a single state, threshold is 0
    if start & (start - 1) == 0:
        return 0
    
    visited = {start: 0}
    q = deque([start])
    
    # Precompute transition masks for each letter and each state
    # a_mask[s] = bitmask of states reachable from state s via 'a'
    # Since each state goes to exactly one state, this is just 1 << a_trans[s]
    a_masks = [1 << a_trans[i] for i in range(N)]
    b_masks = [1 << b_trans[i] for i in range(N)]
    
    while q:
        curr = q.popleft()
        curr_dist = visited[curr]
        
        # Check if we have a single state (reset achieved)
        if curr & (curr - 1) == 0:
            return curr_dist
        
        # Apply letter 'a': for each state in curr, go to a_trans[state]
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
    Uses the known Černý automaton construction which achieves (N-1)^2 = 121.
    """
    # The Černý automaton achieves the maximum possible threshold of (N-1)^2.
    # For N=12, this is 121.
    # Construction:
    # a[i] = (i + 1) % N  (cycle)
    # b[0] = 0
    # b[i] = i - 1 for i > 0  (decrement with 0 as fixed point)
    
    a = [(i + 1) % N for i in range(N)]
    b = [0] * N
    for i in range(1, N):
        b[i] = i - 1
        
    # Verify the threshold to ensure correctness
    score = calculate_threshold(a, b)
    
    if score == (N - 1) ** 2:
        return [a, b]
    else:
        # Fallback: try random search
        start_time = time.time()
        deadline = start_time + 80.0
        best_score = score
        best_a = a
        best_b = b
        
        for _ in range(10000):
            if time.time() > deadline:
                break
            ra = [random.randint(0, N-1) for _ in range(N)]
            rb = [random.randint(0, N-1) for _ in range(N)]
            s = calculate_threshold(ra, rb)
            if s > best_score:
                best_score = s
                best_a = ra
                best_b = rb
                
        return [best_a, best_b]

# EVOLVE-BLOCK-START
# The function construct() is defined above.
# EVOLVE-BLOCK-END
