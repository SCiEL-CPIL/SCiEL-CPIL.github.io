import time
import random
from collections import deque

N = 12

def compute_threshold_bitmask(sigma0, sigma1):
    """
    Computes the shortest reset word length using BFS on the subset automaton.
    Uses bitmasks for efficiency.
    """
    # Start with the full set of states: 111...1 (N bits)
    full_mask = (1 << N) - 1
    
    # Precompute transitions for all possible subsets? No, too many (2^12 = 4096).
    # We can precompute the image of each state for each letter.
    # But we need the image of *subsets*.
    # Image of subset S under letter L is { sigma_L(x) for x in S }.
    
    # To speed up, we can precompute the transition of every possible bitmask.
    # There are 2^12 = 4096 masks.
    # For each mask, and each letter, compute the next mask.
    # This is feasible.
    
    trans = [[0] * 2 for _ in range(1 << N)]
    
    # Precompute single state transitions
    # Actually, we can just compute on the fly or precompute a table.
    # Let's precompute `next_mask[mask][letter]`
    
    # Optimization: 
    # next_mask[mask][0] = OR of (1 << sigma0[i]) for all i in mask
    # next_mask[mask][1] = OR of (1 << sigma1[i]) for all i in mask
    
    # We can compute this efficiently.
    for mask in range(1 << N):
        m0 = 0
        m1 = 0
        for i in range(N):
            if (mask >> i) & 1:
                m0 |= (1 << sigma0[i])
                m1 |= (1 << sigma1[i])
        trans[mask][0] = m0
        trans[mask][1] = m1
        
    # BFS from full_mask
    # Goal: reach any singleton mask (1, 2, 4, ..., 1 << (N-1))
    # Singletons are masks with exactly one bit set.
    
    visited = [False] * (1 << N)
    visited[full_mask] = True
    
    queue = deque()
    queue.append((full_mask, 0))
    
    while queue:
        mask, dist = queue.popleft()
        
        # Check if singleton
        # A mask is a singleton if mask & (mask - 1) == 0
        if mask != 0 and (mask & (mask - 1)) == 0:
            return dist
            
        for letter in [0, 1]:
            next_mask = trans[mask][letter]
            if not visited[next_mask]:
                visited[next_mask] = True
                queue.append((next_mask, dist + 1))
                
    return -1 # Not synchronizing

def generate_pawlik_base():
    """Generates the standard Pawlik automaton."""
    sigma0 = [(i + 1) % N for i in range(N)]
    sigma1 = [0] * N
    for i in range(1, N):
        sigma1[i] = i - 1
    return [sigma0, sigma1]

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # 50s safety margin
    
    # Start with the known optimal Pawlik automaton
    best_sigma0, best_sigma1 = generate_pawlik_base()
    best_score = compute_threshold_bitmask(best_sigma0, best_sigma1)
    
    # Perform a few local search steps to see if we can improve (unlikely, but structural change)
    # We will try random mutations of sigma1, keeping sigma0 as the cycle.
    # This is a "Directed Descent" on the space of automata with a cyclic letter.
    
    num_iterations = 0
    while time.time() < deadline:
        # Create a copy
        s0 = best_sigma0[:]
        s1 = best_sigma1[:]
        
        # Randomly mutate a few positions in s1
        # We keep s0 as the cycle to maintain the "Pawlik-like" structure
        # which is known to be good.
        for _ in range(3):
            idx = random.randint(0, N-1)
            new_val = random.randint(0, N-1)
            s1[idx] = new_val
            
        # Check if it synchronizes and compute score
        # To save time, we can do a quick check or just compute.
        # compute_threshold_bitmask is fast enough for a few hundred iterations in 50s?
        # 2^12 states, BFS is O(2^N * N). ~4096 * 12 ~ 50k ops.
        # 50s / 50k = 1000 iterations.
        
        score = compute_threshold_bitmask(s0, s1)
        
        if score > best_score:
            best_score = score
            best_sigma0 = s0
            best_sigma1 = s1
            
        num_iterations += 1
        if num_iterations > 500: # Hard cap to prevent infinite loop if time check fails
            break

    return [best_sigma0, best_sigma1]
