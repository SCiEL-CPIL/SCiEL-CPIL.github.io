import time
import random
import numpy as np
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1

def _compute_threshold_fast(trans0, trans1):
    """
    Computes the exact shortest reset word length using BFS on the subset automaton.
    Uses bitmasks for fast subset transitions.
    Returns -1 if not synchronizing.
    """
    # Precompute transition masks for each state and letter
    # mask_letter[state] = bitmask of destination states
    # Actually, for a deterministic automaton, each state maps to exactly one state per letter.
    # So we can precompute:
    # t0[i] = 1 << dest0[i]
    # t1[i] = 1 << dest1[i]
    # To apply letter L to a subset S (bitmask):
    # new_S = OR of t_L[i] for all i in S
    
    # We can optimize this further by precomputing the transition of every possible subset?
    # 2^12 = 4096 subsets. 4096 * 2 letters = 8192 transitions.
    # This is feasible and makes BFS extremely fast (just array lookups).
    
    # Precompute transition table for all subsets
    # trans_table[subset_mask][letter] = next_subset_mask
    # subset_mask ranges from 0 to 4095
    
    t0 = [1 << trans0[i] for i in range(N)]
    t1 = [1 << trans1[i] for i in range(N)]
    
    # Helper to compute next subset
    def get_next(subset, letter):
        if letter == 0:
            res = 0
            # Unroll or loop? Loop is fine for 12 bits.
            # We can use a lookup table for 8-bit chunks or just iterate bits.
            # Iterating bits:
            s = subset
            while s:
                lsb = s & -s
                idx = (lsb.bit_length() - 1)
                res |= t0[idx]
                s ^= lsb
            return res
        else:
            res = 0
            s = subset
            while s:
                lsb = s & -s
                idx = (lsb.bit_length() - 1)
                res |= t1[idx]
                s ^= lsb
            return res

    # Precompute all transitions to make BFS O(1) per step
    # next_subset[4096][2]
    next_subsets = [[0, 0] for _ in range(1 << N)]
    for s in range(1 << N):
        next_subsets[s][0] = get_next(s, 0)
        next_subsets[s][1] = get_next(s, 1)
        
    # BFS
    # visited[subset_mask] = distance
    visited = [-1] * (1 << N)
    queue = deque()
    
    start = FULL_MASK
    visited[start] = 0
    queue.append(start)
    
    while queue:
        current = queue.popleft()
        dist = visited[current]
        
        # Check if singleton
        if current & (current - 1) == 0 and current != 0:
            return dist
            
        for letter in range(2):
            nxt = next_subsets[current][letter]
            if visited[nxt] == -1:
                visited[nxt] = dist + 1
                queue.append(nxt)
                
    return -1

def _generate_cerny_seed():
    """
    Generates a standard Cerny automaton which has threshold N-1.
    a: cycle (0->1->...->N-1->0)
    b: 0->0, i->i-1 for i>0
    """
    trans0 = [(i + 1) % N for i in range(N)]
    trans1 = [0] * N
    for i in range(1, N):
        trans1[i] = i - 1
    return trans0, trans1

def construct():
    start_time = time.time()
    deadline = 88.0  # 2s margin for safety
    
    # 1. Initialize with known good seeds
    # We will maintain a population of candidates, or just do hill climbing on the best found so far.
    # Hill climbing is usually effective here.
    
    current_0, current_1 = _generate_cerny_seed()
    current_score = _compute_threshold_fast(current_0, current_1)
    
    best_0, best_1 = current_0[:], current_1[:]
    best_score = current_score
    
    rng = random.Random(42)
    
    # Local Search: Hill Climbing with Random Restarts
    # We try to mutate one state's transition for one letter.
    # If it improves, we keep it. If we stagnate, we restart from best or randomize.
    
    stagnation_counter = 0
    max_stagnation = 1000  # After 1000 failures, try a larger jump or restart
    
    while time.time() < deadline:
        # Create a copy for mutation
        mut_0 = current_0[:]
        mut_1 = current_1[:]
        
        # Choose which letter and which state to mutate
        letter = rng.randint(0, 1)
        state = rng.randint(0, N - 1)
        
        if letter == 0:
            old_val = mut_0[state]
            new_val = rng.randint(0, N - 1)
            if new_val == old_val:
                continue
            mut_0[state] = new_val
            score = _compute_threshold_fast(mut_0, mut_1)
        else:
            old_val = mut_1[state]
            new_val = rng.randint(0, N - 1)
            if new_val == old_val:
                continue
            mut_1[state] = new_val
            score = _compute_threshold_fast(mut_0, mut_1)
            
        if score > current_score:
            # Accept
            current_0, current_1 = mut_0, mut_1
            current_score = score
            stagnation_counter = 0
            
            if score > best_score:
                best_score = score
                best_0, best_1 = current_0[:], current_1[:]
                
        else:
            stagnation_counter += 1
            
            # If stagnated too long, perform a "kick" (multiple mutations) or restart
            if stagnation_counter > max_stagnation:
                # Option A: Restart from best
                # Option B: Apply 3-5 random mutations to current
                # Let's try applying a few mutations to escape local optima
                kicks = rng.randint(2, 5)
                temp_0, temp_1 = current_0[:], current_1[:]
                for _ in range(kicks):
                    l = rng.randint(0, 1)
                    s = rng.randint(0, N - 1)
                    if l == 0:
                        temp_0[s] = rng.randint(0, N - 1)
                    else:
                        temp_1[s] = rng.randint(0, N - 1)
                        
                kick_score = _compute_threshold_fast(temp_0, temp_1)
                
                # If the kick is not terrible, accept it to escape the local optimum
                # Threshold: accept if it's at least current_score - 5 or something?
                # Actually, for reset words, we want to maximize. 
                # If kick_score >= 0 (synchronizing), we might accept it if it's not drastically worse.
                # But if we accept a non-synchronizing one (-1), we get stuck.
                # So only accept kicks that are synchronizing and not too bad?
                # Or just restart from best if stuck?
                # Let's restart from best to be safe, or try a random restart from a different seed.
                
                # Let's try restarting from the best found so far
                current_0, current_1 = best_0[:], best_1[:]
                current_score = best_score
                stagnation_counter = 0
                
                # To avoid infinite loops on the same local optima, let's occasionally
                # generate a fresh random candidate biased towards Cerny structures
                if rng.random() < 0.1:
                    # Generate a random permutation for 0 and a "shift-like" for 1
                    t0 = list(range(N))
                    rng.shuffle(t0)
                    shift = rng.randint(1, N - 1)
                    t1 = [(i + shift) % N for i in range(N)]
                    # Break the cycle in t1 to allow sync
                    break_idx = rng.randint(0, N - 1)
                    t1[break_idx] = rng.randint(0, N - 1)
                    
                    rs_score = _compute_threshold_fast(t0, t1)
                    if rs_score > 0:
                         current_0, current_1 = t0, t1
                         current_score = rs_score
                         if current_score > best_score:
                             best_score = current_score
                             best_0, best_1 = current_0[:], current_1[:]

    return [best_0, best_1]

# EVOLVE-BLOCK-END
