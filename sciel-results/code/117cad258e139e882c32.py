import time
import random
import sys
from collections import deque

N = 12
FULL_STATE = (1 << N) - 1
DEADLINE = 85.0  # safety margin under 90s kill

# Precompute singleton masks
SINGLETONS = [1 << i for i in range(N)]

def calculate_threshold(a_trans, b_trans):
    """
    Returns the shortest reset word length for the automaton defined by a_trans and b_trans,
    or -1 if not synchronizing.
    Uses BFS on the subset automaton with bitmask representation.
    """
    a_bits = [1 << a_trans[i] for i in range(N)]
    b_bits = [1 << b_trans[i] for i in range(N)]
    
    # Precompute transition tables for all possible subsets?
    # 2^12 = 4096 subsets.
    # Trans[a][mask] and Trans[b][mask].
    # This takes 4096 * 12 * 2 operations ~ 100k ops. Very fast.
    # Let's do it to speed up BFS significantly.
    
    trans_a = [0] * (1 << N)
    trans_b = [0] * (1 << N)
    
    for mask in range(1 << N):
        if mask == 0:
            continue
        # Compute transition for a
        res_a = 0
        res_b = 0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            res_a |= a_bits[idx]
            res_b |= b_bits[idx]
            m ^= lsb
        trans_a[mask] = res_a
        trans_b[mask] = res_b
        
    # BFS
    visited = [False] * (1 << N)
    visited[FULL_STATE] = True
    queue = deque()
    queue.append((FULL_STATE, 0))
    
    while queue:
        state, dist = queue.popleft()
        if state.bit_count() == 1:
            return dist
            
        next_a = trans_a[state]
        if not visited[next_a]:
            visited[next_a] = True
            queue.append((next_a, dist + 1))
            
        next_b = trans_b[state]
        if not visited[next_b]:
            visited[next_b] = True
            queue.append((next_b, dist + 1))
            
    return -1

def construct():
    start_time = time.time()
    
    # Theoretical max for N=12 is (N-1)^2 = 121.
    # We know Pinsker-like automata can achieve high values.
    # Let's try to find a specific automaton.
    # Known best for N=12 is 121?
    # Actually, for N=12, the max known is likely 121 if the conjecture holds for small N.
    # Let's try to construct one or search for one.
    
    # Strategy:
    # 1. Start with a Pinsker automaton structure.
    #    a: 0->1, 1->2, ..., N-2->N-1, N-1->0 (Cycle)
    #    b: 0->0, 1->0, ..., N-1->0 (Reset to 0? No, that's too fast)
    #    Standard Pinsker:
    #    a: i -> i+1 mod N
    #    b: 0 -> 0, i -> i-1 for i>0? No.
    #    Let's use the known construction for max threshold.
    #    a: (0 1 2 ... N-1)
    #    b: 0->0, 1->0, 2->1, ..., N-1->N-2?
    #    This gives threshold N(N-1)/2? No.
    
    # Let's just do a smart search.
    # Since N=12 is small, we can afford a decent search.
    # We will use Simulated Annealing or Hill Climbing with restarts.
    
    def get_score(a, b):
        return calculate_threshold(a, b)
    
    best_score = -1
    best_a = None
    best_b = None
    
    # Initial guess: Pinsker-like
    a0 = [(i + 1) % N for i in range(N)]
    # b that creates long paths
    # Try b[i] = (i + N - 1) % N (shift back)
    b0 = [(i - 1) % N for i in range(N)]
    score0 = get_score(a0, b0)
    if score0 > best_score:
        best_score = score0
        best_a = a0[:]
        best_b = b0[:]
        
    # Another guess: 
    # a: cycle
    # b: 0->0, 1->0, 2->0... (too fast)
    # b: 0->0, 1->0, 2->1, 3->2... (this is the "Pinsker" one?)
    b1 = [0] + [(i - 1) for i in range(1, N)]
    score1 = get_score(a0, b1)
    if score1 > best_score:
        best_score = score1
        best_a = a0[:]
        best_b = b1[:]
        
    # Local Search
    current_a = best_a[:]
    current_b = best_b[:]
    current_score = best_score
    
    # We will mutate both a and b occasionally, but mostly b since a is often a cycle.
    # Actually, let's allow mutation of both.
    
    no_improve = 0
    max_no_improve = 2000
    
    while time.time() - start_time < DEADLINE:
        if no_improve > max_no_improve:
            # Restart
            # Try a random restart or a structured restart
            # Let's try a new random a and b
            if random.random() < 0.5:
                current_a = [random.randint(0, N-1) for _ in range(N)]
            else:
                current_a = [(i + 1) % N for i in range(N)] # Keep cycle
            current_b = [random.randint(0, N-1) for _ in range(N)]
            current_score = get_score(current_a, current_b)
            if current_score > best_score:
                best_score = current_score
                best_a = current_a[:]
                best_b = current_b[:]
            no_improve = 0
            continue
            
        # Mutate
        # Pick a random position and letter
        if random.random() < 0.5:
            # Mutate a
            i = random.randint(0, N-1)
            old_val = current_a[i]
            new_val = random.randint(0, N-1)
            if new_val == old_val:
                continue
            current_a[i] = new_val
            new_score = get_score(current_a, current_b)
            if new_score > current_score:
                current_score = new_score
                no_improve = 0
                if new_score > best_score:
                    best_score = new_score
                    best_a = current_a[:]
                    best_b = current_b[:]
                    if best_score == (N-1)**2:
                        break
            else:
                current_a[i] = old_val
                no_improve += 1
        else:
            # Mutate b
            i = random.randint(0, N-1)
            old_val = current_b[i]
            new_val = random.randint(0, N-1)
            if new_val == old_val:
                continue
            current_b[i] = new_val
            new_score = get_score(current_a, current_b)
            if new_score > current_score:
                current_score = new_score
                no_improve = 0
                if new_score > best_score:
                    best_score = new_score
                    best_a = current_a[:]
                    best_b = current_b[:]
                    if best_score == (N-1)**2:
                        break
            else:
                current_b[i] = old_val
                no_improve += 1
                
    if best_a is None:
        # Fallback
        best_a = [(i + 1) % N for i in range(N)]
        best_b = [0] * N
        
    return [best_a, best_b]

