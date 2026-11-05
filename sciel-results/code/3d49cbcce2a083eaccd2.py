import time
import random
import sys
from collections import deque

N = 12
FULL_STATE = (1 << N) - 1
DEADLINE = 55.0  # Safety margin under 60s kill

def calculate_threshold(a_trans, b_trans):
    """
    Returns the shortest reset word length for the automaton defined by a_trans and b_trans,
    or -1 if not synchronizing. Uses fast bitmask BFS.
    """
    # Precompute bit masks for single state transitions
    a_bits = [1 << a_trans[i] for i in range(N)]
    b_bits = [1 << b_trans[i] for i in range(N)]
    
    # Precompute transition tables for all possible subsets using DP
    trans_a = [0] * (1 << N)
    trans_b = [0] * (1 << N)
    
    # Use DP: trans[mask] = trans[mask without lsb] | bit[lsb_idx]
    for mask in range(1, 1 << N):
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        prev = mask ^ lsb
        trans_a[mask] = trans_a[prev] | a_bits[idx]
        trans_b[mask] = trans_b[prev] | b_bits[idx]
    
    # BFS from FULL_STATE
    visited = bytearray(1 << N)
    visited[FULL_STATE] = 1
    queue = deque()
    queue.append(FULL_STATE)
    dist = 0
    
    while queue:
        level_size = len(queue)
        for _ in range(level_size):
            state = queue.popleft()
            if state.bit_count() == 1:
                return dist
            
            next_a = trans_a[state]
            if not visited[next_a]:
                visited[next_a] = 1
                queue.append(next_a)
                
            next_b = trans_b[state]
            if not visited[next_b]:
                visited[next_b] = 1
                queue.append(next_b)
        
        dist += 1
    
    return -1

def construct():
    start_time = time.time()
    
    def get_score(a, b):
        return calculate_threshold(a, b)

    def random_automaton():
        return [random.randint(0, N-1) for _ in range(N)]

    def pinsker_automaton():
        a = [(i + 1) % N for i in range(N)]
        # b: 0->0, i->i-1 for i>0
        b = [0] + [i-1 for i in range(1, N)]
        return a, b

    def local_search(a, b, max_iters=200):
        """Hill climbing: try single-bit mutations, keep if improves."""
        current_score = get_score(a, b)
        if current_score == -1:
            return a, b, -1
        
        for _ in range(max_iters):
            if time.time() - start_time > DEADLINE:
                break
            
            improved = False
            # Try mutating one position in a
            for i in range(N):
                old_val = a[i]
                for new_val in range(N):
                    if new_val == old_val:
                        continue
                    a[i] = new_val
                    s = get_score(a, b)
                    if s > current_score:
                        current_score = s
                        improved = True
                        break
                    a[i] = old_val
                if improved:
                    break
            
            if not improved:
                # Try mutating one position in b
                for i in range(N):
                    old_val = b[i]
                    for new_val in range(N):
                        if new_val == old_val:
                            continue
                        b[i] = new_val
                        s = get_score(a, b)
                        if s > current_score:
                            current_score = s
                            improved = True
                            break
                        b[i] = old_val
                    if improved:
                        break
            
            if not improved:
                break
        
        return a, b, current_score

    # Start with Pinsker automaton (known to have threshold (N-1)^2 = 121)
    best_a, best_b = pinsker_automaton()
    best_score = get_score(best_a, best_b)
    
    # If Pinsker doesn't give 121, try some structured starts
    if best_score < 100:
        # Try other structured constructions
        for perm_seed in range(10):
            random.seed(perm_seed)
            a = list(range(N))
            random.shuffle(a)
            b = list(range(N))
            random.shuffle(b)
            s = get_score(a, b)
            if s > best_score:
                best_score = s
                best_a = a[:]
                best_b = b[:]
    
    # Large-scale random restart local search
    restarts = 0
    while time.time() - start_time < DEADLINE:
        # Generate a random starting point
        a = random_automaton()
        b = random_automaton()
        
        # Quick check: is it synchronizing?
        s = get_score(a, b)
        
        if s == -1:
            # Not synchronizing; skip to next restart
            restarts += 1
            continue
        
        # Run local search from this point
        a2, b2, s2 = local_search(a[:], b[:], max_iters=100)
        
        if s2 > best_score:
            best_score = s2
            best_a = a2[:]
            best_b = b2[:]
        
        restarts += 1
    
    return [best_a, best_b]
