import time
import random
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1

def _calculate_threshold_fast(a, b):
    """
    Computes the exact reset threshold using BFS on the subset automaton.
    Optimized for speed by computing transitions on-the-fly.
    """
    size = 1 << N
    visited = bytearray(size)
    visited[FULL_MASK] = 1
    
    queue = deque()
    queue.append((FULL_MASK, 0))
    
    while queue:
        mask, depth = queue.popleft()
        
        # Check if single state
        if mask & (mask - 1) == 0:
            return depth
            
        # Compute next state for 'a'
        ma = 0
        m = mask
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            ma |= 1 << a[i]
            m -= lsb
            
        if not visited[ma]:
            visited[ma] = 1
            queue.append((ma, depth + 1))
            
        # Compute next state for 'b'
        mb = 0
        m = mask
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            mb |= 1 << b[i]
            m -= lsb
            
        if not visited[mb]:
            visited[mb] = 1
            queue.append((mb, depth + 1))
            
    return -1

def _random_automaton():
    return [random.randint(0, N-1) for _ in range(N)], [random.randint(0, N-1) for _ in range(N)]

def _mutate_automaton(a, b):
    """
    Mutate by changing a single transition in either 'a' or 'b'.
    """
    a_new = a[:]
    b_new = b[:]
    
    if random.random() < 0.5:
        pos = random.randint(0, N - 1)
        a_new[pos] = random.randint(0, N - 1)
    else:
        pos = random.randint(0, N - 1)
        b_new[pos] = random.randint(0, N - 1)
        
    return a_new, b_new

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best_score = -1
    best_a = None
    best_b = None
    
    # 1. Try the standard Černý-like automaton (Shift)
    # a: i -> i+1, b: i -> i-1 (with 0 fixed)
    a_shift = [(i + 1) % N for i in range(N)]
    b_shift = [0] * N
    for i in range(1, N):
        b_shift[i] = i - 1
        
    s = _calculate_threshold_fast(a_shift, b_shift)
    if s > best_score:
        best_score = s
        best_a = a_shift[:]
        best_b = b_shift[:]
        
    # 2. Try a few random automata to see if we get lucky with a high score
    for _ in range(50):
        a, b = _random_automaton()
        s = _calculate_threshold_fast(a, b)
        if s > best_score:
            best_score = s
            best_a = a
            best_b = b
            
    # 3. Hill Climbing / Local Search
    # We use a "best improvement" strategy with random restarts if stuck.
    current_a = best_a
    current_b = best_b
    current_score = best_score
    
    no_improvement_count = 0
    max_no_improvement = 200
    
    while time.time() < deadline:
        improved = False
        # Try a few mutations from the current best
        for _ in range(10):
            if time.time() > deadline:
                break
                
            child_a, child_b = _mutate_automaton(current_a, current_b)
            child_score = _calculate_threshold_fast(child_a, child_b)
            
            if child_score > current_score:
                current_a = child_a
                current_b = child_b
                current_score = child_score
                improved = True
                no_improvement_count = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_a = current_a
                    best_b = current_b
                break # Restart mutation loop with new best
                
        if not improved:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                # Random restart
                a, b = _random_automaton()
                s = _calculate_threshold_fast(a, b)
                if s > current_score:
                    current_a = a
                    current_b = b
                    current_score = s
                    if current_score > best_score:
                        best_score = current_score
                        best_a = current_a
                        best_b = current_b
                else:
                    # Try mutating the global best again
                    current_a = best_a
                    current_b = best_b
                    current_score = best_score
                no_improvement_count = 0

    return [best_a, best_b]
