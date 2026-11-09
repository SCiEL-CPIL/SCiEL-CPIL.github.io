import time
import random
import sys
from collections import deque

N = 12
DEADLINE = 85.0  # Safety margin before the 90s kill

def compute_threshold_fast(a_map, b_map):
    """
    Computes the exact shortest reset word length using BFS on the subset automaton.
    Uses precomputed transition tables for speed.
    Returns the length or None if not synchronizing.
    """
    start = (1 << N) - 1
    
    # Precompute transitions for all 2^N subsets
    trans_a = [0] * (1 << N)
    trans_b = [0] * (1 << N)
    
    for mask in range(1 << N):
        ta = 0
        tb = 0
        for i in range(N):
            if mask & (1 << i):
                ta |= (1 << a_map[i])
                tb |= (1 << b_map[i])
        trans_a[mask] = ta
        trans_b[mask] = tb
    
    # BFS
    visited = [-1] * (1 << N)
    visited[start] = 0
    queue = deque([start])
    
    while queue:
        state = queue.popleft()
        dist = visited[state]
        
        # Check if singleton
        if state & (state - 1) == 0:
            return dist
            
        next_a = trans_a[state]
        if visited[next_a] == -1:
            visited[next_a] = dist + 1
            queue.append(next_a)
            
        next_b = trans_b[state]
        if visited[next_b] == -1:
            visited[next_b] = dist + 1
            queue.append(next_b)
            
    return None

def compute_threshold(a_map, b_map):
    """
    Computes the exact shortest reset word length using BFS on the subset automaton.
    Returns the length or None if not synchronizing.
    """
    return compute_threshold_fast(a_map, b_map)

def is_synchronizing(a_map, b_map):
    """
    Quick check if the automaton synchronizes.
    """
    return compute_threshold(a_map, b_map) is not None

def construct_pawlik_variant(offset=0):
    """
    Constructs a Pawlik-style automaton with a specific topological arrangement.
    The Pawlik automaton has:
    - a: a long cycle (permutation)
    - b: collapses two adjacent states in the cycle
    
    We vary the offset to create different topological arrangements that
    may force different BFS paths in the subset lattice.
    """
    a = list(range(N))
    b = list(range(N))
    
    # Create a cycle for 'a': 0 -> 1 -> 2 -> ... -> N-1 -> 0
    for i in range(N):
        a[i] = (i + 1) % N
    
    # For 'b', collapse states at positions offset and offset+1
    # This creates a bottleneck that forces synchronization
    collapse_target = offset % N
    for i in range(N):
        if i == offset or i == (offset + 1) % N:
            b[i] = collapse_target
        else:
            b[i] = i
    
    return a, b

def random_automaton():
    """
    Generate a random complete deterministic automaton.
    """
    a = [random.randint(0, N-1) for _ in range(N)]
    b = [random.randint(0, N-1) for _ in range(N)]
    return a, b

def construct():
    """
    Returns the transition tables for a synchronizing automaton with N=12 states,
    maximizing the shortest reset word length.
    
    Strategy: 
    1. Try Pawlik variants as baseline
    2. Use random search with the fast evaluator to explore more structures
    3. Local search around the best found automaton
    """
    start_time = time.time()
    
    best_a = None
    best_b = None
    best_score = 0
    
    # Phase 1: Try Pawlik variants
    for offset in range(N):
        if time.time() > DEADLINE - 30:
            break
        a, b = construct_pawlik_variant(offset)
        score = compute_threshold(a, b)
        
        if score is not None and score > best_score:
            best_score = score
            best_a = a[:]
            best_b = b[:]
    
    # Phase 2: Random search
    iterations = 0
    while time.time() < DEADLINE - 10:
        iterations += 1
        a, b = random_automaton()
        score = compute_threshold(a, b)
        
        if score is not None and score > best_score:
            best_score = score
            best_a = a[:]
            best_b = b[:]
    
    # Phase 3: Local search around the best automaton
    if best_a is not None and time.time() < DEADLINE - 2:
        for _ in range(200):
            if time.time() > DEADLINE - 2:
                break
            
            a = best_a[:]
            b = best_b[:]
            
            # Randomly perturb one transition
            if random.random() < 0.5:
                i = random.randint(0, N-1)
                a[i] = random.randint(0, N-1)
            else:
                i = random.randint(0, N-1)
                b[i] = random.randint(0, N-1)
            
            score = compute_threshold(a, b)
            
            if score is not None and score > best_score:
                best_score = score
                best_a = a[:]
                best_b = b[:]
    
    # Fallback: simple synchronizing automaton
    if best_a is None or best_score == 0:
        best_a = list(range(N))
        best_b = [0] * N
        for i in range(1, N):
            best_b[i] = i - 1
    
    return [best_a, best_b]

if __name__ == "__main__":
    # Quick self-test
    a, b = construct()
    print(f"Best score found: {compute_threshold(a, b)}")
    print(f"Time: {time.time()}")
