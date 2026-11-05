"""
Constructs a 12-state binary deterministic automaton maximizing shortest reset word length.
Combines the fast bitmask BFS from inspiration 2 with the systematic search strategy of inspiration 1.
"""
import time
import random
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1

# EVOLVE-BLOCK-START

def _threshold_fast(a, b, deadline=None):
    """
    Compute the exact reset threshold using BFS on the subset automaton.
    Returns -1 if not synchronizing.
    Uses bitmask-based transitions for speed.
    """
    num_masks = 1 << N
    
    # Pre-compute transitions for all 2^N masks
    trans_a = [0] * num_masks
    trans_b = [0] * num_masks
    
    for mask in range(num_masks):
        ma = 0
        mb = 0
        for i in range(N):
            if mask & (1 << i):
                ma |= 1 << a[i]
                mb |= 1 << b[i]
        trans_a[mask] = ma
        trans_b[mask] = mb
    
    # List-based visited: -1 = unvisited, else depth
    visited = [-1] * num_masks
    visited[FULL_MASK] = 0
    queue = deque([FULL_MASK])
    
    while queue:
        if deadline is not None and time.time() > deadline:
            return -1  # Timeout, return -1 to indicate failure
        
        mask = queue.popleft()
        depth = visited[mask]
        
        # Check if singleton
        if mask & (mask - 1) == 0:
            return depth
        
        # Apply 'a'
        next_a = trans_a[mask]
        if visited[next_a] == -1:
            visited[next_a] = depth + 1
            queue.append(next_a)
            
        # Apply 'b'
        next_b = trans_b[mask]
        if visited[next_b] == -1:
            visited[next_b] = depth + 1
            queue.append(next_b)
    
    return -1

def _is_permutation(trans):
    """Check if a transition function is a permutation."""
    return len(set(trans)) == N

def construct():
    """
    Construct a 12-state binary automaton that synchronizes, maximizing the reset threshold.
    Uses systematic search over structured families with fast bitmask BFS evaluation.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    best_score = 0
    best_a = None
    best_b = None
    
    # Base cycle for letter 'a': a[i] = (i + 1) % N
    base_a = [(i + 1) % N for i in range(N)]
    
    def try_automaton(a, b):
        nonlocal best_score, best_a, best_b
        if _is_permutation(b):
            return
        score = _threshold_fast(a, b, deadline)
        if score > best_score:
            best_score = score
            best_a = a[:]
            best_b = b[:]
    
    # Strategy 1: Single-defect 'b' functions
    # b is mostly identity, with one state k mapping to a different state
    for k in range(N):
        for target in range(N):
            if time.time() > deadline:
                break
            if target == k:
                continue  # Skip identity
            
            b = [i for i in range(N)]
            b[k] = target
            try_automaton(base_a, b)
        
        if time.time() > deadline:
            break
    
    # Strategy 2: Double-defect 'b' functions
    if time.time() < deadline - 15.0:
        for k1 in range(N):
            for k2 in range(k1 + 1, N):
                for t1 in range(N):
                    for t2 in range(N):
                        if time.time() > deadline:
                            break
                        if t1 == k1 and t2 == k2:
                            continue  # Identity
                        if t1 == k2 and t2 == k1:
                            continue  # Swap (permutation)
                        
                        b = [i for i in range(N)]
                        b[k1] = t1
                        b[k2] = t2
                        
                        if _is_permutation(b):
                            continue
                        
                        try_automaton(base_a, b)
                    
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
    
    # Strategy 3: Random search with structured perturbations
    if time.time() < deadline - 5.0:
        random.seed(42)
        
        for _ in range(20000):
            if time.time() > deadline:
                break
            
            # Generate random b with guaranteed non-permutation
            b = [random.randint(0, N-1) for _ in range(N)]
            
            # Force non-permutation by creating a collision
            i, j = random.sample(range(N), 2)
            b[j] = b[i]
            
            if _is_permutation(b):
                continue
            
            try_automaton(base_a, b)
    
    # Fallback: If no valid automaton found, use a known synchronizing one
    if best_a is None:
        best_a = base_a[:]
        best_b = [0] * N  # All states map to 0, definitely synchronizing
    
    return [best_a, best_b]

# EVOLVE-BLOCK-END
