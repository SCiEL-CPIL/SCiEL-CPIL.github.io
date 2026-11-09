"""Rewritten program: Fast bitmask BFS + structured search for Černý automaton N=12."""
import time
import random
from collections import deque

N = 12

def _compute_threshold_bitmask(a_map, b_map):
    """
    BFS on subset automaton using bitmasks.
    Returns the shortest reset word length, or -1 if not synchronizing.
    """
    full_mask = (1 << N) - 1
    
    # Precompute transitions for single states
    # trans_a[s] is the bitmask of the state a(s) maps to (just 1 << a(s))
    # Since we need the image of a set, we OR these single-bit masks.
    # Actually, for a set S, image_a(S) = OR_{s in S} (1 << a(s)).
    
    # To make it even faster, we can precompute the image of every subset?
    # 2^12 = 4096 subsets.
    # For each subset, precompute img_a[subset] and img_b[subset].
    # This takes O(2^N * N) time upfront, which is negligible.
    
    img_a = [0] * (1 << N)
    img_b = [0] * (1 << N)
    
    for mask in range(1 << N):
        ma = 0
        mb = 0
        for i in range(N):
            if mask & (1 << i):
                ma |= (1 << a_map[i])
                mb |= (1 << b_map[i])
        img_a[mask] = ma
        img_b[mask] = mb
        
    # BFS
    # State is a bitmask. Start with full_mask.
    # Goal is any mask with exactly one bit set (popcount 1).
    
    visited = {full_mask: 0}
    queue = deque([full_mask])
    
    while queue:
        curr = queue.popleft()
        dist = visited[curr]
        
        next_a = img_a[curr]
        if next_a not in visited:
            # Check if single state
            if next_a & (next_a - 1) == 0: # popcount 1
                return dist + 1
            visited[next_a] = dist + 1
            queue.append(next_a)
            
        next_b = img_b[curr]
        if next_b not in visited:
            if next_b & (next_b - 1) == 0:
                return dist + 1
            visited[next_b] = dist + 1
            queue.append(next_b)
            
    return -1

def _cherny_construction():
    """Standard Černý automaton."""
    a = [(i + 1) % N for i in range(N)]
    b = [0] * N
    for i in range(1, N):
        b[i] = i - 1
    return a, b

def _pawlik_construction():
    """Pawlik automaton, often achieves (N-1)^2."""
    a = [(i + 1) % N for i in range(N)]
    b = [0] * N
    # b maps 0->0, 1->0, i->i-1 for i>1
    b[0] = 0
    b[1] = 0
    for i in range(2, N):
        b[i] = i - 1
    return a, b

def _double_cycle_construction():
    """A variant with two cycles."""
    a = [(i + 1) % N for i in range(N)]
    b = [0] * N
    # Try to create a structure that forces long path
    # This is a heuristic guess
    for i in range(N):
        b[i] = (i + 2) % N 
    # Modify one to break symmetry and allow reset
    b[0] = 0
    return a, b

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    best_length = -1
    best_a = None
    best_b = None
    
    def try_update(a_map, b_map):
        nonlocal best_length, best_a, best_b
        length = _compute_threshold_bitmask(a_map, b_map)
        if length > best_length:
            best_length = length
            best_a = a_map[:]
            best_b = b_map[:]
            # If we hit the theoretical max, we can stop early
            if best_length >= (N - 1) ** 2:
                return True
        return False

    # 1. Try known constructions
    candidates = [
        _cherny_construction(),
        _pawlik_construction(),
        _double_cycle_construction(),
    ]
    
    for a, b in candidates:
        if try_update(a, b):
            return best_a, best_b
            
    # 2. Systematic search over small variations of Černý structure
    # The Černý structure is a cycle + a "reset" map.
    # We can vary the reset map slightly.
    
    base_a = [(i + 1) % N for i in range(N)]
    
    # Iterate over possible b maps that are "almost" the Černý b map
    # b[i] = i-1 for i>1, b[0]=0, b[1]=0
    # Let's perturb b[1] and b[0]
    
    for b0 in range(N):
        for b1 in range(N):
            b_map = [0] * N
            b_map[0] = b0
            b_map[1] = b1
            for i in range(2, N):
                b_map[i] = i - 1
                
            if try_update(base_a, b_map):
                if best_length >= (N - 1) ** 2:
                    return best_a, best_b
                    
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    # 3. Random search with bias towards cycle+reset structures
    # If we haven't found 121, we do random search.
    # Use bitmask BFS which is fast.
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Generate a random permutation for a
        a_map = list(range(N))
        random.shuffle(a_map)
        
        # Generate b_map
        # Bias: create a "reset-like" map
        # Many states map to a small set
        b_map = [random.randint(0, N-1) for _ in range(N)]
        # Force a few to be 0 to help synchronization
        if random.random() < 0.8:
            b_map[random.randint(0, N-1)] = 0
            if random.random() < 0.5:
                b_map[random.randint(0, N-1)] = 0
            
        try_update(a_map, b_map)
        
        # Local search step if we have a decent score
        if best_length > 50 and attempts % 10 == 0:
            # Mutate best_a or best_b
            if random.random() < 0.5:
                new_a = best_a[:]
                idx = random.randint(0, N-1)
                new_a[idx] = random.randint(0, N-1)
                try_update(new_a, best_b)
            else:
                new_b = best_b[:]
                idx = random.randint(0, N-1)
                new_b[idx] = random.randint(0, N-1)
                try_update(best_a, new_b)
                
        if best_length >= (N - 1) ** 2:
            return best_a, best_b

    # Return best found
    if best_a is None:
        # Fallback to a valid synchronizing automaton if search failed completely
        a, b = _cherny_construction()
        best_a = a
        best_b = b
        
    return best_a, best_b

# EVOLVE-BLOCK-END
