import time
import random
from collections import deque
import math

N = 12
FULL = (1 << N) - 1

def _reset_length_bitmask(a, b):
    """
    Exact shortest reset word length via BFS on subset automaton using bitmasks.
    Returns -1 if not synchronizing.
    """
    # Precompute destination bitmasks for each state and letter
    # dest_a[i] is the bitmask of the state a[i] maps to (just 1 << a[i])
    # But for subset transition, we need the union of destinations for all states in the subset.
    # Let's precompute the transition for all 2^N subsets for both letters.
    
    # map_a[mask] = subset reached from 'mask' by applying 'a'
    # map_b[mask] = subset reached from 'mask' by applying 'b'
    
    # Optimization: 
    # Instead of iterating bits for every mask in BFS, we can precompute the transition tables.
    # There are 4096 masks. For each mask, we compute the next mask for 'a' and 'b'.
    # This takes O(2^N * N) which is 4096 * 12 ~ 50k ops. Very fast.
    
    map_a = [0] * (1 << N)
    map_b = [0] * (1 << N)
    
    # dest_a_bits[i] = 1 << a[i]
    # dest_b_bits[i] = 1 << b[i]
    dest_a_bits = [1 << a[i] for i in range(N)]
    dest_b_bits = [1 << b[i] for i in range(N)]
    
    for mask in range(1 << N):
        res_a = 0
        res_b = 0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            res_a |= dest_a_bits[idx]
            res_b |= dest_b_bits[idx]
            m ^= lsb
        map_a[mask] = res_a
        map_b[mask] = res_b
        
    visited = [False] * (1 << N)
    visited[FULL] = True
    queue = deque()
    queue.append((FULL, 0))
    
    while queue:
        mask, dist = queue.popleft()
        if mask & (mask - 1) == 0:
            return dist
        
        nm = map_a[mask]
        if not visited[nm]:
            visited[nm] = True
            queue.append((nm, dist + 1))
            
        nm = map_b[mask]
        if not visited[nm]:
            visited[nm] = True
            queue.append((nm, dist + 1))
            
    return -1

def construct():
    deadline = time.time() + 85.0
    rng = random.Random(42)
    
    # Initialize with a known good structure or random
    # Let's start with a random permutation for a and b
    a = list(range(N))
    rng.shuffle(a)
    b = list(range(N))
    rng.shuffle(b)
    
    best_a = list(a)
    best_b = list(b)
    best_len = _reset_length_bitmask(best_a, best_b)
    
    if best_len == -1:
        # If not synchronizing, try to find one quickly
        # A simple synchronizing automaton: a is cycle, b maps everything to 0
        best_a = list(range(1, N)) + [0]
        best_b = [0] * N
        best_len = _reset_length_bitmask(best_a, best_b)
    
    current_a = list(best_a)
    current_b = list(best_b)
    current_len = best_len
    
    # Simulated Annealing parameters
    # Temperature decay
    T = 10.0
    T_min = 0.01
    alpha = 0.999
    
    steps = 0
    while time.time() < deadline:
        # Mutate
        # Randomly choose which letter to mutate and how
        # 1. Swap two entries in a
        # 2. Swap two entries in b
        # 3. Change one entry in a
        # 4. Change one entry in b
        
        r = rng.random()
        new_a = list(current_a)
        new_b = list(current_b)
        
        if r < 0.4:
            i, j = rng.sample(range(N), 2)
            new_a[i], new_a[j] = new_a[j], new_a[i]
        elif r < 0.8:
            i, j = rng.sample(range(N), 2)
            new_b[i], new_b[j] = new_b[j], new_b[i]
        elif r < 0.9:
            i = rng.randrange(N)
            new_a[i] = rng.randrange(N)
        else:
            i = rng.randrange(N)
            new_b[i] = rng.randrange(N)
        
        new_len = _reset_length_bitmask(new_a, new_b)
        
        # Acceptance criterion
        if new_len > current_len:
            current_a = new_a
            current_b = new_b
            current_len = new_len
            
            if current_len > best_len:
                best_len = current_len
                best_a = list(current_a)
                best_b = list(current_b)
        else:
            # Accept worse solution with probability exp(delta / T)
            # If new_len is -1 (not sync), treat as very bad
            delta = new_len - current_len
            if new_len == -1 or current_len == -1:
                # If one is not synchronizing, be strict
                if new_len != -1 and current_len == -1:
                    current_a = new_a
                    current_b = new_b
                    current_len = new_len
                    if current_len > best_len:
                        best_len = current_len
                        best_a = list(current_a)
                        best_b = list(current_b)
                # Else reject
            else:
                if delta >= 0:
                    current_a = new_a
                    current_b = new_b
                    current_len = new_len
                    if current_len > best_len:
                        best_len = current_len
                        best_a = list(current_a)
                        best_b = list(current_b)
                else:
                    prob = math.exp(delta / T)
                    if rng.random() < prob:
                        current_a = new_a
                        current_b = new_b
                        current_len = new_len
        
        # Cool down
        T = max(T_min, T * alpha)
        steps += 1
        
        # Optional: If we hit 121, we can stop early as it's the theoretical max
        if best_len == 121:
            break

    return [best_a, best_b]
