"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

N = 12

def _compute_reset_length_bitmask(a, b):
    """
    Compute the exact shortest reset word length via BFS on subset automaton using bitmasks.
    Returns -1 if not synchronizing.
    """
    full_mask = (1 << N) - 1
    dest_a = [1 << a[i] for i in range(N)]
    dest_b = [1 << b[i] for i in range(N)]
    
    # BFS
    visited = {full_mask}
    queue = deque([(full_mask, 0)])
    
    while queue:
        mask, dist = queue.popleft()
        if mask & (mask - 1) == 0:
            return dist
        
        new_mask_a = 0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            new_mask_a |= dest_a[idx]
            m ^= lsb
        if new_mask_a not in visited:
            visited.add(new_mask_a)
            queue.append((new_mask_a, dist + 1))
            
        new_mask_b = 0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            new_mask_b |= dest_b[idx]
            m ^= lsb
        if new_mask_b not in visited:
            visited.add(new_mask_b)
            queue.append((new_mask_b, dist + 1))
            
    return -1

def _is_synchronizing_bitmask(a, b):
    """Check if automaton synchronizes using bitmask BFS."""
    full_mask = (1 << N) - 1
    dest_a = [1 << a[i] for i in range(N)]
    dest_b = [1 << b[i] for i in range(N)]
    
    visited = {full_mask}
    queue = deque([full_mask])
    
    while queue:
        mask = queue.popleft()
        if mask & (mask - 1) == 0:
            return True
        
        new_mask_a = 0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            new_mask_a |= dest_a[idx]
            m ^= lsb
        if new_mask_a not in visited:
            visited.add(new_mask_a)
            queue.append(new_mask_a)
            
        new_mask_b = 0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            new_mask_b |= dest_b[idx]
            m ^= lsb
        if new_mask_b not in visited:
            visited.add(new_mask_b)
            queue.append(new_mask_b)
            
    return False

# EVOLVE-BLOCK-START
def construct():
    # Set deadline to 55s to be safely under the 60s evaluator limit
    deadline = time.time() + 55.0
    
    best_a = None
    best_b = None
    best_len = -1
    
    def evaluate(a, b):
        nonlocal best_a, best_b, best_len
        if time.time() > deadline:
            return
        if _is_synchronizing_bitmask(a, b):
            length = _compute_reset_length_bitmask(a, b)
            if length > best_len:
                best_a, best_b, best_len = list(a), list(b), length

    # Strategy 1: Pinwheel construction variants
    # Classic pinwheel: a is a cycle, b maps i to i-1 (with 0->0 or similar)
    # We test shifts of the cycle and variations of the 'b' mapping.
    
    # Standard Pinwheel: a = (i+1)%N, b = i-1 (0->0)
    for shift in range(N):
        a = [(i + 1 + shift) % N for i in range(N)]
        b = [0] + list(range(N - 1))
        evaluate(a, b)
        
        # Variation: b maps i to (i-1)%N (full cycle decrement)
        b2 = [(i - 1) % N for i in range(N)]
        evaluate(a, b2)
        
        # Variation: b maps i to (i-2)%N
        b3 = [(i - 2) % N for i in range(N)]
        evaluate(a, b3)

    # Strategy 2: Rank-2 Cycle Constructions
    # a is a cycle, b collapses two states to the same target
    a_cycle = list(range(1, N)) + [0]
    
    # Try all possible "defects" in the b mapping
    # Standard: b[i] = i-1 for i>0, b[0]=0
    # Defect: b[i] = i-2 for some i, or b[i] = i for some i, etc.
    for i in range(N):
        for j in range(N):
            if time.time() > deadline:
                break
            b = [0] + list(range(N - 1))
            # Perturb b at position i to point to j
            # Ensure it's not just the identity or trivial
            b[i] = j
            evaluate(a_cycle, b)
        if time.time() > deadline:
            break

    # Strategy 3: Systematic search over structured families with parameters
    # Family: a is a cycle with offset, b has a specific structure
    for a_shift in range(N):
        if time.time() > deadline:
            break
        a = [(i + 1 + a_shift) % N for i in range(N)]
        
        # b: mostly i-1, but with a "jump"
        for jump_pos in range(N):
            for jump_dest in range(N):
                if time.time() > deadline:
                    break
                b = [0] + list(range(N - 1))
                b[jump_pos] = jump_dest
                evaluate(a, b)
            if time.time() > deadline:
                break

    # Strategy 4: Reversed Pinwheel
    # a is reverse cycle, b is decrement
    for shift in range(N):
        if time.time() > deadline:
            break
        a = [(-i - 1 - shift) % N for i in range(N)]
        b = [0] + list(range(N - 1))
        evaluate(a, b)
        
        b2 = [(i - 1) % N for i in range(N)]
        evaluate(a, b2)

    # Strategy 5: Random restarts with structured perturbations (limited time)
    rng = random.Random(999)
    attempts = 0
    # Use remaining time for random search if any
    while time.time() < deadline and attempts < 2000:
        attempts += 1
        
        # Start from best known or a structured base
        if best_a and rng.random() < 0.8:
            a = list(best_a)
            b = list(best_b)
        else:
            # Use a random structured base
            base_type = rng.randint(0, 2)
            if base_type == 0:
                a = list(range(1, N)) + [0]
                b = [0] + list(range(N - 1))
            elif base_type == 1:
                a = [(i + 2) % N for i in range(N)]
                b = [0] + list(range(N - 1))
            else:
                a = [rng.randrange(N) for _ in range(N)]
                b = [rng.randrange(N) for _ in range(N)]
        
        # Apply 1-2 random perturbations
        n_perturb = rng.randint(1, 2)
        for _ in range(n_perturb):
            if rng.random() < 0.5:
                idx = rng.randrange(N)
                a[idx] = rng.randrange(N)
            else:
                idx = rng.randrange(N)
                b[idx] = rng.randrange(N)
        
        evaluate(a, b)
    
    if best_a is None:
        # Fallback to a known synchronizing automaton
        best_a = list(range(1, N)) + [0]
        best_b = [0] + list(range(N - 1))
        # Just to be safe, calculate length if not done
        if best_len < 0:
             best_len = _compute_reset_length_bitmask(best_a, best_b)
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
