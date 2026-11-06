import time
import random
import math
from collections import deque

N = 12

def _compute_reset_length(a, b):
    """
    Compute the exact shortest reset word length via BFS on subset automaton using bitmasks.
    Returns -1 if not synchronizing.
    """
    full_mask = (1 << N) - 1
    dest_a = [1 << a[i] for i in range(N)]
    dest_b = [1 << b[i] for i in range(N)]
    
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

def _is_synchronizing(a, b):
    """Quick check if automaton synchronizes using BFS on subset automaton."""
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

def construct():
    deadline = time.time() + 85.0
    rng = random.Random(42)
    
    # Initialize with a known good candidate (Pinwheel variant)
    # a: cycle, b: 0->0, i->i-1
    best_a = [(i + 1) % N for i in range(N)]
    best_b = [0] + [(i - 1) % N for i in range(1, N)]
    best_len = _compute_reset_length(best_a, best_b)
    
    # If the standard Pinwheel is weak, try other variants
    # Variant 2: a: cycle, b: 0->N-1, i->i-1 (b is cycle? No, 0->N-1, 1->0... is a cycle)
    # If b is a permutation, it must not be a cycle that doesn't mix with a?
    # Actually, if b is a permutation, the automaton is a group if a is too.
    # If a is a permutation and b is not, it can synchronize.
    
    current_a = list(best_a)
    current_b = list(best_b)
    current_len = best_len
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    while time.time() < deadline - 5.0 and T > min_T:
        # Generate neighbor
        a_new = list(current_a)
        b_new = list(current_b)
        
        # Mutate 1-3 positions
        n_muts = rng.randint(1, 3)
        for _ in range(n_muts):
            if rng.random() < 0.5:
                pos = rng.randrange(N)
                a_new[pos] = rng.randrange(N)
            else:
                pos = rng.randrange(N)
                b_new[pos] = rng.randrange(N)
        
        # Check synchronization
        if not _is_synchronizing(a_new, b_new):
            # If not synchronizing, reject (or accept with very low prob to escape? 
            # But non-sync is score -inf, so reject)
            continue
        
        new_len = _compute_reset_length(a_new, b_new)
        
        if new_len > current_len:
            current_a, current_b, current_len = a_new, b_new, new_len
            if new_len > best_len:
                best_a, best_b, best_len = list(current_a), list(current_b), new_len
        else:
            # Accept worse solution with probability
            delta = new_len - current_len
            if delta < 0:
                prob = math.exp(delta / max(T, 1e-10))
                if rng.random() < prob:
                    current_a, current_b, current_len = a_new, b_new, new_len
            # else (new_len == current_len) accept to move around plateau
            else:
                if rng.random() < 0.5:
                    current_a, current_b, current_len = a_new, b_new, new_len
        
        T *= cooling
    
    # Final polish: Local search on best candidate
    if time.time() < deadline - 2.0:
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for letter_idx in range(2):
                orig = list(best_a) if letter_idx == 0 else list(best_b)
                for pos in range(N):
                    for new_val in range(N):
                        if new_val == orig[pos]:
                            continue
                        if letter_idx == 0:
                            test_a = list(best_a)
                            test_a[pos] = new_val
                            test_b = best_b
                        else:
                            test_a = best_a
                            test_b = list(best_b)
                            test_b[pos] = new_val
                        
                        if _is_synchronizing(test_a, test_b):
                            l = _compute_reset_length(test_a, test_b)
                            if l > best_len:
                                if letter_idx == 0:
                                    best_a = test_a
                                else:
                                    best_b = test_b
                                best_len = l
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
    
    if best_len < 0:
        # Fallback to the initial Pinwheel
        best_a = [(i + 1) % N for i in range(N)]
        best_b = [0] + [(i - 1) % N for i in range(1, N)]
    
    return [best_a, best_b]
