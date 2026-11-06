"""Rewritten program with Simulated Annealing for Černý problem (N=12)."""
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

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 50.0  # Safety margin
    rng = random.Random(42)
    
    # Initialize with the standard pinwheel
    a = [(i + 1) % N for i in range(N)]
    b = [0] + list(range(N - 1))
    
    if not _is_synchronizing(a, b):
        # Fallback to random if pinwheel doesn't sync (should not happen)
        a = [rng.randrange(N) for _ in range(N)]
        b = [rng.randrange(N) for _ in range(N)]
        if not _is_synchronizing(a, b):
            return a, b

    current_len = _compute_reset_length(a, b)
    best_a = list(a)
    best_b = list(b)
    best_len = current_len
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    
    while time.time() < deadline - 5.0 and T > T_min:
        # Generate a neighbor
        new_a = list(a)
        new_b = list(b)
        
        if rng.random() < 0.5:
            idx = rng.randrange(N)
            new_a[idx] = rng.randrange(N)
        else:
            idx = rng.randrange(N)
            new_b[idx] = rng.randrange(N)
        
        # Check if synchronizing
        if _is_synchronizing(new_a, new_b):
            new_len = _compute_reset_length(new_a, new_b)
            
            if new_len > current_len:
                # Accept better
                a = new_a
                b = new_b
                current_len = new_len
                if new_len > best_len:
                    best_a = list(a)
                    best_b = list(b)
                    best_len = new_len
            else:
                # Accept worse with probability exp((new_len - current_len) / T)
                # Note: We want to maximize, so delta is negative for worse.
                delta = new_len - current_len
                if delta < 0:
                    prob = math.exp(delta / T)
                    if rng.random() < prob:
                        a = new_a
                        b = new_b
                        current_len = new_len
        
        T *= cooling
    
    # Final local search / hill climbing to squeeze out more
    if time.time() < deadline - 2.0:
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for letter_idx in range(2):
                orig = list(a) if letter_idx == 0 else list(b)
                for pos in range(N):
                    for new_val in range(N):
                        if new_val == orig[pos]:
                            continue
                        if letter_idx == 0:
                            a[pos] = new_val
                        else:
                            b[pos] = new_val
                        
                        if _is_synchronizing(a, b):
                            length = _compute_reset_length(a, b)
                            if length > current_len:
                                current_len = length
                                if length > best_len:
                                    best_a = list(a)
                                    best_b = list(b)
                                    best_len = length
                                improved = True
                                break
                        if letter_idx == 0:
                            a[pos] = orig[pos]
                        else:
                            b[pos] = orig[pos]
                    if time.time() > deadline - 2.0:
                        break
                if time.time() > deadline - 2.0:
                    break
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
