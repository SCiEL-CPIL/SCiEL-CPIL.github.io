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

    # Strategy 1: Systematic enumeration of known structural families
    # 1a. Pinwheel variants
    for shift in range(N):
        a = [(i + 1 + shift) % N for i in range(N)]
        b = [0] + list(range(N - 1))
        evaluate(a, b)
        
    # 1b. Rank-2 Cycle Variants
    a_cycle = list(range(1, N)) + [0]
    b_variants = [
        [0] + list(range(N - 1)),
        list(range(N - 1)) + [0],
        [N - 1 - i for i in range(N)],
        [(i + N // 2) % N for i in range(N)],
        [0] * N,
    ]
    for b in b_variants:
        evaluate(a_cycle, b)

    # 1c. Systematic search over 'b' for fixed 'a' cycle with 1-2 perturbations
    base_a = list(range(1, N)) + [0]
    base_b = [0] + list(range(N - 1))
    
    # 1-1 perturbations
    for i in range(N):
        for new_dest in range(N):
            if new_dest == base_b[i]:
                continue
            b_copy = base_b[:]
            b_copy[i] = new_dest
            evaluate(base_a, b_copy)
            
    # 2-2 perturbations (limited)
    if time.time() < deadline - 10:
        for i in range(N):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, N):
                if time.time() > deadline - 5:
                    break
                for new_dest_i in [0, (base_b[i] + 1) % N, (base_b[i] - 1) % N]:
                    for new_dest_j in [0, (base_b[j] + 1) % N, (base_b[j] - 1) % N]:
                        if new_dest_i == base_b[i] and new_dest_j == base_b[j]:
                            continue
                        b_copy = base_b[:]
                        b_copy[i] = new_dest_i
                        b_copy[j] = new_dest_j
                        evaluate(base_a, b_copy)

    # Fallback if nothing found yet
    if best_a is None:
        best_a = base_a[:]
        best_b = base_b[:]
        best_len = _compute_reset_length_bitmask(best_a, best_b)
        if best_len <= 0:
            return [base_a, base_b]

    # Strategy 2: Simulated Annealing with fast evaluation
    # We use the exact reset length as the score (higher is better).
    # This is a "maximization" problem.
    
    current_a = best_a[:]
    current_b = best_b[:]
    current_score = best_len
    
    rng = random.Random(42)
    
    # Initial temperature
    T = 10.0
    cooling = 0.99
    
    steps = 0
    max_steps = 20000 # Safety cap, time is the real limit
    
    while time.time() < deadline and steps < max_steps:
        steps += 1
        
        # Pick a random mutation
        state = rng.randint(0, N - 1)
        letter = rng.randint(0, 1)
        
        if letter == 0:
            old_val = current_a[state]
            new_val = rng.randint(0, N - 1)
            if new_val == old_val:
                continue
            current_a[state] = new_val
            new_score = _compute_reset_length_bitmask(current_a, current_b)
            
            if new_score > current_score:
                current_score = new_score
                if current_score > best_len:
                    best_len = current_score
                    best_a = current_a[:]
                    best_b = current_b[:]
            else:
                # Accept worse moves with probability exp((new - old) / T)
                # Note: new_score could be -1 (not synchronizing). 
                # If it's -1, delta is very negative, prob ~ 0.
                delta = new_score - current_score
                if delta < 0:
                    # If delta is -inf (i.e. -1 vs positive), prob is 0
                    if new_score == -1:
                        prob = 0.0
                    else:
                        prob = min(1.0, max(0.0, delta / T)) # Simple linear or exp?
                        # Standard SA uses exp(delta/T). Since delta is negative, exp is < 1.
                        # Let's use exp for proper SA behavior.
                        prob = min(1.0, max(0.0, delta / T)) 
                        # Actually, let's stick to standard exp(delta/T) for negative delta
                        # But if delta is large negative, exp is 0.
                        # Let's use a simplified acceptance:
                        # If delta < 0: accept with prob exp(delta/T)
                        # If delta >= 0: accept always.
                        # Since delta is int, and T is float.
                        # delta can be -10000 if we lose synchronicity.
                        # Let's just use exp(delta/T) but clamp delta to avoid underflow issues if T is small.
                        # Actually, if delta is -1 (score 100 -> 99), exp(-1/10) ~ 0.9
                        # If delta is -10000 (score 100 -> -1), exp(-10000/10) = 0.
                        # This works.
                        import math
                        if delta < -100 * T:
                             prob = 0.0
                        else:
                             prob = math.exp(delta / T)
                else:
                    prob = 1.0
                
                if rng.random() < prob:
                    # Accept
                    current_score = new_score
                    # Do NOT update best if worse
                else:
                    # Reject
                    current_a[state] = old_val
        else:
            old_val = current_b[state]
            new_val = rng.randint(0, N - 1)
            if new_val == old_val:
                continue
            current_b[state] = new_val
            new_score = _compute_reset_length_bitmask(current_a, current_b)
            
            if new_score > current_score:
                current_score = new_score
                if current_score > best_len:
                    best_len = current_score
                    best_a = current_a[:]
                    best_b = current_b[:]
            else:
                delta = new_score - current_score
                if delta < 0:
                    if new_score == -1:
                        prob = 0.0
                    else:
                        import math
                        if delta < -100 * T:
                             prob = 0.0
                        else:
                             prob = math.exp(delta / T)
                else:
                    prob = 1.0
                
                if rng.random() < prob:
                    current_score = new_score
                else:
                    current_b[state] = old_val
                    
        # Cool down
        T *= cooling
        if T < 0.1:
            T = 0.1 # Floor temperature to allow some exploration
            
        # Restart logic: If stuck or low T, restart from best
        if steps % 1000 == 0:
            # Occasionally restart from global best to escape local optima
            if rng.random() < 0.5:
                current_a = best_a[:]
                current_b = best_b[:]
                current_score = best_len
                T = 5.0 # Reset temperature for new phase

    return [best_a, best_b]
# EVOLVE-BLOCK-END
