"""Construct a set of 20 distinct positive integers minimizing max(|A+A|, |A*A|)."""
import time
import random
from itertools import combinations, product

# EVOLVE-BLOCK-START

def _sumset_size(A):
    """Compute |A+A| with equal summands allowed (diagonal pairs included)."""
    s = set()
    for i in range(len(A)):
        for j in range(i, len(A)):
            s.add(A[i] + A[j])
    return len(s)

def _prodset_size(A):
    """Compute |A*A| with equal factors allowed (diagonal pairs included)."""
    s = set()
    for i in range(len(A)):
        for j in range(i, len(A)):
            s.add(A[i] * A[j])
    return len(s)

def _objective(A):
    """max(|A+A|, |A*A|)."""
    return max(_sumset_size(A), _prodset_size(A))

def _is_valid(A):
    """Check 20 distinct positive integers <= 10000."""
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    if any(x < 1 or x > 10000 for x in A):
        return False
    return True

def _greedy_init():
    """Build an initial set greedily, trying to keep both sets small."""
    A = [1]
    for _ in range(19):
        best_val = None
        best_cost = float('inf')
        # Try candidates in a smart order: small numbers first, then spread
        candidates = []
        for v in range(2, 101):
            if v not in A:
                candidates.append(v)
        # Also add some larger candidates for spread
        for v in range(100, 1000, 10):
            if v not in A:
                candidates.append(v)
        for v in range(1000, 10000, 100):
            if v not in A:
                candidates.append(v)
        
        random.shuffle(candidates)
        for v in candidates[:200]:
            new_A = A + [v]
            cost = _objective(new_A)
            if cost < best_cost:
                best_cost = cost
                best_val = v
        if best_val is None:
            # Fallback: pick a random unused number
            used = set(A)
            while True:
                v = random.randint(1, 10000)
                if v not in used:
                    best_val = v
                    break
        A.append(best_val)
    return sorted(A)

def _swap_move(A):
    """Swap one element with a random new value."""
    idx = random.randrange(20)
    old = A[idx]
    while True:
        new = random.randint(1, 10000)
        if new != old and new not in A:
            break
    A[idx] = new
    return sorted(A)

def _nudge_move(A):
    """Nudge one element by a small amount."""
    idx = random.randrange(20)
    delta = random.choice([-3, -2, -1, 1, 2, 3, 5, 10])
    new_val = A[idx] + delta
    if new_val < 1 or new_val > 10000:
        return None
    if new_val in A:
        return None
    A[idx] = new_val
    return sorted(A)

def _replace_move(A):
    """Replace one element with a random one."""
    idx = random.randrange(20)
    old = A[idx]
    while True:
        new = random.randint(1, 10000)
        if new != old and new not in A:
            break
    A[idx] = new
    return sorted(A)

def _anneal(A, start_temp, end_temp, steps, deadline):
    """Simulated annealing with nudge, swap, and replace moves."""
    current = list(A)
    current_cost = _objective(current)
    best = list(current)
    best_cost = current_cost
    temp = start_temp
    
    for step in range(steps):
        if time.time() > deadline:
            break
        # Choose move type
        r = random.random()
        if r < 0.5:
            new_A = _nudge_move(list(current))
            if new_A is None:
                continue
        elif r < 0.8:
            new_A = _swap_move(list(current))
        else:
            new_A = _replace_move(list(current))
        
        new_cost = _objective(new_A)
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < min(1.0, exp(-delta / max(temp, 1e-9))):
            current = new_A
            current_cost = new_cost
            if new_cost < best_cost:
                best = list(current)
                best_cost = new_cost
        
        # Cool down
        temp = start_temp * (end_temp / start_temp) ** (step / max(steps - 1, 1))
    
    return best, best_cost

def _exp(x):
    """Safe exponential for acceptance probability."""
    if x < -50:
        return 0.0
    import math
    return math.exp(x)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    random.seed(42)
    
    # Phase 1: Generate multiple greedy initial solutions
    best_set = None
    best_cost = float('inf')
    
    for attempt in range(5):
        if time.time() > deadline:
            break
        A = _greedy_init()
        cost = _objective(A)
        if cost < best_cost:
            best_cost = cost
            best_set = list(A)
    
    # Phase 2: Simulated annealing from best initial
    if time.time() > deadline:
        return best_set
    
    # Run annealing with decreasing temperature
    start_temp = 50.0
    end_temp = 0.01
    steps = 2000
    
    # Multiple restarts if time permits
    num_restarts = 3
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        if r == 0:
            A = list(best_set)
        else:
            A = _greedy_init()
            cost = _objective(A)
            if cost < best_cost:
                best_cost = cost
                best_set = list(A)
        
        result, cost = _anneal(A, start_temp, end_temp, steps, deadline)
        if cost < best_cost:
            best_cost = cost
            best_set = list(result)
    
    # Phase 3: Local refinement - try all single-element changes
    if time.time() < deadline - 5:
        current = list(best_set)
        current_cost = best_cost
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for idx in range(20):
                if time.time() > deadline - 2:
                    break
                old = current[idx]
                # Try nearby values
                for delta in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, -10, 10, -20, 20]:
                    new_val = old + delta
                    if new_val < 1 or new_val > 10000:
                        continue
                    if new_val in current:
                        continue
                    current[idx] = new_val
                    new_cost = _objective(current)
                    if new_cost < current_cost:
                        current_cost = new_cost
                        improved = True
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_set = list(current)
                        break
                    else:
                        current[idx] = old
    
    return best_set

# EVOLVE-BLOCK-END
