"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import numpy as np

# EVOLVE-BLOCK-START
def _count_progressions_fast(arr):
    """Count distinct d > 0 such that a, a+d, a+2d are all in arr. Fast version using set."""
    s = set(arr)
    distinct_d = set()
    for a in arr:
        for b in arr:
            if b > a:
                d = b - a
                if d > 0 and (a + 2 * d) in s:
                    distinct_d.add(d)
    return len(distinct_d)

def _count_progressions(arr):
    """Count distinct d > 0 such that a, a+d, a+2d are all in arr."""
    s = set(arr)
    distinct_d = set()
    n = len(arr)
    for i in range(n):
        a = arr[i]
        for j in range(i + 1, n):
            b = arr[j]
            d = b - a
            if a + 2 * d in s:
                distinct_d.add(d)
    return len(distinct_d)

def _evaluate(arr):
    """Evaluate a candidate set. Returns score."""
    s = set(arr)
    distinct_d = set()
    for a in arr:
        for b in arr:
            if b > a:
                d = b - a
                if d > 0 and (a + 2 * d) in s:
                    distinct_d.add(d)
    return len(distinct_d)

def _simulated_annealing(arr, start_time, deadline, initial_temp=10.0, cooling=0.9995):
    """Simulated annealing to improve the set."""
    arr = list(arr)
    best_arr = list(arr)
    best_score = _evaluate(arr)
    current_score = best_score
    current_arr = list(arr)
    temp = initial_temp
    n = len(arr)
    s = set(arr)
    
    while time.time() < deadline - 2.0:
        # Pick a random element to modify
        idx = random.randint(0, n - 1)
        old_val = current_arr[idx]
        
        # Try a new value
        new_val = random.randint(1, 512)
        if new_val == old_val or new_val in s:
            continue
        
        # Remove old, add new
        s.discard(old_val)
        s.add(new_val)
        current_arr[idx] = new_val
        
        new_score = _evaluate(current_arr)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < pow(2.718281828, delta / temp):
            # Accept
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_arr = list(current_arr)
        else:
            # Reject: revert
            current_arr[idx] = old_val
            s.discard(new_val)
            s.add(old_val)
        
        temp *= cooling
        if temp < 0.01:
            temp = 0.01
    
    return best_arr, best_score

def _local_search(arr, start_time, deadline):
    """Local search: try replacing each element with better alternatives."""
    arr = list(arr)
    best_arr = list(arr)
    best_score = _evaluate(arr)
    n = len(arr)
    s = set(arr)
    
    improved = True
    while improved and time.time() < deadline - 3.0:
        improved = False
        for i in range(n):
            if time.time() > deadline - 3.0:
                break
            old_val = arr[i]
            best_new_val = old_val
            best_new_score = best_score
            
            # Try several candidate replacements
            candidates = []
            # Try values that might form progressions
            for other in arr:
                if other == old_val:
                    continue
                # Try values that would create a 3-term AP with other
                # If other - old_val = d, then old_val - d and other + d
                d = other - old_val
                if d > 0:
                    c1 = old_val - d
                    c2 = other + d
                    if 1 <= c1 <= 512 and c1 not in s:
                        candidates.append(c1)
                    if 1 <= c2 <= 512 and c2 not in s:
                        candidates.append(c2)
                elif d < 0:
                    d = -d
                    c1 = old_val + d
                    c2 = other - d
                    if 1 <= c1 <= 512 and c1 not in s:
                        candidates.append(c1)
                    if 1 <= c2 <= 512 and c2 not in s:
                        candidates.append(c2)
            
            # Also try some random values
            for _ in range(5):
                rv = random.randint(1, 512)
                if rv not in s and rv != old_val:
                    candidates.append(rv)
            
            for cand in candidates:
                if cand == old_val or cand in s:
                    continue
                # Try this candidate
                s.discard(old_val)
                s.add(cand)
                arr[i] = cand
                sc = _evaluate(arr)
                if sc > best_new_score:
                    best_new_score = sc
                    best_new_val = cand
                # Revert for next try
                arr[i] = old_val
                s.discard(cand)
                s.add(old_val)
            
            if best_new_val != old_val:
                arr[i] = best_new_val
                s.discard(old_val)
                s.add(best_new_val)
                best_score = best_new_score
                best_arr = list(arr)
                improved = True
    
    return best_arr, best_score

def _make_initial_set():
    """Create a good initial set."""
    random.seed(42)
    # Start with a structured set: two interleaved APs
    # Set 1: 1, 3, 5, ..., 63 (odd numbers up to 63) - 32 elements
    # But let's try something that might give more distinct d
    
    # Try: a set that's dense in [1, 32] but also has spread elements
    # Actually, let's try multiple initial constructions
    
    candidates = []
    
    # Construction 1: [1, 2, ..., 32]
    candidates.append(list(range(1, 33)))
    
    # Construction 2: Two blocks
    for gap in range(10, 50):
        block1 = list(range(1, 17))  # 16 elements
        block2 = list(range(1 + gap, 1 + gap + 16))  # 16 elements
        if max(block2) <= 512:
            candidates.append(block1 + block2)
    
    # Construction 3: Geometric-like
    for base in range(1, 20):
        arr = []
        x = base
        while x <= 512 and len(arr) < 32:
            arr.append(x)
            x = x * 2 + random.randint(1, 5)
        while len(arr) < 32:
            v = random.randint(1, 512)
            if v not in arr:
                arr.append(v)
        candidates.append(arr[:32])
    
    # Construction 4: Random structured
    for _ in range(20):
        arr = random.sample(range(1, 513), 32)
        candidates.append(arr)
    
    # Evaluate and return best
    best = max(candidates, key=lambda x: _evaluate(x))
    return best

def construct():
    deadline = time.time() + 145.0  # Safety margin before 150s limit
    
    # Get initial set
    arr = _make_initial_set()
    best_arr = list(arr)
    best_score = _evaluate(arr)
    
    # Phase 1: Local search to quickly improve
    if time.time() < deadline - 30:
        best_arr, best_score = _local_search(best_arr, time.time(), deadline - 30)
    
    # Phase 2: Simulated annealing with multiple restarts
    num_restarts = 5
    remaining_time = deadline - time.time() - 5
    time_per_restart = remaining_time / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Start from best or slightly perturbed
        if r == 0:
            start_arr = list(best_arr)
        else:
            start_arr = list(best_arr)
            # Perturb
            for _ in range(random.randint(1, 5)):
                idx = random.randint(0, 31)
                new_val = random.randint(1, 512)
                if new_val not in start_arr:
                    start_arr[idx] = new_val
        
        sa_deadline = min(deadline - 5, time.time() + time_per_restart)
        sa_arr, sa_score = _simulated_annealing(start_arr, time.time(), sa_deadline)
        
        if sa_score > best_score:
            best_score = sa_score
            best_arr = sa_arr
    
    # Phase 3: Final local search polish
    if time.time() < deadline - 3:
        best_arr, best_score = _local_search(best_arr, time.time(), deadline - 1)
    
    return list(best_arr)
# EVOLVE-BLOCK-END
