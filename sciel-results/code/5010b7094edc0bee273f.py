import random
import time
import math

def _set_size_sum(A):
    """Compute |A+A| with equal summands allowed (i <= j pairs)."""
    s = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            s.add(ai + A[j])
    return len(s)

def _set_size_prod(A):
    """Compute |A*A| with equal factors allowed (i <= j pairs)."""
    s = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            s.add(ai * A[j])
    return len(s)

def _objective(A):
    """Return max(|A+A|, |A*A|)."""
    return max(_set_size_sum(A), _set_size_prod(A))

def _generate_geometric_base():
    """Generate a geometric-like sequence of 20 distinct positive integers <= 10000."""
    # Try various bases and starting points
    best = None
    best_obj = float('inf')
    for base in [1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0]:
        for start in range(1, 30):
            A = []
            x = float(start)
            for _ in range(20):
                v = int(round(x))
                if v < 1:
                    v = 1
                if v > 10000:
                    break
                if v not in A:
                    A.append(v)
                x *= base
            if len(A) == 20:
                obj = _objective(A)
                if obj < best_obj:
                    best_obj = obj
                    best = list(A)
    if best is None:
        # Fallback: simple geometric
        A = []
        x = 1.0
        for _ in range(20):
            v = int(round(x))
            if v < 1:
                v = 1
            if v > 10000:
                v = 10000
            if v not in A:
                A.append(v)
            else:
                A.append(v + 1)
            x *= 1.3
        # Ensure distinct and <= 10000
        seen = set()
        final = []
        for v in A:
            if v not in seen and 1 <= v <= 10000:
                seen.add(v)
                final.append(v)
        while len(final) < 20:
            final.append(final[-1] + 1)
        best = final[:20]
    return best

def _local_search(A, deadline, rng):
    """Local search with simulated annealing to minimize max(|A+A|, |A*A|)."""
    A = list(A)
    A.sort()
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    T = 50.0  # Initial temperature
    cooling = 0.999
    
    start_time = time.time()
    
    while time.time() < deadline:
        # Pick a random element to perturb
        idx = rng.randint(0, len(A) - 1)
        old_val = A[idx]
        
        # Generate new value: try nearby values or random
        if rng.random() < 0.7:
            # Local move: small perturbation
            delta = rng.randint(-5, 5)
            new_val = old_val + delta
        else:
            # Larger jump
            new_val = rng.randint(1, 10000)
        
        if new_val < 1 or new_val > 10000:
            new_val = max(1, min(10000, new_val))
        
        # Check distinctness
        A[idx] = new_val
        if len(set(A)) != len(A):
            A[idx] = old_val
            continue
        
        A.sort()
        new_obj = _objective(A)
        
        if new_obj <= current_obj or rng.random() < math.exp(-(new_obj - current_obj) / max(T, 0.01)):
            current_obj = new_obj
            if new_obj < best_obj:
                best_obj = new_obj
                best_A = list(A)
        else:
            A[idx] = old_val
            A.sort()
        
        T *= cooling
        if T < 0.01:
            T = 0.01
    
    return best_A, best_obj

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    rng = random.Random(42)
    
    # Start from a good geometric base
    A = _generate_geometric_base()
    best_A = list(A)
    best_obj = _objective(A)
    
    # Run local search
    A, obj = _local_search(A, deadline, rng)
    if obj < best_obj:
        best_obj = obj
        best_A = A
    
    # Try a few more restarts if time permits
    for restart in range(10):
        if time.time() > deadline - 5.0:
            break
        
        # Generate a new geometric base with different parameters
        base = 1.0 + rng.uniform(0.05, 1.0)
        start_val = rng.randint(1, 50)
        A = []
        x = float(start_val)
        for _ in range(20):
            v = int(round(x))
            if v < 1:
                v = 1
            if v > 10000:
                break
            if v not in A:
                A.append(v)
            x *= base
        
        if len(A) < 20:
            # Fill in missing values
            existing = set(A)
            v = 1
            while len(A) < 20:
                if v not in existing and v <= 10000:
                    A.append(v)
                v += 1
            A.sort()
        
        A, obj = _local_search(A, min(deadline, time.time() + 5.0), rng)
        if obj < best_obj:
            best_obj = obj
            best_A = A
    
    return best_A

# EVOLVE-BLOCK-START
# The construct function above is the main entry point
# EVOLVE-BLOCK-END
