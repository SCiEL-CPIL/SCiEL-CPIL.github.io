"""Signed autocorrelation minimization with time-budgeted local search."""
import numpy as np
import time

N = 400

def _score(x):
    """Compute max|autoconv| / (sum(x))^2. Returns inf if sum is 0."""
    s = x.sum()
    if abs(s) < 1e-12:
        return float('inf')
    # Autoconvolution: (x * x)[k] = sum_i x[i]*x[i+k]
    ac = np.convolve(x, x, mode='full')
    return np.max(np.abs(ac)) / (s * s)

def _random_signed():
    """Random ±1 sequence."""
    return np.random.choice([-1.0, 1.0], size=N)

def _random_balanced():
    """Random ±1 with approximately equal counts."""
    x = np.random.choice([-1.0, 1.0], size=N)
    # Try to balance
    diff = x.sum()
    if diff > 0:
        idx = np.where(x == 1.0)[0]
        flip = np.random.choice(idx, size=int(abs(diff)) // 2, replace=False)
        x[flip] = -1.0
    elif diff < 0:
        idx = np.where(x == -1.0)[0]
        flip = np.random.choice(idx, size=int(abs(diff)) // 2, replace=False)
        x[flip] = 1.0
    return x

def _structured_sequence(seed_val):
    """Generate structured low-autocorrelation sequence using quadratic residues."""
    rng = np.random.RandomState(seed_val)
    # Use a pseudo-random but structured approach:
    # Start with random, then apply Walsh-Hadamard-like patterns
    x = np.zeros(N)
    # Mix of different frequency patterns
    for i in range(N):
        # Use multiple sinusoidal components with irrational-like frequencies
        phase1 = 2.0 * np.pi * i * (1 + np.sqrt(2)) / N
        phase2 = 2.0 * np.pi * i * (1 + np.sqrt(3)) / N
        val = np.sin(phase1) + np.cos(phase2) + rng.randn() * 0.5
        x[i] = 1.0 if val >= 0 else -1.0
    return x

def _local_search(x, time_limit):
    """Iterative local search: flip individual elements to reduce score."""
    current_score = _score(x)
    x = x.copy()
    start_time = time.time()
    
    # Precompute contributions for efficiency
    # When we flip x[i] -> -x[i], the change in autoconv is:
    # new_ac = ac - 2*x[i]*convolve(delta_i, x, 'full') where delta_i has 1 at position i
    # More precisely: ac_new[k] = ac[k] - 2*x[i]*sum_j x[j]*delta_i[j+k]
    # = ac[k] - 2*x[i]*x[i+k] for the shifted part
    
    # For efficiency, we'll just recompute score after each flip for simplicity
    # but use early termination
    
    no_improve_count = 0
    max_no_improve = 5000
    best_score = current_score
    best_x = x.copy()
    
    while time.time() - start_time < time_limit:
        improved = False
        # Try flipping random positions
        for _ in range(200):  # Try 200 random flips per iteration
            i = np.random.randint(0, N)
            x[i] = -x[i]
            new_score = _score(x)
            if new_score < current_score - 1e-15:
                current_score = new_score
                improved = True
                no_improve_count = 0
                if current_score < best_score:
                    best_score = current_score
                    best_x = x.copy()
            else:
                x[i] = -x[i]  # Revert
                no_improve_count += 1
        
        if not improved:
            # Try a few more structured moves
            for _ in range(100):
                i = np.random.randint(0, N)
                j = (i + np.random.randint(1, N)) % N
                # Flip two positions
                x[i] = -x[i]
                x[j] = -x[j]
                new_score = _score(x)
                if new_score < current_score - 1e-15:
                    current_score = new_score
                    improved = True
                    no_improve_count = 0
                    if current_score < best_score:
                        best_score = current_score
                        best_x = x.copy()
                else:
                    x[i] = -x[i]
                    x[j] = -x[j]
                    no_improve_count += 1
            
            if not improved:
                no_improve_count += 100
        
        if no_improve_count > max_no_improve:
            break
    
    return best_x, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin
    
    np.random.seed(42)
    
    best_score = float('inf')
    best_x = None
    
    # Phase 1: Generate diverse initial candidates
    candidates = []
    
    # Random sequences
    for _ in range(5):
        candidates.append(_random_signed())
    
    # Balanced random
    for _ in range(5):
        candidates.append(_random_balanced())
    
    # Structured sequences
    for seed_val in range(10):
        candidates.append(_structured_sequence(seed_val))
    
    # Evaluate all candidates
    for x in candidates:
        if time.time() > deadline:
            break
        s = _score(x)
        if s < best_score:
            best_score = s
            best_x = x.copy()
    
    # Phase 2: Local search on best candidates
    # Sort candidates by score, search the best ones
    scored_candidates = []
    for x in candidates:
        s = _score(x)
        scored_candidates.append((s, x))
    scored_candidates.sort(key=lambda t: t[0])
    
    # Local search on top candidates
    time_per_search = 8.0  # seconds per local search
    for s, x in scored_candidates[:5]:  # Top 5
        if time.time() > deadline - time_per_search - 5:
            break
        remaining = deadline - time.time() - 5.0
        if remaining < 2.0:
            break
        new_x, new_s = _local_search(x, remaining)
        if new_s < best_score:
            best_score = new_s
            best_x = new_x
    
    # Phase 3: If time remains, try more local search iterations on the best
    if best_x is not None and time.time() < deadline - 10.0:
        remaining = deadline - time.time() - 5.0
        if remaining > 5.0:
            new_x, new_s = _local_search(best_x, remaining)
            if new_s < best_score:
                best_score = new_s
                best_x = new_x
    
    # Phase 4: Final polish - try flipping single elements on best solution
    if best_x is not None and time.time() < deadline - 2.0:
        x = best_x.copy()
        s = best_score
        while time.time() < deadline - 2.0:
            improved = False
            for _ in range(50):
                i = np.random.randint(0, N)
                x[i] = -x[i]
                new_s = _score(x)
                if new_s < s - 1e-15:
                    s = new_s
                    improved = True
                    if s < best_score:
                        best_score = s
                        best_x = x.copy()
                else:
                    x[i] = -x[i]
            if not improved:
                break
    
    if best_x is None:
        return np.ones(N)
    
    return best_x
