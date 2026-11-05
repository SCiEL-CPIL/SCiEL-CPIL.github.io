import random
import time
import math

def calc_max_disc(signs, n=256, limit=None):
    """Compute max absolute signed sum over any AP in [0, n)."""
    max_disc = 1
    for d in range(1, n):
        for r in range(d):
            curr = 0
            min_p = 0
            max_p = 0
            idx = r
            while idx < n:
                curr += signs[idx]
                if curr < min_p:
                    min_p = curr
                if curr > max_p:
                    max_p = curr
                idx += d
            disc = max_p - min_p
            if disc > max_disc:
                max_disc = disc
                if limit is not None and max_disc > limit:
                    return max_disc
    return max_disc

def calc_max_disc_fast(signs, n=256):
    """Faster: only check steps up to a reasonable bound for initial random, but full for final.
    Actually for correctness we need all. But we can optimize inner loop."""
    max_disc = 1
    for d in range(1, n):
        for r in range(d):
            curr = 0
            min_p = 0
            max_p = 0
            idx = r
            while idx < n:
                curr += signs[idx]
                if curr < min_p:
                    min_p = curr
                if curr > max_p:
                    max_p = curr
                idx += d
            disc = max_p - min_p
            if disc > max_disc:
                max_disc = disc
    return max_disc

def get_worst_aps(signs, n=256, top_k=10):
    """Return list of (disc, d, r) for worst APs."""
    results = []
    for d in range(1, n):
        for r in range(d):
            curr = 0
            min_p = 0
            max_p = 0
            idx = r
            while idx < n:
                curr += signs[idx]
                if curr < min_p:
                    min_p = curr
                if curr > max_p:
                    max_p = curr
                idx += d
            disc = max_p - min_p
            results.append((disc, d, r))
    results.sort(reverse=True)
    return results[:top_k]

def greedy_repair(signs, n=256, max_iters=200, time_budget=10.0):
    """Try to reduce discrepancy by flipping signs that appear in worst APs."""
    start = time.time()
    current_disc = calc_max_disc(signs)
    best_signs = signs[:]
    best_disc = current_disc
    
    for it in range(max_iters):
        if time.time() - start > time_budget:
            break
        if current_disc <= 1:
            break
            
        # Get worst APs
        worst = get_worst_aps(signs, n, top_k=5)
        if not worst or worst[0][0] <= 1:
            break
            
        # Collect indices from worst APs
        idx_set = set()
        for disc, d, r in worst:
            idx = r
            while idx < n:
                idx_set.add(idx)
                idx += d
        
        # Try flipping each index in the set
        best_flip = None
        best_new_disc = current_disc
        for idx in idx_set:
            signs[idx] *= -1
            # Quick check with limit
            new_disc = calc_max_disc(signs, n, limit=current_disc)
            signs[idx] *= -1
            if new_disc < best_new_disc:
                best_new_disc = new_disc
                best_flip = idx
        
        if best_flip is not None:
            signs[best_flip] *= -1
            current_disc = best_new_disc
            if current_disc < best_disc:
                best_disc = current_disc
                best_signs = signs[:]
        else:
            break
            
    return best_signs, best_disc

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55  # 55s margin for 60s limit
    
    # Initial solution: use a structured pattern then randomize slightly
    # Try Thue-Morse like or just random
    random.seed(42)  # Deterministic for reproducibility in initial
    signs = [random.choice((-1, 1)) for _ in range(n)]
    current_disc = calc_max_disc(signs)
    best_disc = current_disc
    best_signs = signs[:]
    
    # Simulated Annealing with multi-point flips
    T = 8.0
    T_min = 0.01
    cooling = 0.9995
    
    # Track consecutive rejections for adaptive restart
    consec_rej = 0
    
    while time.time() < deadline:
        # Decide flip size: mostly 1, sometimes 2 or 3
        if random.random() < 0.7:
            flip_count = 1
        elif random.random() < 0.8:
            flip_count = 2
        else:
            flip_count = 3
            
        # Pick indices to flip
        if flip_count == 1:
            # Bias towards indices in worst APs
            if random.random() < 0.5:
                worst = get_worst_aps(signs, n, top_k=3)
                idx_set = set()
                for disc, d, r in worst:
                    idx = r
                    while idx < n:
                        idx_set.add(idx)
                        idx += d
                if idx_set:
                    i = random.choice(list(idx_set))
                else:
                    i = random.randint(0, n - 1)
            else:
                i = random.randint(0, n - 1)
            indices = [i]
        else:
            indices = random.sample(range(n), flip_count)
        
        # Apply flips
        for idx in indices:
            signs[idx] *= -1
            
        # Calculate new discrepancy
        limit = current_disc + int(T * 5) + 1
        new_disc = calc_max_disc(signs, n, limit)
        
        delta = new_disc - current_disc
        
        if delta <= 0 or random.random() < math.exp(-delta / T):
            # Accept
            current_disc = new_disc
            if current_disc < best_disc:
                best_disc = current_disc
                best_signs = signs[:]
            consec_rej = 0
        else:
            # Reject
            for idx in indices:
                signs[idx] *= -1
            consec_rej += 1
            
        T *= cooling
        if T < T_min:
            T = T_min
        
        # Periodic greedy repair
        if consec_rej > 500 or (consec_rej > 200 and time.time() < deadline - 15):
            # Do a greedy repair
            repaired, rep_disc = greedy_repair(signs, n, max_iters=50, time_budget=5.0)
            if rep_disc < current_disc:
                signs = repaired
                current_disc = rep_disc
                if current_disc < best_disc:
                    best_disc = current_disc
                    best_signs = signs[:]
            consec_rej = 0
            T = max(T, 2.0)  # Reheat slightly
    
    # Final polish: greedy repair with more time
    remaining = deadline - time.time()
    if remaining > 2:
        repaired, rep_disc = greedy_repair(signs, n, max_iters=500, time_budget=max(1.0, remaining - 1))
        if rep_disc < best_disc:
            best_signs = repaired
            best_disc = rep_disc
    
    return best_signs
