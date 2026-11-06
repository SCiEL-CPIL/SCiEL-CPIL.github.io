"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def _score_set(s):
    """Count distinct d > 0 such that a, a+d, a+2d are all in s for some a."""
    sset = set(s)
    diffs = set()
    for a in s:
        for d in range(1, 513 - 2*a):
            if a + d in sset and a + 2*d in sset:
                diffs.add(d)
    return len(diffs)

def _score_set_fast(s):
    """Faster scoring using numpy for the inner loop."""
    arr = np.array(sorted(s))
    sset = set(s)
    diffs = set()
    for i, a in enumerate(arr):
        # d must satisfy a+2d <= 512, so d <= (512-a)//2
        max_d = (512 - a) // 2
        if max_d <= 0:
            continue
        # For each d, check if a+d and a+2d are in s
        # We can vectorize: for d in 1..max_d, check membership
        # But numpy set membership is slow; let's use a boolean array
        # Actually, let's just use the set-based approach but with early termination
        for d in range(1, max_d + 1):
            if (a + d) in sset and (a + 2 * d) in sset:
                diffs.add(d)
    return len(diffs)

def _score_set_bitmask(s):
    """Score using a bitset representation for fast membership."""
    # Create a boolean array of size 513 (indices 0..512)
    bitset = np.zeros(513, dtype=bool)
    for x in s:
        bitset[x] = True
    diffs = set()
    for a in s:
        max_d = (512 - a) // 2
        if max_d <= 0:
            continue
        # Vectorized check: for d in 1..max_d, is bitset[a+d] and bitset[a+2d]?
        d_vals = np.arange(1, max_d + 1)
        # Check a+d
        idx1 = a + d_vals
        # Check a+2d
        idx2 = a + 2 * d_vals
        # Filter to valid indices
        valid = (idx1 <= 512) & (idx2 <= 512)
        if not np.any(valid):
            continue
        idx1_v = idx1[valid]
        idx2_v = idx2[valid]
        # Check membership
        mask = bitset[idx1_v] & bitset[idx2_v]
        if np.any(mask):
            d_found = d_vals[valid][mask]
            diffs.update(d_found.tolist())
    return len(diffs)

def _hill_climb(s, max_iter=5000, time_limit=10.0):
    """Hill-climbing with random restarts within a time limit."""
    start_time = time.time()
    best_score = _score_set_bitmask(s)
    best_s = list(s)
    current_s = list(s)
    current_score = best_score
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Try a random swap or move
        idx = random.randint(0, len(current_s) - 1)
        old_val = current_s[idx]
        
        # Try replacing with a random value not in the set
        new_val = random.randint(1, 512)
        if new_val in current_s:
            continue
        
        current_s[idx] = new_val
        new_score = _score_set_bitmask(current_s)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_s = list(current_s)
        else:
            # Revert
            current_s[idx] = old_val
    
    return best_s, best_score

def _simulated_annealing(s, max_iter=20000, time_limit=10.0):
    """Simulated annealing for global exploration."""
    start_time = time.time()
    best_score = _score_set_bitmask(s)
    best_s = list(s)
    current_s = list(s)
    current_score = best_score
    
    T_start = 1.0
    T_end = 0.01
    cooling = (T_end / T_start) ** (1.0 / max_iter)
    T = T_start
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        T *= cooling
        
        # Generate a neighbor: swap two elements or replace one
        if random.random() < 0.5:
            # Replace one element
            idx = random.randint(0, len(current_s) - 1)
            old_val = current_s[idx]
            new_val = random.randint(1, 512)
            if new_val in current_s:
                continue
            current_s[idx] = new_val
            new_score = _score_set_bitmask(current_s)
            delta = new_score - current_score
            if delta > 0 or random.random() < np.exp(delta / T):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_s = list(current_s)
            else:
                current_s[idx] = old_val
        else:
            # Swap two elements
            i, j = random.sample(range(len(current_s)), 2)
            current_s[i], current_s[j] = current_s[j], current_s[i]
            new_score = _score_set_bitmask(current_s)
            delta = new_score - current_score
            if delta > 0 or random.random() < np.exp(delta / T):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_s = list(current_s)
            else:
                current_s[i], current_s[j] = current_s[j], current_s[i]
    
    return best_s, best_score

def construct():
    """Construct a 32-element set in 1..512 maximizing distinct 3-AP differences."""
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    # Initial set: use a structured starting point
    # Start with a set that has some structure
    initial_set = list(range(1, 33))
    
    best_overall = list(initial_set)
    best_score_overall = _score_set_bitmask(initial_set)
    
    # Phase 1: Hill climbing from multiple random starts
    num_starts = 5
    for i in range(num_starts):
        if time.time() > deadline:
            break
        
        if i == 0:
            s = list(initial_set)
        else:
            s = [random.randint(1, 512) for _ in range(32)]
            # Ensure distinct
            while len(set(s)) < 32:
                s[random.randint(0, 31)] = random.randint(1, 512)
        
        s, score = _hill_climb(s, max_iter=3000, time_limit=10.0)
        if score > best_score_overall:
            best_score_overall = score
            best_overall = list(s)
    
    # Phase 2: Simulated annealing from the best found
    if time.time() < deadline:
        s, score = _simulated_annealing(best_overall, max_iter=30000, time_limit=30.0)
        if score > best_score_overall:
            best_score_overall = score
            best_overall = list(s)
    
    # Phase 3: More hill climbing from the best
    if time.time() < deadline:
        s, score = _hill_climb(best_overall, max_iter=5000, time_limit=30.0)
        if score > best_score_overall:
            best_score_overall = score
            best_overall = list(s)
    
    # Final: try a few more SA restarts if time permits
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        for _ in range(3):
            if time.time() > deadline:
                break
            # Perturb the best solution
            s = list(best_overall)
            for _ in range(5):
                idx = random.randint(0, 31)
                s[idx] = random.randint(1, 512)
            while len(set(s)) < 32:
                s[random.randint(0, 31)] = random.randint(1, 512)
            
            s, score = _simulated_annealing(s, max_iter=10000, time_limit=min(15.0, remaining_time / 3))
            if score > best_score_overall:
                best_score_overall = score
                best_overall = list(s)
            remaining_time = deadline - time.time()
    
    return best_overall
# EVOLVE-BLOCK-END
