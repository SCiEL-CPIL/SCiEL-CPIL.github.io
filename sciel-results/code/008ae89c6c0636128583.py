import random
import time
from itertools import combinations

def _is_sidon(subset):
    """Check if subset has all unique pair sums (including diagonals)."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(S):
    """Find the largest Sidon subset size of S."""
    n = len(S)
    for k in range(n, 0, -1):
        for comb in combinations(S, k):
            if _is_sidon(list(comb)):
                return k
    return 0

def _collision_score(S):
    """Count total number of sum collisions (pairs of pairs that share a sum).
    Higher score = more collisions = harder to find large Sidon subsets."""
    sums = {}
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            sums[s] = sums.get(s, 0) + 1
    # Count collisions: for each sum that appears c times, C(c,2) is the number
    # of colliding pairs
    collisions = 0
    for c in sums.values():
        if c > 1:
            collisions += c * (c - 1) // 2
    return collisions

def _max_sidon_size_fast(S):
    """Faster check: only check k=6 and k=5 (we expect max to be 5 or less)."""
    n = len(S)
    # Check k=6 first (most likely to be the max if >5)
    for comb in combinations(S, 6):
        if _is_sidon(list(comb)):
            return 6
    # Check k=5
    for comb in combinations(S, 5):
        if _is_sidon(list(comb)):
            return 5
    # Check k=4
    for comb in combinations(S, 4):
        if _is_sidon(list(comb)):
            return 4
    return 3

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    random.seed(42)
    
    # Start with a dense set in a small range to maximize collisions
    # Try to find a set where max Sidon size is 5
    best_set = list(range(12))
    best_size = _max_sidon_size_fast(best_set)
    
    # First, do a quick random search for a good starting point
    for _ in range(200):
        if time.time() > deadline:
            break
        # Dense random set in [0, 30]
        s = random.sample(range(31), 12)
        s.sort()
        size = _max_sidon_size_fast(s)
        if size < best_size:
            best_size = size
            best_set = s
            if best_size <= 5:
                break
    
    # Simulated annealing to minimize max Sidon size
    current_set = best_set[:]
    current_size = best_size
    current_collisions = _collision_score(current_set)
    
    T = 10.0  # Initial temperature
    T_min = 0.01
    cooling = 0.999
    
    for iteration in range(50000):
        if time.time() > deadline:
            break
        
        # Pick a random element to swap
        idx = random.randint(0, 11)
        old_val = current_set[idx]
        
        # Try a new value: either nearby or random
        if random.random() < 0.7:
            # Local move: shift by small amount
            new_val = old_val + random.randint(-5, 5)
            new_val = max(0, min(128, new_val))
        else:
            # Global move
            new_val = random.randint(0, 128)
        
        if new_val in current_set:
            continue
        
        # Create new set
        new_set = current_set[:]
        new_set[idx] = new_val
        new_set.sort()
        
        # Evaluate
        new_collisions = _collision_score(new_set)
        delta_collisions = new_collisions - current_collisions
        
        # We want to INCREASE collisions (make it harder to find Sidon subsets)
        # So we accept if collisions increase, or with some probability if they decrease
        if delta_collisions >= 0:
            # Accept: collisions increased or stayed same
            current_set = new_set
            current_collisions = new_collisions
            # Check actual max Sidon size occasionally
            if iteration % 10 == 0 or delta_collisions > 5:
                new_size = _max_sidon_size_fast(new_set)
                if new_size < best_size:
                    best_size = new_size
                    best_set = new_set[:]
                    if best_size <= 5:
                        # Great! We found a set with max Sidon size 5
                        # Try to push it lower
                        T = 5.0
                current_size = new_size
        else:
            # Collisions decreased: accept with probability exp(delta/T)
            # delta is negative, so exp(negative/T) < 1
            prob = min(1.0, max(0.0, 2.718281828 ** (delta_collisions / T)))
            if random.random() < prob:
                current_set = new_set
                current_collisions = new_collisions
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Final verification and local polish
    final_size = _max_sidon_size_fast(best_set)
    if final_size < best_size:
        best_size = final_size
    
    # Try a few more targeted improvements if we have time
    if time.time() < deadline - 5:
        for _ in range(500):
            if time.time() > deadline - 2:
                break
            idx = random.randint(0, 11)
            old_val = best_set[idx]
            for offset in range(-3, 4):
                new_val = old_val + offset
                if new_val < 0 or new_val > 128:
                    continue
                if new_val in best_set:
                    continue
                new_set = best_set[:]
                new_set[idx] = new_val
                new_set.sort()
                new_size = _max_sidon_size_fast(new_set)
                if new_size < best_size:
                    best_size = new_size
                    best_set = new_set
                    break
    
    return best_set
