"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def _is_sidon(subset):
    """Check if a subset is Sidon (all pair sums including diagonals are unique)."""
    n = len(subset)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _largest_sidon_size(arr):
    """Find the size of the largest Sidon subset of arr (12 elements)."""
    n = len(arr)
    # Try from largest possible down
    for k in range(n, 1, -1):
        for combo in combinations(arr, k):
            if _is_sidon(combo):
                return k
    return 1  # Any single element is Sidon

def _count_collision_pairs(arr):
    """Count the number of violating quadruples (a,b,c,d) where a+b=c+d with {a,b}!= {c,d}."""
    sums = {}
    for i in range(len(arr)):
        for j in range(i, len(arr)):
            s = arr[i] + arr[j]
            if s in sums:
                sums[s] += 1
            else:
                sums[s] = 1
    # Count pairs of pairs with same sum
    collisions = 0
    for s, cnt in sums.items():
        if cnt > 1:
            collisions += cnt * (cnt - 1) // 2
    return collisions

def _generate_structured_candidates():
    """Generate structured candidate sets with high collision density."""
    candidates = []
    
    # Arithmetic progressions and near-progressions
    for start in range(0, 20):
        for step in [1, 2, 3]:
            if start + 11 * step <= 30:
                ap = [start + i * step for i in range(12)]
                candidates.append(ap)
    
    # Dense clusters in small ranges
    for base in range(0, 15):
        # Cluster around base with small offsets
        offsets = list(range(12))
        cluster = [base + o for o in offsets if base + o <= 30]
        if len(cluster) == 12:
            candidates.append(cluster)
    
    # Mixed: some AP, some random
    random.seed(123)
    for _ in range(50):
        base = random.randint(0, 10)
        set_vals = set()
        while len(set_vals) < 12:
            # Mix of structured and random
            if random.random() < 0.5:
                val = base + random.randint(0, 8)
            else:
                val = random.randint(0, 30)
            if val not in set_vals and 0 <= val <= 30:
                set_vals.add(val)
        candidates.append(list(set_vals))
    
    return candidates

def construct():
    """Find 12 distinct integers in [0,128] minimizing the largest Sidon subset size."""
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    # Phase 1: Evaluate structured candidates
    best_set = None
    best_score = 13  # Max possible is 12, use 13 as infinity
    
    structured = _generate_structured_candidates()
    for cand in structured:
        if len(set(cand)) != 12:
            continue
        if any(x < 0 or x > 128 for x in cand):
            continue
        score = _largest_sidon_size(cand)
        if score < best_score:
            best_score = score
            best_set = cand[:]
        if best_score <= 4:
            break
    
    # If no structured candidate worked, use random
    if best_set is None:
        best_set = list(range(12))
        best_score = _largest_sidon_size(best_set)
    
    # Phase 2: Local search to improve
    random.seed(42)
    current = best_set[:]
    current_score = best_score
    
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        improved = False
        # Try random swaps within small range [0, 35] to maintain density
        for _ in range(2000):
            if time.time() >= deadline:
                break
            i = random.randint(0, 11)
            # Try swapping current[i] with a random value in dense range
            new_val = random.randint(0, 35)
            if new_val in current:
                continue
            
            old_val = current[i]
            current[i] = new_val
            new_score = _largest_sidon_size(current)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
                if new_score < best_score:
                    best_score = new_score
                    best_set = current[:]
                    no_improve_count = 0
                    if best_score <= 4:
                        break
            else:
                # Revert
                current[i] = old_val
        
        if not improved:
            no_improve_count += 1
        else:
            no_improve_count = 0
        
        # If we've found a good score, do more targeted search
        if best_score <= 4:
            break
    
    # Verify best_set is valid
    assert len(set(best_set)) == 12
    assert all(0 <= x <= 128 for x in best_set)
    
    return best_set
# EVOLVE-BLOCK-END
