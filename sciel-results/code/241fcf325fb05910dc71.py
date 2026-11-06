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
            if start + 11 * step <= 128:
                ap = [start + i * step for i in range(12)]
                candidates.append(ap)
    
    # Dense clusters in small ranges
    for base in range(0, 120):
        cluster = [base + o for o in range(12)]
        if all(0 <= x <= 128 for x in cluster):
            candidates.append(cluster)
    
    # Mixed: some AP, some random
    random.seed(123)
    for _ in range(100):
        base = random.randint(0, 10)
        set_vals = set()
        while len(set_vals) < 12:
            # Mix of structured and random
            if random.random() < 0.5:
                val = base + random.randint(0, 8)
            else:
                val = random.randint(0, 30)
            if val not in set_vals and 0 <= val <= 128:
                set_vals.add(val)
        candidates.append(list(set_vals))
    
    # Additional structured candidates: multiple AP segments
    for _ in range(50):
        set_vals = set()
        while len(set_vals) < 12:
            seg_start = random.randint(0, 110)
            seg_step = random.randint(1, 3)
            seg_len = random.randint(3, 6)
            for i in range(seg_len):
                v = seg_start + i * seg_step
                if 0 <= v <= 128 and v not in set_vals:
                    set_vals.add(v)
                    if len(set_vals) >= 12:
                        break
        if len(set_vals) == 12:
            candidates.append(list(set_vals))
    
    return candidates

def construct():
    """Find 12 distinct integers in [0,128] minimizing the largest Sidon subset size."""
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill
    
    # Phase 1: Evaluate structured candidates
    best_set = None
    best_score = 13  # Max possible is 12, use 13 as infinity
    best_collisions = -1
    
    structured = _generate_structured_candidates()
    for cand in structured:
        if len(set(cand)) != 12:
            continue
        if any(x < 0 or x > 128 for x in cand):
            continue
        coll = _count_collision_pairs(cand)
        # Use collision count as a fast filter first
        if coll > best_collisions or best_set is None:
            score = _largest_sidon_size(cand)
            if score < best_score:
                best_score = score
                best_set = cand[:]
                best_collisions = coll
        elif coll >= best_collisions:
            score = _largest_sidon_size(cand)
            if score < best_score:
                best_score = score
                best_set = cand[:]
                best_collisions = coll
        if best_score <= 4:
            break
    
    # If no structured candidate worked, use random
    if best_set is None:
        best_set = list(range(12))
        best_score = _largest_sidon_size(best_set)
        best_collisions = _count_collision_pairs(best_set)
    
    # Phase 2: Local search to improve
    rng = random.Random(42)
    current = best_set[:]
    current_score = best_score
    current_coll = best_collisions
    
    no_improve_count = 0
    max_no_improve = 300
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        improved = False
        
        # Dynamic inner loop size based on remaining time
        remaining = deadline - time.time()
        inner_steps = min(2000, int(remaining * 30))
        if inner_steps < 10:
            break
        
        for _ in range(inner_steps):
            if time.time() >= deadline - 2:
                break
            i = rng.randint(0, 11)
            # Try swapping current[i] with a random value in dense range
            # Use a mix of dense local moves and occasional global moves
            if rng.random() < 0.7:
                new_val = rng.randint(0, 40)
            else:
                new_val = rng.randint(0, 128)
            
            if new_val in current:
                continue
            
            old_val = current[i]
            current[i] = new_val
            new_coll = _count_collision_pairs(current)
            
            # Only compute expensive _largest_sidon_size if collision count improved
            # or periodically
            if new_coll > current_coll:
                new_score = _largest_sidon_size(current)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if new_score < best_score:
                        best_score = new_score
                        best_set = current[:]
                        best_collisions = new_coll
                        no_improve_count = 0
                        if best_score <= 4:
                            break
                else:
                    no_improve_count += 1
                current_coll = new_coll
            elif new_coll == current_coll:
                # Occasionally check score even if collisions didn't change
                if rng.random() < 0.1:
                    new_score = _largest_sidon_size(current)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        if new_score < best_score:
                            best_score = new_score
                            best_set = current[:]
                            best_collisions = new_coll
                            no_improve_count = 0
                            if best_score <= 4:
                                break
                    else:
                        no_improve_count += 1
            else:
                # Revert
                current[i] = old_val
                no_improve_count += 1
        
        if not improved:
            no_improve_count += 1
        else:
            no_improve_count = 0
        
        # If we've found a good score, do more targeted search
        if best_score <= 4:
            break
        
        # Restart strategy when stuck
        if no_improve_count > 100:
            if rng.random() < 0.5:
                # Perturb best set
                current = best_set[:]
                for _ in range(rng.randint(2, 5)):
                    i = rng.randrange(12)
                    new_val = rng.randint(0, 128)
                    while new_val in current:
                        new_val = rng.randint(0, 128)
                    current[i] = new_val
            else:
                # Random restart with dense bias
                current = []
                while len(current) < 12:
                    if rng.random() < 0.6:
                        v = rng.randint(0, 35)
                    else:
                        v = rng.randint(0, 128)
                    if v not in current:
                        current.append(v)
            
            current_score = _largest_sidon_size(current)
            current_coll = _count_collision_pairs(current)
            no_improve_count = 0
    
    # Final polish: try all single-element swaps from best_set
    if time.time() < deadline - 5:
        for i in range(12):
            if time.time() >= deadline - 2:
                break
            old_val = best_set[i]
            for new_val in range(0, 129):
                if new_val == old_val or new_val in best_set:
                    continue
                candidate = best_set[:]
                candidate[i] = new_val
                coll = _count_collision_pairs(candidate)
                if coll > best_collisions:
                    score = _largest_sidon_size(candidate)
                    if score < best_score:
                        best_score = score
                        best_set = candidate
                        best_collisions = coll
                        if best_score <= 4:
                            break
            if best_score <= 4:
                break
    
    # Verify best_set is valid
    assert len(set(best_set)) == 12
    assert all(0 <= x <= 128 for x in best_set)
    
    return best_set
# EVOLVE-BLOCK-END
