"""Optimized solver for minimizing max Sidon subset size in a 12-element set."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 60s kill

    def max_sidon_size(arr):
        """Exact largest Sidon subset size using bitmask DP."""
        n = len(arr)
        arr_sorted = sorted(arr)
        
        # Precompute: for each element i, which other elements j (j>i) 
        # have a sum collision with element i in a set S?
        # For a candidate subset represented as bitmask, check if adding element i
        # creates a collision.
        
        # For each pair (i,j) with i<j, store the sum arr[i]+arr[j]
        # For diagonal: arr[i]+arr[i]
        # A collision occurs if two different pairs (i,j) and (k,l) have the same sum
        # where the pairs are distinct as unordered pairs.
        
        # Key insight: for a set S, it's Sidon iff all sums a+b (a<=b, a,b in S) are unique.
        # Number of such sums = |S|*(|S|+1)/2.
        
        # We'll use backtracking with memoization on (idx, frozenset of used sums) - too expensive.
        # Instead, use backtracking with pruning.
        
        best_size = 0
        
        def backtrack(idx, current, current_sums):
            nonlocal best_size
            k = len(current)
            if k > best_size:
                best_size = k
            
            if idx >= n:
                return
            
            # Pruning: can't beat best_size
            if k + (n - idx) <= best_size:
                return
            
            for i in range(idx, n):
                elem = arr_sorted[i]
                # Check if adding elem keeps Sidon property
                # New sums: elem+elem, and elem+current[j] for each j in current
                # These must not collide with existing sums or each other
                new_sums = [elem + elem]
                for j in current:
                    new_sums.append(elem + j)
                
                # Check for collisions within new_sums
                if len(set(new_sums)) != len(new_sums):
                    continue
                # Check collisions with existing sums
                if any(s in current_sums for s in new_sums):
                    continue
                
                backtrack(i + 1, current + [elem], current_sums | set(new_sums))
        
        backtrack(0, [], set())
        return best_size

    def greedy_collision_set():
        """Greedily build a set maximizing pair-sum collisions."""
        used = set()
        result = []
        # Start with 0 to anchor
        result.append(0)
        used.add(0)
        
        all_sums = {}  # sum -> count of pairs producing it
        
        for _ in range(11):
            best_val = None
            best_collisions = -1
            for v in range(1, 129):
                if v in used:
                    continue
                # Count how many new sums would collide with existing sums
                collisions = 0
                for u in result:
                    s = v + u
                    if s in all_sums:
                        collisions += 1
                    # Also check diagonal
                s_diag = v + v
                if s_diag in all_sums:
                    collisions += 1
                # Also check if any two new sums collide with each other
                new_sums = [v + u for u in result] + [v + v]
                if len(set(new_sums)) != len(new_sums):
                    collisions += 10  # penalty for internal collision
                
                if collisions > best_collisions:
                    best_collisions = collisions
                    best_val = v
            
            if best_val is not None:
                result.append(best_val)
                used.add(best_val)
                # Update all_sums
                for u in result[:-1]:
                    s = best_val + u
                    all_sums[s] = all_sums.get(s, 0) + 1
                s_diag = best_val + best_val
                all_sums[s_diag] = all_sums.get(s_diag, 0) + 1
        
        return result

    def local_search(initial_set, max_iters=20000):
        """Local search: try single-element replacements to reduce max Sidon size."""
        current = sorted(initial_set)
        current_score = max_sidon_size(current)
        best = current[:]
        best_score = current_score
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            # Try replacing one element
            for idx in range(12):
                if time.time() > deadline:
                    break
                old_val = current[idx]
                # Try a few candidate replacements
                for new_val in random.sample(range(0, 129), 8):
                    if new_val == old_val or new_val in set(current):
                        continue
                    
                    trial = current[:]
                    trial[idx] = new_val
                    trial.sort()
                    
                    s = max_sidon_size(trial)
                    if s < current_score:
                        current = trial
                        current_score = s
                        if s < best_score:
                            best_score = s
                            best = trial[:]
                        improved = True
                        break
                if improved:
                    break
            
            if not improved and iteration % 50 == 0:
                # Occasionally restart with a new greedy set
                if random.random() < 0.3:
                    new_greedy = greedy_collision_set()
                    s = max_sidon_size(new_greedy)
                    if s < current_score:
                        current = sorted(new_greedy)
                        current_score = s
                        if s < best_score:
                            best_score = s
                            best = current[:]
        
        return best, best_score

    # Phase 1: Generate diverse starting points
    random.seed(42)
    candidates = []
    
    # Greedy collision set
    g1 = greedy_collision_set()
    candidates.append(g1)
    
    # AP-based sets (arithmetic progressions create many collisions)
    candidates.append(list(range(0, 12)))
    candidates.append(list(range(0, 24, 2)))
    candidates.append([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15])
    candidates.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    
    # Sets with many pairs summing to same value
    # e.g., {0,1,2,...,5} and {10,11,12,...,15} have lots of cross-sums
    candidates.append([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15])
    candidates.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    
    # Modular structure: multiples that create sum collisions
    candidates.append([0, 2, 4, 6, 8, 10, 20, 22, 24, 26, 28, 30])
    candidates.append([0, 1, 3, 4, 6, 7, 10, 11, 13, 14, 16, 17])
    
    # Random sets with clustering
    for _ in range(5):
        base = random.randint(0, 20)
        spread = random.randint(1, 3)
        cand = []
        for i in range(12):
            v = base + i * spread + random.randint(-1, 1)
            v = max(0, min(128, v))
            cand.append(v)
        # Ensure distinct
        cand = list(set(cand))
        while len(cand) < 12:
            v = random.randint(0, 128)
            if v not in cand:
                cand.append(v)
        cand = cand[:12]
        candidates.append(cand)
    
    # Phase 2: Local search from each candidate
    best_overall = None
    best_overall_score = 12
    
    for cand in candidates:
        if time.time() > deadline:
            break
        if len(set(cand)) != 12 or any(x < 0 or x > 128 for x in cand):
            continue
        
        result, score = local_search(cand, max_iters=3000)
        if score < best_overall_score:
            best_overall_score = score
            best_overall = result
    
    if best_overall is None:
        best_overall = list(range(12))
        best_overall_score = 12
    
    return best_overall
# EVOLVE-BLOCK-END
