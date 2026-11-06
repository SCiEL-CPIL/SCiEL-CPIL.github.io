import itertools
import time
import random

def _is_sidon(subset):
    """Check if a subset has all pair sums (including diagonals) unique."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_subset_size(S):
    """Find the largest Sidon subset of S using backtracking with pruning."""
    S = sorted(S)
    n = len(S)
    
    # Quick upper bound: if all pair sums are unique, it's Sidon
    if _is_sidon(S):
        return n
    
    # Try sizes from n-1 down to 1
    for size in range(n - 1, 0, -1):
        # Use backtracking with early termination
        def backtrack(start, current):
            if len(current) == size:
                return True
            remaining_needed = size - len(current)
            remaining_available = n - start
            if remaining_available < remaining_needed:
                return False
            
            for i in range(start, n):
                # Pruning: check if adding S[i] to current keeps it Sidon
                candidate = current + [S[i]]
                if _is_sidon(candidate):
                    if backtrack(i + 1, candidate):
                        return True
            return False
        
        if backtrack(0, []):
            return size
    
    return 1

def construct():
    start = time.time()
    deadline = start + 80  # Safety margin before 90s kill
    
    # Generate structured candidates that create many sum collisions
    candidates = []
    
    # 1. Arithmetic progressions (dense sum structure)
    for a in range(0, 100):
        for d in range(1, 12):
            c = [a + i * d for i in range(12)]
            if all(0 <= x <= 128 for x in c):
                candidates.append(tuple(c))
    
    # 2. Symmetric pairs around various centers
    for center in range(20, 110):
        for spread in range(1, 15):
            c = []
            for k in range(6):
                a = center - spread * (5 - k)
                b = center + spread * (5 - k)
                if 0 <= a <= 128 and 0 <= b <= 128 and a != b:
                    c.append(a)
                    c.append(b)
            if len(c) == 12 and len(set(c)) == 12:
                candidates.append(tuple(sorted(c)))
    
    # 3. Clustered sets (groups of close numbers)
    for base in range(0, 100, 3):
        for offset in range(0, 30):
            c = []
            for group in range(4):
                g_base = base + group * 25 + offset
                for k in range(3):
                    val = g_base + k
                    if 0 <= val <= 128 and val not in c:
                        c.append(val)
            if len(c) == 12:
                candidates.append(tuple(sorted(c)))
    
    # 4. Mixed: some AP, some random
    rng = random.Random(12345)
    for _ in range(500):
        # Start with an AP of 6, add 6 random
        a = rng.randint(0, 80)
        d = rng.randint(1, 8)
        c = [a + i * d for i in range(6)]
        remaining = [x for x in range(129) if x not in c]
        c.extend(rng.sample(remaining, 6))
        if all(0 <= x <= 128 for x in c) and len(set(c)) == 12:
            candidates.append(tuple(sorted(c)))
    
    # Deduplicate
    candidates = list(set(candidates))
    
    # Evaluate all candidates
    best_set = None
    best_size = 12
    
    for cand in candidates:
        if time.time() > deadline:
            break
        sz = max_sidon_subset_size(list(cand))
        if sz < best_size:
            best_size = sz
            best_set = list(cand)
            if best_size <= 4:
                break
    
    # If we found a good set, try to improve with local search
    if best_set is not None and best_size <= 5:
        current = sorted(best_set)
        best_score = best_size
        in_set = set(current)
        
        while time.time() < deadline:
            improved = False
            for i in range(12):
                if time.time() > deadline:
                    break
                old_val = current[i]
                for new_val in range(129):
                    if new_val in in_set:
                        continue
                    candidate = list(current)
                    candidate[i] = new_val
                    candidate = sorted(candidate)
                    score = max_sidon_subset_size(candidate)
                    if score < best_score:
                        best_score = score
                        best_set = candidate
                        in_set.discard(old_val)
                        in_set.add(new_val)
                        current = candidate
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
    
    # If no good set found, try random search
    if best_set is None or best_size > 4:
        if best_set is None:
            best_set = list(range(12))
            best_size = max_sidon_subset_size(best_set)
        
        for iteration in range(5000):
            if time.time() > deadline:
                break
            cand = sorted(rng.sample(range(129), 12))
            # Quick heuristic: count sum collisions
            sums = {}
            for i in range(12):
                for j in range(i, 12):
                    s = cand[i] + cand[j]
                    sums[s] = sums.get(s, 0) + 1
            max_collide = max(sums.values())
            if max_collide >= 3:  # Many collisions likely means small Sidon subset
                sz = max_sidon_subset_size(cand)
                if sz < best_size:
                    best_size = sz
                    best_set = cand
                    if best_size <= 3:
                        break
    
    return sorted(best_set)
