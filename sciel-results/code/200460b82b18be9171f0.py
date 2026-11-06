import time
import random
from itertools import combinations

def is_sidon(subset):
    """Check if subset has all unordered pair sums unique (including diagonals)."""
    n = len(subset)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def is_sidon_fast(subset):
    """Faster Sidon check using sorted array and early exit."""
    n = len(subset)
    if n < 2:
        return True
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(S):
    """Find the largest Sidon subset size of S using backtracking with pruning."""
    S = sorted(S)
    n = len(S)
    
    # First check if the whole set is Sidon
    if is_sidon_fast(S):
        return n
    
    # Try k = n-1, n-2, ... down to 2
    for k in range(n - 1, 1, -1):
        found = _find_sidon_of_size(S, k)
        if found:
            return k
    return 2

def _find_sidon_of_size(S, k):
    """Backtracking search for a Sidon subset of exactly size k."""
    n = len(S)
    if k > n:
        return False
    if k <= 1:
        return True
    
    # Use backtracking with pruning
    best = [False]
    
    def backtrack(start, current, sums):
        if best[0]:
            return
        if len(current) == k:
            best[0] = True
            return
        # Pruning: not enough elements left
        if len(current) + (n - start) < k:
            return
        # Try adding each remaining element
        for i in range(start, n):
            elem = S[i]
            # Check if adding elem keeps Sidon property
            valid = True
            for j in range(len(current)):
                s = elem + current[j]
                if s in sums:
                    valid = False
                    break
            if valid:
                s = elem + elem
                if s in sums:
                    valid = False
            if valid:
                # Add elem
                sums.add(elem + elem)
                for j in range(len(current)):
                    sums.add(elem + current[j])
                current.append(elem)
                
                backtrack(i + 1, current, sums)
                
                if best[0]:
                    return
                # Undo
                current.pop()
                for j in range(len(current)):
                    sums.discard(elem + current[j])
                sums.discard(elem + elem)
    
    backtrack(0, [], set())
    return best[0]

def count_sum_collisions(S):
    """Count the number of duplicate pair sums in S (higher = more collisions = harder for Sidon)."""
    sums = {}
    for i in range(len(S)):
        for j in range(i, len(S)):
            s = S[i] + S[j]
            sums[s] = sums.get(s, 0) + 1
    collisions = sum(v - 1 for v in sums.values() if v > 1)
    return collisions

def greedy_collision_set(deadline):
    """Build a set greedily by maximizing sum collisions."""
    # Start with elements that create many collisions
    # Strategy: pick elements close together to create sum overlaps
    
    best_set = list(range(12))
    best_collisions = count_sum_collisions(best_set)
    
    start_time = time.time()
    
    # Try multiple random restarts of greedy construction
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Random initial pair
        a = random.randint(0, 128)
        b = random.randint(0, 128)
        if a == b:
            continue
        current = [a, b]
        used = {a, b}
        
        # Greedily add elements that maximize collisions with existing set
        for _ in range(10):
            if time.time() > deadline:
                break
            
            best_elem = None
            best_score = -1
            
            # Candidate elements: near existing elements and random
            candidates = set()
            for x in current:
                for d in range(-5, 6):
                    y = x + d
                    if 0 <= y <= 128 and y not in used:
                        candidates.add(y)
            for _ in range(20):
                y = random.randint(0, 128)
                if y not in used:
                    candidates.add(y)
            
            for elem in candidates:
                new_set = current + [elem]
                coll = count_sum_collisions(new_set)
                if coll > best_score:
                    best_score = coll
                    best_elem = elem
            
            if best_elem is not None:
                current.append(best_elem)
                used.add(best_elem)
        
        if len(current) == 12:
            coll = count_sum_collisions(current)
            if coll > best_collisions:
                best_collisions = coll
                best_set = current[:]
    
    return best_set

def structured_collision_sets():
    """Generate structured sets known to have many sum collisions."""
    candidates = []
    
    # 1. Dense clusters: groups of close numbers
    for base in range(0, 120, 5):
        s = set()
        # Add 3-4 clusters of 3-4 close numbers
        remaining = 12
        while remaining > 0 and len(s) < 12:
            cluster_start = random.randint(0, 128)
            cluster_size = min(remaining, random.choice([2, 3, 4]))
            for i in range(cluster_size):
                val = cluster_start + i
                if 0 <= val <= 128:
                    s.add(val)
            remaining = 12 - len(s)
        if len(s) == 12:
            candidates.append(list(s))
    
    # 2. Symmetric pairs around a center
    for center in range(20, 110, 5):
        s = set()
        pairs = 6
        for i in range(pairs):
            d = random.randint(1, 15)
            if center - d >= 0 and center + d <= 128:
                s.add(center - d)
                s.add(center + d)
        while len(s) < 12:
            s.add(random.randint(0, 128))
        if len(s) == 12:
            candidates.append(list(s))
    
    # 3. Modular arithmetic patterns (mod small m)
    for m in [3, 4, 5, 6, 7]:
        for offset in range(m):
            s = set()
            # Pick values that are all ≡ offset (mod m) or close
            for _ in range(12):
                val = offset + m * random.randint(0, (128 - offset) // m)
                s.add(val)
            if len(s) == 12:
                candidates.append(list(s))
    
    # 4. Quadratic residues / structured
    for a in range(1, 10):
        for b in range(1, 10):
            s = set()
            for i in range(12):
                val = a * i * i + b * i
                if 0 <= val <= 128:
                    s.add(val)
            while len(s) < 12:
                s.add(random.randint(0, 128))
            if len(s) == 12:
                candidates.append(list(s))
    
    return candidates

def local_search_improve(S, deadline):
    """Try to improve a set by swapping elements to increase collisions."""
    S = sorted(S)
    best_score = count_sum_collisions(S)
    best_set = S[:]
    start_time = time.time()
    
    for _ in range(500):
        if time.time() > deadline:
            break
        
        # Pick a random element to replace
        idx = random.randint(0, 11)
        old_val = S[idx]
        
        # Try replacing with nearby values
        for new_val in range(max(0, old_val - 10), min(129, old_val + 11)):
            if new_val == old_val:
                continue
            if new_val in S:
                continue
            
            new_S = S[:]
            new_S[idx] = new_val
            new_S.sort()
            
            coll = count_sum_collisions(new_S)
            if coll > best_score:
                best_score = coll
                best_set = new_S[:]
                S = new_S
                break
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safe margin before 90s kill
    
    # Initial best: arithmetic progression
    best_set = list(range(12))
    best_score = max_sidon_size(best_set)
    
    # Phase 1: Generate many collision-heavy candidates
    candidates = []
    
    # Structured collision sets
    candidates.extend(structured_collision_sets())
    
    # Greedy collision set
    greedy_set = greedy_collision_set(deadline)
    candidates.append(greedy_set)
    
    # Random sets (some may have good collision structure)
    for _ in range(500):
        candidates.append(random.sample(range(129), 12))
    
    # AP-based
    for step in range(1, 11):
        if 11 * step <= 128:
            candidates.append(list(range(0, 12 * step, step)))
    
    # Evaluate candidates
    for candidate in candidates:
        if time.time() > deadline:
            break
        
        if len(set(candidate)) != 12:
            continue
        if any(x < 0 or x > 128 for x in candidate):
            continue
        
        score = max_sidon_size(candidate)
        
        if score < best_score:
            best_score = score
            best_set = candidate[:]
            if best_score == 2:
                break
    
    # Phase 2: Local search on best found
    if time.time() < deadline - 5:
        improved = local_search_improve(best_set, deadline)
        score = max_sidon_size(improved)
        if score < best_score:
            best_score = score
            best_set = improved
    
    # Phase 3: More greedy restarts if time allows
    if time.time() < deadline - 10:
        for _ in range(50):
            if time.time() > deadline:
                break
            g = greedy_collision_set(deadline)
            if len(set(g)) == 12:
                score = max_sidon_size(g)
                if score < best_score:
                    best_score = score
                    best_set = g[:]
                    if best_score == 2:
                        break
    
    return best_set
