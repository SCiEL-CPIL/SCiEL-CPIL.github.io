"""Search for 12-element ambient set in [0,128] minimizing largest Sidon subset."""
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    def is_sidon(subset):
        """Check if subset has all unordered pair sums (with repetition) unique."""
        s = sorted(subset)
        sums = set()
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                sm = s[i] + s[j]
                if sm in sums:
                    return False
                sums.add(sm)
        return True

    def max_sidon_size_fast(ambient):
        """Find largest Sidon subset via checking all subsets from largest to smallest."""
        ambient = sorted(ambient)
        n = len(ambient)
        for size in range(n, 0, -1):
            for combo in itertools.combinations(range(n), size):
                s = [ambient[i] for i in combo]
                sums = set()
                ok = True
                for i in range(len(s)):
                    for j in range(i, len(s)):
                        sm = s[i] + s[j]
                        if sm in sums:
                            ok = False
                            break
                        sums.add(sm)
                    if not ok:
                        break
                if ok:
                    return size
        return 0

    # Key insight: to minimize max Sidon subset, we want many "conflicts"
    # A conflict is when 2a = b+c for distinct a,b,c in the set (3-term AP centered at a)
    # This means the diagonal sum of a equals the pair sum of b,c
    
    # Strategy: construct sets with many 3-term APs to force conflicts
    
    def count_3term_aps(ambient):
        """Count 3-term arithmetic progressions in ambient."""
        s = set(ambient)
        count = 0
        for a in ambient:
            for b in ambient:
                if a != b:
                    c = 2*b - a
                    if c in s and c != a and c != b:
                        count += 1
        return count // 2

    def count_conflicts(ambient):
        """Count all pair-sum conflicts (including diagonals)."""
        ambient = sorted(ambient)
        n = len(ambient)
        sum_count = {}
        for i in range(n):
            for j in range(i, n):
                sm = ambient[i] + ambient[j]
                sum_count[sm] = sum_count.get(sm, 0) + 1
        conflicts = 0
        for sm, cnt in sum_count.items():
            if cnt > 1:
                conflicts += cnt - 1
        return conflicts

    # Generate structured candidates with many APs
    candidates = []
    
    # Pure arithmetic progressions with various steps
    for d in range(1, 11):
        for start in range(0, 129 - 11*d):
            cand = [start + i*d for i in range(12)]
            if all(0 <= x <= 128 for x in cand):
                candidates.append(cand)
    
    # Two interleaved APs
    for d1 in range(1, 8):
        for d2 in range(d1+1, 12):
            for start in range(0, 129):
                cand = []
                for i in range(6):
                    cand.append(start + i*d1)
                for i in range(6):
                    cand.append(start + 1 + i*d2)
                if len(set(cand)) == 12 and all(0 <= x <= 128 for x in cand):
                    candidates.append(cand)
    
    # Dense small sets: use numbers 0-20 with many overlaps
    for base in range(0, 10):
        cand = [base + i for i in range(12)]
        if all(0 <= x <= 128 for x in cand):
            candidates.append(cand)
    
    # Mix of two APs with different starts
    for d in range(2, 8):
        for s1 in range(0, 20):
            for s2 in range(s1+1, min(s1+d*3, 129)):
                cand1 = [s1 + i*d for i in range(6)]
                cand2 = [s2 + i*d for i in range(6)]
                cand = cand1 + cand2
                if len(set(cand)) == 12 and all(0 <= x <= 128 for x in cand):
                    candidates.append(cand)
    
    # Evaluate all candidates
    best_set = None
    best_size = 12
    
    for cand in candidates:
        if len(cand) == 12 and len(set(cand)) == 12 and all(0 <= x <= 128 for x in cand):
            sz = max_sidon_size_fast(cand)
            if sz < best_size:
                best_size = sz
                best_set = list(cand)
            if best_size <= 5:  # Good enough, stop early
                break
    
    if best_set is None:
        best_set = list(range(12))
        best_size = max_sidon_size_fast(best_set)
    
    # Local search to improve
    current = list(best_set)
    current_size = best_size
    random.seed(42)
    
    for iteration in range(10000):
        if time.time() > deadline - 5:
            break
        if current_size <= 4:
            break
        
        # Try replacing one element with a random value
        i = random.randint(0, 11)
        new_val = random.randint(0, 128)
        if new_val in current:
            continue
        trial = list(current)
        trial[i] = new_val
        if len(set(trial)) != 12:
            continue
        
        trial_size = max_sidon_size_fast(trial)
        if trial_size < current_size:
            current = trial
            current_size = trial_size
            if current_size < best_size:
                best_size = current_size
                best_set = list(current)
    
    # Systematic local search: try all values for each position
    for round_num in range(3):
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(12):
            if time.time() > deadline - 5:
                break
            for new_val in range(0, 129):
                if new_val == current[i] or new_val in current:
                    continue
                trial = list(current)
                trial[i] = new_val
                trial_size = max_sidon_size_fast(trial)
                if trial_size < current_size:
                    current = trial
                    current_size = trial_size
                    improved = True
                    if current_size < best_size:
                        best_size = current_size
                        best_set = list(current)
                    break
        if not improved:
            break
    
    if best_set is None:
        best_set = current
    
    # Verify
    assert len(best_set) == 12
    assert len(set(best_set)) == 12
    assert all(0 <= x <= 128 for x in best_set)
    
    return best_set
