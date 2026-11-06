"""Search for 12-element ambient set in [0,128] minimizing largest Sidon subset."""
import time
import random
import itertools
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

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

    def max_sidon_size_bruteforce(ambient):
        """Find largest Sidon subset of ambient (12 elements) via backtracking."""
        ambient = sorted(ambient)
        n = len(ambient)
        best = 0
        # Try subsets by size descending
        # Precompute: for each pair (i,j) i<=j, the sum
        # Use backtracking with pruning
        # Sort by value
        # We'll use a recursive approach: build Sidon set greedily with backtracking
        # To be fast, we check incrementally
        
        # Bitmask approach: 2^12 = 4096 subsets, check each
        # But we want the MAXIMUM, so iterate sizes from 12 down
        for size in range(n, 0, -1):
            found = False
            for combo in itertools.combinations(range(n), size):
                sub = [ambient[i] for i in combo]
                if is_sidon(sub):
                    best = size
                    found = True
                    break
            if found:
                break
        return best

    def max_sidon_size_fast(ambient):
        """Faster: use bitmask DP or clever pruning."""
        ambient = sorted(ambient)
        n = len(ambient)
        # Precompute pair sums for all pairs
        # For a candidate subset, we need all pair sums unique
        # Use bitmask: for each element, track which sums are used
        # Since n=12, max subset size 12, pair sums up to 12*13/2 = 78
        
        # Approach: iterate all 2^12 = 4096 subsets, check Sidon property
        # But we want max, so we can do it smartly
        # Precompute for each pair (i,j), i<=j: sum = ambient[i]+ambient[j]
        # For a subset S, Sidon iff all pair sums in S are distinct
        
        # Fast check: for a given subset (bitmask), check if any two pairs give same sum
        # Two pairs (i,j) and (k,l) with i<=j, k<=l, and (i,j)!=(k,l) give same sum
        # if ambient[i]+ambient[j] == ambient[k]+ambient[l]
        
        # Precompute conflict matrix: for each pair of pairs, do they conflict?
        # Actually simpler: for each sum value, which pairs produce it?
        # If any sum is produced by 2+ distinct pairs within the subset, not Sidon
        
        # Let's just do the 4096 subsets approach with early termination
        # But we can be smarter: use the fact that if a subset of size k is not Sidon,
        # no superset is Sidon (monotonicity)
        
        # Actually monotonicity works the other way: if S is Sidon, all subsets are Sidon.
        # So we want the largest Sidon subset.
        
        # Let me use a different approach: 
        # For each element, decide include or exclude, with pruning
        # But for n=12, brute force over all subsets is fine if we check efficiently
        
        # Efficient check: for subset with elements, compute all pair sums, check duplicates
        # Max 78 sums for size 12
        
        best = 0
        # Iterate over all subsets, from largest to smallest
        # Use Gray code or just combinations
        for size in range(n, 0, -1):
            for combo in itertools.combinations(range(n), size):
                # Check if this combo is Sidon
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
                    best = size
                    return best
        return best

    # The key insight: to minimize max Sidon subset, we want many "conflicts"
    # A conflict is when 2a = b+c for distinct a,b,c in the set (3-term AP centered at a)
    # Or more generally, when a+b = c+d for distinct pairs
    # The diagonal pairs (a,a) give sum 2a, which conflicts with (b,c) if b+c=2a
    
    # Strategy: construct sets with many 3-term APs
    # A 3-term AP {x, x+d, x+2d} gives conflict: 2(x+d) = x + (x+2d)
    # So the center element's diagonal conflicts with the pair of the other two
    
    # Let's try to maximize the number of 3-term APs in a 12-element set in [0,128]
    
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
        return count // 2  # each AP counted twice (once from each end)

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
                conflicts += cnt - 1  # number of "extra" pairs causing conflict
        return conflicts

    # Local search: start with a random set, try to swap elements to reduce max Sidon size
    # First, find a good starting point
    
    # Let's try structured sets with many APs
    # e.g., multiples: {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22} - all even
    # This has many APs
    
    # Actually, let me think about what minimizes the largest Sidon subset
    # If the entire 12-set is NOT Sidon, then max <= 11
    # If every 11-subset is NOT Sidon, then max <= 10
    # etc.
    
    # The minimum possible is... well, a 2-element set is always Sidon (sums: 2a, a+b, 2b - need 2a != a+b != 2b, i.e., a!=b, which holds)
    # A 3-element set {a,b,c} is Sidon iff 2a != a+b, a+b != 2b, 2b != b+c, etc.
    # i.e., a != b, a+b != 2b (a!=b), b+c != 2b (c!=b), and also 2a != b+c, a+c != 2b, etc.
    # For {a,b,c}: sums are 2a, a+b, 2b, a+c, b+c, 2c
    # All distinct iff: a<b<c, and a+c != 2b (no 3-term AP), and 2a != a+b (a!=b), etc.
    # So a 3-set is non-Sidon iff it forms a 3-term AP
    
    # So to make the whole 12-set non-Sidon, we just need one 3-term AP
    # To make all 11-subsets non-Sidon, we need that removing any one element still leaves a 3-term AP
    # This is harder
    
    # Let me just do a search
    
    best_set = None
    best_size = 12
    
    # Try some structured sets first
    # Set with many APs: use a base and add structured elements
    candidates = []
    
    # Arithmetic progression: {0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 128}? No, 128-0=128, 12*12=144>128
    # {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110} - AP with d=10, 12 terms, max=110
    candidates.append(list(range(0, 120, 10)))
    
    # {1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111}
    candidates.append(list(range(1, 112, 10)))
    
    # Mix: some AP plus some scattered
    # Let's try to be clever: use pairs that create many conflicts
    
    # Actually, let me just do random search with local optimization
    random.seed(42)
    
    def evaluate_set(s):
        """Return max Sidon subset size."""
        return max_sidon_size_fast(s)
    
    # Try initial candidates
    for cand in candidates:
        if len(cand) == 12 and all(0 <= x <= 128 for x in cand) and len(set(cand)) == 12:
            sz = evaluate_set(cand)
            if sz < best_size:
                best_size = sz
                best_set = list(cand)
    
    # Local search
    current = list(best_set) if best_set else list(range(12))
    current_size = evaluate_set(current)
    
    for iteration in range(5000):
        if time.time() > deadline - 5:
            break
        if current_size <= 4:  # Can't do much better
            break
        
        # Try a random swap or move
        i = random.randint(0, 11)
        op = random.random()
        
        if op < 0.7:
            # Swap: replace one element with a random value
            new_val = random.randint(0, 128)
            if new_val in current:
                continue
            trial = list(current)
            trial[i] = new_val
            if len(set(trial)) != 12:
                continue
        else:
            # Swap two elements
            j = random.randint(0, 11)
            if i == j:
                continue
            trial = list(current)
            trial[i], trial[j] = trial[j], trial[i]
        
        trial_size = evaluate_set(trial)
        if trial_size < current_size:
            current = trial
            current_size = trial_size
            if current_size < best_size:
                best_size = current_size
                best_set = list(current)
    
    # Also try a more systematic search: for each position, try all possible values
    # This is expensive but let's do a few rounds
    for round_num in range(5):
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(12):
            if time.time() > deadline - 5:
                break
            for new_val in range(0, 129):
                if new_val == current[i]:
                    continue
                if new_val in current:
                    continue
                trial = list(current)
                trial[i] = new_val
                if len(set(trial)) != 12:
                    continue
                trial_size = evaluate_set(trial)
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
