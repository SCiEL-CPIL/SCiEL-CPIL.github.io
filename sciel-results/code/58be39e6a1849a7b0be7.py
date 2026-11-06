"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from functools import lru_cache

# EVOLVE-BLOCK-START
import time as _time

def _largest_sidon_bruteforce(s):
    """Compute largest subset of s with all unordered pair sums unique INCLUDING diagonals.
    Pair sums: for i <= j, sum = s[i] + s[j]. All must be distinct."""
    n = len(s)
    # Use bitmask DP / brute force over subsets since n=12 is small
    # A subset T is "Sidon with diagonals" if all s[i]+s[j] for i<=j in T are unique
    best = 0
    # Enumerate all subsets by size descending
    for k in range(n, 0, -1):
        for sub in itertools.combinations(range(n), k):
            sums = set()
            ok = True
            for a_idx in range(len(sub)):
                for b_idx in range(a_idx, len(sub)):
                    v = s[sub[a_idx]] + s[sub[b_idx]]
                    if v in sums:
                        ok = False
                        break
                    sums.add(v)
            if ok:
                best = k
                return best
    return best

def _check_sidon_with_diagonals(subset):
    """Check if a subset has all pair sums (i<=j) unique."""
    sums = set()
    for i in range(len(subset)):
        for j in range(i, len(subset)):
            v = subset[i] + subset[j]
            if v in sums:
                return False
            sums.add(v)
    return True

def construct():
    start_time = _time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    # Map integers to 2D grid over F_7 x F_7
    # n -> (n % 7, n // 7) but n//7 can be up to 18, so use mod 7 for second coord too
    # Actually for [0,128], n//7 ranges 0..18. Let's use a different mapping.
    # Better: use n mod 7 and (n // 7) mod 7, but this isn't injective for [0,128].
    # Alternative: just search directly in [0,128] with the Sidon constraint.
    
    # Strategy: Random search with local optimization
    # Key insight: we want to MINIMIZE the largest Sidon subset.
    # The baseline range(12) = [0,1,2,...,11] has many collisions.
    # We want a set where ANY subset of size k has a pair-sum collision.
    
    # Let's think: if we take a set where many elements share the same value mod some m,
    # then pairs from the same residue class will collide.
    
    # Actually, the simplest approach: use a set with lots of arithmetic structure.
    # If we take elements in an arithmetic progression, sums collide heavily.
    # But we need exactly 12 distinct integers in [0,128].
    
    # Key observation: The maximum Sidon subset size is what we minimize.
    # For a "random" set of 12 in [0,128], the largest Sidon subset is likely large (~8-10).
    # We want it small, like 4 or 5.
    
    # Strategy: Use a set with many additive dependencies.
    # If we can make it so that for any 6 elements, there's a pair-sum collision,
    # then the max Sidon subset is at most 5.
    
    # Approach: Place elements in pairs that share sum properties.
    # E.g., if a+b = c+d for many quadruples, then we can't have {a,b,c,d} as Sidon.
    
    # Let's try: take 6 pairs where each pair sums to the same value S.
    # Then any selection containing both elements of two different pairs has a collision.
    # Max Sidon subset can contain at most one from each pair = 6 elements max.
    # But wait, we need ALL pair sums unique including diagonals.
    # If we have pairs (a1,b1), (a2,b2), ..., (a6,b6) all with sum S,
    # then selecting a1 and a2: a1+a2 might not collide with anything.
    # The collision is a1+b1 = a2+b2 = S.
    # So a Sidon subset can include at most one from each pair.
    # But it could include one from each of all 6 pairs = 6 elements.
    # We need to check if those 6 elements form a valid Sidon set.
    
    # Better: use more overlapping structure.
    
    # Let me just do a smart random search.
    
    best_set = list(range(12))
    best_size = _largest_sidon_bruteforce(best_set)
    
    # Try structured sets first
    # Set 1: all even numbers 0,2,4,...,22 -> same as range(12) scaled
    # Set 2: numbers with many shared residues mod small numbers
    
    # Try: 12 numbers where many are congruent mod 2, mod 3, etc.
    # If all 12 are even, then all pair sums are even, but that doesn't directly help.
    
    # Key idea: use a set where elements form many 4-cycles in the sum graph.
    # a+b = c+d means {a,b,c,d} is not Sidon.
    
    # Let's try a grid-based approach:
    # Pick 12 points in a 4x3 grid: values = i*32 + j for i in 0..3, j in 0..2
    # This gives 12 values in [0, 3*32+2] = [0,98]
    # Sums: (i*32+j) + (i'*32+j') = (i+i')*32 + (j+j')
    # Collision when i+i' = i'+i'' and j+j' = j'+j'' (for different pairs)
    # This is like a 2D Sidon problem.
    
    candidates = []
    
    # Candidate 1: 4x3 grid
    grid_4x3 = [i * 32 + j for i in range(4) for j in range(3)]
    candidates.append(grid_4x3)
    
    # Candidate 2: 3x4 grid with different scaling
    grid_3x4 = [i * 33 + j for i in range(3) for j in range(4)]
    candidates.append(grid_3x4)
    
    # Candidate 3: powers of 2 style but compressed
    # 1, 2, 4, 8, 16, 32, 64, 5, 10, 20, 40, 80
    candidates.append([1, 2, 4, 8, 16, 32, 64, 5, 10, 20, 40, 80])
    
    # Candidate 4: arithmetic progression
    candidates.append(list(range(0, 120, 10)))  # 0,10,20,...,110
    
    # Candidate 5: two interleaved APs
    candidates.append([0, 10, 20, 30, 40, 50, 1, 11, 21, 31, 41, 51])
    
    # Candidate 6: Fibonacci-like
    fibs = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    # Make distinct
    fibs_distinct = []
    for f in fibs:
        if f not in fibs_distinct:
            fibs_distinct.append(f)
    while len(fibs_distinct) < 12:
        fibs_distinct.append(fibs_distinct[-1] + fibs_distinct[-2] if len(fibs_distinct) >= 2 else len(fibs_distinct))
    candidates.append(fibs_distinct[:12])
    
    # Evaluate all candidates
    for cand in candidates:
        if len(set(cand)) != 12:
            continue
        if any(x < 0 or x > 128 for x in cand):
            continue
        sz = _largest_sidon_bruteforce(cand)
        if sz < best_size:
            best_size = sz
            best_set = cand[:]
    
    # Now do random search with local improvement
    rng = random.Random(42)
    
    for trial in range(5000):
        if _time.time() > deadline:
            break
        
        # Generate random candidate
        if trial % 3 == 0:
            # Grid-based random
            rows = rng.randint(2, 5)
            cols = 12 // rows if 12 % rows == 0 else rng.randint(2, 6)
            if rows * cols < 12:
                rows = 3
                cols = 4
            scale = rng.randint(10, 20)
            cand = []
            for i in range(rows):
                for j in range(min(cols, 12 - len(cand))):
                    val = i * scale + j * rng.randint(1, 3)
                    if val <= 128 and val not in cand:
                        cand.append(val)
            while len(cand) < 12:
                v = rng.randint(0, 128)
                if v not in cand:
                    cand.append(v)
        elif trial % 3 == 1:
            # Random with clustering
            centers = rng.sample(range(0, 129), 4)
            cand = []
            for c in centers:
                for offset in range(3):
                    v = c + rng.randint(-2, 2)
                    if 0 <= v <= 128 and v not in cand:
                        cand.append(v)
            while len(cand) < 12:
                v = rng.randint(0, 128)
                if v not in cand:
                    cand.append(v)
        else:
            # Pure random
            cand = rng.sample(range(129), 12)
        
        sz = _largest_sidon_bruteforce(cand)
        if sz < best_size:
            best_size = sz
            best_set = cand[:]
        
        # Local improvement: try swapping one element
        if sz <= best_size:
            improved = True
            while improved and _time.time() < deadline:
                improved = False
                current = best_set[:]
                current_sz = best_size
                for idx in range(12):
                    if _time.time() > deadline:
                        break
                    old_val = current[idx]
                    for new_val in range(129):
                        if new_val == old_val or new_val in current:
                            continue
                        trial_set = current[:]
                        trial_set[idx] = new_val
                        trial_sz = _largest_sidon_bruteforce(trial_set)
                        if trial_sz < current_sz:
                            current = trial_set
                            current_sz = trial_sz
                            improved = True
                            break
                if current_sz < best_size:
                    best_size = current_sz
                    best_set = current[:]
    
    return best_set
# EVOLVE-BLOCK-END
