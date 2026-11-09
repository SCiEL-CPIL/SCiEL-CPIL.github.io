"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import time
import random

# EVOLVE-BLOCK-START

def _is_sidon4_fast(a, b, c, d):
    """Check if 4-element set {a,b,c,d} (sorted) has all pair sums unique."""
    # 10 pair sums: a+a, a+b, a+c, a+d, b+b, b+c, b+d, c+c, c+d, d+d
    # Check for collisions efficiently
    s1 = a + a
    s2 = a + b
    s3 = a + c
    s4 = a + d
    s5 = b + b
    s6 = b + c
    s7 = b + d
    s8 = c + c
    s9 = c + d
    s10 = d + d
    
    # Since a < b < c < d (sorted), we can reason about which collisions are possible
    # s1 < s2 < s3 < s4 (since a < b < c < d)
    # s2 < s5 (a < b)
    # s3 < s6 (a < b)
    # s4 < s7 (a < b)
    # s5 < s6 < s7
    # s6 < s8 (b < c)
    # s7 < s9 (b < c)
    # s8 < s9 < s10
    
    # Possible collisions to check:
    # s1 == s2? No (a < b)
    # s1 == s3? No
    # s1 == s4? No
    # s1 == s5? a+a == b+b -> 2a == 2b -> a==b, impossible
    # s1 == s6? a+a == b+c
    # s1 == s7? a+a == b+d
    # s1 == s8? a+a == c+c -> a==c, impossible
    # s1 == s9? a+a == c+d
    # s1 == s10? a+a == d+d -> a==d, impossible
    # s2 == s3? No (b < c)
    # s2 == s4? No
    # s2 == s5? a+b == b+b -> a==b, impossible
    # s2 == s6? a+b == b+c -> a==c, impossible
    # s2 == s7? a+b == b+d -> a==d, impossible
    # s2 == s8? a+b == c+c
    # s2 == s9? a+b == c+d
    # s2 == s10? a+b == d+d
    # s3 == s4? No
    # s3 == s5? a+c == b+b
    # s3 == s6? a+c == b+c -> a==b, impossible
    # s3 == s7? a+c == b+d
    # s3 == s8? a+c == c+c -> a==c, impossible
    # s3 == s9? a+c == c+d -> a==d, impossible
    # s3 == s10? a+c == d+d
    # s4 == s5? a+d == b+b
    # s4 == s6? a+d == b+c
    # s4 == s7? a+d == b+d -> a==b, impossible
    # s4 == s8? a+d == c+c
    # s4 == s9? a+d == c+d -> a==c, impossible
    # s4 == s10? a+d == d+d -> a==d, impossible
    # s5 == s6? b+b == b+c -> b==c, impossible
    # s5 == s7? b+b == b+d -> b==d, impossible
    # s5 == s8? b+b == c+c -> b==c, impossible
    # s5 == s9? b+b == c+d
    # s5 == s10? b+b == d+d
    # s6 == s7? b+c == b+d -> c==d, impossible
    # s6 == s8? b+c == c+c -> b==c, impossible
    # s6 == s9? b+c == c+d -> b==d, impossible
    # s6 == s10? b+c == d+d
    # s7 == s8? b+d == c+c
    # s7 == s9? b+d == c+d -> b==c, impossible
    # s7 == s10? b+d == d+d -> b==d, impossible
    # s8 == s9? c+c == c+d -> c==d, impossible
    # s8 == s10? c+c == d+d -> c==d, impossible
    # s9 == s10? c+d == d+d -> c==d, impossible
    
    # So the possible collisions are:
    # s1==s6: 2a == b+c
    # s1==s7: 2a == b+d
    # s1==s9: 2a == c+d
    # s2==s8: a+b == 2c
    # s2==s9: a+b == c+d
    # s2==s10: a+b == 2d
    # s3==s5: a+c == 2b
    # s3==s7: a+c == b+d
    # s3==s10: a+c == 2d
    # s4==s5: a+d == 2b
    # s4==s6: a+d == b+c
    # s4==s8: a+d == 2c
    # s5==s9: 2b == c+d
    # s5==s10: 2b == 2d -> b==d, impossible
    # s6==s10: b+c == 2d
    # s7==s8: b+d == 2c
    
    if (2*a == b+c or 2*a == b+d or 2*a == c+d or
        a+b == 2*c or a+b == c+d or a+b == 2*d or
        a+c == 2*b or a+c == b+d or a+c == 2*d or
        a+d == 2*b or a+d == b+c or a+d == 2*c or
        2*b == c+d or b+c == 2*d or b+d == 2*c):
        return False
    return True

def _has_sidon4(arr):
    """Check if any 4-subset of arr is Sidon. Returns True if exists."""
    n = len(arr)
    for i in range(n - 3):
        a = arr[i]
        for j in range(i + 1, n - 2):
            b = arr[j]
            for k in range(j + 1, n - 1):
                c = arr[k]
                for l in range(k + 1, n):
                    d = arr[l]
                    if _is_sidon4_fast(a, b, c, d):
                        return True
    return False

def _max_sidon_size(candidates):
    """Find the largest Sidon subset size of the given 12-element set."""
    for size in range(12, 0, -1):
        for combo in itertools.combinations(candidates, size):
            n = len(combo)
            sums = set()
            ok = True
            for i in range(n):
                for j in range(i, n):
                    s = combo[i] + combo[j]
                    if s in sums:
                        ok = False
                        break
                    sums.add(s)
                if not ok:
                    break
            if ok:
                return size
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    best_set = list(range(12))
    best_size = _max_sidon_size(best_set)  # should be 4
    
    # Strategy 1: Try all consecutive blocks
    for start in range(0, 118):
        if time.time() > deadline:
            break
        candidates = list(range(start, start + 12))
        if not _has_sidon4(candidates):
            ms = _max_sidon_size(candidates)
            if ms < best_size:
                best_size = ms
                best_set = candidates
                if best_size <= 3:
                    return best_set
    
    # Strategy 2: Arithmetic progressions
    for d in range(1, 12):
        for a in range(0, 129 - 11 * d):
            if time.time() > deadline:
                break
            candidates = [a + i * d for i in range(12)]
            if not _has_sidon4(candidates):
                ms = _max_sidon_size(candidates)
                if ms < best_size:
                    best_size = ms
                    best_set = candidates
                    if best_size <= 3:
                        return best_set
    
    # Strategy 3: Dense random search - bias toward small ranges
    random.seed(42)
    for attempt in range(100000):
        if time.time() > deadline:
            break
        # Choose a small range to maximize collision probability
        max_val = random.choice([15, 20, 25, 30, 40, 50, 70, 100, 128])
        candidates = sorted(random.sample(range(max_val + 1), 12))
        if not _has_sidon4(candidates):
            ms = _max_sidon_size(candidates)
            if ms < best_size:
                best_size = ms
                best_set = candidates
                if best_size <= 3:
                    return best_set
    
    # Strategy 4: Local search
    if best_size > 3:
        current_set = best_set[:]
        for iteration in range(20000):
            if time.time() > deadline:
                break
            idx = random.randint(0, 11)
            old_val = current_set[idx]
            found = False
            for _ in range(30):
                if time.time() > deadline:
                    break
                new_val = random.randint(0, 128)
                if new_val in current_set:
                    continue
                neighbor = current_set[:]
                neighbor[idx] = new_val
                neighbor.sort()
                if not _has_sidon4(neighbor):
                    ms = _max_sidon_size(neighbor)
                    if ms < best_size:
                        best_size = ms
                        best_set = neighbor[:]
                        if best_size <= 3:
                            return best_set
                    current_set = neighbor
                    found = True
                    break
            if not found and random.random() < 0.2:
                idx2 = random.randint(0, 11)
                if idx2 != idx:
                    neighbor = current_set[:]
                    neighbor[idx], neighbor[idx2] = neighbor[idx2], neighbor[idx]
                    neighbor.sort()
                    if not _has_sidon4(neighbor):
                        ms = _max_sidon_size(neighbor)
                        if ms < best_size:
                            best_size = ms
                            best_set = neighbor[:]
                            if best_size <= 3:
                                return best_set
                        current_set = neighbor
    
    # Strategy 5: Structured candidates with many additive relations
    structured_candidates = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13],
        [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
        [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
        [0, 2, 4, 5, 7, 9, 10, 12, 14, 15, 17, 18],
        [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
        [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17],
        [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21],
        [0, 2, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27],
        [0, 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16],
        [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 17],
    ]
    for cand in structured_candidates:
        if time.time() > deadline:
            break
        if len(set(cand)) == 12 and max(cand) <= 128:
            if not _has_sidon4(cand):
                ms = _max_sidon_size(cand)
                if ms < best_size:
                    best_size = ms
                    best_set = cand[:]
                    if best_size <= 3:
                        return best_set
    
    return best_set

# EVOLVE-BLOCK-END
