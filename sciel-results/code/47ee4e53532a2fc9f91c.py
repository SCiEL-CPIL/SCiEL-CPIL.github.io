import time
import random
import itertools

# EVOLVE-BLOCK-START

def _count_diffs(s):
    """Count the number of distinct d > 0 such that a, a+d, a+2d are all in s."""
    s_set = set(s)
    diffs = set()
    for a in s:
        for b in s:
            if b > a:
                d = b - a
                if a + 2 * d in s_set:
                    diffs.add(d)
    return len(diffs)

def _construct_greedy(n=32, max_val=512, time_limit=140.0):
    """
    Greedy / local search approach to maximize the number of distinct 3-term progression differences.
    
    Strategy:
    1. Start with a random set of 32 integers in 1..512.
    2. Use local search: try swapping elements to improve the score.
    3. Also try a more structured approach: use arithmetic progressions with different steps.
    """
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def time_left():
        return time_limit - elapsed()
    
    # Method 1: Local search from a good initial set
    # Good initial sets: use small arithmetic progressions or structured sets
    
    best_set = None
    best_score = 0
    
    # Try several initial configurations
    initial_sets = []
    
    # 1. Small arithmetic progression
    initial_sets.append(list(range(1, 33)))
    
    # 2. Two interleaved APs
    initial_sets.append(list(range(1, 33, 2)) + list(range(2, 34, 2)))  # same as 1..32
    
    # 3. Geometric-like spread
    initial_sets.append([1, 2, 4, 8, 16, 32, 64, 128, 256, 3, 6, 12, 24, 48, 96, 192,
                         5, 10, 20, 40, 80, 160, 320, 7, 14, 28, 56, 112, 224, 448, 9, 18])
    # Filter to valid range and ensure 32 distinct
    initial_sets[2] = [x for x in initial_sets[2] if 1 <= x <= 512][:32]
    while len(initial_sets[2]) < 32:
        initial_sets[2].append(random.randint(1, 512))
    initial_sets[2] = list(set(initial_sets[2]))[:32]
    while len(initial_sets[2]) < 32:
        initial_sets[2].append(random.randint(1, 512))
    initial_sets[2] = list(set(initial_sets[2]))[:32]
    
    # 4. Random sets
    for _ in range(5):
        s = set()
        while len(s) < 32:
            s.add(random.randint(1, 512))
        initial_sets.append(list(s))
    
    # 5. Structured: union of small APs with different d
    # d=1: 1,2,3,...,16 (16 terms)
    # d=2: 17,19,21,...,49 (16 terms)
    ap1 = list(range(1, 17))  # 1..16
    ap2 = list(range(17, 50, 2))  # 17,19,...,49
    initial_sets.append(ap1 + ap2)
    
    # d=1: 1..10, d=3: 11,14,17,...,56 (16 terms), d=5: 58,63,...,133 (16 terms)
    ap3 = list(range(1, 11))  # 1..10
    ap4 = list(range(11, 57, 3))  # 11,14,...,56
    ap5 = list(range(58, 134, 5))  # 58,63,...,133
    initial_sets.append(ap3 + ap4 + ap5)
    
    # d=1: 1..8, d=2: 9,11,...,39 (16 terms), d=4: 41,45,...,101 (16 terms)
    ap6 = list(range(1, 9))  # 1..8
    ap7 = list(range(9, 40, 2))  # 9,11,...,39
    ap8 = list(range(41, 102, 4))  # 41,45,...,101
    initial_sets.append(ap6 + ap7 + ap8)
    
    # Evaluate initial sets
    for s in initial_sets:
        if len(s) != 32 or len(set(s)) != 32:
            continue
        if any(x < 1 or x > 512 for x in s):
            continue
        sc = _count_diffs(s)
        if sc > best_score:
            best_score = sc
            best_set = list(s)
    
    # Local search
    current_set = best_set[:]
    current_score = best_score
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time_left() > 5.0 and no_improve_count < max_no_improve:
        improved = False
        
        # Try swapping each element with a random other value
        for i in range(32):
            if time_left() < 5.0:
                break
            for _ in range(20):  # Try 20 random swaps per position
                if time_left() < 5.0:
                    break
                new_val = random.randint(1, 512)
                if new_val in current_set:
                    continue
                old_val = current_set[i]
                current_set[i] = new_val
                new_score = _count_diffs(current_set)
                if new_score > current_score:
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_set = current_set[:]
                    improved = True
                else:
                    current_set[i] = old_val
        
        if not improved:
            no_improve_count += 1
        else:
            no_improve_count = 0
    
    # Also try a more systematic local search: for each element, try all possible values
    # This is expensive, so do it only if we have time
    if time_left() > 20.0:
        current_set = best_set[:]
        current_score = best_score
        for i in range(32):
            if time_left() < 10.0:
                break
            old_val = current_set[i]
            for new_val in range(1, 513):
                if new_val == old_val:
                    continue
                if new_val in current_set:
                    continue
                current_set[i] = new_val
                new_score = _count_diffs(current_set)
                if new_score > current_score:
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_set = current_set[:]
                else:
                    current_set[i] = old_val
    
    return best_set

def construct():
    """
    Return exactly 32 distinct integers in 1..512 that maximize the number of
    distinct d > 0 such that a, a+d, a+2d are all in the set for some a.
    """
    result = _construct_greedy()
    # Ensure validity
    if len(result) != 32 or len(set(result)) != 32 or any(x < 1 or x > 512 for x in result):
        # Fallback
        return list(range(1, 33))
    return result

# EVOLVE-BLOCK-END
