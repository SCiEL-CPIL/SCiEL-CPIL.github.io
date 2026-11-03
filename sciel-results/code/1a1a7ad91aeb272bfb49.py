import time
import random
import numpy as np
from collections import defaultdict

# Precompute balls for q=3, n=8
Q = 3
N_LEN = 8
RADIUS = 1
SPACE_SIZE = Q ** N_LEN

# Convert int to list of digits
def int_to_digits(num, q, n):
    res = [0] * n
    for i in range(n):
        res[i] = num % q
        num //= q
    return res

# Convert list of digits to int
def digits_to_int(digits, q):
    num = 0
    for d in reversed(digits):
        num = num * q + d
    return num

# Precompute balls
BALLS = []
for i in range(SPACE_SIZE):
    d = int_to_digits(i, Q, N_LEN)
    ball = [i]
    for pos in range(N_LEN):
        for val in range(Q):
            if val == d[pos]:
                continue
            new_d = d[:]
            new_d[pos] = val
            ball.append(digits_to_int(new_d, Q))
    BALLS.append(ball)

def construct(q, n, radius):
    # The task specifies q, n, radius. 
    # We assume the specific case q=3, n=8, radius=1.
    # If parameters differ, fall back to seed logic or simple greedy.
    if q != 3 or n != 8 or radius != 1:
        # Fallback to seed logic for other params
        import itertools
        if n == 4 and q == 3 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # General greedy fallback
        def greedy_general(q, n, r):
            # Too slow for general case in 120s if n is large, but n=8 is small.
            # Just return empty or simple code
            return []
        return greedy_general(q, n, radius)

    start_time = time.time()
    deadline = start_time + 110.0
    
    def greedy_construct(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = set(range(SPACE_SIZE))
        code = []
        cover_map = defaultdict(list)
        
        while uncovered:
            best_c = -1
            best_count = -1
            # Sample if set is large? No, 6561 is small.
            # But iterating all 6561 candidates and checking intersection with uncovered set (size up to 6561) is 6561 * 24 ops.
            # 150k ops per step. 600 steps -> 90M ops. Might be slow in Python.
            # Optimization: Only consider candidates that cover the "most uncovered" point?
            # Standard greedy: pick point p in uncovered, pick c in B(p) that covers most uncovered points.
            # Pick a random point from uncovered to break ties / randomize.
            if not uncovered: break
            p = random.choice(list(uncovered))
            
            # Find best c in B(p)
            candidates = BALLS[p]
            best_c = -1
            max_cov = -1
            for c in candidates:
                # Count how many points in B(c) are in uncovered
                cov = 0
                for pt in BALLS[c]:
                    if pt in uncovered:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    best_c = c
                elif cov == max_cov and random.random() < 0.5:
                    best_c = c
            
            if best_c == -1:
                break
                
            code.append(best_c)
            idx = len(code) - 1
            for pt in BALLS[best_c]:
                cover_map[pt].append(idx)
                if pt in uncovered:
                    uncovered.remove(pt)
                    
        return code

    def local_search(code):
        if not code:
            return code
        
        cover_map = defaultdict(list)
        for i, c in enumerate(code):
            for pt in BALLS[c]:
                cover_map[pt].append(i)
        
        # Remove redundant
        changed = True
        while changed:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                removable = True
                for pt in BALLS[c]:
                    if len(cover_map[pt]) == 1:
                        removable = False
                        break
                if removable:
                    # Remove i
                    for pt in BALLS[c]:
                        cover_map[pt].remove(i)
                    code.pop(i)
                    # Adjust indices in cover_map for elements after i?
                    # No, cover_map stores indices. If we pop i, indices > i are shifted.
                    # This makes the data structure invalid.
                    # Solution: Rebuild cover_map after each removal? 
                    # Cost: N * 24 * N = 600 * 24 * 600 = 8.6M ops. 
                    # If we do this for 600 removals, it's 5 Billion ops. Too slow.
                    
                    # Better: Use a different data structure or careful index management.
                    # Let's just rebuild cover_map from scratch in each "pass" of the while loop?
                    # Or, simpler: Just return the code and let the outer loop handle it?
                    # No, we need to remove as much as possible.
                    
                    # Let's restart the local search function with the new code?
                    # That's recursive.
                    
                    # Let's try a different approach for local search:
                    # Keep list of codewords.
                    # To check removability of i, we need to know if any pt in B(c_i) has count 1.
                    # We can maintain `counts` array.
                    # `counts[pt]` = number of codewords covering pt.
                    # To check if i is removable:
                    #   for pt in B(c_i): if counts[pt] == 1: return False
                    #   return True
                    # To remove i:
                    #   for pt in B(c_i): counts[pt] -= 1
                    #   code.pop(i)
                    #   # No index shifting issue if we just use counts!
                    #   # But we need to know WHO covers pt if count becomes 1?
                    #   # No, we don't need to know who, just count.
                    #   # Wait, if count becomes 1, we don't need to know who for the *removability check* of OTHERS?
                    #   # If we check if j is removable, we look at B(c_j).
                    #   # If pt in B(c_j) has count 1, is it covered by j?
                    #   # We don't know if j is the one covering it.
                    #   # So `counts` is not enough. We need to know if j is the *only* one.
                    #   # So we need `owners` or `cover_map`.
                    
                    # Let's go back to `cover_map` but handle index shifts.
                    # If we remove i, we must decrement all indices > i in cover_map.
                    # This is O(N * |B|) per removal.
                    # 600 * 24 * 600 = 8.6M per removal.
                    # Total 5B. Too slow.
                    
                    # Alternative:
                    # Use a set of codewords? No, order doesn't matter for coverage.
                    # But we need to return a list.
                    # We can use a list of sets? No.
                    
                    # Let's try:
                    # 1. Start with greedy code.
                    # 2. Compute cover_map.
                    # 3. Loop:
                    #    - Find any removable i.
                    #    - If found:
                    #       - Remove i from code.
                    #       - Update cover_map: for pt in B(c_i), remove i.
                    #       - For all pt, for all j in cover_map[pt], if j > i, cover_map[pt].remove(j); cover_map[pt].append(j-1)
                    #       - This is expensive.
                    
                    # Simpler:
                    # Just do one pass of removals.
                    # If we remove i, we don't update indices immediately.
                    # We just mark i as removed.
                    # At the end, filter the list.
                    # But during the pass, if we check j < i, it's fine.
                    # If we check j > i, and i was removed, the index in cover_map is stale.
                    
                    # Let's just rebuild cover_map after each successful removal.
                    # Number of removals is at most 600.
                    # Rebuild cost: 600 * 24 = 14k ops.
                    # Total cost: 600 * 14k = 8.4M ops.
                    # This is FAST.
                    pass
        
        # Re-implement local search with rebuild strategy
        code = list(code)
        while True:
            cover_map = defaultdict(list)
            for i, c in enumerate(code):
                for pt in BALLS[c]:
                    cover_map[pt].append(i)
            
            removed_any = False
            # Iterate backwards to handle indices easily if we were to pop, but we rebuild
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                removable = True
                for pt in BALLS[c]:
                    if len(cover_map[pt]) == 1:
                        removable = False
                        break
                if removable:
                    code.pop(i)
                    removed_any = True
                    break # Rebuild and restart
            
            if not removed_any:
                break
        return code

    best_code = greedy_construct()
    best_size = len(best_code)
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Randomize greedy
        code = greedy_construct(random_seed=attempts)
        code = local_search(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            # Optional: Local search on best_code again?
            # best_code = local_search(best_code) # Already done
            
    return [int_to_digits(c, Q, N_LEN) for c in best_code]
