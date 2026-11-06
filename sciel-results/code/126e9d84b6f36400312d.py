import time
import random
from itertools import combinations

def is_valid_incremental(S, x):
    """
    Check if S U {x} is valid, given S is valid.
    S is a list of distinct ints. x is not in S.
    Returns True if valid, False otherwise.
    """
    n = len(S)
    
    # 1. Check if x divides sum of any 3 distinct in S
    # Manual triple loop for speed
    for i in range(n):
        si = S[i]
        for j in range(i+1, n):
            sj = S[j]
            for k in range(j+1, n):
                sk = S[k]
                if (si + sj + sk) % x == 0:
                    return False
            
    # 2. Check if any a in S divides x + y + z for y,z in S\{a}
    # For each a (index i), check pairs (j, k) with j, k != i
    for i in range(n):
        a = S[i]
        target = (-x) % a
        # If a == 1, target is 0. 
        # We need (y+z) % 1 == 0, which is always true.
        # So if 1 is in S, and x is anything, and there are at least 2 other elements, it's invalid?
        # Wait. If a=1, 1 divides ANY sum.
        # So if 1 is in S, we need NO triples of OTHER elements?
        # Condition: 1 must not divide sum of any 3 DISTINCT OTHER.
        # Since 1 divides everything, this means there must be NO 3 distinct other elements.
        # So if 1 is in S, |S| must be <= 3?
        # If |S| >= 3, there are 3 other elements. Their sum is divisible by 1. Invalid.
        # So 1 can only be in S if |S| <= 3.
        # This is a strong constraint.
        
        # Optimization: If a == 1, return False if n >= 3 (since there are at least 2 others, 
        # wait, we need 3 OTHERS. So if n-1 >= 3, i.e., n >= 4, it's invalid.
        # If n=3, there are 2 others. No triple of 3 others. So valid?
        # Yes, "sum of any k=3 DISTINCT OTHER". If there aren't 3 others, the condition is vacuously true.
        # So if a=1, it is valid only if number of other elements < 3.
        # i.e., n - 1 < 3 => n < 4.
        if a == 1:
            if n >= 4:
                return False
            # If n < 4, no triple of 3 others exists, so condition holds for a=1.
            # We still need to check other a's? 
            # If n < 4, the loops below will run but find no pairs (if n=3, pairs exist but we need 3 others for a? 
            # No, the condition is about a dividing sum of 3 others.
            # If n=3, for a, there are 2 others. No triple. So valid.
            # So we can return True immediately if a=1 and n<4?
            # We still need to check if OTHER elements in S divide sums.
            # But if n<4, there are no triples of 3 others for ANY element.
            # So if n < 4, the set is ALWAYS valid?
            # Let's check: S={1,2,3}. 
            # a=1: others {2,3}. No 3 others. OK.
            # a=2: others {1,3}. No 3 others. OK.
            # a=3: others {1,2}. No 3 others. OK.
            # So any set of size < 4 is valid.
            # This means we can always add elements until size 3.
            # The constraint kicks in at size 4.
            pass

        # Check pairs
        for j in range(n):
            if j == i: continue
            y = S[j]
            for k in range(j+1, n):
                if k == i: continue
                z = S[k]
                if (y + z) % a == target:
                    return False
                    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s margin
    
    best_set = []
    best_size = 0
    
    # Greedy Phase
    # 1. Descending
    S = []
    for x in range(192, 0, -1):
        if time.time() > deadline: break
        if len(S) == 48: break
        if is_valid_incremental(S, x):
            S.append(x)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S[:]
        
    # 2. Ascending
    S = []
    for x in range(1, 193):
        if time.time() > deadline: break
        if len(S) == 48: break
        if is_valid_incremental(S, x):
            S.append(x)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S[:]

    # 3. Randomized Greedy
    for _ in range(20):
        if time.time() > deadline: break
        S = []
        candidates = list(range(1, 193))
        random.shuffle(candidates)
        for x in candidates:
            if len(S) == 48: break
            if is_valid_incremental(S, x):
                S.append(x)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
        if best_size == 48:
            return best_set

    # Local Search Phase
    current_set = best_set[:]
    current_size = best_size
    S_set = set(current_set)
    
    no_improve_count = 0
    
    while time.time() < deadline:
        if no_improve_count > 100:
            break
            
        improved = False
        
        # 1. Try to add
        missing = [x for x in range(1, 193) if x not in S_set]
        if missing:
            random.shuffle(missing)
            for x in missing:
                if time.time() > deadline: break
                if len(current_set) == 48: break
                if is_valid_incremental(current_set, x):
                    current_set.append(x)
                    S_set.add(x)
                    current_size += 1
                    improved = True
                    if current_size > best_size:
                        best_size = current_size
                        best_set = current_set[:]
                    break
                    
        if improved:
            no_improve_count = 0
            if best_size == 48:
                return best_set
            continue
            
        # 2. Try swaps
        # Only if we have elements to remove and candidates to add
        if current_set and missing:
            for _ in range(200): # Limit swaps per iteration
                if time.time() > deadline: break
                
                a = random.choice(current_set)
                b = random.choice(missing)
                
                # Check validity of S\{a} U {b}
                # S_minus_a is valid.
                S_minus_a = [v for v in current_set if v != a]
                
                if is_valid_incremental(S_minus_a, b):
                    # Perform swap
                    current_set = S_minus_a + [b]
                    S_set.discard(a)
                    S_set.add(b)
                    improved = True
                    # Size remains same
                    break
                    
        if not improved:
            no_improve_count += 1
            # If stuck, maybe try a different random seed or restart?
            # For now, just break if stuck for a while
            if no_improve_count > 50:
                break
                
    return best_set
