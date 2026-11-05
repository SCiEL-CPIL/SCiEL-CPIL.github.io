import random
import time

def construct():
    start_time = time.time()
    timeout = 55  # 5 seconds margin
    
    N = 192
    # Initial set: Greedy from large to small
    S = set()
    for x in range(N, 0, -1):
        # Check if x can be added
        # We need to check if any a in S divides sum of 3 distinct others in S U {x}
        # And if x divides sum of 3 distinct others in S
        # This is expensive. 
        # Let's just start with a random valid set or empty.
        pass
        
    # Let's use a faster heuristic start.
    # Pick random numbers from high range.
    S = set(random.sample(range(96, N+1), 50))
    # Clean S to be valid
    # Remove elements that cause conflicts
    # Simplest: Start empty, add random valid elements.
    S = set()
    candidates = list(range(1, N+1))
    random.shuffle(candidates)
    for x in candidates:
        if len(S) >= 48: break
        # Check if adding x is valid
        # Check x against S
        # Check S against x
        if _can_add(S, x):
            S.add(x)
            
    # Local Search
    best_S = set(S)
    best_size = len(S)
    
    while time.time() - start_time < timeout:
        # Try to add
        added = False
        for x in range(N, 0, -1):
            if x not in S:
                if _can_add(S, x):
                    S.add(x)
                    added = True
                    break
        if added:
            if len(S) > best_size:
                best_S = set(S)
                best_size = len(S)
                # Restart or continue?
                continue
                
        # Try to swap
        swapped = False
        # Randomize order for swap
        S_list = list(S)
        random.shuffle(S_list)
        for y in S_list:
            for x in range(N, 0, -1):
                if x in S: continue
                # Try S' = S - {y} + {x}
                S_temp = set(S)
                S_temp.remove(y)
                S_temp.add(x)
                if is_valid(S_temp):
                    S = S_temp
                    swapped = True
                    break
            if swapped:
                break
                
        if not added and not swapped:
            # Stuck, perturb
            # Remove a random element
            if S:
                y = random.choice(list(S))
                S.remove(y)
                
    return list(best_S)

def _can_add(S, x):
    # Check if x divides sum of 3 distinct in S
    if len(S) < 3:
        pass # trivially true
    else:
        # Check if x | sum(b,c,d) for distinct b,c,d in S
        # Use the target sum optimization
        # Targets: k*x for k such that k*x in [min_sum, max_sum]
        min_sum = sum(sorted(S)[:3])
        max_sum = sum(sorted(S)[-3:])
        k_min = (min_sum + x - 1) // x
        k_max = max_sum // x
        for k in range(k_min, k_max + 1):
            target = k * x
            if _has_3sum(S, target):
                return False
                
    # Check if any a in S divides sum of 3 distinct in S U {x}
    # The 3 distinct can include x or not.
    # Case 1: 3 distinct from S (already checked for S, but S was valid, so ok)
    # Case 2: 2 from S, 1 is x. Sum = b+c+x. Check if a | b+c+x.
    # Case 3: 1 from S, 2 are x? No, x is distinct element.
    # So we check for each a in S:
    # Does a divide (b+c+x) for distinct b,c in S?
    # Or a divide (b+x+x)? No, distinct.
    # So just b,c distinct in S.
    for a in S:
        # Check if exists distinct b,c in S such that a | (b+c+x)
        # i.e., b+c ≡ -x (mod a)
        # We can precompute pair sums mod a?
        # For small S, just check.
        # Optimize: 
        # We need to know if there are distinct b,c in S with (b+c) % a == (-x) % a.
        # Let target_res = (-x) % a
        # Use a set of residues? 
        # Since S is small, iterate pairs.
        # But we do this for each a.
        # O(N^3) total.
        
        # Faster: 
        # For fixed a, we want to know if there are distinct b,c in S such that (b+c) % a == target_res.
        # We can iterate b in S, and check if there exists c in S, c!=b, such that (c - (target_res - b)) % a == 0.
        # i.e., c ≡ target_res - b (mod a).
        # We can keep a count of residues of S mod a.
        # But we need distinct b,c.
        
        # Let's just do the pair iteration if S is small.
        found = False
        # To speed up, we can break early.
        # 
        # Optimization: 
        # Precompute residues of S mod a? 
        # 
        # Let's use a helper.
        if _has_pair_sum_mod(S, a, (-x) % a):
            return False
            
    return True

def _has_3sum(S, target):
    # Check if there are 3 distinct elements in S summing to target
    # S is a set
    # O(N^2)
    # Sort S for two pointers? Or just set lookup.
    # Set lookup is O(1).
    # Iterate i, j. Check if target - S[i] - S[j] in S.
    # Ensure distinct.
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        for j in range(i+1, n):
            rem = target - S_list[i] - S_list[j]
            if rem in S:
                # Check distinct
                if rem != S_list[i] and rem != S_list[j]:
                    return True
    return False

def _has_pair_sum_mod(S, a, target_res):
    # Check if there exist distinct b, c in S such that (b+c) % a == target_res
    # S is a set
    # O(N^2)
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        b = S_list[i]
        # We need c such that (b+c) % a == target_res
        # c ≡ target_res - b (mod a)
        needed = (target_res - b) % a
        # Check if there is c in S, c != b, such that c % a == needed
        # We can iterate S_list
        for j in range(n):
            if i == j: continue
            c = S_list[j]
            if c % a == needed:
                return True
    return False

def is_valid(S):
    # Full validation for safety
    for a in S:
        others = [x for x in S if x != a]
        if len(others) < 3: continue
        # Check if a divides sum of any 3 distinct in others
        min_sum = sum(sorted(others)[:3])
        max_sum = sum(sorted(others)[-3:])
        k_min = (min_sum + a - 1) // a
        k_max = max_sum // a
        for k in range(k_min, k_max + 1):
            target = k * a
            if _has_3sum(set(others), target):
                return False
    return True
