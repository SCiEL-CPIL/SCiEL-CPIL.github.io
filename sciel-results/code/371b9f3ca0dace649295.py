import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    n = 256
    
    # Precompute: for each a, the set of sums in [2, 2n] divisible by a
    divs = [set() for _ in range(n + 1)]
    for a in range(1, n + 1):
        for mult in range(2, 2 * n // a + 1):
            v = a * mult
            if v <= 2 * n:
                divs[a].add(v)
    
    def check_decreasing_add(S, x):
        """Check if x can be added when all elements in S are > x.
        Only need to check: for all b,c in S with b<=c, x ∤ (b+c)."""
        for i in range(len(S)):
            b = S[i]
            for j in range(i, len(S)):
                c = S[j]
                if (b + c) in divs[x]:
                    return False
        return True
    
    def full_check_add(S_set, x):
        """Full check for adding x to S_set (x can be anywhere in ordering)."""
        S_list = list(S_set)
        
        # Case 1: x is 'a' (smallest in triple)
        for b in S_list:
            if b <= x:
                continue
            for c in S_list:
                if c < b:
                    continue
                if (b + c) in divs[x]:
                    return False
        
        # Case 2: x is 'b' (middle)
        for a in S_list:
            if a >= x:
                continue
            for c in S_list:
                if c < x:
                    continue
                if (x + c) in divs[a]:
                    return False
        
        # Case 3: x is 'c' (largest)
        for a in S_list:
            for b in S_list:
                if a >= b:
                    continue
                if b >= x:
                    continue
                if (b + x) in divs[a]:
                    return False
        
        return True
    
    def full_validate(S):
        """Full validation of set S."""
        S_sorted = sorted(S)
        k = len(S_sorted)
        for i in range(k):
            a = S_sorted[i]
            for j in range(i + 1, k):
                b = S_sorted[j]
                for l in range(j, k):
                    c = S_sorted[l]
                    if (b + c) % a == 0:
                        return False
        return True
    
    best_set = []
    best_size = 0
    rng = random.Random(42)
    
    # Strategy 1: Decreasing greedy - only check Case 1 (fastest)
    S = []
    for x in range(256, 0, -1):
        if time.time() > deadline - 5.0:
            break
        if check_decreasing_add(S, x):
            S.append(x)
    
    if len(S) > best_size:
        best_size = len(S)
        best_set = list(S)
    
    # Strategy 2: Start with upper half [129,256], try to add lower elements
    for trial in range(10):
        if time.time() > deadline - 5.0:
            break
        
        S = list(range(129, 257))
        S_set = set(S)
        
        # Try adding elements from 128 down to 1
        for x in range(128, 0, -1):
            if time.time() > deadline - 5.0:
                break
            if full_check_add(S_set, x):
                S.append(x)
                S_set.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
    
    # Strategy 3: Different upper ranges
    for upper_start in [140, 150, 160, 170, 180, 190, 200]:
        if time.time() > deadline - 5.0:
            break
        
        S = list(range(upper_start, 257))
        S_set = set(S)
        
        # Validate base first
        if not full_validate(S):
            # Fix: greedily build from top
            S = []
            S_set = set()
            for x in range(256, upper_start - 1, -1):
                if check_decreasing_add(S, x):
                    S.append(x)
                    S_set.add(x)
        
        # Try adding lower elements
        for x in range(upper_start - 1, 0, -1):
            if time.time() > deadline - 5.0:
                break
            if full_check_add(S_set, x):
                S.append(x)
                S_set.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
    
    # Strategy 4: Randomized orderings
    for trial in range(30):
        if time.time() > deadline - 5.0:
            break
        
        S = []
        S_set = set()
        candidates = list(range(1, 257))
        rng.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline - 5.0:
                break
            if full_check_add(S_set, x):
                S.append(x)
                S_set.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
    
    # Strategy 5: Weighted random orderings (bias toward larger numbers)
    for trial in range(20):
        if time.time() > deadline - 5.0:
            break
        
        S = []
        S_set = set()
        candidates = list(range(1, 257))
        # Create weighted sampling: larger numbers more likely
        weights = [float(x) ** 2 for x in candidates]
        total_w = sum(weights)
        
        # Sample without replacement using weights
        remaining = list(range(1, 257))
        remaining_weights = [float(x) ** 2 for x in remaining]
        
        while remaining:
            if time.time() > deadline - 5.0:
                break
            total_w = sum(remaining_weights)
            r = rng.random() * total_w
            cum = 0
            for i, w in enumerate(remaining_weights):
                cum += w
                if r <= cum:
                    x = remaining[i]
                    if full_check_add(S_set, x):
                        S.append(x)
                        S_set.add(x)
                    remaining.pop(i)
                    remaining_weights.pop(i)
                    break
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
    
    # Final validation
    if not full_validate(best_set):
        # Fallback: decreasing greedy
        S = []
        for x in range(256, 0, -1):
            if check_decreasing_add(S, x):
                S.append(x)
        best_set = S
    
    return best_set
