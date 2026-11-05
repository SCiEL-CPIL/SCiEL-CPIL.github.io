import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 256
    
    def is_valid_set(S):
        """Check if S (as a set) satisfies the property: for all a<b<=c in S, a ∤ (b+c)."""
        S_list = sorted(S)
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            # For all j > i, k >= j: check (S_list[j] + S_list[k]) % a != 0
            for j in range(i + 1, m):
                b = S_list[j]
                # Check b+b (j=k=j)
                if (b + b) % a == 0:
                    return False
                for k in range(j + 1, m):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    def can_add(S, x):
        """Check if x can be added to set S without violating the property."""
        if x in S:
            return False
        S_list = sorted(S)
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            if a >= x:
                # a >= x, so a cannot be the smallest in any triple involving x as b or c
                # But wait: if a > x, then in any triple a < b <= c, if x is b or c,
                # we need a < x for a to be the smallest. Since a >= x, a is not < x,
                # so a cannot be the 'a' in a triple where x is b or c.
                # However, we also need to check if x could be the 'a' in a triple.
                # If x is 'a', then we need x < b <= c where b,c in S.
                # So we need to check: for all b, c in S with x < b <= c, x ∤ (b+c)
                # This is a separate case. Let's handle it below.
                pass
            else:
                # a < x
                # Case 1: x is 'c' (largest). Need a < b <= x for b in S.
                # Check: for all b in S with a < b <= x, (b + x) % a != 0
                for j in range(m):
                    b = S_list[j]
                    if a < b <= x:
                        if (b + x) % a == 0:
                            return False
                # Case 2: x is 'b' (middle). Need a < x <= c for c in S.
                # Check: for all c in S with c >= x, (x + c) % a != 0
                for j in range(m):
                    c = S_list[j]
                    if c >= x:
                        if (x + c) % a == 0:
                            return False
        # Case 3: x is 'a' (smallest). Need x < b <= c for b, c in S.
        # Check: for all b, c in S with x < b <= c, (b + c) % x != 0
        for i in range(m):
            b = S_list[i]
            if b > x:
                # Check b+b
                if (b + b) % x == 0:
                    return False
                for j in range(i + 1, m):
                    c = S_list[j]
                    if (b + c) % x == 0:
                        return False
        return True
    
    best_size = 0
    best_S = set()
    
    # Strategy 1: Start with upper half, greedily add from below
    for restart in range(50):
        if time.time() > deadline - 5.0:
            break
        
        rng = random.Random(restart * 1000 + 42)
        
        # Start with a random subset of the upper portion
        base_start = n - rng.randint(40, 80)
        S = set(range(base_start, n + 1))
        
        # Verify and fix if needed
        if not is_valid_set(S):
            S_list = sorted(S)
            valid_S = set()
            for x in S_list:
                if can_add(valid_S, x):
                    valid_S.add(x)
            S = valid_S
        
        # Greedily add remaining elements in random order
        candidates = [x for x in range(1, base_start) if x not in S]
        rng.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 3.0:
                break
            if can_add(S, x):
                S.add(x)
        
        # Try to add more in another pass
        candidates2 = [x for x in range(1, n + 1) if x not in S]
        rng.shuffle(candidates2)
        for x in candidates2:
            if time.time() > deadline - 3.0:
                break
            if can_add(S, x):
                S.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    # Strategy 2: Start with a random subset, greedily add
    for restart in range(50):
        if time.time() > deadline - 5.0:
            break
        
        rng = random.Random(restart * 2000 + 99)
        
        # Start with a random subset of size ~30
        S = set(rng.sample(range(1, n + 1), rng.randint(20, 40)))
        
        # Verify and fix if needed
        if not is_valid_set(S):
            S_list = sorted(S)
            valid_S = set()
            for x in S_list:
                if can_add(valid_S, x):
                    valid_S.add(x)
            S = valid_S
        
        # Greedily add remaining elements in random order
        candidates = [x for x in range(1, n + 1) if x not in S]
        rng.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 3.0:
                break
            if can_add(S, x):
                S.add(x)
        
        # Try to add more in another pass
        candidates2 = [x for x in range(1, n + 1) if x not in S]
        rng.shuffle(candidates2)
        for x in candidates2:
            if time.time() > deadline - 3.0:
                break
            if can_add(S, x):
                S.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    # Strategy 3: Start with all odd numbers (they have good properties), then add evens
    if time.time() < deadline - 5.0:
        S = set(range(1, n + 1, 2))  # All odd numbers
        if not is_valid_set(S):
            S_list = sorted(S)
            valid_S = set()
            for x in S_list:
                if can_add(valid_S, x):
                    valid_S.add(x)
            S = valid_S
        
        candidates = [x for x in range(2, n + 1, 2) if x not in S]
        rng = random.Random(12345)
        rng.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 3.0:
                break
            if can_add(S, x):
                S.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    # Strategy 4: Start with numbers > n/3, then add smaller
    if time.time() < deadline - 5.0:
        S = set(range(86, n + 1))  # Numbers > 256/3 ≈ 85.33
        if not is_valid_set(S):
            S_list = sorted(S)
            valid_S = set()
            for x in S_list:
                if can_add(valid_S, x):
                    valid_S.add(x)
            S = valid_S
        
        candidates = [x for x in range(1, 86) if x not in S]
        rng = random.Random(54321)
        rng.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 3.0:
                break
            if can_add(S, x):
                S.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    if best_size == 0:
        # Fallback: return a valid set (upper half)
        return list(range(129, 257))
    
    # Final verification
    if not is_valid_set(best_S):
        # Should not happen, but fallback
        S = set(range(129, 257))
        return sorted(S)
    
    return sorted(best_S)
