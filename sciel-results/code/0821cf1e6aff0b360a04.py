import time
import random

def construct():
    deadline = time.time() + 85  # Safety margin for 90s limit
    
    def can_add(a, S):
        """
        Check if adding 'a' to valid set 'S' keeps it valid.
        S must be a set for O(1) lookup.
        """
        # Case 1: a is the smallest (x)
        # 1/a = 1/b + 1/c => (b-a)(c-a) = a^2
        a2 = a * a
        for b in S:
            if b <= a:
                continue
            d = b - a
            if a2 % d != 0:
                continue
            c = a + a2 // d
            if c > a and c != b and c in S:
                return False
        
        # Case 2: a is the middle (y)
        # 1/x = 1/a + 1/c => c = a*x / (a-x)
        # x < a < c
        for x in S:
            if x >= a:
                continue
            denom = a - x
            num = a * x
            if num % denom != 0:
                continue
            c = num // denom
            if c > a and c in S:
                return False
        
        # Case 3: a is the largest (z)
        # 1/x = 1/y + 1/a => y = a*x / (a-x)
        # x < y < a
        for x in S:
            if x >= a:
                continue
            denom = a - x
            num = a * x
            if num % denom != 0:
                continue
            y = num // denom
            if x < y < a and y in S:
                return False
        
        return True

    # Start with a strong base: All odds + Powers of 2
    S = set()
    for i in range(1, 181):
        if i % 2 != 0:
            S.add(i)
        else:
            # Check if power of 2
            p = i
            is_pow2 = True
            while p % 2 == 0:
                p //= 2
            if p == 1:
                S.add(i)
    
    # Greedily add remaining numbers
    # Sort candidates by some heuristic? Maybe just random or ascending.
    # Ascending might be better to find small conflicts early?
    # Or descending to keep large numbers?
    # Let's try ascending.
    for a in range(1, 181):
        if time.time() > deadline:
            break
        if a not in S:
            if can_add(a, S):
                S.add(a)
    
    # Local search
    random.seed(42)
    best_S = set(S)
    best_size = len(S)
    
    all_nums = set(range(1, 181))
    
    # Hill climbing
    for _ in range(10000): # Limit iterations
        if time.time() > deadline:
            break
        
        excluded = all_nums - S
        if not excluded:
            break
        
        a = random.choice(list(excluded))
        if can_add(a, S):
            S.add(a)
            if len(S) > best_size:
                best_S = set(S)
                best_size = len(S)
        else:
            # Try swap
            if S:
                b = random.choice(list(S))
                S.remove(b)
                if can_add(a, S):
                    S.add(a)
                    if len(S) > best_size:
                        best_S = set(S)
                        best_size = len(S)
                else:
                    S.add(b)
    
    return sorted(best_S)
