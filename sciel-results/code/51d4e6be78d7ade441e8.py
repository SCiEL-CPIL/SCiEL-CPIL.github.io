import time

def construct():
    start = time.time()
    deadline = start + 80  # safety margin
    
    def can_add(a, S):
        """
        Check if adding 'a' to valid set 'S' keeps it valid.
        Uses algebraic structure: 1/a=1/b+1/c iff (b-a)(c-a)=a^2 for a<b<c.
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
        for x in S:
            if x >= a:
                continue
            d = a - x
            if d == 0:
                continue
            num = a * x
            if num % d != 0:
                continue
            c = num // d
            if c > a and c != x and c in S:
                return False
        
        # Case 3: a is the largest (z)
        # 1/x = 1/y + 1/a => y = a*x / (a-x)
        for x in S:
            if x >= a:
                continue
            d = a - x
            if d == 0:
                continue
            num = a * x
            if num % d != 0:
                continue
            y = num // d
            if y > x and y < a and y != x and y in S:
                return False
        
        return True
    
    # Precompute all valid triples for verification
    all_triples = []
    for a in range(1, 181):
        a2 = a * a
        for d in range(1, a2 + 1):
            if a2 % d != 0:
                continue
            b = a + d
            c = a + a2 // d
            if b > a and c > b and b <= 180 and c <= 180:
                all_triples.append((a, b, c))
    
    def is_valid(S):
        for (a, b, c) in all_triples:
            if a in S and b in S and c in S:
                return False
        return True
    
    odds = set(i for i in range(1, 181) if i % 2 == 1)
    evens = list(range(2, 181, 2))
    
    # Strategy 1: Start with odds, add evens ascending
    result1 = set(odds)
    for x in evens:
        if time.time() > deadline - 5:
            break
        if can_add(x, result1):
            result1.add(x)
    
    best = result1
    best_size = len(best)
    
    # Strategy 2: Start with odds, add evens descending
    if time.time() < deadline - 10:
        result2 = set(odds)
        for x in sorted(evens, reverse=True):
            if time.time() > deadline - 5:
                break
            if can_add(x, result2):
                result2.add(x)
        if len(result2) > best_size:
            best = result2
            best_size = len(best)
    
    # Strategy 3: Start empty, add all numbers ascending
    if time.time() < deadline - 10:
        result3 = set()
        for x in range(1, 181):
            if time.time() > deadline - 5:
                break
            if can_add(x, result3):
                result3.add(x)
        if len(result3) > best_size:
            best = result3
            best_size = len(best)
    
    # Strategy 4: Start empty, add all numbers descending
    if time.time() < deadline - 10:
        result4 = set()
        for x in range(180, 0, -1):
            if time.time() > deadline - 5:
                break
            if can_add(x, result4):
                result4.add(x)
        if len(result4) > best_size:
            best = result4
            best_size = len(best)
    
    # Strategy 5: Local search - remove one, re-add greedily
    current = set(best)
    if time.time() < deadline - 3:
        for _ in range(5):
            if time.time() > deadline - 3:
                break
            improved = False
            for remove_elem in sorted(current):
                if time.time() > deadline - 3:
                    break
                trial = current - {remove_elem}
                for x in range(1, 181):
                    if x in trial:
                        continue
                    if can_add(x, trial):
                        trial.add(x)
                if len(trial) > len(current):
                    current = trial
                    improved = True
                    break
            if not improved:
                break
    
    if len(current) > len(best):
        best = current
    
    # Final verification
    if not is_valid(best):
        best = odds
    
    return list(best)
