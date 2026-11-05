import time
import random
import math

def construct():
    deadline = time.time() + 50  # Safety margin: return well before 60s
    
    def get_divisors(n):
        """Return a list of divisors of n."""
        divs = []
        i = 1
        while i * i <= n:
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
            i += 1
        return divs

    def find_conflict(a, S):
        """
        Check if adding 'a' to valid set 'S' creates a conflict.
        Returns (b, c) if conflict exists, else None.
        S must be a set.
        Optimized to iterate over divisors of a^2 instead of S.
        """
        a2 = a * a
        divs = get_divisors(a2)
        
        # Case 1: a is the smallest (x)
        # 1/a = 1/b + 1/c => (b-a)(c-a) = a^2
        # Let b = a + d, c = a + a^2/d.
        # We need d | a^2, d > 0.
        # b, c > a. Distinct: b != c => d != a^2/d => d^2 != a^2 => d != a.
        # Also b, c must be in S.
        for d in divs:
            if d >= a: # If d >= a, then c = a + a^2/d <= a + a = 2a. 
                       # If d == a, b=c=a+a=2a, not distinct.
                       # If d > a, then c < b. We can just check all d.
                       # Actually, b and c are symmetric in the equation 1/a = 1/b+1/c.
                       # We just need to check if BOTH are in S.
                       # d must be such that b=a+d and c=a+a^2/d are integers (they are).
                       # We require b != c, so d != a.
                       if d == a:
                           continue
            b = a + d
            c = a + a2 // d
            if b > 180 or c > 180:
                continue
            if b in S and c in S:
                return (b, c)
        
        # Case 2: a is the middle (y)
        # 1/x = 1/a + 1/c => c = ax / (a-x)
        # Let k = a-x. Then x = a-k. c = a(a-k)/k = a^2/k - a.
        # k must divide a^2.
        # x < a => k > 0.
        # c > a => a^2/k - a > a => a^2/k > 2a => k < a/2.
        # x >= 1 => a-k >= 1 => k <= a-1.
        for k in divs:
            if k >= a: # k < a/2 is stricter
                continue
            if k == 0:
                continue
            # Check if k < a/2
            if 2 * k >= a:
                continue
            x = a - k
            c = a2 // k - a
            if x < 1 or c > 180:
                continue
            if x in S and c in S:
                return (x, c)
                
        # Case 3: a is the largest (z)
        # 1/x = 1/y + 1/a => y = ax / (a-x)
        # Let k = a-x. Then x = a-k. y = a(a-k)/k = a^2/k - a.
        # k must divide a^2.
        # x < y < a.
        # y < a => a^2/k - a < a => a^2/k < 2a => k > a/2.
        # x < y => a-k < a^2/k - a => 2a < k + a^2/k.
        # x >= 1 => k <= a-1.
        for k in divs:
            if k <= a // 2: # k > a/2
                continue
            if k >= a:
                continue
            x = a - k
            y = a2 // k - a
            if x < 1 or y > 180:
                continue
            if x < y < a and x in S and y in S:
                return (x, y)
        
        return None

    # Start with a known valid set: all numbers > 90
    # This ensures no 1/a = 1/b + 1/c because 1/b + 1/c <= 2/91 < 1/45.5, 
    # while 1/a >= 1/180. Wait.
    # If a, b, c > 90, then 1/a, 1/b, 1/c < 1/90.
    # 1/b + 1/c < 2/90 = 1/45.
    # 1/a >= 1/180.
    # Is it possible 1/a = 1/b + 1/c?
    # Max RHS is 2/91 approx 0.0219.
    # Min LHS is 1/180 approx 0.0055.
    # So yes, it is possible. e.g. 1/46 = 1/92 + 1/92 (but distinct elements required).
    # 1/47 = 1/94 + 1/94 (not distinct).
    # Let's check if {91..180} is valid.
    # We need distinct a,b,c.
    # Smallest a is 91. 1/91 approx 0.0109.
    # Max 1/b+1/c with b,c > 91 is 2/92 approx 0.0217.
    # So 1/91 could equal 1/b+1/c?
    # If b=92, c=92, sum=1/46. 1/46 != 1/91.
    # If b=92, c=184 (out of range).
    # Generally, for large numbers, 1/b+1/c is small.
    # Let's just use the greedy approach from 1 to 180 to be safe, or start with a smaller set.
    # Actually, starting with empty and adding 1..180 is safer and fast enough with optimized conflict check.
    
    S = set()
    
    # Greedily add numbers 1 to 180
    for a in range(1, 181):
        if time.time() > deadline:
            break
        if find_conflict(a, S) is None:
            S.add(a)
    
    best_S = set(S)
    best_size = len(S)
    
    all_nums = set(range(1, 181))
    
    random.seed(42)
    
    # Local search with targeted swaps
    # With fast conflict check, we can do more iterations
    stuck_count = 0
    max_stuck = 2000
    
    while time.time() < deadline:
        excluded = all_nums - S
        if not excluded:
            break
        
        # Try to add a random excluded element
        a = random.choice(list(excluded))
        conflict = find_conflict(a, S)
        
        if conflict is None:
            S.add(a)
            if len(S) > best_size:
                best_S = set(S)
                best_size = len(S)
            stuck_count = 0
        else:
            b, c = conflict
            # Try to remove b and add a
            S.remove(b)
            new_conflict = find_conflict(a, S)
            if new_conflict is None:
                S.add(a)
                if len(S) > best_size:
                    best_S = set(S)
                    best_size = len(S)
                stuck_count = 0
            else:
                # Revert b
                S.add(b)
                # Try to remove c and add a
                S.remove(c)
                new_conflict2 = find_conflict(a, S)
                if new_conflict2 is None:
                    S.add(a)
                    if len(S) > best_size:
                        best_S = set(S)
                        best_size = len(S)
                    stuck_count = 0
                else:
                    # Revert c
                    S.add(c)
                    # If targeted swap failed, try a random swap as fallback
                    if S:
                        b_rand = random.choice(list(S))
                        S.remove(b_rand)
                        new_conflict3 = find_conflict(a, S)
                        if new_conflict3 is None:
                            S.add(a)
                            if len(S) > best_size:
                                best_S = set(S)
                                best_size = len(S)
                            stuck_count = 0
                        else:
                            S.add(b_rand)
                            stuck_count += 1
                            
        # If stuck for too long, restart from best state with perturbation
        if stuck_count > max_stuck:
            S = set(best_S)
            if S:
                for _ in range(5):
                    if not S: break
                    S.remove(random.choice(list(S)))
            stuck_count = 0

    return sorted(best_S)
