def construct():
    n = 256
    candidates = list(range(2, n + 1))
    
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(n)
    
    # Start with all primes
    S = set(primes)
    S_list = list(S)
    
    def is_valid_set(S):
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                for k in range(j + 1, len(S_list)):
                    a, b, c = S_list[i], S_list[j], S_list[k]
                    if (b * c) % a == 0:
                        return False
        return True

    # Precompute conflicts for each candidate with current S
    # To speed up, we just check compatibility
    
    def get_conflicts(x, S):
        """
        Returns the set of elements in S that conflict with x.
        Conflicts:
        1. s | x or x | s
        2. For any t in S \ {s}, s | x*t or x | s*t or t | x*s
        Note: If s | x, then s | x*t is always true. So s is a conflict.
        If x | s, then x | s*t is always true. So s is a conflict.
        """
        conflicts = set()
        for s in S:
            if s == x:
                continue
            # Check direct divisibility
            if s % x == 0 or x % s == 0:
                conflicts.add(s)
                continue
            
            # Check product divisibility with any other t in S
            # We need to check if there EXISTS a t in S \ {s, x} such that:
            # s | x*t OR x | s*t OR t | x*s
            
            # Optimization: 
            # If s | x*t, since gcd(s, x) might not be 1, this is tricky.
            # Just iterate t? That's O(N^2) per conflict check.
            # N=54. 54*54 = 2916 ops. For each of 200 candidates, 200*2916 = 580k ops.
            # This is fast enough.
            
            found_conflict = False
            for t in S:
                if t == s or t == x:
                    continue
                if (x * t) % s == 0:
                    found_conflict = True
                    break
                if (s * t) % x == 0:
                    found_conflict = True
                    break
                if (x * s) % t == 0:
                    found_conflict = True
                    break
            
            if found_conflict:
                conflicts.add(s)
        
        return conflicts

    # Local Search
    improved = True
    while improved:
        improved = False
        # Try adding each candidate not in S
        for x in candidates:
            if x in S:
                continue
            
            conflicts = get_conflicts(x, S)
            
            # If we add x, we must remove all conflicts.
            # Net change in size: 1 - len(conflicts)
            # We only proceed if size increases or stays same (for diversity)?
            # We want to maximize size. So strictly increase.
            if len(conflicts) < 1:
                # No conflicts, just add
                S.add(x)
                improved = True
            elif len(conflicts) == 0: # redundant
                pass
            else:
                # If we remove len(conflicts) elements and add 1, 
                # we gain if 1 > len(conflicts). Impossible if conflicts >= 1.
                # Wait, if len(conflicts) == 1, size stays same.
                # If len(conflicts) == 0, size increases.
                # So we can only add if conflicts is empty?
                # OR we can swap: remove 1, add 1 (size same).
                # OR remove 2, add 1 (size decreases).
                
                # So simple addition only works if conflicts is empty.
                # But we can do SWAPS.
                # Let's stick to addition first.
                pass
        
        # If no simple addition, try swaps?
        # For now, let's try to remove elements to allow additions?
        # Actually, the above loop only adds if conflicts is empty.
        # This will stop at the prime set if no composite is conflict-free.
        
        # Let's try a different move: 
        # Remove a subset R from S, add a subset A from candidates\S.
        # If |A| > |R|, size increases.
        
        # Heuristic: Try to add x, remove minimal conflicts.
        # If conflicts is not empty, we can't add x without removing them.
        # If we remove them, we lose |conflicts| and gain 1.
        # This is bad if |conflicts| >= 1.
        
        # So, how to get > 54?
        # We must find a composite x such that it conflicts with 0 primes?
        # If x is composite, x = p1*p2.
        # If p1, p2 in S, then p1 | x*p2? 
        # p1 | p1*p2*p2. Yes.
        # So p1 is a conflict.
        # So any composite with prime factors in S has conflicts.
        # Thus, we MUST remove prime factors to add composites.
        # Removing 1 prime (say p) allows adding p^2?
        # Conflicts of p^2 with S\{p}:
        # s | p^2 * t?
        # If s is prime q != p, q | p^2 t? No.
        # p^2 | q t? No.
        # t | p^2 q? No.
        # So p^2 has NO conflicts with other primes.
        # So: Remove p, Add p^2.
        # Size change: -1 + 1 = 0.
        # This doesn't increase size.
        
        # Can we remove p and add TWO numbers?
        # e.g. Remove 2. Add 4 and 6?
        # 6 = 2*3. Requires removing 3 as well?
        # If we remove 2 and 3.
        # Add 4 and 9?
        # 4 conflicts with?
        # 9 conflicts with?
        # 4 and 9 conflict with each other?
        # 4 | 9*t? No.
        # 9 | 4*t? No.
        # t | 36? No.
        # So 4 and 9 are compatible.
        # Remove 2, 3 (2 elements). Add 4, 9 (2 elements).
        # Size change 0.
        
        # It seems size 54 is a local maximum for this swap strategy.
        # To get 56, we need a different structure.
        
        # What if we don't start with primes?
        # What if we take all numbers > 128?
        # We found that {129..256} is invalid due to 256|192*160.
        # But if we remove 256, 192, 160?
        # Let's try a Greedy on "Large Numbers" sorted by value descending.
        
        S_alt = set()
        # Sort candidates by value descending
        for x in sorted(candidates, reverse=True):
            # Check if x can be added to S_alt
            # S_alt is likely to be large, so check is expensive.
            # But S_alt grows from empty.
            
            # Quick check: 
            # If S_alt is empty, add.
            # Else, check conflicts.
            
            # Reuse get_conflicts but it assumes S is set.
            # We need a version that checks if x is compatible with S_alt.
            
            compatible = True
            for s in S_alt:
                if s == x: continue
                if s % x == 0 or x % s == 0:
                    compatible = False
                    break
                # Check product with others
                # This is O(N^2) per x.
                # If N=100, 100*100=10k. 255 candidates * 10k = 2.5M. OK.
                
                # Optimization: 
                # If x is large, s is large.
                # s | x*t implies s/gcd(s,x) | t.
                # Let g = gcd(s,x). s' = s/g.
                # We need s' | t.
                # So if there exists t in S_alt \ {s} such that s' | t, then conflict.
                
                g = __import__('math').gcd(s, x)
                s_prime = s // g
                if s_prime == 1:
                    continue # s divides x, handled above? 
                    # If s|x, then s%x==0 caught above.
                    # So s_prime > 1.
                
                # Check if any t in S_alt (t != s) is divisible by s_prime
                # Also check x | s*t -> x/gcd(x,s) | t.
                g2 = __import__('math').gcd(x, s)
                x_prime = x // g2
                # x_prime == s_prime.
                
                # And t | x*s.
                # This is harder to precompute.
                
                # Let's just do the O(N) check for t.
                for t in S_alt:
                    if t == s or t == x:
                        continue
                    if (x * t) % s == 0:
                        compatible = False
                        break
                    if (s * t) % x == 0:
                        compatible = False
                        break
                    if (x * s) % t == 0:
                        compatible = False
                        break
                if not compatible:
                    break
            
            if compatible:
                S_alt.add(x)
        
        if len(S_alt) > len(S):
            S = S_alt
    
    # Verify
    if not is_valid_set(S):
        # Fallback to primes
        S = set(primes)
        
    return list(S)
