def construct():
    from math import isqrt
    
    def gen_powerful(limit):
        """Generate all powerful numbers up to limit."""
        pw = set()
        # A powerful number: for each prime p, exponent >= 2 if p divides it
        # Form: product of p_i^{e_i} where e_i >= 2
        # Generate by: for each squarefree part s, multiply by squares
        # Simpler: generate all numbers of form a^2 * b^3 where b is squarefree? No.
        # Actually, powerful numbers are exactly those where in prime factorization, 
        # every exponent is >= 2.
        # Generate: iterate over possible "core" = product of p^2 for subset of primes,
        # then multiply by p for additional powers.
        
        # Better: for each integer m from 1 to limit, check if powerful? Too slow.
        
        # Standard method: powerful numbers up to N can be generated as:
        # For each squarefree d, and for each k such that d * k^2 * (something) <= N
        # Actually, every powerful number a can be written uniquely as a = s * t^2 where s is powerful and squarefree? No.
        
        # Simpler approach: 
        # A number a is powerful iff for every prime p|a, p^2|a.
        # Generate by: a = product of p_i^{e_i}, e_i >= 2.
        # We can generate by iterating over the "squarefree kernel" but that's complex.
        
        # Let's use: powerful numbers are of the form x^2 * y^3 where y is squarefree? 
        # No, that's not right either.
        
        # Correct: a powerful number has the form: for its prime factorization, 
        # a = p1^{e1} * p2^{e2} * ... where ei >= 2.
        # We can generate by: let a = m^2 * n where n is such that in the factorization of n,
        # every prime that appears does so to an even power? No.
        
        # Let me just do: for a to be powerful, sqrt(a) must be an integer or... no.
        # Check: a is powerful iff for all primes p | a, p^2 | a.
        # Equivalent: a / rad(a)^2 is an integer? Where rad is radical. Hmm.
        
        # Practical: generate all a = u^2 * v^3 where v is squarefree and gcd(u,v)=1? 
        # This is a known parametrization but let me verify.
        # If a is powerful, write a = product p_i^{e_i}, e_i >= 2.
        # Let e_i = 2*f_i + g_i where g_i in {0,1}. Then a = (product p_i^{f_i})^2 * product p_i^{g_i}.
        # So a = u^2 * w where w is squarefree (product of p_i with g_i=1).
        # But w being squarefree and u can share primes with w.
        # Actually this works: a = u^2 * w, w squarefree, and that's it.
        # Check: a=12=2^2*3. u^2*w: w=3, u^2=4, u=2. 4*3=12. ✓
        # a=72=2^3*3^2. w=2, u^2=36, u=6. 36*2=72. ✓
        # a=25=5^2. w=1, u=5. 25*1=25. ✓
        # So: powerful numbers up to L = {u^2 * w : w squarefree, u^2*w <= L}
        
        squarefree = []
        for w in range(1, limit + 1):
            # Check if w is squarefree
            n = w
            is_sf = True
            p = 2
            while p * p <= n:
                if n % (p * p) == 0:
                    is_sf = False
                    break
                n //= p
                p += 1 if p == 2 else 2
            if is_sf:
                squarefree.append(w)
        
        pw = set()
        max_u = isqrt(limit)
        for u in range(1, max_u + 1):
            u2 = u * u
            for w in squarefree:
                a = u2 * w
                if a > limit:
                    break
                pw.add(a)
        return sorted(pw)
    
    # Generate powerful numbers up to a limit
    # Let's try limit = 500000
    limit = 500000
    pw = gen_powerful(limit)
    pw_set = set(pw)
    
    # Count pairs
    max_sum = 2 * limit
    counts = [0] * (max_sum + 1)
    
    n = len(pw)
    for i in range(n):
        a = pw[i]
        for j in range(i, n):
            b = pw[j]
            s = a + b
            if s > max_sum:
                break
            counts[s] += 1
    
    # Find best
    best_N = 2
    best_count = 0
    for s in range(2, max_sum + 1):
        if counts[s] > best_count:
            best_count = counts[s]
            best_N = s
    
    # Collect pairs
    pairs = []
    for a in pw:
        b = best_N - a
        if a <= b and b in pw_set:
            pairs.append([a, b])
    
    return {'N': best_N, 'pairs': pairs}
