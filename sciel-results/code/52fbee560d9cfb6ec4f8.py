import time
from math import gcd

def construct():
    start = time.time()
    deadline = start + 80
    
    n = 256
    candidates = list(range(2, n + 1))
    
    # Precompute: for each a, for each b != a, whether a | b*c for some c
    # More precisely: for each a, precompute which pairs (b,c) in [2,n]^2 have a | b*c
    # We'll use this for fast conflict checking during greedy
    
    # Precompute divisibility: divs[x] = set of y in [2,n] such that x | y
    # Actually, for the greedy check: when adding a, we need to check for all pairs (b,c) in selected:
    # a ∤ b*c, b ∤ a*c, c ∤ a*b
    # We can precompute for each a: the set of b values such that for SOME c, a | b*c
    # But that's not quite right. Let's think differently.
    
    # For fast checking: precompute for each a, a bitmask or set of "conflict partners"
    # Actually, let's precompute: for each a, for each b, the set of c values where a | b*c
    # a | b*c iff (a // gcd(a,b)) | c
    # So for given a,b: m = a//gcd(a,b), and c must be a multiple of m
    
    # Let's precompute a table: conflict_table[a][b] = set of c values in [2,n] where a | b*c
    # This is O(n^2 * n) = O(n^3) which is 16M for n=256, too slow in Python
    
    # Better: for each a, precompute for each b the value m = a//gcd(a,b)
    # Then during greedy, when checking if a can be added:
    # For each pair (b,c) in selected, check if c is a multiple of m_a_b
    # And symmetrically for the other two conditions
    
    # Let's precompute m_table[a][b] = a // gcd(a,b) for all a,b in [2,n]
    m_table = [[0] * (n + 1) for _ in range(n + 1)]
    for a in range(2, n + 1):
        for b in range(2, n + 1):
            m_table[a][b] = a // gcd(a, b)
    
    # Precompute conflict counts for ordering
    # For each a, count how many pairs (b,c) have a | b*c
    conflict_counts = [0] * (n + 1)
    for a in range(2, n + 1):
        cnt = 0
        for b in range(2, n + 1):
            if b == a:
                continue
            m = m_table[a][b]
            if m == 1:
                cnt += n - 2  # all c except a and b
            else:
                # count multiples of m in [2,n], excluding a and b if they are multiples
                count_c = n // m
                if a % m == 0:
                    count_c -= 1
                if b % m == 0:
                    count_c -= 1
                if count_c < 0:
                    count_c = 0
                cnt += count_c
        conflict_counts[a] = cnt
    
    # Try multiple orderings and keep the best
    orderings = [
        sorted(range(2, n + 1), key=lambda x: (conflict_counts[x], x)),
        sorted(range(2, n + 1), key=lambda x: (conflict_counts[x], -x)),
        sorted(range(2, n + 1)),
        sorted(range(2, n + 1), reverse=True),
    ]
    
    best_selected = []
    
    for ordering in orderings:
        if time.time() > deadline - 20:
            break
        
        selected = []
        
        for a in ordering:
            if time.time() > deadline - 25:
                break
            
            valid = True
            sel = selected
            k = len(sel)
            for i in range(k):
                b = sel[i]
                for j in range(i + 1, k):
                    c = sel[j]
                    # Check a | b*c
                    if (b * c) % a == 0:
                        valid = False
                        break
                    # Check b | a*c
                    if (a * c) % b == 0:
                        valid = False
                        break
                    # Check c | a*b
                    if (a * b) % c == 0:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                selected.append(a)
        
        if len(selected) > len(best_selected):
            best_selected = list(selected)
    
    # Final verification
    sel = best_selected
    for i in range(len(sel)):
        a = sel[i]
        for j in range(len(sel)):
            b = sel[j]
            if i == j:
                continue
            for k in range(len(sel)):
                c = sel[k]
                if i == k or j == k:
                    continue
                if (b * c) % a == 0:
                    # Fallback to primes
                    is_prime = [True] * (n + 1)
                    is_prime[0] = False
                    is_prime[1] = False
                    for p in range(2, int(n**0.5) + 1):
                        if is_prime[p]:
                            for q in range(p * p, n + 1, p):
                                is_prime[q] = False
                    return [x for x in range(2, n + 1) if is_prime[x]]
    
    if not best_selected:
        # Fallback to primes
        is_prime = [True] * (n + 1)
        is_prime[0] = False
        is_prime[1] = False
        for p in range(2, int(n**0.5) + 1):
            if is_prime[p]:
                for q in range(p * p, n + 1, p):
                    is_prime[q] = False
        best_selected = [x for x in range(2, n + 1) if is_prime[x]]
    
    return list(best_selected)
