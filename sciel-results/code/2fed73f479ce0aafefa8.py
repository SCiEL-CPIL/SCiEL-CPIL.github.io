import time

def construct():
    N = 180
    # Precompute all triples (a, b, c) with a < b < c
    # 1/a = 1/b + 1/c
    # a = bc / (b+c)
    # Iterate a from 1 to N
    # b, c > a
    # b+c = ab/c => c(b-a) = ab => c = ab / (b-a)
    # Let k = b-a > 0. b = a+k.
    # c = a(a+k)/k = a^2/k + a.
    # So k must divide a^2.
    # c = a + a^2/k.
    # We need c <= N and c > b.
    # c > b => a + a^2/k > a+k => a^2/k > k => a^2 > k^2 => a > k.
    # So k < a.
    # Also k | a^2.
    
    triples = []
    for a in range(1, N + 1):
        for k in range(1, a):
            if a * a % k == 0:
                c = a + a * a // k
                b = a + k
                if c <= N and b < c:
                    # Ensure distinct: b != c is guaranteed by k != a? 
                    # b=c => a+k = a+a^2/k => k = a^2/k => k^2=a^2 => k=a.
                    # But k < a, so b != c.
                    # Also a < b is guaranteed by k > 0.
                    triples.append((a, b, c))
    
    # Greedy construction
    # Sort candidates by value descending? 
    # If we pick large numbers first, we leave more room for smaller ones?
    # Actually, smaller numbers have more constraints.
    # If we pick large numbers first, they don't conflict with each other often (as shown above, a>120 is safe).
    # Then we pick smaller numbers, checking if their partners are already picked.
    
    S = set()
    # Map a to its partners
    partners = {a: set() for a in range(1, N + 1)}
    for a, b, c in triples:
        partners[a].add(b)
        partners[a].add(c)
        # Note: b and c might be partners of other 'a's, but the constraint is only on the min element 'a'.
        # So we only need to check 'a's partners.
        
    # Iterate from N down to 1
    for a in range(N, 0, -1):
        # Check if any pair of partners of 'a' is in S
        # Wait, the condition is: NO THREE DISTINCT elements a,b,c in S.
        # If we add 'a', we must ensure that there is no b,c in S such that {a,b,c} is a triple.
        # Since a is the minimum, we only look at triples where a is the min.
        # So we check if there exists b,c in S such that (a,b,c) is in triples.
        
        # Optimization: for a fixed a, the partners are specific.
        # We need to check if ANY pair of partners is in S?
        # No, we need to check if the SPECIFIC pairs (b,c) that form a triple with a are in S.
        # Let's store the pairs.
        pass

    # Let's restructure to store pairs for each a
    pairs_for_a = {a: [] for a in range(1, N + 1)}
    for a, b, c in triples:
        pairs_for_a[a].append((b, c))

    S = set()
    for a in range(N, 0, -1):
        conflict = False
        for b, c in pairs_for_a[a]:
            if b in S and c in S:
                conflict = True
                break
        if not conflict:
            S.add(a)
            
    return sorted(S)
