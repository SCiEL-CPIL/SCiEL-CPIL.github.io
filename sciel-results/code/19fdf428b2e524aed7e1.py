<<<<<<< SEARCH
import math

def construct():
    N = 180
    S = set(range(91, N + 1))
    
    for a in range(90, 0, -1):
        a2 = a * a
        conflict = False
        
        # d must be such that a+d <= N => d <= N-a
        # and a+a2/d <= N => a2/d <= N-a => d >= a2/(N-a)
        
        min_d = max(1, int(math.ceil(a2 / (N - a))) if N > a else 1)
        max_d = N - a
        
        for d in range(min_d, max_d + 1):
            if a2 % d == 0:
                d1 = d
                d2 = a2 // d
                
                b1 = a + d1
                c1 = a + d2
                
                if b1 == c1:
                    continue
                
                # We know b1 <= N and c1 <= N from min_d/max_d constraints roughly
                # But d2 might be small, so c1 might be small?
                # c1 = a + a2/d. Since d >= min_d, a2/d <= N-a, so c1 <= N.
                # b1 = a + d. Since d <= max_d, b1 <= N.
                
                if b1 in S and c1 in S:
                    conflict = True
                    break
                
        if not conflict:
            S.add(a)
            
    return list(S)
=======
import math
import time

def construct():
    N = 180
    deadline = time.time() + 50  # safety margin before 60s limit
    
    # Precompute all valid triples (a, b, c) with a < b < c, all distinct,
    # and 1/a = 1/b + 1/c
    # 1/a = 1/b + 1/c => c = ab/(a+b), so ab/(a+b) must be integer > b
    # Equivalently: a*b = a*c + b*c => a*b - a*c - b*c = 0
    # Let's enumerate: for each a in [1,N], for each b in [a+1, N],
    # check if c = a*b/(a+b) is an integer in [b+1, N]
    
    # Actually, let me think more carefully.
    # 1/a = 1/b + 1/c where a, b, c are distinct.
    # WLOG assume a < b < c (since the equation is symmetric in b,c but not in a).
    # Wait, 1/a = 1/b + 1/c. Since b,c > a (because 1/b + 1/c < 1/a means a < b and a < c),
    # so a is the smallest. Then b and c are both > a.
    # c = ab/(a+b). For c to be integer, (a+b) must divide ab.
    # Note that gcd(a, a+b) = gcd(a, b), so let g = gcd(a,b), a = g*a', b = g*b' with gcd(a',b')=1.
    # Then c = g*a'*b'/(a'+b'). Since gcd(a', a'+b') = gcd(a', b') = 1, we need (a'+b') | b'.
    # But a' < a'+b', so (a'+b') | b' implies a'+b' <= b', i.e., a' <= 0, contradiction.
    # Wait, that's wrong. Let me redo.
    # c = ab/(a+b). Let g = gcd(a, b). a = g*a', b = g*b', gcd(a',b')=1.
    # c = g*a'*b'/(a'+b'). We need (a'+b') | g*a'*b'.
    # Since gcd(a'+b', a') = gcd(b', a') = 1 and gcd(a'+b', b') = gcd(a', b') = 1,
    # we need (a'+b') | g.
    # So c = g*a'*b'/(a'+b') where (a'+b') divides g.
    
    # Let me just brute force enumerate all triples.
    triples = []  # list of (a, b, c) with a < b < c
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            num = a * b
            den = a + b
            if num % den == 0:
                c = num // den
                if c > b and c <= N:
                    triples.append((a, b, c))
    
    # For each element x, find which triples it participates in
    # An element can appear as a, b, or c in a triple
    # We want to build a large set with no triple fully contained.
    
    # Greedy: start with all elements, then remove elements to break all triples.
    # Better: minimum-degree greedy - at each step, pick the element that is in the
    # fewest "active" triples, add it, and mark those triples as broken.
    
    # Actually, let's use a different greedy: 
    # Start with S = [N+1, ..., N] (large numbers, fewer conflicts)
    # Then try to add smaller numbers in order, checking for conflicts.
    # The current program does this but only checks one direction.
    
    # Let me try: start with all large numbers, then greedily add smaller ones
    # that don't create a triple with existing elements.
    
    S = set(range(N // 2 + 1, N + 1))  # start with [91, ..., 180]
    
    # For each candidate a from 90 down to 1, check if adding it creates a conflict
    # A conflict exists if there are b, c in S such that (a,b,c) is a triple.
    # Since a is the smallest in any triple it's part of, we check:
    # For each b in S with b > a, compute c = a*b/(a+b) and check if c is in S and c > b.
    # Also check if a could be b or c in a triple where the smallest is already in S.
    # Wait, if a is not the smallest, say triple is (a', b, c) with a' < a, and a is b or c.
    # So we need to check all three roles.
    
    # Precompute for each element x, the list of triples it's in
    # And for each pair (x, y), whether they form part of a triple
    
    # For efficiency, for each a, precompute which pairs (b,c) in S would conflict
    # When adding a, we need: no b,c in S such that {a,b,c} is a triple.
    
    # Let's precompute: for each a, the set of (b,c) pairs that form a triple with a
    # where a is the smallest: (a, b, c) with a < b < c
    # where a is the middle: (a', a, c) with a' < a < c  
    # where a is the largest: (a', b, a) with a' < b < a
    
    # Actually, let's just check all triples involving a.
    
    # Build a mapping: for each element, list of triples it participates in
    elem_triples = {x: [] for x in range(1, N + 1)}
    for (a, b, c) in triples:
        elem_triples[a].append((b, c))  # a is smallest, need b and c in S
        elem_triples[b].append((a, c))  # a is smallest, need a and c in S  
        elem_triples[c].append((a, b))  # a is smallest, need a and b in S
    
    # Now greedily add elements from 90 down to 1
    for a in range(90, 0, -1):
        if time.time() > deadline:
            break
        
        conflict = False
        for (x, y) in elem_triples[a]:
            if x in S and y in S:
                conflict = True
                break
        
        if not conflict:
            S.add(a)
    
    return list(S)
>>>>>>> REPLACE

Let me reconsider the approach. The current program scores 1.6778 and the inspirations score 1.6889. Let me look more carefully at what might improve things. The key insight is that we want to maximize the set size. Let me try a smarter greedy that considers the "cost" of adding each element in terms of how many future elements it would block.

