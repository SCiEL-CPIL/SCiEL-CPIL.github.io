def construct():
    import time
    import random
    from math import gcd
    from itertools import combinations

    deadline = time.time() + 85

    n = 128
    # Precompute GCD table
    gcd_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(1, n+1):
            gcd_table[i][j] = gcd(i, j)

    # Identify all "bad" triples: (a, b, c) with a < b < c
    # such that gcd(a,b) == gcd(a,c) == gcd(b,c)
    # We want to avoid these.
    # Let's model this as a graph problem or hypergraph problem.
    # Since the constraint is on triples, it's a 3-uniform hypergraph.
    # However, we can simplify.
    
    # Observation: If we pick all numbers that are powers of distinct primes, 
    # any two have gcd 1. So all pairwise gcds are 1.
    # Then for any triple, all three pairwise gcds are 1. This is a BAD triple.
    # So we cannot just pick coprime numbers.
    
    # The condition is: NOT (gcd(a,b) == gcd(a,c) == gcd(b,c)).
    # Let g_ab = gcd(a,b), g_ac = gcd(a,c), g_bc = gcd(b,c).
    # We need g_ab != g_ac or g_ab != g_bc or g_ac != g_bc.
    
    # Let's look at the structure of numbers.
    # If a, b, c are all even, gcds are at least 2.
    # If a, b, c are all multiples of 3, gcds are at least 3.
    
    # Let's try to construct a large set.
    # Strategy: Partition [1, 128] into sets such that within each set,
    # no triple has equal pairwise gcds.
    
    # A known construction for this type of problem (avoiding equal gcd triples)
    # often involves selecting numbers based on their largest prime factor or similar.
    
    # Let's try a greedy approach with local search, but optimized.
    # The current program scores 1.5. Let's see if we can improve the search.
    
    # Precompute for each pair (a,b) the value gcd(a,b).
    # For a candidate x to be added to S, we check all pairs (a,b) in S.
    # This is O(|S|^2) per insertion.
    
    # Better: Maintain a count of "bad" potential triples?
    # Or just use a fast check.
    
    # Let's try a different angle:
    # What if we take all numbers that are NOT divisible by any small prime?
    # Or all numbers that are powers of 2?
    # {1, 2, 4, 8, 16, 32, 64}.
    # gcd(1,2)=1, gcd(1,4)=1, gcd(2,4)=2. Not all equal.
    # gcd(2,4)=2, gcd(4,8)=4, gcd(2,8)=2. Not all equal.
    # Actually, for powers of 2: gcd(2^i, 2^j) = 2^min(i,j).
    # For triple i<j<k: gcd(2^i, 2^j)=2^i, gcd(2^i, 2^k)=2^i, gcd(2^j, 2^k)=2^j.
    # Since i < j, 2^i != 2^j. So powers of 2 are SAFE.
    # Same for powers of any prime.
    
    # What about mixing primes?
    # If we take {2^i, 3^j}, gcd is 1.
    # Triple: 2^a, 2^b, 3^c.
    # gcd(2^a, 2^b) = 2^min(a,b)
    # gcd(2^a, 3^c) = 1
    # gcd(2^b, 3^c) = 1
    # Are they equal? Only if 2^min(a,b) == 1, i.e., min(a,b)=0.
    # So if we include 1 (2^0 or 3^0), we might have issues.
    # If a,b >= 1, then gcd(2^a, 2^b) >= 2, while others are 1. Safe.
    
    # So we can combine powers of different primes, as long as we don't include 1?
    # Or if we include 1, we must be careful.
    # If 1 is in S, and a,b are powers of 2 (>=2), then:
    # gcd(1,a)=1, gcd(1,b)=1, gcd(a,b)>=2. Not equal. Safe.
    
    # It seems we can take:
    # All powers of 2: 1, 2, 4, 8, 16, 32, 64 (7 numbers)
    # All powers of 3: 3, 9, 27, 81 (4 numbers)
    # All powers of 5: 5, 25 (2 numbers)
    # All powers of 7: 7 (1 number)
    # All powers of 11: 11 (1 number)
    # ...
    
    # Let's check if mixing these creates bad triples.
    # Triple from three different prime bases: p^a, q^b, r^c.
    # gcds are all 1. BAD!
    # So we cannot have numbers from three different prime bases.
    
    # So we can pick numbers from AT MOST TWO prime bases.
    # Base 2: 1, 2, 4, 8, 16, 32, 64
    # Base 3: 3, 9, 27, 81
    # If we take all powers of 2 and all powers of 3:
    # Triples within powers of 2: Safe (as shown above).
    # Triples within powers of 3: Safe.
    # Triples with two from 2 and one from 3: Safe (gcds are 2^k, 1, 1).
    # Triples with one from 2 and two from 3: Safe.
    # Triples with one from 2, one from 3, one from... wait, we only have two bases.
    
    # So the set {1, 2, 4, 8, 16, 32, 64, 3, 9, 27, 81} is valid?
    # Size = 7 + 4 = 11.
    # This is small. We want to maximize cardinality.
    
    # What if we use composite numbers?
    # The condition is about gcds.
    
    # Let's look at the current best score 1.5.
    # The seed program likely gets a decent set via greedy + hill climb.
    # Let's try to improve the hill climb.
    
    # Key insight: The constraint is local.
    # Let's try to build a set that is large.
    
    # Alternative idea:
    # Consider the set of all numbers in [1, 128] that are NOT divisible by 6?
    # Or something similar.
    
    # Let's stick to the hypergraph independent set approach but optimize it.
    # The current program uses a simple greedy + hill climb.
    # Let's try a more sophisticated local search: Simulated Annealing or Tabu Search.
    
    # Or, let's try to find a structural pattern.
    # If we take all odd numbers:
    # gcds are odd.
    # Triple 3, 9, 27: gcds 3, 3, 9. Not equal.
    # Triple 1, 3, 9: gcds 1, 1, 3. Not equal.
    # Triple 3, 15, 45: gcds 3, 3, 15. Not equal.
    # Is the set of all odd numbers valid?
    # Let's check a potential bad triple: a, b, c all odd.
    # We need gcd(a,b) = gcd(a,c) = gcd(b,c) = g.
    # This implies g divides a, b, c.
    # Let a = g*a', b = g*b', c = g*c'.
    # gcd(a', b') = gcd(a', c') = gcd(b', c') = 1.
    # So a', b', c' are pairwise coprime.
    # Example: g=3, a'=1, b'=5, c'=7.
    # a=3, b=15, c=21.
    # gcd(3,15)=3, gcd(3,21)=3, gcd(15,21)=3.
    # So {3, 15, 21} is a bad triple!
    # So all odd numbers is NOT valid.
    
    # How to avoid this?
    # We need to ensure that for any g, the set of numbers divisible by g
    # does not contain three numbers that are pairwise coprime when divided by g.
    
    # This is complex. Let's go back to local search.
    # The current program's hill climb is:
    # 1. Try to add any element that doesn't create a bad triple.
    # 2. Repeat until no more can be added.
    
    # Improvement:
    # 1. Start with a good initial set.
    # 2. Use a "kick" phase: remove a few random elements, then re-add greedily.
    # 3. Use a "swap" phase: try swapping one element for another.
    
    # Let's implement a more robust local search.
    
    # Precompute bad triples for faster checking?
    # There are C(128, 3) ~ 340k triples.
    # We can precompute for each pair (a,b) the list of c such that (a,b,c) is bad.
    # Or for each x, the list of pairs (a,b) such that (a,b,x) is bad.
    
    # Let's precompute:
    # bad_pairs_for_x[x] = set of pairs (a,b) such that gcd(a,b)==gcd(a,x)==gcd(b,x)
    
    # This will speed up can_add significantly.
    
    bad_pairs_for_x = [set() for _ in range(n+1)]
    # For each triple (a,b,c) with a<b<c
    # If it's bad, add (a,b) to bad_pairs_for_x[c], (a,c) to bad_pairs_for_x[b], (b,c) to bad_pairs_for_x[a]
    # Actually, we just need to check if adding x to S creates a bad triple.
    # A bad triple involving x must have two other elements a, b in S.
    # So we need to check if there exists a, b in S such that (a,b,x) is bad.
    # This is equivalent to: does S contain a pair (a,b) that is in bad_pairs_for_x[x]?
    
    # Let's build bad_pairs_for_x.
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            for c in range(b+1, n+1):
                g_ab = gcd_table[a][b]
                g_ac = gcd_table[a][c]
                g_bc = gcd_table[b][c]
                if g_ab == g_ac == g_bc:
                    bad_pairs_for_x[c].add((a,b))
                    bad_pairs_for_x[b].add((a,c))
                    bad_pairs_for_x[a].add((b,c))
    
    # Now can_add(x, S) is:
    # For each pair (a,b) in S, check if (a,b) is in bad_pairs_for_x[x].
    # This is still O(|S|^2) if we iterate over all pairs in S.
    # But we can do better.
    # Let's maintain a set of pairs in S? No, S is a set of elements.
    
    # Alternative:
    # For each x, we want to know if S contains any pair from bad_pairs_for_x[x].
    # We can precompute for each x, the list of bad pairs.
    # Then can_add(x, S) checks if any of these pairs is a subset of S.
    
    # To make this fast, we can represent S as a bitset or just a set.
    # Checking if a pair (a,b) is in S is O(1) if S is a set.
    # So can_add is O(|bad_pairs_for_x[x]|).
    # How large is bad_pairs_for_x[x]?
    # In the worst case, it could be large, but hopefully sparse.
    
    def can_add(x, S_set):
        for a, b in bad_pairs_for_x[x]:
            if a in S_set and b in S_set:
                return False
        return True

    def hill_climb(S):
        S = list(S)
        S_set = set(S)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in range(1, n+1):
                if x not in S_set and can_add(x, S_set):
                    S.append(x)
                    S_set.add(x)
                    improved = True
            if improved:
                S.sort()
        return S

    # Initial construction
    best_S = []
    
    # Try multiple random orders
    rng = random.Random(123)
    for _ in range(50):
        if time.time() >= deadline - 5:
            break
        order = list(range(1, n+1))
        rng.shuffle(order)
        S = []
        S_set = set()
        for x in order:
            if x not in S_set and can_add(x, S_set):
                S.append(x)
                S_set.add(x)
        S = hill_climb(S)
        if len(S) > len(best_S):
            best_S = S
    
    # Local search: kick and re-greedy
    for _ in range(200):
        if time.time() >= deadline - 5:
            break
        if not best_S:
            continue
        
        # Random kick: remove k elements
        k = rng.randint(2, 5)
        to_remove = rng.sample(best_S, k)
        S = [x for x in best_S if x not in to_remove]
        S_set = set(S)
        
        # Re-greedy
        for x in range(1, n+1):
            if x not in S_set and can_add(x, S_set):
                S.append(x)
                S_set.add(x)
        
        S = hill_climb(S)
        if len(S) > len(best_S):
            best_S = S
    
    return best_S
