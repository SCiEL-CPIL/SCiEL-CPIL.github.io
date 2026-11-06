import time
import numpy as np

def construct():
    # Deadline guard
    start_time = time.time()
    timeout = 140.0
    
    # Try Paley tournaments for primes p from 31 to 83
    # We need a list of primes
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    primes = [p for p in range(31, 84) if is_prime(p) and p % 4 == 3]
    # Paley tournaments are defined for p = 3 mod 4.
    # Primes 31, 43, 59, 67, 79 are 3 mod 4.
    # 37 is 1 mod 4. 41 is 1 mod 4. 47 is 3 mod 4.
    # Wait, Paley tournament exists for any prime p = 3 mod 4.
    # For p = 1 mod 4, the quadratic residues form a symmetric relation, so it's not a tournament (it would have 2-cycles).
    # So we only check p = 3 mod 4.
    
    # Let's check the primes:
    # 31 (3 mod 4)
    # 43 (3 mod 4)
    # 47 (3 mod 4)
    # 59 (3 mod 4)
    # 67 (3 mod 4)
    # 79 (3 mod 4)
    
    # The seed uses 83. 83 is 3 mod 4.
    
    for p in primes:
        if time.time() - start_time > timeout:
            break
            
        # Construct Paley tournament
        # i -> j iff (j - i) % p is a quadratic residue
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
        
        # Check validity
        # Optimization: 
        # A 4-set {a,b,c,d} is dominated if there is v in V \ {a,b,c,d} 
        # such that v->a, v->b, v->c, v->d.
        # This is equivalent to: The intersection of the out-neighborhoods of a,b,c,d is non-empty 
        # and does not contain a,b,c,d (which is guaranteed if out-neighborhoods don't contain self).
        
        # For each 4-set, compute intersection of out-neighbors.
        # n is small (31-83). Number of 4-sets is ~ 30k for n=83.
        # Intersection of 4 sets of size ~p/2.
        # This is fast enough.
        
        valid = True
        # Precompute out-neighborhoods as bitmasks or sets?
        # Sets are fine for n<100.
        out_neigh = [set() for _ in range(p)]
        for i in range(p):
            for j in range(p):
                if i != j:
                    if (j - i) % p in squares:
                        out_neigh[i].add(j)
        
        # Check all 4-sets
        # To speed up, we can iterate and break early.
        # Using combinations
        from itertools import combinations
        for quad in combinations(range(p), 4):
            # Intersection of out_neigh[v] for v in quad
            # Start with out_neigh[quad[0]]
            common = out_neigh[quad[0]]
            for v in quad[1:]:
                common.intersection_update(out_neigh[v])
                if not common:
                    break
            if not common:
                valid = False
                break
        
        if valid:
            # Return orientation
            orientation = []
            for i in range(p):
                for j in range(i + 1, p):
                    # 0 means i -> j
                    if (j - i) % p in squares:
                        orientation.append(0)
                    else:
                        orientation.append(1)
            return {'n': p, 'orientation': orientation}
            
    # Fallback to seed Paley 83
    p = 83
    squares = {x * x % p for x in range(1, p)}
    orientation = [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]
    return {'n': p, 'orientation': orientation}
