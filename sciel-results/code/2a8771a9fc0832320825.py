import time
import math
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s kill

    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]

    primes = get_primes(256)
    small_primes = [p for p in primes if p <= 16]  # 2, 3, 5, 7, 11, 13
    large_primes = [p for p in primes if p > 16]

    # Generate candidate pool U for composites/small primes
    # U contains numbers in [2, 256] whose prime factors are all in small_primes
    U = []
    for x in range(2, 257):
        y = x
        valid = True
        for p in small_primes:
            while y % p == 0:
                y //= p
        if y == 1:
            U.append(x)

    # Precompute conflict relationships
    # conflict[x][y] = True if x and y cannot both be in C
    # This happens if:
    # 1. x divides y or y divides x (antichain violation)
    # 2. There exists some z in U such that z divides x*y (but this is about triples, not pairs)
    # Actually, the constraint is: for all distinct a,b,c in the set, a !| b*c
    # This is a 3-wise constraint, not pairwise.
    # However, we can precompute for each pair (x,y) which z's would conflict:
    # If x | y*z or y | x*z or z | x*y, then {x,y,z} is invalid.
    
    # Let's precompute for each x in U, the set of y in U such that x | y * z for some z
    # Actually, let's just precompute the "forbidden triples" or use a faster check.
    
    # A number x conflicts with a pair (y,z) if x | y*z.
    # Let's precompute divisors of each number in U.
    
    # Better approach: Precompute for each x in U, the set of y in U such that 
    # there exists z in U where x | y*z.
    # But z varies, so this is complex.
    
    # Let's use a different precomputation:
    # For each x in U, precompute which pairs (y,z) in U are "forbidden" with x.
    # forbidden[x] = set of (y,z) such that x | y*z (and y,z are distinct and different from x)
    
    # Actually, the condition is: for all distinct a,b,c in C, a !| b*c.
    # So if we want to add x to C, we need to check that for all pairs (b,c) in C, 
    # x !| b*c, and for all y in C, x !| y*? (no, the triple must be distinct).
    # When adding x to C, the new triples involving x are (x, b, c) for all distinct b,c in C.
    # So we need x !| b*c for all distinct b,c in C.
    # Also, existing triples in C must remain valid (they do, since we're only adding x).
    # And we need the antichain condition: x !| y and y !| x for all y in C.
    
    # Precompute: for each x in U, and for each pair (b,c) in U, does x | b*c?
    # This is |U|^3 which might be too slow if |U| is large.
    # |U| is the count of numbers in [2,256] with prime factors only in {2,3,5,7,11,13}.
    # Let's estimate: these are 13-smooth numbers.
    # Actually, let's just compute it on the fly with precomputed multiples.
    
    # Precompute for each x in U, the set of y in U such that x | y * z for some z in U.
    # Hmm, that's not quite right either.
    
    # Let's precompute: for each x in U, the set of "conflicting pairs" (b,c) in U such that x | b*c.
    # We can do this by: for each x, for each b in U, find all c in U such that x | b*c.
    # x | b*c means b*c is a multiple of x.
    # So c must be a multiple of x/gcd(x,b) divided by... 
    # Actually, x | b*c iff (x/gcd(x,b)) | c.
    # Let g = gcd(x, b), then x | b*c iff (x/g) | c.
    # So for each x and b, the conflicting c's are multiples of (x/gcd(x,b)) that are in U.
    
    # Let's precompute this.
    
    U_set = set(U)
    U_list = list(U)
    U_size = len(U_list)
    
    # Precompute gcd for all pairs? No, let's compute on the fly.
    # Precompute: for each x in U, for each b in U, the value x/gcd(x,b)
    # And then the set of c in U that are multiples of that value.
    
    # Actually, let's precompute for each x in U:
    # conflict_pairs[x] = set of (b, c) in U x U such that x | b*c
    # But this is too large. Instead, let's precompute:
    # for each x in U, for each b in U, the set of c in U such that x | b*c
    # Then when adding x to C, we check if any pair (b,c) in C x C is in this set.
    
    # Let's precompute: for each x in U, a set of "forbidden pairs" as a bitmask or set of pairs.
    # Since |U| is at most 255, we can use a set of tuples.
    
    # Precompute multiples for each divisor
    # For each d in [1, 256], precompute the set of numbers in U that are multiples of d.
    multiples_in_U = {}
    for d in range(1, 257):
        multiples_in_U[d] = set()
        for m in range(d, 257, d):
            if m in U_set:
                multiples_in_U[d].add(m)
    
    # Now, for each x in U, precompute the set of pairs (b, c) in U x U such that x | b*c.
    # x | b*c iff (x/gcd(x,b)) | c
    # So for each b, c must be in multiples_in_U[x/gcd(x,b)]
    
    # Let's precompute this as a set of (b, c) pairs for each x.
    # But this might be large. Let's see.
    
    # Actually, let's just precompute for each x in U, a list of (b, c) pairs.
    # We'll store it as a set for fast lookup.
    
    conflict_pairs = {}
    for x in U_list:
        cp = set()
        for b in U_list:
            g = math.gcd(x, b)
            needed = x // g
            # c must be a multiple of needed
            for c in multiples_in_U[needed]:
                if c != b and c != x:  # distinctness
                    cp.add((b, c))
                    cp.add((c, b))  # symmetric
        conflict_pairs[x] = cp
    
    # Now, can_add_to_set(x, C) becomes:
    # 1. Check antichain: for all y in C, x !| y and y !| x
    # 2. Check that no pair (b, c) in C x C is in conflict_pairs[x]
    
    # Precompute antichain conflicts: for each x, the set of y such that x|y or y|x
    antichain_conflicts = {}
    for x in U_list:
        ac = set()
        for y in U_list:
            if x != y and (x % y == 0 or y % x == 0):
                ac.add(y)
        antichain_conflicts[x] = ac
    
    def can_add_to_set(x, C):
        C_set = set(C)
        # Antichain check
        if C_set & antichain_conflicts[x]:
            return False
        # Product check: no pair (b, c) in C x C should be in conflict_pairs[x]
        # We need to check all pairs in C
        C_list = list(C)
        n = len(C_list)
        for i in range(n):
            for j in range(i + 1, n):
                b = C_list[i]
                c = C_list[j]
                if (b, c) in conflict_pairs[x] or (c, b) in conflict_pairs[x]:
                    return False
        return True

    # Greedy approach
    current_C = []
    U_sorted = sorted(U, reverse=True)
    
    for x in U_sorted:
        if can_add_to_set(x, current_C):
            current_C.append(x)
    
    best_C_size = len(current_C)
    best_C = current_C[:]
    
    # Randomized restarts
    for _ in range(2000):
        if time.time() > deadline:
            break
        random.shuffle(U_sorted)
        c = []
        for x in U_sorted:
            if can_add_to_set(x, c):
                c.append(x)
        if len(c) > best_C_size:
            best_C_size = len(c)
            best_C = c[:]
    
    # Local search: try to improve by swapping
    def local_search_improve(C, U_pool, max_iterations=200):
        improved = True
        iterations = 0
        C_list = C[:]
        while improved and iterations < max_iterations:
            if time.time() > deadline:
                break
            improved = False
            iterations += 1
            C_set = set(C_list)
            for i, remove_elem in enumerate(C_list):
                for add_elem in U_pool:
                    if add_elem in C_set:
                        continue
                    new_C = [x for x in C_list if x != remove_elem]
                    if can_add_to_set(add_elem, new_C):
                        new_C.append(add_elem)
                        C_list = new_C
                        C_set = set(C_list)
                        improved = True
                        break
                if improved:
                    break
        return C_list
    
    if time.time() < deadline:
        improved_C = local_search_improve(best_C, U, max_iterations=100)
        if len(improved_C) > best_C_size:
            best_C_size = len(improved_C)
            best_C = improved_C
    
    # Try different sorting strategies
    if time.time() < deadline:
        for sort_key in [lambda x: x, lambda x: -x]:
            if time.time() > deadline:
                break
            U_alt = sorted(U, key=sort_key)
            c = []
            for x in U_alt:
                if can_add_to_set(x, c):
                    c.append(x)
            if len(c) > best_C_size:
                best_C_size = len(c)
                best_C = c[:]
    
    # Combine with large primes
    result = list(large_primes) + list(best_C)
    result = list(set(result))  # ensure distinctness
    result = [int(x) for x in result]
    
    return result
