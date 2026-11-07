"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    N = 256
    # Initial set: all primes in [2, 256]
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
        
    primes = [i for i in range(2, N + 1) if is_prime(i)]
    S = set(primes)
    
    # Precompute divisibility: divs[a] contains all b in [2, N] such that a divides b
    # Actually, we need to check a | bc.
    # We can just use modulo.
    
    def is_valid_set(S):
        # Check all triples a, b, c in S, distinct
        # For each a, check if a divides b*c for any distinct b, c in S
        # We can optimize by iterating a, then pairs (b,c)
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(n):
                if j == i: continue
                b = S_list[j]
                for k in range(j+1, n):
                    if k == i: continue
                    c = S_list[k]
                    if a % (b * c) == 0: # Wait, a divides bc means bc % a == 0
                        return False
        return True

    # Faster validity check for a specific change
    # Given S_old (valid), x in S_old, y not in S_old
    # Check if (S_old - {x}) U {y} is valid
    def check_swap(S_old, x, y):
        # S_old is a set
        # New elements involving y:
        # 1. y | bc for distinct b, c in S_old - {x}
        # 2. b | yc for distinct b, c in S_old - {x}
        # 3. c | yb for distinct b, c in S_old - {x}
        # Note: b and c are from S_old - {x}
        # We can iterate over pairs (b, c) in S_old - {x}
        S_rem = S_old - {x}
        S_rem_list = list(S_rem)
        m = len(S_rem_list)
        for i in range(m):
            b = S_rem_list[i]
            for j in range(i+1, m):
                c = S_rem_list[j]
                # Check y | bc
                if (b * c) % y == 0:
                    return False
                # Check b | yc
                if (y * c) % b == 0:
                    return False
                # Check c | yb
                if (y * b) % c == 0:
                    return False
        return True

    # Check 1-2 exchange: remove x, add y, z
    def check_exchange(S_old, x, y, z):
        S_rem = S_old - {x}
        S_rem_list = list(S_rem)
        m = len(S_rem_list)
        # New elements: y, z
        # Triples involving y or z:
        # 1. y | b*c, y | b*z, y | z*c (where b,c in S_rem)
        # 2. z | b*c, z | b*y, z | y*c
        # 3. b | y*c, b | y*z, b | z*c
        # 4. c | y*b, c | y*z, c | z*b
        # Basically, for all triples (u, v, w) in (S_rem U {y, z}) that contain y or z,
        # check if u | v*w.
        
        # To avoid duplicates, we can just iterate all triples in S_new that contain y or z.
        S_new_list = S_rem_list + [y, z]
        n = len(S_new_list)
        # Indices of y and z in S_new_list are m and m+1
        for i in range(n):
            u = S_new_list[i]
            # If u is not y and not z, and the triple doesn't contain y or z, skip?
            # Actually, if u is in S_rem, the triple (u, v, w) is valid in S_old unless v or w is y or z.
            # So we only need to check triples where at least one of v,w is y or z?
            # No, the condition is "for every three distinct members a,b,c".
            # If a is in S_rem, and b,c are in S_rem, it was already valid.
            # So we only need to check triples where at least one element is y or z.
            # Cases:
            # 1. a=y, b,c in S_rem
            # 2. a=z, b,c in S_rem
            # 3. a in S_rem, b=y, c in S_rem
            # 4. a in S_rem, b=z, c in S_rem
            # 5. a=y, b=z, c in S_rem
            # 6. a=z, b=y, c in S_rem
            # 7. a in S_rem, b=y, c=z
            # 8. a in S_rem, b=z, c=y
            
            # We can just iterate all pairs (v, w) from S_new_list distinct from u,
            # and check if u | v*w.
            # But we can filter: if u, v, w are all in S_rem, skip.
            for j in range(n):
                if j == i: continue
                v = S_new_list[j]
                for k in range(j+1, n):
                    if k == i: continue
                    w = S_new_list[k]
                    # Check if all three are in S_rem
                    if u in S_rem and v in S_rem and w in S_rem:
                        continue
                    if (v * w) % u == 0:
                        return False
        return True

    # Local Search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_list = list(S)
        candidates = [i for i in range(2, N + 1) if i not in S]
        
        # 1-1 Swaps
        for x in S_list:
            for y in candidates:
                if check_swap(S, x, y):
                    S.remove(x)
                    S.add(y)
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
            
        # 1-2 Exchanges
        for x in list(S):
            for y in candidates:
                if y in S: continue
                for z in candidates:
                    if z in S or z == y: continue
                    if check_exchange(S, x, y, z):
                        S.remove(x)
                        S.add(y)
                        S.add(z)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

    return list(S)

# EVOLVE-BLOCK-END
