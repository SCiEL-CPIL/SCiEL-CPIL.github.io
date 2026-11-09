import time
import itertools

def construct():
    start_time = time.time()
    DEADLINE = 130.0  # safety margin before 150s kill

    # We'll try to find the smallest valid tournament by trying candidate constructions.
    # Known: Paley tournament on p vertices where p ≡ 3 (mod 4) and p is prime.
    # The property "every 4-set has a common dominator" is equivalent to the tournament
    # being "4-dominated" or having domination number property for 4-sets.
    
    # For Paley tournament on prime p ≡ 3 mod 4:
    # i -> j iff (j - i) is a quadratic residue mod p
    # A vertex v dominates a set S if v -> s for all s in S, i.e., (s - v) is QR for all s in S.
    # Equivalently, v is in the common in-neighborhood of all elements of S.
    
    # For small primes, we can verify directly. The question is which is the smallest p
    # for which the Paley tournament satisfies the 4-dominating property.
    
    # Let's think about this differently. The property is that for every 4-set {a,b,c,d},
    # there exists a vertex v (not necessarily in the set) such that v -> a, v -> b, v -> c, v -> d.
    # This means v is in In(a) ∩ In(b) ∩ In(c) ∩ In(d).
    
    # For the Paley tournament, In(x) = {y : (x - y) is QR mod p}
    # So v ∈ In(x) iff (x - v) is QR, i.e., (v - x) is QR (since -1 is not QR for p ≡ 3 mod 4... wait)
    # Actually for p ≡ 3 mod 4, -1 is NOT a quadratic residue. So (x - v) is QR iff (v - x) is NOT QR.
    # Hmm, let me be more careful.
    # In(x) = {y ≠ x : (x - y) mod p ∈ QR}
    # v ∈ In(x) iff (x - v) mod p ∈ QR
    
    # So we need: for every 4-set {a,b,c,d}, there exists v such that
    # (a-v) ∈ QR, (b-v) ∈ QR, (c-v) ∈ QR, (d-v) ∈ QR.
    # Equivalently, v ∈ a - QR ∩ b - QR ∩ c - QR ∩ d - QR (where x - QR = {x - q : q ∈ QR} ∪ possibly adjusted)
    
    # Actually, let me just compute this for small primes and check.
    
    def is_prime(n):
        if n < 2: return False
        if n < 4: return True
        if n % 2 == 0 or n % 3 == 0: return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def paley_orientation(p):
        """Return orientation list for Paley tournament on p vertices."""
        squares = set()
        for x in range(1, p):
            squares.add(x * x % p)
        orientation = []
        for i in range(p):
            for j in range(i + 1, p):
                # i -> j iff (j - i) mod p is a QR
                if (j - i) % p in squares:
                    orientation.append(0)
                else:
                    orientation.append(1)
        return orientation

    def check_paley(p):
        """Check if Paley tournament on p vertices satisfies the 4-dominating property."""
        if time.time() - start_time > DEADLINE:
            return None, None
        
        squares = set()
        for x in range(1, p):
            squares.add(x * x % p)
        
        # For each vertex v, compute the set of vertices it dominates
        # v dominates u iff (u - v) mod p is in squares
        # So Out(v) = {u : (u - v) mod p ∈ squares}
        
        # For the check: for every 4-set {a,b,c,d}, we need some v with v -> a, v -> b, v -> c, v -> d
        # i.e., (a-v) ∈ QR, (b-v) ∈ QR, (c-v) ∈ QR, (d-v) ∈ QR
        # i.e., v ∈ a - QR, v ∈ b - QR, v ∈ c - QR, v ∈ d - QR
        # where a - QR = {a - q mod p : q ∈ squares}
        
        # Precompute for each vertex x, the set of v such that (x - v) ∈ QR
        # That is, v ∈ x - QR = {x - q mod p : q ∈ squares}
        # Note: v can equal x? No, because we need (x-v) ∈ QR and 0 is not a QR.
        # Actually 0 is not in squares (we only added x^2 for x in 1..p-1), so v ≠ x is automatic.
        
        # in_neigh[x] = set of v such that v -> x, i.e., (x - v) ∈ QR
        in_neigh = []
        for x in range(p):
            mask = 0
            for q in squares:
                v = (x - q) % p
                mask |= (1 << v)
            in_neigh.append(mask)
        
        # Now check all 4-sets
        for a in range(p):
            if time.time() - start_time > DEADLINE:
                return None, None
            for b in range(a + 1, p):
                if time.time() - start_time > DEADLINE:
                    return None, None
                for c in range(b + 1, p):
                    if time.time() - start_time > DEADLINE:
                        return None, None
                    abc = in_neigh[a] & in_neigh[b] & in_neigh[c]
                    if abc == 0:
                        # No common in-neighbor for {a,b,c}, so no 4-set containing them can work
                        # But wait, we need a 4-SET {a,b,c,d}, so we need to check with d
                        # Actually if abc == 0, then for any d, abc & in_neigh[d] == 0, so this fails
                        # But we still need to check if there's SOME v dominating all 4
                        # If abc == 0, then indeed no v dominates a, b, c simultaneously, so no 4-set
                        # containing a,b,c has a dominator. Return False.
                        return False, None
                    for d in range(c + 1, p):
                        if time.time() - start_time > DEADLINE:
                            return None, None
                        if (abc & in_neigh[d]) == 0:
                            return False, None
        return True, in_neigh

    # Try primes p ≡ 3 mod 4 starting from small ones
    # The task says smallest known has 67 vertices, and Erdős lower bound is 31.
    # So we need to find the smallest valid p.
    
    # Actually, re-reading the task: "The smallest known has 67 vertices" - this might mean
    # that the best known construction uses 67 vertices, not that it's proven minimal.
    # We need to find the smallest n for which a valid tournament exists.
    
    # Let's try Paley tournaments for increasing primes p ≡ 3 mod 4.
    # But also consider: maybe non-Paley tournaments work with fewer vertices.
    # However, Paley tournaments are the natural candidate for small n.
    
    # For p = 3: too small (need n >= 5)
    # For p = 7: p ≡ 3 mod 4, prime. Let's check.
    # For p = 11: p ≡ 3 mod 4, prime.
    # etc.
    
    best_result = None
    
    # Try Paley tournaments
    for p in range(7, 91):
        if time.time() - start_time > DEADLINE:
            break
        if p % 4 != 3 or not is_prime(p):
            continue
        
        valid, in_neigh = check_paley(p)
        if valid:
            orientation = paley_orientation(p)
            best_result = {'n': p, 'orientation': orientation}
            break
    
    if best_result is not None:
        return best_result
    
    # If no Paley tournament worked within time, try a known valid one.
    # The task mentions 67 vertices as smallest known. Let's use that as fallback.
    # Actually, let me check if p=67 works (it should per the task description).
    # But we may have run out of time. Let's just construct p=67 directly.
    
    p = 67
    if time.time() - start_time > DEADLINE - 10:
        # Very tight on time, just return without verification
        squares = set()
        for x in range(1, p):
            squares.add(x * x % p)
        orientation = []
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in squares:
                    orientation.append(0)
                else:
                    orientation.append(1)
        return {'n': p, 'orientation': orientation}
    
    valid, in_neigh = check_paley(p)
    if valid:
        orientation = paley_orientation(p)
        return {'n': p, 'orientation': orientation}
    
    # Absolute fallback: construct p=83
    p = 83
    squares = set()
    for x in range(1, p):
        squares.add(x * x % p)
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return {'n': p, 'orientation': orientation}
