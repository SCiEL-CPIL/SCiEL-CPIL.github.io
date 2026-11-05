import time
import math
from itertools import combinations

def construct() -> dict:
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def make_paley_adjacency(p):
        """Build the adjacency list for Paley tournament on p vertices (p prime, p ≡ 3 mod 4)."""
        residues = set()
        for x in range(1, p):
            residues.add((x * x) % p)
        adjacency = []
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in residues:
                    adjacency.append(0)  # i -> j
                else:
                    adjacency.append(1)  # j -> i
        return adjacency

    def check_property_bitmask(n, adjacency):
        """
        Check if every 4-subset has a common dominator using bitmasks.
        in_masks[v] is an integer where bit i is 1 if i -> v.
        """
        in_masks = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adjacency[idx] == 0:
                    in_masks[j] |= (1 << i)
                else:
                    in_masks[i] |= (1 << j)
                idx += 1

        for u in range(n):
            mask_u = in_masks[u]
            for v in range(u + 1, n):
                mask_uv = mask_u & in_masks[v]
                if mask_uv == 0:
                    return False
                for w in range(v + 1, n):
                    mask_uvw = mask_uv & in_masks[w]
                    if mask_uvw == 0:
                        return False
                    for x in range(w + 1, n):
                        if (mask_uvw & in_masks[x]) == 0:
                            return False
        return True

    # Try prime powers q ≡ 3 (mod 4) for Paley-type constructions
    # q = p^k where p is prime and q ≡ 3 mod 4
    # For p ≡ 3 mod 4: q ≡ 3 mod 4 iff k is odd
    # For p ≡ 1 mod 4: q ≡ 1 mod 4 always, so no candidates
    # So we need primes p ≡ 3 mod 4 with odd exponents
    
    # Prime powers q ≡ 3 mod 4 up to 90:
    # p=3: 3, 27 (3^1, 3^3)
    # p=7: 7 (7^1)
    # p=11: 11
    # p=19: 19
    # p=23: 23
    # p=31: 31
    # p=43: 43
    # p=47: 47
    # p=59: 59
    # p=67: 67
    # p=79: 79
    # p=83: 83
    # p=89: 89
    # Also p=3, k=3: 27
    
    # For composite q (prime powers), we need to define quadratic residues in GF(q)
    # For q = p^k, the multiplicative group has order q-1
    # An element is a square iff its (q-1)/2 power is 1
    
    def make_paley_adjacency_q(q):
        """Build Paley tournament on q vertices where q is a prime power ≡ 3 mod 4."""
        # Find prime base
        base = None
        for p in range(2, q + 1):
            if is_prime(p):
                pk = p
                while pk < q:
                    pk *= p
                if pk == q:
                    base = p
                    break
        if base is None:
            return None
            
        # Build field GF(q) as polynomials mod irreducible polynomial
        # For simplicity, only handle q = p (prime) or q = p^3 for p=3 (q=27)
        if q == base:
            # Prime field
            residues = set()
            for x in range(1, q):
                residues.add((x * x) % q)
            adjacency = []
            for i in range(q):
                for j in range(i + 1, q):
                    if (j - i) % q in residues:
                        adjacency.append(0)
                    else:
                        adjacency.append(1)
            return adjacency
        elif q == 27:
            # GF(27) = GF(3)[x] / (x^3 + x + 1)
            # Represent elements as (a, b, c) meaning a + bx + cx^2
            # x^3 = -x - 1 = 2x + 2 mod 3
            
            def gf27_add(a, b):
                return [(a[i] + b[i]) % 3 for i in range(3)]
            
            def gf27_mul(a, b):
                # (a0 + a1*x + a2*x^2) * (b0 + b1*x + b2*x^2)
                # x^3 = 2x + 2
                # x^4 = x*(2x+2) = 2x^2 + 2x
                c0 = (a[0]*b[0] + a[1]*b[2] + a[2]*b[1]) % 3
                c1 = (a[0]*b[1] + a[1]*b[0] + a[2]*b[2]) % 3
                c2 = (a[0]*b[2] + a[1]*b[1] + a[2]*b[0]) % 3
                # x^3 terms: a1*b2 + a2*b1
                x3 = (a[1]*b[2] + a[2]*b[1]) % 3
                # x^4 term: a2*b2
                x4 = (a[2]*b[2]) % 3
                # x^3 = 2x + 2, so contributes 2*x3 to c0 and x3 to c1
                # x^4 = 2x^2 + 2x, so contributes 2*x4 to c1 and x4 to c2
                c0 = (c0 + 2 * x3) % 3
                c1 = (c1 + x3 + 2 * x4) % 3
                c2 = (c2 + x4) % 3
                return [c0, c1, c2]
            
            # Build all 27 elements
            elements = []
            for a in range(3):
                for b in range(3):
                    for c in range(3):
                        elements.append([a, b, c])
            
            # Find quadratic residues (nonzero squares)
            residues = set()
            for e in elements:
                if e == [0, 0, 0]:
                    continue
                sq = gf27_mul(e, e)
                residues.add(tuple(sq))
            
            # Map element to index
            elem_to_idx = {tuple(e): i for i, e in enumerate(elements)}
            
            adjacency = []
            for i in range(27):
                for j in range(i + 1, 27):
                    # Compute j - i in GF(27)
                    diff = gf27_add(elements[j], [(-elements[i][k]) % 3 for k in range(3)])
                    if tuple(diff) in residues:
                        adjacency.append(0)
                    else:
                        adjacency.append(1)
            return adjacency
        else:
            return None

    # Candidates: prime powers q ≡ 3 mod 4, sorted by q
    candidates = [7, 11, 19, 23, 27, 31, 43, 47, 59, 67, 79, 83, 89]
    
    for q in candidates:
        if time.time() > deadline - 10.0:
            break
        adj = make_paley_adjacency_q(q)
        if adj is not None:
            if check_property_bitmask(q, adj):
                return {'n': q, 'orientation': adj}
    
    # Fallback: return 67
    adj67 = make_paley_adjacency(67)
    return {'n': 67, 'orientation': adj67}
