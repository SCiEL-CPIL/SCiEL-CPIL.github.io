"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _make_latin(n, a, b):
    """Construct a Latin square using linear formula over Z_n."""
    return [[(a * i + b * j) % n for j in range(n)] for i in range(n)]

def _is_latin(sq, n):
    for i in range(n):
        if sorted(sq[i]) != list(range(n)):
            return False
    for j in range(n):
        col = [sq[i][j] for i in range(n)]
        if sorted(col) != list(range(n)):
            return False
    return True

def _are_orthogonal(sq1, sq2, n):
    pairs = set()
    for i in range(n):
        for j in range(n):
            p = (sq1[i][j], sq2[i][j])
            if p in pairs:
                return False
            pairs.add(p)
    return True

def construct():
    n = 10
    start = time.time()
    deadline = start + 50.0  # safety margin before 60s limit

    # Known MOLS of order 10: we can construct 3 using a product structure.
    # Use the fact that 10 = 2*5 and build squares via linear combinations.
    # Square k: entry (i,j) = (a_k * i + b_k * j) mod 10 won't work directly
    # since Z_10 is not a field. Instead, use Z_5 x Z_2 representation.
    
    # Represent each index 0..9 as (x, y) where x = i % 5, y = i // 5
    # Construct squares using linear maps over Z_5 for the x-component
    
    # Method: For each square, use a different multiplier in Z_5
    # Square m: value at position (i,j) = 5 * ((i//5) + (j//5)) % 2 + (a_m * (i%5) + (j%5)) % 5
    # This needs verification. Let me use a simpler known construction.
    
    # Actually, let's use the standard construction for MOLS when n = p*q.
    # For n=10=2*5, we can get at least min(p-1, q-1) = 4 MOLS? No, that's not right either.
    
    # Let me just try all simple linear squares and check orthogonality.
    # Square defined by (a*i + b*j) % 10 is Latin iff gcd(a,10)=1 and gcd(b,10)=1
    # Two such squares are orthogonal iff... let me just brute force check.
    
    candidates = []
    for a in range(1, 10):
        for b in range(1, 10):
            if __import__('math').gcd(a, 10) != 1 or __import__('math').gcd(b, 10) != 1:
                continue
            sq = [[(a * i + b * j) % 10 for j in range(10)] for i in range(10)]
            if _is_latin(sq, 10):
                candidates.append((a, b, sq))
    
    # Find maximum set of mutually orthogonal squares from candidates
    best_set = []
    for cand in candidates:
        a, b, sq = cand
        ok = True
        for _, _, sq2 in best_set:
            if not _are_orthogonal(sq, sq2, 10):
                ok = False
                break
        if ok:
            best_set.append(cand)
    
    if best_set:
        result = [sq for _, _, sq in best_set]
    else:
        # Fallback: single trivial square
        result = [[(i + j) % 10 for j in range(10)] for i in range(10)]
    
    # Ensure proper types
    result = [[list(map(int, row)) for row in sq] for sq in result]
    
    return result
# EVOLVE-BLOCK-END
