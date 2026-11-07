"""Construct a set of mutually orthogonal Latin squares of order 10."""
import time

def _validate_squares(squares):
    """Validate that all squares are Latin and pairwise orthogonal."""
    n = len(squares[0])
    # Check each square is Latin
    for sq in squares:
        for r in range(n):
            if sorted(sq[r]) != list(range(n)):
                return False
        for c in range(n):
            col = [sq[r][c] for r in range(n)]
            if sorted(col) != list(range(n)):
                return False
    # Check pairwise orthogonality
    k = len(squares)
    for i in range(k):
        for j in range(i + 1, k):
            pairs = set()
            for r in range(n):
                for c in range(n):
                    pairs.add((squares[i][r][c], squares[j][r][c]))
            if len(pairs) != n * n:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin
    
    # Known set of 9 mutually orthogonal Latin squares of order 10
    # These are based on the finite field construction / known maximal set
    # For order 10, the maximum number of MOLS is 9 (since 10 is not a prime power,
    # but we can still have up to 9 MOLS; actually for n=10, max MOLS is 9? 
    # Wait: the maximum number of MOLS of order n is n-1 if n is a prime power.
    # 10 is not a prime power, so the max is less than 9. In fact, for order 10,
    # it's known that at most 2 MOLS exist? No, that's not right.
    # Actually, for order 10, the Euler 36 officers problem shows that there is NO
    # pair of orthogonal Latin squares of order 6, but for order 10, MOLS do exist.
    # The maximum number of MOLS of order 10 is 9? No. Let me recall:
    # For n=10, the maximum number of MOLS is 9? Actually, I think for n=10,
    # there are at least 2 MOLS. Let me just construct a known set.
    
    # Actually, let me use a known construction. For order 10, we can use
    # the fact that 10 = 2 * 5, but that doesn't directly give MOLS.
    # 
    # Let me just hardcode a known set of MOLS for order 10.
    # One known result: there exist 9 MOLS of order 10? No, that's not right.
    # The maximum number of MOLS of order n is at most n-1. For n=10, that's 9.
    # But since 10 is not a prime power, we can't achieve n-1.
    # 
    # Actually, I recall that for n=10, the maximum number of MOLS is 9? 
    # Let me just try to construct as many as possible using a search.
    
    # Let me use a known set. I'll construct squares based on group-based methods
    # and verify orthogonality.
    
    # Square 0: standard addition table mod 10
    sq0 = [[(i + j) % 10 for j in range(10)] for i in range(10)]
    
    # Square 1: multiplication-based? Let's try (i*2 + j) mod 10 - but that's not Latin
    # Actually (i*a + j) mod n is Latin if gcd(a, n) = 1
    # For n=10, a must be coprime to 10, so a in {1, 3, 7, 9}
    # sq_a = [[(i*a + j) % 10 for j in range(10)] for i in range(10)]
    
    # Two such squares with a1, a2 are orthogonal iff... let me check.
    # Pairs: (i*a1 + j, i*a2 + j) mod 10. For fixed pair (x, y), we need unique (i, j).
    # x = i*a1 + j, y = i*a2 + j => x - y = i*(a1 - a2) mod 10
    # So if gcd(a1 - a2, 10) = 1, then i is uniquely determined, and then j is too.
    # So squares with multipliers a1, a2 where gcd(a1-a2, 10)=1 are orthogonal.
    
    # Available multipliers coprime to 10: 1, 3, 7, 9
    # Differences: 3-1=2 (gcd 2), 7-1=6 (gcd 2), 9-1=8 (gcd 2), 
    #              7-3=4 (gcd 2), 9-3=6 (gcd 2), 9-7=2 (gcd 2)
    # None of the differences are coprime to 10! So these linear squares are NOT pairwise orthogonal.
    
    # I need a different approach. Let me use known MOLS for order 10.
    # 
    # Actually, let me just search computationally for MOLS.
    # Start with sq0, then try to find additional squares.
    
    squares = [sq0]
    
    # Try to find additional orthogonal squares using randomized search
    import random
    random.seed(42)
    
    def is_latin(sq):
        n = len(sq)
        for r in range(n):
            if sorted(sq[r]) != list(range(n)):
                return False
        for c in range(n):
            col = [sq[r][c] for r in range(n)]
            if sorted(col) != list(range(n)):
                return False
        return True
    
    def is_orthogonal(sq1, sq2):
        n = len(sq1)
        pairs = set()
        for r in range(n):
            for c in range(n):
                pairs.add((sq1[r][c], sq2[r][c]))
        return len(pairs) == n * n
    
    def find_orthogonal_square(existing_squares, max_iters=100000):
        """Try to find a new Latin square orthogonal to all existing ones."""
        n = 10
        for _ in range(max_iters):
            # Random permutation of symbols in a base square
            perm = list(range(n))
            random.shuffle(perm)
            # Apply random row and column permutations
            row_perm = list(range(n))
            col_perm = list(range(n))
            random.shuffle(row_perm)
            random.shuffle(col_perm)
            
            # Start from sq0 and apply permutations
            base = [[(i + j) % n for j in range(n)] for i in range(n)]
            candidate = [[perm[base[row_perm[r]][col_perm[c]]] for c in range(n)] for r in range(n)]
            
            # Check Latin (should be by construction)
            if not is_latin(candidate):
                continue
            
            # Check orthogonality with all existing
            ok = True
            for ex in existing_squares:
                if not is_orthogonal(candidate, ex):
                    ok = False
                    break
            if ok:
                return candidate
        return None
    
    # Try to find up to 8 more squares
    max_squares = 9
    while len(squares) < max_squares and time.time() < deadline - 5.0:
        new_sq = find_orthogonal_square(squares, max_iters=50000)
        if new_sq is None:
            break
        squares.append(new_sq)
    
    # Validate the result
    if squares and _validate_squares(squares):
        # Convert to ensure pure Python ints
        result = []
        for sq in squares:
            result.append([[int(v) for v in row] for row in sq])
        return result
    else:
        # Fallback: return just the first square
        return [[[(i + j) % 10 for j in range(10)] for i in range(10)]]
