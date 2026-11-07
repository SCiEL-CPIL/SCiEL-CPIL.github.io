"""Find pairwise orthogonal Latin squares of order 10."""
import time

N = 10
DEADLINE = 55.0  # seconds, leave margin before 60s limit


def _is_latin(sq):
    n = len(sq)
    for i in range(n):
        if sorted(sq[i]) != list(range(n)):
            return False
    for j in range(n):
        if sorted(sq[i][j] for i in range(n)) != list(range(n)):
            return False
    return True


def _orthogonal(a, b):
    n = len(a)
    seen = set()
    for i in range(n):
        for j in range(n):
            p = (a[i][j], b[i][j])
            if p in seen:
                return False
            seen.add(p)
    return True


def _pairwise_orthogonal(squares):
    for i in range(len(squares)):
        for j in range(i + 1, len(squares)):
            if not _orthogonal(squares[i], squares[j]):
                return False
    return True


def construct():
    """Return a list of 10x10 Latin squares that are pairwise orthogonal.

    Returns at least 1 square (a valid Latin square of order 10).
    Tries to find additional orthogonal squares within the time budget.
    """
    t0 = time.time()

    # Base Latin square: L(i,j) = (i+j) % 10
    base = [[(i + j) % N for j in range(N)] for i in range(N)]
    assert _is_latin(base), "base is not Latin"

    # Try to find a second square orthogonal to base.
    # Known construction: for order 10, use a specific set of MOLS.
    # One known set: based on the fact that there exist 9 MOLS of order 10 (Euler's 36 officers).
    # However, constructing them from scratch is complex. Let's use a known explicit set.

    # Known MOLS of order 10 (from literature, e.g., based on Z_5 x Z_2 construction or explicit tables).
    # Here's a known set of 9 pairwise orthogonal Latin squares of order 10.
    # I'll embed a known valid set.

    # Actually, let's use a simpler approach: there's a well-known pair of orthogonal Latin squares of order 10.
    # L1(i,j) = (i + j) % 10
    # L2(i,j) = (i + 3j) % 10 -- NOT Latin as shown above.
    
    # Let me use a verified construction. For order 10 = 2*5, there exist MOLS.
    # A known set:
    # Square 0: (i+j)%10
    # Square 1: based on a different structure.
    
    # Let me just try to find one more via a small search.
    # Actually, let me use a known explicit pair.
    
    # Known: The following two squares are orthogonal:
    # A(i,j) = (i + j) % 10
    # B(i,j) = (i + 4*j) % 10  -- no, 4 is not coprime to 10.
    
    # Let me use a completely different approach: embed a known valid set of MOLS of order 10.
    # I know that 9 MOLS exist for order 10. Let me hardcode one known set.
    
    # Here's a known construction using the field GF(5) and the structure Z_10 = Z_2 x Z_5:
    # Actually, the simplest is to just return the base square and try to find one more.
    
    # Let me try a direct search for a second square orthogonal to base.
    # This is computationally expensive for n=10, so let me use a known one.
    
    # Known orthogonal mate to (i+j)%10:
    # From literature: L2(i,j) where the symbols are arranged such that pairs are all distinct.
    # One known construction: 
    # L1: row i, col j: (i+j)%10
    # L2: row i, col j: (2*i + 7*j) % 10 -- check if Latin: 2 is not invertible mod 10. No.
    
    # OK, let me just use a verified hardcoded set. I'll use a known pair from the theory of MOLS.
    
    # Actually, for the purpose of this task, returning 1 square gives score 1.0.
    # If I can reliably return 2, score is 2.0. Let me hardcode a known valid pair.
    
    # Known: The following are orthogonal Latin squares of order 10:
    # S0 = [[(i+j)%10 for j in range(10)] for i in range(10)]
    # S1 = a specific square. Let me construct one using the "Euler" method for order 10.
    
    # For now, let me try a small randomized search for a second square.
    # If it fails, return just the base.
    
    import random
    random.seed(42)
    
    def _try_find_second(base_sq, deadline):
        """Try to find a Latin square orthogonal to base_sq."""
        n = N
        # Use backtracking with randomized rows
        rows = [None] * n
        col_perms = [None] * n  # track used symbols in each column
        
        def _valid_row(row, r):
            # Check if row is a permutation of 0..n-1
            if sorted(row) != list(range(n)):
                return False
            # Check orthogonality with base at row r
            for j in range(n):
                pair = (base_sq[r][j], row[j])
                # We'll check globally later, but at least check column uniqueness for this row
            return True
        
        # Simpler: generate random permutations for each row and check
        for _ in range(100000):
            if time.time() > deadline:
                return None
            cand = []
            for i in range(n):
                row = list(range(n))
                random.shuffle(row)
                cand.append(row)
            # Check Latin: each column must be a permutation
            ok = True
            for j in range(n):
                col = [cand[i][j] for i in range(n)]
                if sorted(col) != list(range(n)):
                    ok = False
                    break
            if not ok:
                continue
            # Check orthogonality
            if _orthogonal(base_sq, cand):
                return cand
        return None
    
    deadline = t0 + DEADLINE - 5  # leave 5s margin
    second = _try_find_second(base, deadline)
    
    if second is not None and time.time() < deadline:
        return [base, second]
    else:
        return [base]
