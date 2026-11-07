"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

# Deadline guard: finish well before the 90s wall-clock kill.
_DEADLINE = time.time() + 80.0  # 80s safety margin

def _is_deadline():
    return time.time() > _DEADLINE

def _is_latin_square(sq, n):
    """Check if sq is a valid n x n Latin square with symbols 0..n-1."""
    for i in range(n):
        row = sq[i]
        if len(row) != n:
            return False
        if sorted(row) != list(range(n)):
            return False
    for j in range(n):
        col = [sq[i][j] for i in range(n)]
        if sorted(col) != list(range(n)):
            return False
    return True

def _is_orthogonal(a, b, n):
    """Check if two n x n arrays are orthogonal (all n^2 pairs distinct)."""
    pairs = set()
    for i in range(n):
        for j in range(n):
            p = (a[i][j], b[i][j])
            if p in pairs:
                return False
            pairs.add(p)
    return True

def _generate_latin_square(n, seed):
    """Generate a random Latin square of order n using random permutation of rows/cols and symbol relabeling."""
    rng = random.Random(seed)
    # Start with the cyclic Latin square
    base = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Apply random row permutation
    row_perm = list(range(n))
    rng.shuffle(row_perm)
    sq = [base[row_perm[i]] for i in range(n)]
    
    # Apply random column permutation
    col_perm = list(range(n))
    rng.shuffle(col_perm)
    sq = [[sq[i][col_perm[j]] for j in range(n)] for i in range(n)]
    
    # Apply random symbol permutation
    sym_perm = list(range(n))
    rng.shuffle(sym_perm)
    sq = [[sym_perm[sq[i][j]] for j in range(n)] for i in range(n)]
    
    return sq

def _generate_latin_square_from_group(n, seed):
    """Generate a Latin square based on a random group-like structure.
    For n=10, we can use the fact that Z_10 is not a group for Latin squares directly,
    but we can use random isotopic transformations."""
    rng = random.Random(seed)
    # Use a random permutation-based construction
    # Start with cyclic, then apply random isotopism
    base = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Random row perm
    rperm = list(range(n))
    rng.shuffle(rperm)
    # Random col perm
    cperm = list(range(n))
    rng.shuffle(cperm)
    # Random symbol perm
    spm = list(range(n))
    rng.shuffle(spm)
    
    sq = [[spm[base[rperm[i]][cperm[j]]] for j in range(n)] for i in range(n)]
    return sq

def _try_build_orthogonal_set(n, max_squares=9, time_limit=80.0):
    """Try to build a set of mutually orthogonal Latin squares."""
    start_time = time.time()
    rng = random.Random(42)
    
    # Start with the standard cyclic Latin square
    L = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [L]
    
    # Try to find orthogonal squares
    attempts = 0
    max_attempts = 500000
    
    while len(squares) < max_squares and attempts < max_attempts:
        if time.time() - start_time > time_limit - 5:
            break
        
        attempts += 1
        
        # Generate a candidate square
        candidate = _generate_latin_square(n, rng.randint(0, 2**31 - 1))
        
        # Check orthogonality with all existing squares
        ok = True
        for existing in squares:
            if not _is_orthogonal(existing, candidate, n):
                ok = False
                break
        
        if ok:
            squares.append(candidate)
    
    return squares

def _try_systematic_search(n, time_limit=80.0):
    """Systematic search using structured generation."""
    start_time = time.time()
    
    # The standard cyclic square
    L0 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Try to find squares orthogonal to L0
    # A square orthogonal to the cyclic one must have specific properties
    # Let's try: for each k in 1..n-1, try square S_k where S_k[i][j] = (i + k*j) % n
    # This won't work for n=10 since it's not prime power, but let's try variations
    
    squares = [L0]
    
    # Try multiplicative variants
    for k in range(1, n):
        if time.time() - start_time > time_limit - 5:
            break
        candidate = [[(i + k * j) % n for j in range(n)] for i in range(n)]
        if _is_latin_square(candidate, n):
            ok = True
            for existing in squares:
                if not _is_orthogonal(existing, candidate, n):
                    ok = False
                    break
            if ok:
                squares.append(candidate)
    
    # Try more structured patterns
    # For n=10, try squares based on different group operations
    # Try: S[i][j] = (a*i + b*j) % n for various a, b
    for a in range(1, n):
        for b in range(1, n):
            if time.time() - start_time > time_limit - 5:
                break
            candidate = [[(a * i + b * j) % n for j in range(n)] for i in range(n)]
            if _is_latin_square(candidate, n):
                ok = True
                for existing in squares:
                    if not _is_orthogonal(existing, candidate, n):
                        ok = False
                        break
                if ok:
                    squares.append(candidate)
    
    return squares

def _hybrid_search(n, time_limit=80.0):
    """Combine systematic and random search."""
    start_time = time.time()
    
    # Start with systematic
    squares = _try_systematic_search(n, time_limit - 10)
    
    if len(squares) >= 9:
        return squares
    
    # Then try random search to add more
    rng = random.Random(12345)
    attempts = 0
    max_attempts = 2000000
    
    while len(squares) < 9 and attempts < max_attempts:
        if time.time() - start_time > time_limit - 5:
            break
        
        attempts += 1
        
        candidate = _generate_latin_square(n, rng.randint(0, 2**31 - 1))
        
        ok = True
        for existing in squares:
            if not _is_orthogonal(existing, candidate, n):
                ok = False
                break
        
        if ok:
            squares.append(candidate)
    
    return squares

def construct():
    n = 10
    
    # Try hybrid search
    squares = _hybrid_search(n, time_limit=75.0)
    
    # Ensure we have at least one valid square
    if not squares:
        squares = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Verify all are valid Latin squares and mutually orthogonal
    valid_squares = []
    for sq in squares:
        if not _is_latin_square(sq, n):
            continue
        ok = True
        for v in valid_squares:
            if not _is_orthogonal(v, sq, n):
                ok = False
                break
        if ok:
            valid_squares.append(sq)
    
    if not valid_squares:
        valid_squares = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Convert to pure Python lists of ints
    result = []
    for sq in valid_squares:
        result.append([[int(sq[i][j]) for j in range(n)] for i in range(n)])
    
    return result
# EVOLVE-BLOCK-END
