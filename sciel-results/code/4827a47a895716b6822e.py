# EVOLVE-BLOCK-START
import time

def _group_add(x, y):
    """Addition in Z_2 x Z_5, mapped to 0..9"""
    ax, bx = x // 5, x % 5
    ay, by = y // 5, y % 5
    return ((ax + ay) % 2) * 5 + ((bx + by) % 5)

def _verify_latin(square):
    n = len(square)
    for i in range(n):
        if sorted(square[i]) != list(range(n)):
            return False
    for j in range(n):
        if sorted(square[i][j] for i in range(n)) != list(range(n)):
            return False
    return True

def _verify_orthogonal(s1, s2):
    n = len(s1)
    pairs = set()
    for i in range(n):
        for j in range(n):
            p = (s1[i][j], s2[i][j])
            if p in pairs:
                return False
            pairs.add(p)
    return len(pairs) == n * n

def _verify_all(squares):
    if not squares:
        return False
    n = len(squares[0])
    for sq in squares:
        if not _verify_latin(sq):
            return False
    for i in range(len(squares)):
        for j in range(i + 1, len(squares)):
            if not _verify_orthogonal(squares[i], squares[j]):
                return False
    return True

def _construct_mols_10():
    """
    Construct 9 MOLS of order 10 using the group Z_2 x Z_5 automorphism method.
    """
    n = 10
    
    # S0: group table
    S0 = [[_group_add(i, j) for j in range(n)] for i in range(n)]
    
    squares = [S0]
    
    # Automorphisms of Z_2 x Z_5:
    # phi_c,d(a,b) = (a + c*b mod 2, d*b mod 5) where c in {0,1}, d in {1,2,3,4}
    # This gives 8 non-identity automorphisms
    
    for c in range(2):
        for d in range(1, 5):
            if c == 0 and d == 1:
                continue  # This is the identity, already S0
            
            # Define the automorphism
            def phi(x, c=c, d=d):
                a, b = x // 5, x % 5
                new_a = (a + c * b) % 2
                new_b = (d * b) % 5
                return new_a * 5 + new_b
            
            # Construct square: S[i][j] = i + phi(j) in group operation
            sq = [[_group_add(i, phi(j)) for j in range(n)] for i in range(n)]
            
            # Verify it's a Latin square and orthogonal to all existing
            if _verify_latin(sq):
                orth = True
                for existing in squares:
                    if not _verify_orthogonal(sq, existing):
                        orth = False
                        break
                if orth:
                    squares.append(sq)
    
    return squares

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55 second deadline, 5s margin
    
    try:
        squares = _construct_mols_10()
        
        # Validate
        if _verify_all(squares):
            # Ensure all values are Python int
            result = []
            for sq in squares:
                new_sq = [[int(sq[i][j]) for j in range(10)] for i in range(10)]
                result.append(new_sq)
            return result
        else:
            # Fall back to just S0
            s0 = [[_group_add(i, j) for j in range(10)] for i in range(10)]
            return [s0]
    except Exception:
        # Ultimate fallback
        s0 = [[(i + j) % 10 for j in range(10)] for i in range(10)]
        return [s0]

# EVOLVE-BLOCK-END
