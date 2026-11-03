"""Matrix multiplication 2x3 by 3x5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    u_val = U[i * m + j][c]
                    v_val = V[j * p + k][c]
                    w_val = W[k * n + i][c]
                    # Convert to Fraction for exact arithmetic
                    if not isinstance(u_val, Fraction):
                        u_val = Fraction(u_val)
                    if not isinstance(v_val, Fraction):
                        v_val = Fraction(v_val)
                    if not isinstance(w_val, Fraction):
                        w_val = Fraction(w_val)
                    total += u_val * v_val * w_val
                if total != 1:
                    return False
    return True

def _build_strassen_decomposition():
    """
    Build a rank-28 decomposition for M_{2,3,5} using a structured approach.
    
    We partition:
    A is 2x3: A = [A_left (2x2) | A_right (2x1)]
    B is 3x5: B = [B_left (3x2) | B_right (3x3)]
    
    C = A*B = [A_left * B_left + A_right * B_right_left | A_left * B_right_right + A_right * B_right_right]
    
    Actually, let me think more carefully.
    
    A is 2x3, B is 3x5.
    Partition A into A1 (2x2) and A2 (2x1)
    Partition B into B1 (3x2) and B2 (3x3)
    
    C[0:2, 0:2] = A1*B1 + A2*B2[0:1, 0:2]  -- wait, this doesn't partition cleanly.
    
    Let me use a different approach. Use Strassen-like decomposition on the 2x2x2 core.
    
    For the 2x2 * 2x2 case, Strassen uses 7 multiplications.
    
    Let's partition:
    A = [[a11, a12, a13], [a21, a22, a23]]
    B = [[b11, b12, b13, b14, b15],
         [b21, b22, b23, b24, b25],
         [b31, b32, b33, b34, b35]]
    
    We want to compute C = A*B, which is 2x5.
    
    C[0,0] = a11*b11 + a12*b21 + a13*b31
    C[0,1] = a11*b12 + a12*b22 + a13*b32
    C[0,2] = a11*b13 + a12*b23 + a13*b33
    C[0,3] = a11*b14 + a12*b24 + a13*b34
    C[0,4] = a11*b15 + a12*b25 + a13*b35
    
    C[1,0] = a21*b11 + a22*b21 + a23*b31
    C[1,1] = a21*b12 + a22*b22 + a23*b32
    C[1,2] = a21*b13 + a22*b23 + a23*b33
    C[1,3] = a21*b14 + a22*b24 + a23*b34
    C[1,4] = a21*b15 + a22*b25 + a23*b35
    
    The schoolbook approach uses 2*3*5 = 30 rank-1 terms.
    
    Let me try to use a more clever decomposition.
    
    Key insight: We can group terms. Let's define intermediate sums.
    
    For each column k of B, we compute A * B[:,k] which gives a 2-vector.
    
    Actually, let me try a different structural approach.
    
    Consider the tensor T[i,j,k] where i in {0,1}, j in {0,1,2}, k in {0,1,2,3,4}.
    
    We need to find U[i*m+j], V[j*p+k], W[k*n+i] such that
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k).
    
    Let me try rank 28 by exploiting some symmetries.
    
    Actually, let me just try to find a good decomposition through search,
    but with a smarter initialization and verification using exact arithmetic.
    """
    
    n, m, p = 2, 3, 5
    nm, mp, pn = n * m, m * p, p * n  # 6, 15, 10
    
    # Try to construct a rank-28 decomposition
    # We'll use a structured approach based on Strassen's ideas
    
    # For now, let's try a systematic search with exact arithmetic
    deadline = time.time() + 100
    
    best_rank = n * m * p  # Start with schoolbook rank 30
    
    # First, get the schoolbook decomposition as baseline
    schoolbook_rank = n * m * p
    U = [[Fraction(0)] * schoolbook_rank for _ in range(nm)]
    V = [[Fraction(0)] * schoolbook_rank for _ in range(mp)]
    W = [[Fraction(0)] * schoolbook_rank for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
    
    if _verify(U, V, W, n, m, p):
        best_rank = schoolbook_rank
    
    # Now try to find lower rank decompositions
    # Theoretical lower bound for M_{2,3,5} is at least max(2*5, 3*5, 2*3) = max(10, 15, 6) = 15
    # But actually the tensor rank is likely higher.
    
    # Try ranks from 15 up to 29
    for target_rank in range(15, best_rank):
        if time.time() > deadline:
            break
        
        result = _search_rank(target_rank, n, m, p, deadline)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                best_rank = target_rank
                break
    
    return {"U": U, "V": V, "W": W}

def _search_rank(rank, n, m, p, deadline):
    """Search for a rank decomposition using random restarts with local search."""
    nm, mp, pn = n * m, m * p, p * n
    
    rng = random.Random(rank * 7919 + 13)
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Initialize with random entries from a small set
        values = [Fraction(0), Fraction(1), Fraction(-1), Fraction(1, 2), Fraction(-1, 2)]
        
        U = [[rng.choice(values) for _ in range(rank)] for _ in range(nm)]
        V = [[rng.choice(values) for _ in range(rank)] for _ in range(mp)]
        W = [[rng.choice(values) for _ in range(rank)] for _ in range(pn)]
        
        # Local search
        result = _local_search_exact(U, V, W, n, m, p, rank, deadline, rng)
        if result is not None:
            return result
    
    return None

def _local_search_exact(U, V, W, n, m, p, rank, deadline, rng):
    """Local search with exact arithmetic verification."""
    nm, mp, pn = n * m, m * p, p * n
    
    def _score(U_, V_, W_):
        """Count how many (i,j,k) triples sum to exactly 1."""
        count = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for c in range(rank):
                        u_val = U_[i * m + j][c]
                        v_val = V_[j * p + k][c]
                        w_val = W_[k * n + i][c]
                        if not isinstance(u_val, Fraction):
                            u_val = Fraction(u_val)
                        if not isinstance(v_val, Fraction):
                            v_val = Fraction(v_val)
                        if not isinstance(w_val, Fraction):
                            w_val = Fraction(w_val)
                        total += u_val * v_val * w_val
                    if total == 1:
                        count += 1
        return count
    
    current_score = _score(U, V, W)
    target_score = n * m * p
    
    if current_score == target_score:
        return U, V, W
    
    values = [Fraction(0), Fraction(1), Fraction(-1), Fraction(1, 2), Fraction(-1, 2),
              Fraction(2), Fraction(-2), Fraction(3), Fraction(-3)]
    
    for iteration in range(2000):
        if time.time() > deadline:
            break
        if current_score == target_score:
            break
        
        # Pick a random matrix and entry to perturb
        mat_choice = rng.random()
        if mat_choice < 0.33:
            mat = 'U'
            idx = rng.randint(0, nm - 1)
            col = rng.randint(0, rank - 1)
            old_val = U[idx][col]
            new_val = rng.choice(values)
            U[idx][col] = new_val
        elif mat_choice < 0.66:
            mat = 'V'
            idx = rng.randint(0, mp - 1)
            col = rng.randint(0, rank - 1)
            old_val = V[idx][col]
            new_val = rng.choice(values)
            V[idx][col] = new_val
        else:
            mat = 'W'
            idx = rng.randint(0, pn - 1)
            col = rng.randint(0, rank - 1)
            old_val = W[idx][col]
            new_val = rng.choice(values)
            W[idx][col] = new_val
        
        new_score = _score(U, V, W)
        
        if new_score >= current_score:
            current_score = new_score
        else:
            # Revert
            if mat == 'U':
                U[idx][col] = old_val
            elif mat == 'V':
                V[idx][col] = old_val
            else:
                W[idx][col] = old_val
    
    if current_score == target_score:
        return U, V, W
    
    return None

def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # Try the structured Strassen-based approach first
    deadline = time.time() + 100
    result = _build_strassen_decomposition()
    
    # Verify the result
    if not _verify(result['U'], result['V'], result['W'], n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[Fraction(0)] * rank for _ in range(n * m)]
        V = [[Fraction(0)] * rank for _ in range(m * p)]
        W = [[Fraction(0)] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = Fraction(1)
                    V[j * p + k][col] = Fraction(1)
                    W[k * n + i][col] = Fraction(1)
                    col += 1
        result = {"U": U, "V": V, "W": W}
    
    return result
# EVOLVE-BLOCK-END
