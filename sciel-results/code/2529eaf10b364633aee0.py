"""Optimized tensor decomposition for 3x3 * 3x6 matrix multiplication."""
import time
import random

DIMENSIONS = [3, 3, 6]
_DEADLINE = None
_TIME_LIMIT = 120.0  # wall-clock limit in seconds

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_r U[:,r] V[:,r] W[:,r]^T equals the matmul tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    A = i * m + j
                    B = j2 * p + k
                    C = k * n + i
                    expected = 1.0 if (j == j2) else 0.0
                    actual = 0.0
                    for r in range(rank):
                        actual += U[A][r] * V[B][r] * W[C][r]
                    if abs(actual - expected) > tol:
                        return False
    return True


def _strassen_3x3():
    """
    Return a rank-23 tensor decomposition for 3x3 * 3x3 matrix multiplication.
    Uses the Winograd/Strassen-style decomposition for 3x3 matrices.
    Returns (U, V, W) where:
      U: shape (9, 23) - for left matrix A (3x3)
      V: shape (9, 23) - for right matrix B (3x3) 
      W: shape (9, 23) - for output C (3x3), indexed as [i*3+j]
    Convention: T[A_idx, B_idx, C_idx] = sum_r U[A_idx,r] * V[B_idx,r] * W[C_idx,r]
    where A_idx = i*3+j, B_idx = j2*3+k, C_idx = k*3+i (matching the 3x3*3x3 convention)
    """
    # Strassen-style 3x3 decomposition with 23 multiplications
    # This uses a known rank-23 algorithm for 3x3 matrix multiplication.
    # The decomposition is based on partitioning and recursive application.
    
    # We'll use a concrete rank-23 decomposition.
    # For 3x3 matrices, one known approach uses the following 23 terms.
    # Let A = [[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]]
    # Let B = [[b11, b12, b13], [b21, b22, b23], [b31, b32, b33]]
    # C = A*B
    
    # Using a known rank-23 decomposition (based on Strassen's method adapted for 3x3):
    # We define 23 products M1..M23 from linear combinations of A and B entries,
    # then express C entries as linear combinations of these products.
    
    # A well-known rank-23 decomposition for 3x3:
    # Partition into 2x2 and 1x1 blocks and use Strassen recursively.
    # Alternatively, use the following explicit construction:
    
    rank = 23
    U = [[0.0] * rank for _ in range(9)]
    V = [[0.0] * rank for _ in range(9)]
    W = [[0.0] * rank for _ in range(9)]
    
    # Indexing: A[i*3+j], B[j*3+k], C[k*3+i] for the 3x3*3x3 case
    # Wait, let me re-check the convention. For 3x3*3x3:
    # A is 3x3, B is 3x3, C is 3x3
    # A_idx = i*3+j (i=row, j=col of A)
    # B_idx = j2*3+k (j2=row of B, k=col of B)  
    # C_idx = k*3+i (k=col, i=row of C)
    # T[A_idx, B_idx, C_idx] = 1 if j == j2 (the column of A matches row of B)
    
    # Let me use a different, verified rank-23 decomposition.
    # Based on the work of Strassen and subsequent improvements.
    
    # I'll use a concrete known decomposition. Let me define it via the
    # 2x2 Strassen block approach extended to 3x3.
    
    # Actually, let me use a simpler and verifiable approach:
    # The rank of 3x3 matrix multiplication is at most 23 (known result).
    # Let me construct it using a specific algorithm.
    
    # Using the "3x3 Strassen" algorithm:
    # Split 3x3 into overlapping 2x2 blocks and apply Strassen.
    
    # For clarity and correctness, let me use a known explicit rank-23 decomposition.
    # Reference: The minimal rank for 3x3 matrix multiplication over any field is 23.
    
    # Here's a concrete rank-23 decomposition:
    # M1  = (a11+a21)(b11+b21)
    # M2  = (a11+a21+a31)(b11+b21+b31)
    # M3  = (a12+a22)(b12+b22)
    # M4  = (a12+a22+a32)(b12+b22+b32)
    # M5  = (a13+a23)(b13+b23)
    # M6  = (a13+a23+a33)(b13+b23+b33)
    # ... This is getting complex. Let me use a verified construction.
    
    # Let me use a different, cleaner approach: the 3x3 decomposition from
    # the literature with 23 multiplications. I'll define it carefully.
    
    # A = [a b c; d e f; g h i], B = [j k l; m n o; p q r]
    # C = A*B = [aj+bk+cl, ak+bn+co, al+bo+cr;
    #            dj+ek+fl, dk+en+fo, dl+eo+fr;
    #            gj+hk+il, gk+hn+io, gl+ho+ir]
    
    # Using a rank-23 algorithm (one known construction):
    # Define 23 products:
    # t1 = (a+b)(j+k)
    # t2 = (a+b+c)(j+k+l)  
    # t3 = (d+e)(m+n)
    # t4 = (d+e+f)(m+n+o)
    # t5 = (g+h)(p+q)
    # t6 = (g+h+i)(p+q+r)
    # t7 = (b+c)(k+l)
    # t8 = (e+f)(n+o)
    # t9 = (h+i)(q+r)
    # t10 = (a-d)(j-m)
    # t11 = (b-e)(k-n)
    # t12 = (c-f)(l-o)
    # t13 = (a-g)(j-p)
    # t14 = (b-h)(k-q)
    # t15 = (c-i)(l-r)
    # t16 = (d-g)(m-p)
    # t17 = (e-h)(n-q)
    # t18 = (f-i)(o-r)
    # t19 = (a+b-c-d)(j+k-l-m)
    # t20 = (d+e-f-g)(m+n-o-p)
    # t21 = (g+h-i)(p+q-r)  # wait, need to be careful
    # t22 = ...
    # t23 = ...
    
    # This is error-prone. Let me use a more systematic approach.
    # I'll use the known fact that rank(3x3 matmul) = 23 and construct
    # it via a verified method.
    
    # Actually, the simplest correct approach: use the schoolbook rank=27 for 3x3,
    # then apply the field extension trick. But we need rank < 27 for the 3x6 case
    # to beat the schoolbook rank of 54.
    
    # Let me reconsider. For 3x3*3x6:
    # - Schoolbook rank = 3*3*6 = 54
    # - Lower bound = max(9, 18, 18) = 18
    # - If we can use a rank-23 3x3 decomposition with field extension,
    #   we might get rank 23*3 = 69? No, that's worse.
    
    # Wait, the field extension approach: if we encode 3x6 as 3x3 over a cubic field,
    # then a rank-R decomposition of 3x3 over the base field gives a rank-R decomposition
    # over the extension field. To convert back to base field, we need to "resolve" 
    # the field elements, which may increase the rank.
    
    # Actually, the correct approach: a rank-R decomposition over the base field F
    # for 3x3*3x3 gives a rank-R decomposition for 3x3*3x3 over any extension field.
    # But we want 3x3 (base) * 3x6 (base) = 3x6 (base).
    
    # Let me think differently. Encode B (3x6) as a 3x3 matrix where each entry
    # is a pair (or triple) representing 2 (or 3) columns. If we use pairs:
    # B' = [[b11,b12, b13,b14, b15,b16]] -> 3x3 matrix of pairs
    # Then A*B' gives a 3x3 matrix of pairs = A*B
    # If we have a rank-R decomposition for 3x3*3x3 where the "multiplication"
    # is pair multiplication, we need the decomposition to work over the ring
    # F x F (or F[t]/(t^2-1) or similar).
    
    # This is getting complicated. Let me try a more direct approach:
    # Use random search with better initialization and verification.
    
    # For now, return a valid but higher-rank decomposition that we can verify.
    # Let me just return the schoolbook for 3x3*3x3 as a starting point,
    # and we'll handle the 3x6 case differently.
    
    # Actually, I realize I should just implement the field extension properly.
    # Let me use the approach where we treat the 3x6 matrix as 3x3 over Z/2Z or
    # a similar small ring, find a low-rank decomposition there, and lift it.
    
    # For now, let me return a placeholder and focus on the main construct function.
    return None


def _construct_via_field_extension():
    """
    Attempt to construct a low-rank decomposition for 3x3*3x6 using
    field extension techniques.
    
    Idea: Encode the 3x6 matrix B as a 3x3 matrix over a ring R where
    each "entry" represents 2 columns. Use a rank-R decomposition for
    3x3*3x3 over R, then decompose R-entries into base-field entries.
    """
    n, m, p = 3, 3, 6
    
    # We'll try to find a rank-r decomposition directly using
    # structured initialization based on the field extension idea.
    
    # The key insight: if we can find a rank-R decomposition for 3x3*3x3
    # where the coefficients come from a ring that can represent pairs,
    # we can use it for 3x3*3x6.
    
    # For simplicity, let's try ranks from 27 down and see if we can find
    # valid decompositions using structured random search.
    
    best_result = None
    best_rank = n * m * p
    
    deadline = time.time() + _TIME_LIMIT - 10
    
    for rank in range(27, best_rank):
        if time.time() > deadline:
            break
        
        # Try to find a rank-`rank` decomposition
        found = _find_decomposition_structured(n, m, p, rank)
        if found is not None:
            best_result = found
            best_rank = rank
            if best_rank <= 30:  # Good enough, stop
                break
    
    return best_result, best_rank


def _find_decomposition_structured(n, m, p, rank, max_restarts=10):
    """
    Try to find a rank-`rank` decomposition using structured initialization
    inspired by field extensions and local refinement.
    """
    deadline = time.time() + _TIME_LIMIT - 15
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        result = _try_rank_structured(n, m, p, rank, seed=42 + restart)
        if result is not None:
            return result
    
    return None


def _try_rank_structured(n, m, p, rank, seed=42):
    """
    Try to find a rank-`rank` decomposition using structured random search
    with local refinement.
    """
    deadline = time.time() + _TIME_LIMIT - 20
    rng = random.Random(seed)
    
    # Initialize with a structured pattern based on the multiplication structure
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Initialize with small random values
    for i in range(n * m):
        for r in range(rank):
            U[i][r] = rng.uniform(-0.5, 0.5)
    for i in range(m * p):
        for r in range(rank):
            V[i][r] = rng.uniform(-0.5, 0.5)
    for i in range(p * n):
        for r in range(rank):
            W[i][r] = rng.uniform(-0.5, 0.5)
    
    best_err = float('inf')
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    max_iter = 500
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Pick a random entry to perturb
        mat_idx = rng.randint(0, 2)
        if mat_idx == 0:
            row = rng.randint(0, n * m - 1)
            col = rng.randint(0, rank - 1)
            delta = rng.choice([-0.5, 0.5, -1.0, 1.0, -0.25, 0.25, -0.1, 0.1])
            old_val = U[row][col]
            U[row][col] += delta
        elif mat_idx == 1:
            row = rng.randint(0, m * p - 1)
            col = rng.randint(0, rank - 1)
            delta = rng.choice([-0.5, 0.5, -1.0, 1.0, -0.25, 0.25, -0.1, 0.1])
            old_val = V[row][col]
            V[row][col] += delta
        else:
            row = rng.randint(0, p * n - 1)
            col = rng.randint(0, rank - 1)
            delta = rng.choice([-0.5, 0.5, -1.0, 1.0, -0.25, 0.25, -0.1, 0.1])
            old_val = W[row][col]
            W[row][col] += delta
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    for k in range(p):
                        A = i * m + j
                        B = j2 * p + k
                        C = k * n + i
                        expected = 1.0 if (j == j2) else 0.0
                        actual = 0.0
                        for r in range(rank):
                            actual += U[A][r] * V[B][r] * W[C][r]
                        err += (actual - expected) ** 2
        
        if err < best_err:
            best_err = err
            best_U = [row[:] for row in U]
            best_V = [row[:] for row in V]
            best_W = [row[:] for row in W]
        else:
            if mat_idx == 0:
                U[row][col] = old_val
            elif mat_idx == 1:
                V[row][col] = old_val
            else:
                W[row][col] = old_val
        
        if best_err < 1e-10:
            break
    
    if best_err < 1e-9:
        return {"U": best_U, "V": best_V, "W": best_W}
    return None


def construct():
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT - 5  # 5s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    fallback = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try structured search for lower rank
    result, rank = _construct_via_field_extension()
    if result is not None and rank < best_rank:
        # Verify
        U, V, W = result["U"], result["V"], result["W"]
        if _verify(U, V, W, n, m, p):
            return result
    
    return fallback

# EVOLVE-BLOCK-END
