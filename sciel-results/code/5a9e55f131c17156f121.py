import time
import numpy as np

DIMENSIONS = [2, 2, 3]

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2x2 matrix by a 2x3 matrix.
    
    Uses a structural decomposition: treat the 2x2 x 2x3 problem as
    combining two 2x2 x 2x2 sub-problems (for columns 0-1 and column 2)
    with shared A-structure. We attempt rank 10 by finding a more
    efficient coupling than the naive 7+4=11 approach.
    
    If rank 10 is not achievable, falls back to rank 11.
    """
    n, m, p = 2, 2, 3
    
    # Try rank 10 first with a different structural pattern
    # The tensor T[i,j,k] = delta[i,j] for the matmul: C[i,k] = sum_j A[i,j]*B[j,k]
    # For 2x2x3: indices i in {0,1}, j in {0,1}, k in {0,1,2}
    # T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    
    # Structural approach: use 10 terms where the last column (k=2) 
    # is handled with only 2 terms (one per output row), and the first 
    # two columns use 8 terms via a modified Strassen-like scheme.
    
    rank = 10
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def set_u(i, j, r, val): U[i*m+j][r] = val
    def set_v(j, k, r, val): V[j*p+k][r] = val
    def set_w(k, i, r, val): W[k*n+i][r] = val

    # Terms 0-7: Handle C[:, 0:2] (two output columns) with 8 terms
    # This is a 2x2 x 2x2 problem, normally rank 7 (Strassen).
    # We use 8 terms here to allow the last column to share structure.
    
    # Term 0: A00*B00 -> C00, A11*B11 -> C11
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1); set_w(1, 1, 0, 1)

    # Term 1: (A10+A11)*B00 -> C10, -C11
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1); set_w(1, 1, 1, -1)

    # Term 2: A00*(B01-B11) -> C01, C11
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1); set_w(1, 1, 2, 1)

    # Term 3: A11*(B10-B00) -> C00, C01
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1); set_w(0, 1, 3, 1)

    # Term 4: (A00+A01)*B11 -> -C00, C10
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1); set_w(1, 0, 4, 1)

    # Term 5: (A10-A00)*(B00+B01) -> C11
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    # Term 6: (A01-A11)*(B10+B11) -> C00
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)

    # Term 7: A01*B10 -> C00, C01 (new term to help with column 2 sharing)
    set_u(0, 1, 7, 1)
    set_v(1, 0, 7, 1)
    set_w(0, 0, 7, 1); set_w(1, 0, 7, 1)

    # Terms 8-9: Handle C[:, 2] (third column) with 2 terms
    # Term 8: A00*B02 + A01*B12 -> C02
    set_u(0, 0, 8, 1); set_u(0, 1, 8, 1)
    set_v(0, 2, 8, 1); set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1)

    # Term 9: A10*B02 + A11*B12 -> C12
    set_u(1, 0, 9, 1); set_u(1, 1, 9, 1)
    set_v(0, 2, 9, 1); set_v(1, 2, 9, 1)
    set_w(2, 1, 9, 1)

    # Verify the decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta(i==j)
    # Actually, the matmul tensor is: T[i,j,k] = 1 if i==j (since C[i,k] = sum_j A[i,j]B[j,k])
    # Wait, no. The tensor for matmul A(n x m) * B(m x p) = C(n x p) is:
    # T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
    # So T[i,j,k] = 1 for all valid i,j,k (it's a 3-tensor where each slice is identity-like)
    
    # More precisely: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # No wait, that's not right either. The tensor is defined such that:
    # <T, A ⊗ B> = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] = sum_{i,k} C[i,k] * 1
    # Actually, the standard definition: T[i,j,k] = delta_{i,j} would give trace, not matmul.
    
    # For matrix multiplication A (n x m) * B (m x p) = C (n x p):
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T satisfies: C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (each term A[i,j]*B[j,k] contributes to C[i,k])
    
    # Let's verify: for each (i,j,k), the contribution should be 1
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
        if not valid:
            break
    if not valid:
        # Fall back to rank 11
        rank = 11
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def set_u(i, j, r, val): U[i*m+j][r] = val
        def set_v(j, k, r, val): V[j*p+k][r] = val
        def set_w(k, i, r, val): W[k*n+i][r] = val

        set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
        set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
        set_w(0, 0, 0, 1); set_w(1, 1, 0, 1)

        set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
        set_v(0, 0, 1, 1)
        set_w(0, 1, 1, 1); set_w(1, 1, 1, -1)

        set_u(0, 0, 2, 1)
        set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
        set_w(1, 0, 2, 1); set_w(1, 1, 2, 1)

        set_u(1, 1, 3, 1)
        set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
        set_w(0, 0, 3, 1); set_w(0, 1, 3, 1)

        set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
        set_v(1, 1, 4, 1)
        set_w(0, 0, 4, -1); set_w(1, 0, 4, 1)

        set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
        set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
        set_w(1, 1, 5, 1)

        set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
        set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
        set_w(0, 0, 6, 1)

        set_u(0, 0, 7, 1)
        set_v(0, 2, 7, 1)
        set_w(2, 0, 7, 1)

        set_u(0, 1, 8, 1)
        set_v(1, 2, 8, 1)
        set_w(2, 0, 8, 1)

        set_u(1, 0, 9, 1)
        set_v(0, 2, 9, 1)
        set_w(2, 1, 9, 1)

        set_u(1, 1, 10, 1)
        set_v(1, 2, 10, 1)
        set_w(2, 1, 10, 1)

    return {"U": U, "V": V, "W": W}
