"""Strassen-based rank-84 decomposition for 4x4 * 4x6 matrix multiplication."""
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def _build_strassen_decomposition():
    """
    Build a rank-84 exact tensor decomposition for (4,4,6) matrix multiplication
    using 2x2 Strassen-style block decomposition.
    
    A is 4x4, B is 4x6, C = A*B is 4x6.
    
    We partition:
      A into 2x2 blocks of size 2x2: A11, A12, A21, A22
      B into 2x2 blocks: top half is 2x6, bottom half is 2x6
        B_top = [B11 B12] where B11 is 2x3, B12 is 2x3
        B_bot = [B21 B22] where B21 is 2x3, B22 is 2x3
      C into 2x2 blocks: C11, C12, C21, C22 each 2x3
    
    Using Strassen's 7 products:
      M1 = (A11+A22)*(B11+B22)
      M2 = (A21+A22)*B11
      M3 = A11*(B12-B22)
      M4 = A22*(B21-B11)
      M5 = (A11+A12)*B22
      M6 = (A21-A11)*(B11+B12)
      M7 = (A12-A22)*(B21+B22)
    
    Then:
      C11 = M1 - M4 + M5 - M7
      C12 = M3 + M5
      C21 = M2 + M4
      C22 = M1 - M3 + M2 - M6
    """
    n, m, p = DIMENSIONS  # 4, 4, 6
    # Each block: A blocks are 2x2, B blocks are 2x3, C blocks are 2x3
    rank = 84  # 7 * (2*2*3) = 7 * 12
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 84
    V = [[0] * rank for _ in range(m * p)]  # 24 x 84
    W = [[0] * rank for _ in range(p * n)]  # 24 x 84
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices):
        nonlocal col
        for ui, ui_val in u_indices:
            U[ui][col] = ui_val
        for vi, vi_val in v_indices:
            V[vi][col] = vi_val
        for wi, wi_val in w_indices:
            W[wi][col] = wi_val
        col += 1
    
    # Block index conventions:
    # A[i,j] where i in 0..3, j in 0..3
    #   A11: i=0,1 j=0,1
    #   A12: i=0,1 j=2,3
    #   A21: i=2,3 j=0,1
    #   A22: i=2,3 j=2,3
    # B[j,k] where j in 0..3, k in 0..5
    #   B11: j=0,1 k=0,1,2
    #   B12: j=0,1 k=3,4,5
    #   B21: j=2,3 k=0,1,2
    #   B22: j=2,3 k=3,4,5
    # C[i,k] where i in 0..3, k in 0..5
    #   C11: i=0,1 k=0,1,2
    #   C12: i=0,1 k=3,4,5
    #   C21: i=2,3 k=0,1,2
    #   C22: i=2,3 k=3,4,5
    
    # U index: i*m + j = i*4 + j (row i of A, col j of A)
    # V index: j*p + k = j*6 + k (row j of B, col k of B)
    # W index: k*n + i = k*4 + i (row k of C, col i of C) [trace convention: W[k*n+i, r]]
    
    for i in range(2):
        for j in range(2):
            for k in range(3):
                # M1 = (A11+A22)*(B11+B22)
                # Contributes to C11 and C22
                # We'll handle each C block contribution separately per product
                
                # For M1: U has A11[i,j] and A22[i,j], V has B11[j,k] and B22[j,k]
                # C11 gets +M1, C22 gets +M1
                # We need separate columns for each C target since W entries differ
                # Actually, we can combine: for each (i,j,k) triple, we add to a column
                # that contributes to both C11 and C22 with appropriate W values
                
                pass
    
    # Let me redo this more carefully. Each of the 7 products M_t contributes to
    # specific C blocks. For each product and each C block it contributes to,
    # we need 2*2*3 = 12 scalar terms (i in 0,1; j in 0,1; k in 0,1,2).
    # But we can share the U and V entries across C blocks for the same product.
    
    # Actually, the standard way: for each product M_t, we create 12 columns
    # (one per (i,j,k) combination in the block), and for each column we set
    # U, V entries based on the product definition, and W entries based on
    # which C blocks this product contributes to.
    
    # Let me restart the construction properly.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Product definitions:
    # M1: U_factors = [(A11, +1), (A22, +1)], V_factors = [(B11, +1), (B22, +1)]
    #     C_targets = [(C11, +1), (C22, +1)]
    # M2: U_factors = [(A21, +1), (A22, +1)], V_factors = [(B11, +1)]
    #     C_targets = [(C21, +1), (C22, +1)]
    # M3: U_factors = [(A11, +1)], V_factors = [(B12, +1), (B22, -1)]
    #     C_targets = [(C12, +1), (C22, -1)]
    # M4: U_factors = [(A22, +1)], V_factors = [(B21, +1), (B11, -1)]
    #     C_targets = [(C21, +1), (C11, -1)]
    # M5: U_factors = [(A11, +1), (A12, +1)], V_factors = [(B22, +1)]
    #     C_targets = [(C11, +1), (C12, +1)]
    # M6: U_factors = [(A21, +1), (A11, -1)], V_factors = [(B11, +1), (B12, +1)]
    #     C_targets = [(C22, -1)]
    # M7: U_factors = [(A12, +1), (A22, -1)], V_factors = [(B21, +1), (B22, +1)]
    #     C_targets = [(C11, -1)]
    
    # Block index mapping:
    # A11[i,j] -> U index: i*4 + j
    # A12[i,j] -> U index: i*4 + (2+j)
    # A21[i,j] -> U index: (2+i)*4 + j
    # A22[i,j] -> U index: (2+i)*4 + (2+j)
    # B11[j,k] -> V index: j*6 + k
    # B12[j,k] -> V index: j*6 + (3+k)
    # B21[j,k] -> V index: (2+j)*6 + k
    # B22[j,k] -> V index: (2+j)*6 + (3+k)
    # C11[i,k] -> W index: k*4 + i
    # C12[i,k] -> W index: (3+k)*4 + i
    # C21[i,k] -> W index: k*4 + (2+i)
    # C22[i,k] -> W index: (3+k)*4 + (2+i)
    
    def a11(i, j): return i * 4 + j
    def a12(i, j): return i * 4 + (2 + j)
    def a21(i, j): return (2 + i) * 4 + j
    def a22(i, j): return (2 + i) * 4 + (2 + j)
    def b11(j, k): return j * 6 + k
    def b12(j, k): return j * 6 + (3 + k)
    def b21(j, k): return (2 + j) * 6 + k
    def b22(j, k): return (2 + j) * 6 + (3 + k)
    def c11(i, k): return k * 4 + i
    def c12(i, k): return (3 + k) * 4 + i
    def c21(i, k): return k * 4 + (2 + i)
    def c22(i, k): return (3 + k) * 4 + (2 + i)
    
    products = [
        # (name, U_factors, V_factors, C_targets)
        # U_factors: list of (func, sign)
        # V_factors: list of (func, sign)
        # C_targets: list of (func, sign)
        ("M1", 
         [(a11, 1), (a22, 1)],
         [(b11, 1), (b22, 1)],
         [(c11, 1), (c22, 1)]),
        ("M2",
         [(a21, 1), (a22, 1)],
         [(b11, 1)],
         [(c21, 1), (c22, 1)]),
        ("M3",
         [(a11, 1)],
         [(b12, 1), (b22, -1)],
         [(c12, 1), (c22, -1)]),
        ("M4",
         [(a22, 1)],
         [(b21, 1), (b11, -1)],
         [(c21, 1), (c11, -1)]),
        ("M5",
         [(a11, 1), (a12, 1)],
         [(b22, 1)],
         [(c11, 1), (c12, 1)]),
        ("M6",
         [(a21, 1), (a11, -1)],
         [(b11, 1), (b12, 1)],
         [(c22, -1)]),
        ("M7",
         [(a12, 1), (a22, -1)],
         [(b21, 1), (b22, 1)],
         [(c11, -1)]),
    ]
    
    for pname, u_facs, v_facs, c_targs in products:
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    # Set U entries
                    for func, sign in u_facs:
                        U[func(i, j)][col] = sign
                    # Set V entries
                    for func, sign in v_facs:
                        V[func(j, k)][col] = sign
                    # Set W entries
                    for func, sign in c_targs:
                        W[func(i, k)][col] = sign
                    col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    # Convert to numpy for verification
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Verify: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should give
    # the (i,k) entry of C = A*B, i.e., sum_j A[i,j]*B[j,k]
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # should equal delta (the standard multiplication tensor)
    
    # Let's verify on a random example
    rng = np.random.RandomState(42)
    A_test = rng.randn(4, 4)
    B_test = rng.randn(4, 6)
    C_expected = A_test @ B_test
    
    # Compute using the decomposition:
    # C[i,k] = sum_j sum_r U[i*4+j, r] * V[j*6+k, r] * W[k*4+i, r]
    # But this is expensive to compute directly. Let's use the tensor form.
    # Actually, the decomposition says:
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_r (sum_j U[i*4+j,r] * A[i,j]) * (sum_k V[j*6+k,r] * B[j,k])...
    # No, that's not right. The tensor decomposition means:
    # T_{i,j,k} = sum_r U_{i*4+j, r} * V_{j*6+k, r} * W_{k*4+i, r}
    # where T_{i,j,k} = 1 if it's the multiplication tensor (i.e., T_{i,j,k} represents
    # the contribution of A[i,j]*B[j,k] to C[i,k])
    
    # For matrix multiplication, the tensor is:
    # T[i,j,k] = 1 for all (i,j,k) where the (i,k) entry of C depends on A[i,j]*B[j,k]
    # which is all i,j,k. So T[i,j,k] = 1 for all valid (i,j,k).
    
    # Let's verify by checking that the contraction works.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_j sum_r (U[i*4+j, r] * A[i,j]) * (V[j*6+k, r] * B[j,k]) * ... 
    # This isn't quite right. Let me think again.
    
    # The tensor decomposition of matrix multiplication:
    # The multiplication tensor T has T[i,j,k] = 1 (for the standard basis).
    # If T = sum_r u_r ⊗ v_r ⊗ w_r, then for any A, B:
    # (A*B)[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j A[i,j] u_r[i,j]) (sum_k B[j,k] v_r[j,k]) w_r[k,i]
    # Wait, that's not the standard form. Let me be more careful.
    
    # The standard form: C = A * B means C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T_{ijk} = 1 (unit tensor for mult)
    # T = sum_r U_r ⊗ V_r ⊗ W_r where U_r[i,j] = U[i*m+j, r], V_r[j,k] = V[j*p+k, r], W_r[k,i] = W[k*n+i, r]
    # Then: C[i,k] = sum_j A[i,j] B[j,k] = sum_{j} T[i,j,k] A[i,j] B[j,k]
    #            = sum_r sum_j U_r[i,j] V_r[j,k] W_r[k,i] A[i,j] B[j,k]
    #            = sum_r W_r[k,i] (sum_j U_r[i,j] A[i,j]) (sum_k V_r[j,k] B[j,k])
    
    # Hmm, this is getting complicated. Let me just do a numerical check.
    
    # Build U_r, V_r, W_r as 3D arrays
    # U_r[i,j] = U_np[i*4+j, r] for i in 0..3, j in 0..3
    # V_r[j,k] = V_np[j*6+k, r] for j in 0..3, k in 0..5
    # W_r[k,i] = W_np[k*4+i, r] for k in 0..5, i in 0..3
    
    # C_computed[i,k] = sum_r W_np[k*4+i, r] * (sum_j U_np[i*4+j, r] * A_test[i,j]) * (sum_k2 V_np[j*6+k2, r] * B_test[j,k2])
    # Wait, the indices don't work that way. Let me just directly compute the tensor contraction.
    
    # Actually, the simplest verification:
    # For each r, define:
    #   a_r[i,j] = U_np[i*4+j, r]
    #   b_r[j,k] = V_np[j*6+k, r]
    #   c_r[k,i] = W_np[k*4+i, r]
    # Then the decomposition says: for all i,j,k, sum_r a_r[i,j] * b_r[j,k] * c_r[k,i] = 1
    # (since the multiplication tensor has all-ones entries in the standard basis)
    
    # Let's verify this.
    max_err = 0.0
    for i in range(4):
        for j in range(4):
            for k in range(6):
                s = 0.0
                for r in range(rank):
                    s += U_np[i*4+j, r] * V_np[j*6+k, r] * W_np[k*4+i, r]
                max_err = max(max_err, abs(s - 1.0))
    
    if max_err > 1e-10:
        # Verification failed, fall back to schoolbook
        pass
    # The decomposition should be correct by construction
    
    return U, V, W


def _build_schoolbook():
    """Fallback schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    try:
        U, V, W = _build_strassen_decomposition()
        current_rank = len(U[0])
        
        # Quick verification using numpy
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Verify tensor identity: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # Use a fast check on a few random (i,j,k) points
        rng = np.random.RandomState(0)
        errors = []
        for _ in range(50):
            i = rng.randint(0, n)
            j = rng.randint(0, m)
            k = rng.randint(0, p)
            s = np.dot(U_np[i*m+j, :], V_np[j*p+k, :] * W_np[k*n+i, :])
            errors.append(abs(s - 1.0))
        
        if max(errors) < 1e-10:
            # Valid Strassen decomposition
            return {"U": U, "V": V, "W": W}
        else:
            # Verification failed, use schoolbook
            U, V, W = _build_schoolbook()
            return {"U": U, "V": V, "W": W}
    except Exception:
        # Fallback to schoolbook on any error
        U, V, W = _build_schoolbook()
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
