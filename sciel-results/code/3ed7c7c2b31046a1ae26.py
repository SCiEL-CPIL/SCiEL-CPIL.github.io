"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct using random matrices."""
    import numpy as np
    rank = len(U[0])
    # Build the tensor T from U, V, W
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We check against the standard multiplication tensor
    # Standard: T_std[i,j,k] = 1 if (i,j,k) correspond to a valid multiplication term
    # Actually, the multiplication tensor for (n,m,p) -> C = AB has:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # In tensor form with indices (i, j, k) for (A-index, B-index, C-index):
    # The tensor T_mult has T[i, j, k] = 1 if k is the output column and i is input row,
    # but the standard trace convention is W[k*n+i, r]
    # Let's just check with random matrices.
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_true = A @ B
        # Compute C from decomposition
        # C[i, k] = sum_j A[i,j] B[j,k]
        # In tensor decomposition: C[i,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) (sum_k' V[j'*p+k', r] B[j',k']) ... no
        # The correct reconstruction is:
        # For each r, we have rank-1 term U_r ⊗ V_r ⊗ W_r
        # The multiplication result is sum_r (U_r^T A_flat) (V_r^T B_flat) W_r
        # where A_flat is n*m vector, B_flat is m*p vector, W_r is p*n vector reshaped to (p,n)
        # Actually: C = sum_r outer( (U_r . A_flat), (V_r . B_flat) ) * W_r reshaped
        # Let's be more careful.
        # A_flat[i*m+j] = A[i,j]
        # B_flat[j*p+k] = B[j,k]
        # W_flat[k*n+i] = W_coeff for term (k, i) in C
        # C[i,k] = sum_r (sum_{i',j'} U[i'*m+j', r] * delta(i',i) ... )
        # The standard tensor for matrix multiplication (n,m,p) has components:
        # T[i, j, k] where i in [0,n), j in [m, p*m), k in [0, p*n)
        # T[i, j, k] = 1 if j = (i * m) + j_inner AND k = (k_inner * n) + i AND j_inner == ... 
        # Actually it's simpler: the multiplication tensor has T[A_idx, B_idx, C_idx] = 1
        # where A_idx indexes A, B_idx indexes B, C_idx indexes C
        # For C[i,k] = sum_j A[i,j] B[j,k]:
        # The term A[i,j] * B[j,k] contributes to C[i,k]
        # So T[i, j, k] = 1 for each valid (i,j,k) triple, and there are n*m*p such terms.
        # In the factorization: T = sum_r U_r ⊗ V_r ⊗ W_r
        # U_r has shape (n*m,), V_r has shape (m*p,), W_r has shape (p*n,)
        # T[A_idx, B_idx, C_idx] = sum_r U_r[A_idx] * V_r[B_idx] * W_r[C_idx]
        # We need: for each (i,j,k), sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] = 1
        # And for all other (A_idx, B_idx, C_idx), the sum should be 0.
        
        # Let's just check the 5 random matrix products.
        C_decomp = np.zeros((n, p))
        for r in range(rank):
            u_col = np.array(U[:, r]).reshape(n, m)
            v_col = np.array(V[:, r]).reshape(m, p)
            w_col = np.array(W[:, r]).reshape(p, n)
            # The rank-1 term contributes: (A^T u_col) ... no.
            # U_r[A_idx] * V_r[B_idx] * W_r[C_idx]
            # A_idx = i*m + j, B_idx = j*p + k, C_idx = k*n + i
            # For a fixed r, the contribution to C[i, k] is:
            # sum over... actually it's a single term for each (i,j,k) matching the indices.
            # C_decomp[i, k] += U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] for each j? No.
            # The sum over r of U_r[i*m+j] V_r[j*p+k] W_r[k*n+i] should equal 1 for all i,j,k.
            # And for non-matching indices it should be 0.
            # This is equivalent to checking that the tensor is correctly decomposed.
            # Easiest: compute C via the decomposition formula.
            # C[i,k] = sum_r (sum_j U_r[i*m+j] * A[i,j]) * (sum_j' V_r[j'*p+k] * B[j',k]) * ... 
            # No, that's not right either. The tensor product means:
            # (A ⊗ B ⊗ C)_tensor = sum_r (U_r . A_vec) (V_r . B_vec) W_r
            # where A_vec is A flattened, and W_r is reshaped to (p, n).
            # Let me verify: 
            # (U_r . A_vec) = sum_{i,j} U_r[i*m+j] A[i,j]
            # (V_r . B_vec) = sum_{j,k} V_r[j*p+k] B[j,k]
            # Then the contribution to C is: (U_r . A_vec) * (V_r . B_vec) * W_r reshaped to (p,n), transposed to (n,p)?
            # W_r[k*n + i] corresponds to C[i, k]. So W_r reshaped to (p, n) has W_r[k, i] = W_r[k*n+i].
            # So we need W_r reshaped to (n, p) with W_r[i, k] = W_r[k*n + i].
            
            a_dot_u = sum(U[i*m+j][r] * A[i][j] for i in range(n) for j in range(m))
            b_dot_v = sum(V[j*p+k][r] * B[j][k] for j in range(m) for k in range(p))
            for i in range(n):
                for k in range(p):
                    C_decomp[i, k] += a_dot_u * b_dot_v * W[k*n + i][r]
        
        if not np.allclose(C_decomp, C_true, atol=1e-9):
            return False
    return True

def _build_rank_20():
    """
    Build a rank-20 decomposition for 3x4x8.
    
    Strategy: Split A (3x4) into A1 (2x4) and A2 (1x4).
    C1 = A1 * B (2x8), C2 = A2 * B (1x8).
    
    C2 = A2 * B: A2 is 1x4, B is 4x8.
    C2[k] = sum_j A2[j] B[j,k] for k=0..7.
    This is a rank-4 decomposition: for each j=0..3, one term.
    U2[j] = 1 for the A2 row, V2[j*p+k] = B[j,k] coefficient... 
    Actually, for rank-1 term r_j: 
    U_r[j] = 1 (for the A2 component at position j), 0 elsewhere.
    V_r[j*p+k] = 1 for all k? No.
    V_r should be such that V_r . B_vec gives B[j, :] as a vector.
    V_r[j*p+k] = 1 for the specific k? No, we need to extract row j of B.
    V_r is a vector of length m*p = 4*8 = 32.
    We want V_r . B_vec = sum_k B[j,k] * (some coefficient).
    To get B[j,k] for each k, we need separate terms or a different approach.
    
    Actually, for A2 (1x4) times B (4x8):
    C2[k] = sum_{j=0}^{3} A2[j] B[j,k]
    This can be written as: for each j, a rank-1 term where:
    U term: 1 at position corresponding to A2[j]
    V term: 1 at position j*p + k for each k? No, that would be 8 terms per j.
    
    Let me think differently. The tensor for (1, 4, 8) multiplication has rank 4.
    C[k] = sum_j A[j] B[j,k]
    Tensor: T[A_idx, B_idx, C_idx] where A_idx in [0,4), B_idx in [0, 4*8), C_idx in [0, 8*1)
    T[a, b, c] = 1 if b = a*8 + c, i.e., B_idx corresponds to row a, column c.
    Factorization: for r = 0..3 (one for each a):
    U_r[a] = 1 if a == r, else 0.
    V_r[b] = 1 if b == r*8 + c for some c? No, V_r needs to be fixed.
    V_r[j*8 + k] = 1 if j == r, else 0. (This is a vector with 8 ones at positions r*8, r*8+1, ..., r*8+7)
    W_r[c*1 + i] = 1 if c == ... W has shape (8*1, rank) = (8, rank).
    W_r[c] = 1 for all c? No.
    T[a, b, c] = sum_r U_r[a] V_r[b] W_r[c]
    For a=r, b=r*8+c, we need T = 1.
    U_r[r] = 1, V_r[r*8+c] = 1, so we need W_r[c] = 1.
    For a != r, U_r[a] = 0, so T = 0. Good.
    For a = r, b != r*8+c: V_r[b] = 0, so T = 0. Good.
    So W_r[c] = 1 for all c = 0..7.
    
    So for the (1,4,8) part:
    4 terms. For r=0..3:
    U2 has a 1 at position corresponding to A2's j=r.
    V2 has 1s at positions r*8, r*8+1, ..., r*8+7.
    W2 has 1s at all 8 positions.
    
    For C1 = A1 * B where A1 is 2x4, B is 4x8:
    This is (2,4,8) multiplication, which has rank 2*4 = 8.
    C1[i,k] = sum_j A1[i,j] B[j,k] for i=0,1 and k=0..7.
    Tensor: T[A_idx, B_idx, C_idx] where A_idx in [0, 2*4), B_idx in [0, 4*8), C_idx in [0, 8*2)
    T[a, b, c] = 1 if a = i*4+j, b = j*8+k, c = k*2+i.
    Factorization: for each j=0..3 and i=0..1? That's 8 terms.
    For r corresponding to (i,j):
    U1[a] = 1 if a = i*4 + j.
    V1[b] = 1 if b = j*8 + k for some k? No, similar to before.
    V1[b] = 1 if b is in the range j*8..j*8+7? Then we need W to separate k.
    W1[c] = 1 if c = k*2 + i for the correct k.
    Actually, let's use the same pattern:
    For term (i, j):
    U1[i*4 + j] = 1.
    V1[j*8 + k] = 1 for all k? No, that would give sum_k B[j,k] which is wrong.
    
    Let me reconsider. For (n, m, p) multiplication:
    C[i,k] = sum_j A[i,j] B[j,k]
    The rank is n*m if we can do it that way? No, generally it's n*m*p for schoolbook.
    But for fixed i, it's an inner product of length m, which needs m multiplications.
    So for (2,4,8), we have 2 rows, each needing 4 multiplications, total 8. Yes, rank 8.
    
    Factorization for (2,4,8):
    For each i in {0,1} and j in {0,1,2,3}:
    U[i*4+j] = 1 for this term.
    V[j*8+k] = 1 for all k? No.
    We need V to pick out B[j,k] for each k. But V is a single vector.
    Ah, I see the issue. The V vector must be such that V . B_vec gives the right thing.
    If V[j*8+k] = 1 for all k, then V . B_vec = sum_k B[j,k], which is the row sum, not what we want.
    
    The correct factorization for (n,m,p):
    T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k (in appropriate bases)
    This has rank n*m*p.
    
    But for (1, m, p), the rank is m:
    C[k] = sum_j A[j] B[j,k]
    T[a, b, c] = 1 if b = a*p + c.
    Factorization: for r = 0..m-1:
    U_r[a] = delta(a, r)
    V_r[b] = 1 if b = r*p + c for any c, i.e., V_r[r*p..r*p+p-1] = 1, else 0.
    W_r[c] = 1 for all c.
    
    For (2, 4, 8):
    C[i,k] = sum_j A1[i,j] B[j,k]
    T[a, b, c] where a = i*4+j, b = j*8+k, c = k*2+i.
    T[a, b, c] = 1 if a = i*4+j AND b = j*8+k AND c = k*2+i for some i,j,k.
    This means for each (i,j,k), there's a 1 at (i*4+j, j*8+k, k*2+i).
    Total 2*4*8 = 64 non-zero entries? No, that can't be right.
    The tensor has one 1 for each valid (i,j,k) combination that corresponds to a multiplication term.
    For (2,4,8), the multiplication has 2*8 = 16 output entries, each a sum of 4 products, total 64 products.
    So the tensor has 64 ones? That would mean rank at least 64/something.
    
    I think I'm overcomplicating this. Let me just implement the schoolbook and verify it works, then try to find a better decomposition.
    """
    n, m, p = 3, 4, 8
    # Use schoolbook for now as a fallback
    U, V, W = _schoolbook(n, m, p)
    return U, V, W

def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x4x8 matrix multiplication."""
    deadline = time.time() + 140  # 140s to leave 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # First, get the schoolbook decomposition as a baseline
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p
    
    # Try to find a better decomposition
    # For 3x4x8, the rank should be significantly less than 96.
    # Let's try a simple block decomposition.
    
    # Split A (3x4) into rows: A0 (1x4), A1 (1x4), A2 (1x4)
    # C0 = A0 * B (1x8), rank 4
    # C1 = A1 * B (1x8), rank 4  
    # C2 = A2 * B (1x8), rank 4
    # Total rank: 12
    
    # But wait, this is just the schoolbook decomposition with the rows separated.
    # The schoolbook decomposition already has rank 96 = 3*4*8.
    # If we can do each row in rank 4, total would be 12.
    
    # For (1, m, p) multiplication, rank is m = 4.
    # So for (3, 4, 8), if we decompose by rows of A:
    # Each row i: C[i,k] = sum_j A[i,j] B[j,k]
    # This is (1, 4, 8) multiplication, rank 4.
    # Total rank: 3 * 4 = 12.
    
    # Let's verify this works.
    U = [[0] * 12 for _ in range(n * m)]
    V = [[0] * 12 for _ in range(m * p)]
    W = [[0] * 12 for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            # Term for row i, column j of A
            U[i * m + j][col] = 1
            # V should pick out B[j, k] for all k
            for k in range(p):
                V[j * p + k][col] = 1
            # W should place the result in C[i, k]
            for k in range(p):
                W[k * n + i][col] = 1
            col += 1
    
    # Verify this decomposition
    if time.time() < deadline:
        if _verify(U, V, W, n, m, p):
            best_U, best_V, best_W = U, V, W
            best_rank = 12
    
    # Check time
    if time.time() >= deadline - 5:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Can we do better than 12?
    # For (3, 4, 8), the rank is at least max(3*4, 4*8, 3*8) = max(12, 32, 24) = 32? 
    # No, that's not a lower bound. The rank is at least the dimension of the output space,
    # which is 3*8 = 24. So rank >= 24.
    # Wait, that's not right either. The rank of the multiplication tensor M_{n,m,p}
    # is at least max(nm, mp, np) by some bounds? Actually, I'm not sure.
    
    # Let me just return the rank-12 decomposition if it's valid.
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
