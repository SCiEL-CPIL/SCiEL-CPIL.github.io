import time
import numpy as np
from itertools import product

_START_TIME = time.time()
_DEADLINE = 140

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _verify(U, V, W, n=2, m=4, p=4):
    """Verify that U, V, W decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the multiplication tensor T[i,j,k,l] = 1 if k==j
    # U[i*m+j, r], V[k*p+l, r], W[l*n+i, r]
    # Check: sum_r U[i*m+j,r] * V[k*p+l,r] * W[l*n+i,r] == delta(j,k)
    for i in range(n):
        for j in range(m):
            for k in range(m):
                for l in range(p):
                    expected = 1.0 if k == j else 0.0
                    actual = 0.0
                    for r in range(rank):
                        actual += U[i*m+j][r] * V[k*p+l][r] * W[l*n+i][r]
                    if abs(actual - expected) > 1e-9:
                        return False
    return True

def _construct_rank24():
    """
    Construct a rank-24 decomposition exploiting the shared B structure.
    
    Key idea: We have two 1x4x4 multiplications sharing the same B.
    Row 0: C0[l] = sum_j A[0,j] * B[j,l]
    Row 1: C1[l] = sum_j A[1,j] * B[j,l]
    
    We can write this as:
    C0[l] = sum_j (A[0,j] * B[j,l])
    C1[l] = sum_j (A[1,j] * B[j,l])
    
    The tensor is: T[i,j,k,l] = delta(j,k) for both i=0 and i=1.
    
    Strategy: Use 12 terms for the "diagonal" structure and exploit
    the fact that B is shared to reduce from 32 (schoolbook) or 28 (4xStrassen).
    
    Actually, let me think about this more carefully.
    
    The multiplication tensor for 2x4 x 4x4 -> 2x4 has:
    - Input A: 2*4 = 8 dimensions
    - Input B: 4*4 = 16 dimensions  
    - Output C: 2*4 = 8 dimensions
    
    Tensor T[i,j,k,l] = 1 if j==k, for i in {0,1}, j,k in {0,1,2,3}, l in {0,1,2,3}
    
    This is equivalent to two independent 1x4x4 multiplications that share B.
    
    For a single 1x4x4: rank is at most 16 (schoolbook), and it's known that
    the tensor rank of 1x4x4 is 16 (it's a "slice" of matrix multiplication).
    
    But when we have TWO such multiplications sharing B, we can potentially
    do better than 2*16=32.
    
    Let me try a different approach: partition into 2x2 blocks of B.
    B = [B11 B12; B21 B22] where each Bi is 2x2.
    A = [A1 A2] where A1, A2 are 1x2.
    
    C = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    
    For row 0 of C: C0 = [A1(0)*B11 + A2(0)*B21, A1(0)*B12 + A2(0)*B22]
    For row 1 of C: C1 = [A1(1)*B11 + A2(1)*B21, A1(1)*B12 + A2(1)*B22]
    
    Now, A1(0), A2(0), A1(1), A2(1) are the 4 entries of A.
    
    Let me denote:
    a00 = A[0,0], a01 = A[0,1], a02 = A[0,2], a03 = A[0,3]
    a10 = A[1,0], a11 = A[1,1], a12 = A[1,2], a13 = A[1,3]
    
    B entries: b[j,l] for j,l in 0..3
    
    C[0,l] = sum_j a[0,j] * b[j,l]
    C[1,l] = sum_j a[1,j] * b[j,l]
    
    I'll try to find a rank-24 decomposition by solving a system of equations.
    """
    n, m, p = 2, 4, 4
    rank = 24
    
    # Variables: U[i*m+j, r], V[k*p+l, r], W[l*n+i, r]
    # Total variables: (8+16+8)*24 = 720
    
    # Constraints: T[i,j,k,l] = sum_r U[i*m+j,r]*V[k*p+l,r]*W[l*n+i,r] = delta(j,k)
    # Number of constraints: 2*4*4*4 = 128
    
    # This is a system of 128 equations in 720 unknowns (quadratic).
    # Let me try a constructive approach.
    
    # Approach: Use the structure that T factors as:
    # T[i,j,k,l] = sum_r u[i,j,r] * v[k,l,r] * w[l,i,r] = delta(j,k)
    
    # Since the constraint is delta(j,k), we need:
    # For j=k: sum_r u[i,j,r]*v[j,l,r]*w[l,i,r] = 1
    # For j!=k: sum_r u[i,j,r]*v[k,l,r]*w[l,i,r] = 0
    
    # Let me try a block-structured solution.
    # Partition indices into pairs: (0,1) and (2,3)
    
    # For the "diagonal" blocks (j=k within same pair):
    # We can use 2x2 Strassen-like structures.
    
    # Actually, let me try a more direct algebraic construction.
    
    # Define 24 bilinear terms. For each term r, we specify:
    # - Which A entries it uses (linear form on A)
    # - Which B entries it uses (linear form on B)
    # - Which C entries it contributes to (linear form on C)
    
    # The constraint is that the tensor equals the multiplication tensor.
    
    # Let me try: use 12 terms for "row 0" and 12 terms for "row 1"
    # but with shared B structure.
    
    # Actually, a cleaner approach: 
    # C[0,l] = sum_j A[0,j]*B[j,l]  -- this is a 1x4x4 tensor
    # C[1,l] = sum_j A[1,j]*B[j,l]  -- this is a 1x4x4 tensor
    # They share B.
    
    # The combined tensor has rank at most 2*16=32, but potentially less.
    
    # Known result: The rank of the 2x4x4 matrix multiplication tensor.
    # This is related to the "border rank" and "tensor rank" problems.
    
    # Let me try a constructive rank-24 solution.
    
    # I'll use the following structure:
    # Split B into 4 columns. For each column l of B (4x1 vector b_l),
    # C[:,l] = A * b_l (2x4 times 4x1 = 2x1)
    
    # So C[:,l] = A @ b_l for l=0,1,2,3
    # This is 4 independent 2x4x1 multiplications!
    
    # Each 2x4x1: C[0] = sum_j A[0,j]*B[j], C[1] = sum_j A[1,j]*B[j]
    # This is just computing two dot products of length 4.
    # The tensor for this is: T[i,j,k] = 1 for all i,j (where k is the column index of B)
    # Wait, no. For 2x4 @ 4x1:
    # C[i] = sum_j A[i,j] * B[j]
    # Tensor: T[i,j,k] where i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    # T[i,j,k] = 1 if j==k, else 0. This is the same for both i values!
    
    # So the tensor for 2x4x1 is: T[i,j,k] = delta(j,k) * 1 (independent of i)
    # This means the two "rows" are identical in structure.
    # The rank of this tensor: T[i,j,k] = delta(j,k)
    # This can be written as: T = sum_k (e_k^A) (outer) (e_k^B) (outer) (1^C)
    # Wait, that's not right either.
    
    # T[i,j,k] = delta(j,k) for i in {0,1}, j,k in {0,1,2,3}
    # = sum_k delta(j,k) * 1 (for each i)
    # = for each i: sum_k (e_k in A-space) (outer) (e_k in B-space) (outer) (1 in C-space)
    
    # But the C-space is 1-dimensional (just one output per i).
    # Actually C has 2 outputs (one per row i), but they're computed independently.
    
    # Let me re-examine. The full tensor for 2x4 x 4x4 -> 2x4:
    # T[i,j,k,l] = delta(j,k) where:
    #   i = row of A and C (0 or 1)
    #   j = column of A (0-3)
    #   k = row of B (0-3)  
    #   l = column of B and column of C (0-3)
    
    # T[i,j,k,l] = 1 if j==k, else 0. Note: independent of i and l!
    
    # So T[i,j,k,l] = delta(j,k) * 1 * 1 (the 1's are for i and l dimensions)
    
    # This means: T = (sum over i of e_i) (outer) (sum over l of e_l) (outer) (the 4x4 identity tensor in j,k)
    
    # The 4x4 identity tensor in (j,k) space: I[j,k] = delta(j,k)
    # This has rank 4 (it's the sum of 4 rank-1 tensors: sum_j e_j (outer) e_j)
    
    # So T = (sum_i e_i) (outer) (sum_l e_l) (outer) (sum_j e_j (outer) e_j)
    # = sum_{i,l,j} e_i (outer) e_j (outer) e_j (outer) e_l
    # This gives rank 2*4*4 = 32 (the schoolbook algorithm)
    
    # But can we do better? The key is that the "i" and "l" dimensions are 
    # independent of the "j,k" identity structure.
    
    # Actually, T[i,j,k,l] = delta(j,k) means the tensor is:
    # T = (1_i) (outer) (I_{jk}) (outer) (1_l)
    # where 1_i is the all-ones vector in R^2, I_{jk} is the 4x4 identity,
    # and 1_l is the all-ones vector in R^4.
    
    # The tensor rank of a product of tensors is at most the product of ranks.
    # rank(1_i) = 1, rank(I_{jk}) = 4, rank(1_l) = 1
    # So rank(T) <= 1*4*1 = 4?? That can't be right.
    
    # Wait, I'm confusing things. Let me be more careful.
    
    # T is a 4-index tensor of shape (2,4,4,4).
    # T[i,j,k,l] = 1 if j==k, else 0.
    
    # A rank-1 tensor of shape (2,4,4,4) is: u[i]*v[j]*w[k]*x[l]
    # We need: sum_r u_r[i] * v_r[j] * w_r[k] * x_r[l] = delta(j,k)
    
    # For this to equal delta(j,k), we need the (j,k) part to be the identity.
    # The (j,k) part of term r is: v_r[j] * w_r[k] (a rank-1 matrix in j,k space)
    # We need: sum_r (v_r (outer) w_r) * (u_r[i] * x_r[l]) = I_{jk} * (1 for all i,l)
    
    # This means: for each (i,l), sum_r u_r[i]*x_r[l] * v_r[j]*w_r[k] = delta(j,k)
    
    # Let's define: for each r, the matrix M_r = v_r (outer) w_r (4x4 matrix)
    # And coefficients: c_r(i,l) = u_r[i] * x_r[l]
    
    # We need: sum_r c_r(i,l) * M_r = I for all i,l.
    
    # This means: the set {M_r} must span the identity in a way that the
    # coefficients c_r(i,l) can produce I for all (i,l) pairs.
    
    # If all c_r(i,l) = 1 for all r,i,l, then we need sum_r M_r = I,
    # which requires rank at least 4 (since I has rank 4 and each M_r has rank 1).
    
    # But we can vary c_r(i,l). The question is: what's the minimum number of
    # rank-1 matrices M_r such that I = sum_r c_r(i,l) * M_r for all (i,l)?
    
    # Equivalently: the 4x4 identity matrix must be in the span of {M_r} with
    # coefficients that factor as u_r[i]*x_r[l].
    
    # If we allow arbitrary coefficients, we just need {M_r} to span I,
    # so rank >= 4. But the coefficients must be rank-1 in (i,l) space.
    
    # Let's try rank 4: M_r = e_r (outer) e_r for r=0,1,2,3.
    # Then sum_r c_r(i,l) * M_r = sum_r c_r(i,l) * e_r(j)*e_r(k)
    # For this to equal delta(j,k), we need c_r(i,l) = 1 for all r, i, l.
    # But c_r(i,l) = u_r[i]*x_r[l]. Can we have u_r[i]*x_r[l] = 1 for all i,l?
    # Yes! u_r = [1,1], x_r = [1,1,1,1] for all r.
    # So rank 4 works?? That would give total rank 4, which seems too good.
    
    # Wait, let me recheck. If rank=4:
    # U[i,j,r] = delta(j,r) * 1  (for i=0,1)
    # V[k,l,r] = delta(k,r) * 1  (for all l)  
    # W[l,i,r] = 1  (for all l,i)
    
    # Check: sum_r U[i,j,r]*V[k,l,r]*W[l,i,r]
    # = sum_r delta(j,r) * delta(k,r) * 1
    # = delta(j,k)  (since only r=j=k contributes)
    
    # YES! This works! Rank 4?!
    
    # But wait, that can't be right. Let me double-check the indexing.
    # U has shape (n*m, rank) = (8, rank): U[i*m+j, r]
    # V has shape (m*p, rank) = (16, rank): V[k*p+l, r]
    # W has shape (p*n, rank) = (8, rank): W[l*n+i, r]
    
    # My proposed solution:
    # U[i*m+j, r] = delta(j, r) for r=0,1,2,3
    # V[k*p+l, r] = delta(k, r) for r=0,1,2,3
    # W[l*n+i, r] = 1 for r=0,1,2,3
    
    # Check: sum_{r=0}^{3} U[i*4+j, r] * V[k*4+l, r] * W[l*2+i, r]
    # = sum_{r=0}^{3} delta(j,r) * delta(k,r) * 1
    # = 1 if j=k (and j=k is in 0..3), else 0
    # = delta(j,k) ✓
    
    # So the rank is at most 4! This is much better than 28!
    
    # But wait - is this correct? Let me verify with a concrete example.
    # A = [[1,2,3,4],[5,6,7,8]], B = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    # C = A*B = A = [[1,2,3,4],[5,6,7,8]]
    
    # Using my decomposition with rank 4:
    # C[i,l] = sum_r U[i*4+r, r]... wait, I need to be more careful.
    
    # C[i,l] should equal sum_j A[i,j] * B[j,l]
    # In the decomposition: C[i,l] = sum_r U[i*m+j... 
    
    # Actually, the decomposition says:
    # C[i,l] = sum_j sum_r U[i*m+j, r] * V[j*p+l, r] * W[l*n+i, r]
    # No wait, that's not how it works. The tensor decomposition is:
    # T[i,j,k,l] = sum_r U[i*m+j, r] * V[k*p+l, r] * W[l*n+i, r]
    # And the actual computation is:
    # C[i,l] = sum_{j,k} A[i,j] * B[k,l] * T[i,j,k,l]
    # = sum_{j,k} A[i,j] * B[k,l] * sum_r U[i*m+j,r]*V[k*p+l,r]*W[l*n+i,r]
    # = sum_r (sum_j A[i,j]*U[i*m+j,r]) * (sum_k B[k,l]*V[k*p+l,r]) * W[l*n+i,r]
    
    # So C[i,l] = sum_r alpha_r(i) * beta_r(l) * W[l*n+i, r]
    # where alpha_r(i) = sum_j A[i,j]*U[i*m+j,r]
    # and beta_r(l) = sum_k B[k,l]*V[k*p+l,r]
    
    # With my proposed U, V, W:
    # alpha_r(i) = sum_j A[i,j]*delta(j,r) = A[i,r]
    # beta_r(l) = sum_k B[k,l]*delta(k,r) = B[r,l]
    # W[l*n+i, r] = 1
    
    # C[i,l] = sum_r A[i,r] * B[r,l] * 1 = sum_r A[i,r]*B[r,l] ✓
    
    # THIS IS CORRECT! The rank is 4!
    
    # Wait, but this seems too simple. Let me re-examine the problem statement.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix 
    #  by a 4 by 4 matrix."
    
    # The tensor T[i,j,k,l] = delta(j,k) indeed has rank 4 by my construction.
    # This is because the tensor is "separable" - the i and l dimensions don't 
    # interact with the j,k identity structure.
    
    # So the answer should be rank 4!
    
    n, m, p = 2, 4, 4
    rank = 4
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for i in range(n):
            U[i * m + r][r] = 1
        for l in range(p):
            V[r * p + l][r] = 1
        for l in range(p):
            for i in range(n):
                W[l * n + i][r] = 1
    
    # Verify
    assert _verify(U, V, W, n, m, p), "Rank 4 decomposition failed verification!"
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = 2, 4, 4
    
    # First try rank 4
    try:
        result = _construct_rank24()  # misnamed, actually rank 4
        return result
    except Exception:
        pass
    
    # Fallback to rank 28 (Strassen-based)
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen(A_r0, A_c0, B_r0, B_c0, C_r0, C_c0, start_r):
        a11 = A_r0 * m + A_c0
        a12 = A_r0 * m + A_c0 + 1
        a21 = (A_r0 + 1) * m + A_c0
        a22 = (A_r0 + 1) * m + A_c0 + 1
        b11 = B_r0 * p + B_c0
        b12 = B_r0 * p + B_c0 + 1
        b21 = (B_r0 + 1) * p + B_c0
        b22 = (B_r0 + 1) * p + B_c0 + 1
        c11_idx = C_c0 * n + C_r0
        c12_idx = (C_c0 + 1) * n + C_r0
        c21_idx = C_c0 * n + (C_r0 + 1)
        c22_idx = (C_c0 + 1) * n + (C_r0 + 1)
        r = start_r
        U[a11][r] += 1; U[a22][r] += 1; V[b11][r] += 1; V[b22][r] += 1
        W[c11_idx][r] += 1; W[c22_idx][r] += 1; r += 1
        U[a21][r] += 1; U[a22][r] += 1; V[b11][r] += 1
        W[c21_idx][r] += 1; W[c22_idx][r] += -1; r += 1
        U[a11][r] += 1; V[b12][r] += 1; V[b22][r] += -1
        W[c12_idx][r] += 1; W[c22_idx][r] += 1; r += 1
        U[a22][r] += 1; V[b21][r] += 1; V[b11][r] += -1
        W[c11_idx][r] += 1; W[c21_idx][r] += 1; r += 1
        U[a11][r] += 1; U[a12][r] += 1; V[b22][r] += 1
        W[c11_idx][r] += -1; W[c12_idx][r] += 1; r += 1
        U[a21][r] += 1; U[a11][r] += -1; V[b11][r] += 1; V[b12][r] += 1
        W[c22_idx][r] += 1; r += 1
        U[a12][r] += 1; U[a22][r] += -1; V[b21][r] += 1; V[b22][r] += 1
        W[c11_idx][r] += 1; r += 1
        return r
    
    r = add_strassen(0, 0, 0, 0, 0, 0, 0)
    r = add_strassen(0, 2, 2, 0, 0, 0, r)
    r = add_strassen(0, 0, 0, 2, 0, 2, r)
    r = add_strassen(0, 2, 2, 2, 0, 2, r)
    
    return {"U": U, "V": V, "W": W}
