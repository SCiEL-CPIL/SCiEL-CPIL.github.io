"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    start_time = time.time()
    deadline = 145.0

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    # ---- Build base decomposition: 4 Strassen blocks (28) + 6 compressed direct terms = 34 ----
    base_rank = 34
    U = [[0.0] * base_rank for _ in range(n * m)]
    V = [[0.0] * base_rank for _ in range(m * p)]
    W = [[0.0] * base_rank for _ in range(p * n)]
    col = 0

    def strassen_block(y, z, w):
        nonlocal col
        y00, y01, y10, y11 = y
        z00, z01, z10, z11 = z
        w00, w01, w10, w11 = w
        # M1 = (y00+y11)(z00+z11)
        U[u_idx(*y00)][col] += 1; U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1; V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] += 1; W[w_idx(*w11)][col] += 1
        col += 1
        # M2 = (y10+y11)z00
        U[u_idx(*y10)][col] += 1; U[u_idx(*y11)][col] += 1
        V[v_idx(*z00)][col] += 1
        W[w_idx(*w10)][col] += 1; W[w_idx(*w11)][col] += -1
        col += 1
        # M3 = y00(z01-z11)
        U[u_idx(*y00)][col] += 1
        V[v_idx(*z01)][col] += 1; V[v_idx(*z11)][col] += -1
        W[w_idx(*w01)][col] += 1; W[w_idx(*w11)][col] += 1
        col += 1
        # M4 = y11(z10-z00)
        U[u_idx(*y11)][col] += 1
        V[v_idx(*z10)][col] += 1; V[v_idx(*z00)][col] += -1
        W[w_idx(*w00)][col] += 1; W[w_idx(*w10)][col] += 1
        col += 1
        # M5 = (y00+y01)z11
        U[u_idx(*y00)][col] += 1; U[u_idx(*y01)][col] += 1
        V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] += -1; W[w_idx(*w01)][col] += 1
        col += 1
        # M6 = (y10-y00)(z00+z01)
        U[u_idx(*y10)][col] += 1; U[u_idx(*y00)][col] += -1
        V[v_idx(*z00)][col] += 1; V[v_idx(*z01)][col] += 1
        W[w_idx(*w11)][col] += 1
        col += 1
        # M7 = (y01-y11)(z10+z11)
        U[u_idx(*y01)][col] += 1; U[u_idx(*y11)][col] += -1
        V[v_idx(*z10)][col] += 1; V[v_idx(*z11)][col] += 1
        W[w_idx(*w00)][col] += 1
        col += 1

    AL = [(0,0),(0,1),(1,0),(1,1)]
    AR = [(0,2),(0,3),(1,2),(1,3)]
    B11 = [(0,0),(0,1),(1,0),(1,1)]
    B12 = [(0,2),(0,3),(1,2),(1,3)]
    B21 = [(2,0),(2,1),(3,0),(3,1)]
    B22 = [(2,2),(2,3),(3,2),(3,3)]
    C11 = [(0,0),(1,0),(0,1),(1,1)]
    C12 = [(2,0),(3,0),(2,1),(3,1)]

    strassen_block(AL, B11, C11)
    strassen_block(AR, B21, C11)
    strassen_block(AL, B12, C12)
    strassen_block(AR, B22, C12)

    # Compressed direct terms for k=4 using Strassen on 2x2x1 blocks
    # Left half: A[:,0:2] * B[0:2,4] -> C[:,0:2] (2x2x1)
    # Use 3-term decomposition:
    # C00 = A00*B00, C01 = A01*B01, C10 = A10*B00, C11 = A11*B01
    # With Strassen-like: M1=(A00+A11)*B00, M2=(A00+A11)*B01, M3=(A00+A11)*(B00+B01)
    # C00 = M1 - M3 + A11*B01... this doesn't work cleanly for 1-column
    # Better: just use 3 terms directly:
    # T1 = (A00+A11)*B00 -> contributes to C00 and C10
    # T2 = (A00+A11)*B01 -> contributes to C01 and C11
    # T3 = (A00+A11)*(B00+B01) -> subtract to get correct values
    # Actually for 2x2x1, rank is 3 (it's a 2x2 matrix times a vector)
    # C = A @ b where A is 2x2, b is 2x1
    # C0 = A00*b0 + A01*b1
    # C1 = A10*b0 + A11*b1
    # This needs 4 scalar multiplications in general, but tensor rank of 2x2x1 is 3
    # Using: M1 = (A00+A11)*b0, M2 = (A00+A11)*b1, M3 = (A00+A11)*(b0+b1)
    # C0 = M1 - M3 + A11*b1... no
    # Let me just use the standard 3-term:
    # M1 = A00*b0, M2 = A01*b1, M3 = (A10+A11)*(b0+b1)
    # C0 = M1 + M2, C1 = M3 - M1 - M2 + A10*b0 + A11*b1... not clean
    # 
    # For 2x2x1, the tensor is: T[i,j,0] = A[i,j]*b[j]
    # This is rank 2 actually! C0 = A00*b0 + A01*b1, C1 = A10*b0 + A11*b1
    # We can write: C0 = (A00*b0 + A01*b1), C1 = (A10*b0 + A11*b1)
    # As a tensor: sum over k of U[i,k]*V[j,k]*W[0,k]
    # With k=0: U[i,0]*V[j,0]*W[0,0] -> this gives A[i,j]*b[j] only if we set it up right
    # Actually 2x2x1 has rank 2: use two terms
    # Term 1: covers column j=0: U[i,0]=A[i,0], V[0,0]=b[0], V[1,0]=0, W[0,0]=1 -> contributes A[i,0]*b[0] to C[i]
    # Term 2: covers column j=1: U[i,1]=A[i,1], V[0,1]=0, V[1,1]=b[1], W[0,1]=1 -> contributes A[i,1]*b[1] to C[i]
    # So rank 2 for each 2x2x1 block!
    # Left half: 2 terms, Right half: 2 terms, Total: 4 terms for k=4
    # But wait, we need to verify this is correct for the full tensor
    
    # Let me just use 4 terms for the left 2x2x1 and 4 for right 2x2x1 = 8 terms (same as before)
    # Or try 2+2=4 terms total
    
    # Actually, let me think again. The tensor for multiplying 2x4 by 4x1 is 2x4x1.
    # T[i,j,0] = A[i,j]*B[j,0]
    # This can be decomposed as: sum_k U[i,k]*V[j,k]*W[0,k]
    # We need: for all i,j: sum_k U[i,k]*V[j,k] = A[i,j]*B[j,0]
    # This is a matrix factorization: A*B_col = U * V^T where V is 4xK, W is 1xK
    # The rank of A*B_col as a matrix is at most min(2,4)=2, so K=2 suffices!
    # U is 2x2, V is 4x2, W is 1x2
    # A*B_col = [[A00*B00 + A01*B10 + A02*B20 + A03*B30],
    #             [A10*B00 + A11*B10 + A12*B20 + A13*B30]]
    # = U * V^T where U is 2x2, V^T is 2x4
    # Let U = [[1, 0], [0, 1]], then V^T = B_col^T * I... no
    # V^T[i,j] should satisfy sum_k U[i,k]*V^T[k,j] = A[i,j]*B[j,0]
    # If U = I, then V^T[i,j] = A[i,j]*B[j,0], but V depends on both A and B, which is not allowed
    # In tensor decomposition, U depends on A indices, V on B indices, W on C indices
    # U[i,k] is coefficient for A[i,j] part, V[j,k] for B[j,k] part
    # So: sum_k U[i,k]*V[j,k] = A[i,j]*B[j,0] for all i,j
    # This means the matrix M[i,j] = A[i,j]*B[j,0] must have rank <= K
    # M = A * diag(B_col) which has rank at most min(rank(A), rank(diag(B_col)))
    # If B_col is full (all nonzero), rank(diag(B_col)) = 4, rank(A) <= 2, so rank(M) <= 2
    # So K=2 suffices!
    # U[i,k] and V[j,k] are fixed coefficients (don't depend on A,B values)
    # We need: for all i,j, sum_k U[i,k]*V[j,k] = A[i,j]*B[j,0]
    # But A[i,j] and B[j,0] are variables! This can't work with fixed U,V unless...
    # Oh wait, in the tensor decomposition, U, V, W are FIXED matrices that define the algorithm
    # The decomposition is: C[i,0] = sum_j,k U[i,k]*V[j,k]*W[0,k] * A[i,j]*B[j,0]... no
    # The tensor T[i,j,k] = delta[i,j,k] where T[i,j,k]=1 if the multiplication uses A[i,j]*B[j,k]
    # For 2x4x1: T[i,j,0] = 1 for all i,j (since C[i,0] = sum_j A[i,j]*B[j,0])
    # We need: T[i,j,0] = sum_r U[i,r]*V[j,r]*W[0,r]
    # So: 1 = sum_r U[i,r]*V[j,r]*W[0,r] for all i in {0,1}, j in {0,1,2,3}
    # This is: U * (V * W^T) = ones(2x4) where V*W^T is 4x1... 
    # Let w = W[0,:] (vector of length R), then V*w is a 4-vector, and U*(V*w) = ones(2x4)
    # U is 2xR, V is 4xR, w is R-vector
    # V*w is a 4-vector v, U*v = ones(2x4) means each row of U dotted with v gives all 1s
    # Row i of U dotted with v = 1 for all j... but the result should be a 2x4 matrix of 1s
    # U*v gives a 2-vector, not 2x4. I'm confusing myself.
    # 
    # Let me restart. The tensor for matrix multiplication C = A*B where A is n x m, B is m x p:
    # T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
    # Decomposition: T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # For n=2, m=4, p=5: T is 2x4x5, all ones
    # We need: for all i,j,k: 1 = sum_r U[i,r]*V[j,r]*W[k,r]
    # This is a 3-tensor of all ones with rank R
    # 
    # For the k=4 slice: T[i,j,4] = 1 = sum_r U[i,r]*V[j,r]*W[4,r]
    # Let w4 = W[:,4] (the 4th column of W, shape R)
    # Then: 1 = sum_r U[i,r]*V[j,r]*w4[r] for all i,j
    # This means: U * diag(w4) * V^T = ones(2x4)
    # The rank of ones(2x4) is 1, so we can do this with R=1 if w4 is nonzero!
    # U[i,0]*w4[0]*V[j,0] = 1 for all i,j
    # Set U[i,0]=1, V[j,0]=1, w4[0]=1 -> works!
    # But this is just one term. The issue is that the same U,V,W columns are shared across all k
    # 
    # OK so the full problem is: find U(2xR), V(4xR), W(5xR) such that
    # sum_r U[i,r]*V[j,r]*W[k,r] = 1 for all i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4}
    # This is the all-ones 2x4x5 tensor. Its rank is what we're looking for.
    # 
    # The block-Strassen approach gives 36. Can we do better?
    # The rank of the all-ones n x m x p tensor is min(n*m, m*p, p*n) in general? No.
    # For all-ones tensor, the rank is at most min(n*m, m*p, p*n) but can be less.
    # Actually, the all-ones tensor has rank 1! Because 1*1*1 = 1.
    # Wait, that's trivially true but doesn't help because U,V,W must be fixed and the
    # decomposition must work for ALL input matrices, not just the all-ones case.
    # 
    # I'm confusing the tensor itself with the decomposition. The tensor T[i,j,k]=1 represents
    # the multiplication map. The decomposition T = sum_r U_r ⊗ V_r ⊗ W_r means
    # C[i,k] = sum_j,r U[i,r]*V[j,r]*W[k,r]*A[i,j]*B[j,k]
    # For this to equal sum_j A[i,j]*B[j,k], we need:
    # sum_r U[i,r]*V[j,r]*W[k,r] = 1 for all i,j,k
    # So yes, we need the all-ones tensor decomposition.
    # 
    # The rank of the all-ones 2x4x5 tensor: since it's separable as 1⊗1⊗1, rank 1!
    # But that can't be right because then any matrix multiplication would have rank 1.
    # 
    # OH WAIT. I'm making a fundamental error. The tensor for matrix multiplication is NOT
    # all ones. Let me re-read the task.
    # 
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 5 matrix"
    # The multiplication tensor has T[i,j,k] = 1 if and only if the term A[i,j]*B[j,k] appears in C[i,k]
    # For standard matrix multiplication, C[i,k] = sum_j A[i,j]*B[j,k], so T[i,j,k] = 1 for ALL i,j,k
    # where i is row of A (0..1), j is column of A / row of B (0..3), k is column of B (0..4)
    # So T is indeed all ones of shape 2x4x5.
    # 
    # And the rank of the all-ones tensor of any shape is 1, since it's rank-1 (outer product of all-ones vectors).
    # But that would mean we can do any matrix multiplication with rank 1, which is absurd.
    # 
    # I think the issue is that U, V, W must have specific structures related to the input/output.
    # Let me re-read: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # U has shape (n*m, R) = (8, R)
    # V has shape (m*p, R) = (20, R)  
    # W has shape (p*n, R) = (10, R)
    # 
    # And the decomposition is: for the multiplication tensor T[i,j,k] (i in 0..n-1, j in 0..m-1, k in 0..p-1)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # 
    # So T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For standard matmul, T[i,j,k] = 1 for all valid i,j,k
    # 
    # So we need: for all i in 0..1, j in 0..3, k in 0..4:
    # 1 = sum_r U[i*4+j, r] * V[j*5+k, r] * W[k*2+i, r]
    # 
    # This is NOT the same as the all-ones tensor in a simple sense because the indices are coupled.
    # U[i*4+j, r] depends on both i and j, V[j*5+k, r] depends on both j and k, W[k*2+i, r] depends on both k and i.
    # 
    # So it's a more complex constraint. The block-Strassen approach exploits this structure.
    # 
    # Given the complexity, let me just implement the 34-rank base (using 2 terms per 2x2x1 half)
    # and then try greedy elimination to find if we can go lower.

    # For k=4, left half (j=0,1): T[i,j,4] = 1 for i in {0,1}, j in {0,1}
    # Need: 1 = sum_r U[i*4+j, r] * V[j*5+4, r] * W[4*2+i, r]
    # Let's use 2 terms: r1, r2
    # U[0*4+0, r1]=1, U[1*4+0, r1]=1, V[0*5+4, r1]=1, V[1*5+4, r1]=0, W[8+0, r1]=1, W[8+1, r1]=1
    # U[0*4+1, r2]=1, U[1*4+1, r2]=1, V[0*5+4, r2]=0, V[1*5+4, r2]=1, W[8+0, r2]=1, W[8+1, r2]=1
    # Check: for i=0,j=0: U[0,r1]*V[4,r1]*W[8,r1] + U[0,r2]*V[4,r2]*W[8,r2] = 1*1*1 + 0*0*1 = 1 ✓
    # for i=0,j=1: U[1,r1]*V[9,r1]*W[8,r1] + U[1,r2]*V[9,r2]*W[8,r2] = 1*0*1 + 1*1*1 = 1 ✓
    # for i=1,j=0: U[4,r1]*V[4,r1]*W[9,r1] + U[4,r2]*V[4,r2]*W[9,r2] = 1*1*1 + 0*0*1 = 1 ✓
    # for i=1,j=1: U[5,r1]*V[9,r1]*W[9,r1] + U[5,r2]*V[9,r2]*W[9,r2] = 1*0*1 + 1*1*1 = 1 ✓
    # 
    # So 2 terms suffice for the left half! Similarly 2 for the right half. Total 4 for k=4.
    # Base rank: 28 + 4 = 32

    # Left half k=4 (j=0,1)
    U[u_idx(0,0)][col] += 1; U[u_idx(1,0)][col] += 1
    V[v_idx(0,4)][col] += 1
    W[w_idx(4,0)][col] += 1; W[w_idx(4,1)][col] += 1
    col += 1

    U[u_idx(0,1)][col] += 1; U[u_idx(1,1)][col] += 1
    V[v_idx(1,4)][col] += 1
    W[w_idx(4,0)][col] += 1; W[w_idx(4,1)][col] += 1
    col += 1

    # Right half k=4 (j=2,3)
    U[u_idx(0,2)][col] += 1; U[u_idx(1,2)][col] += 1
    V[v_idx(2,4)][col] += 1
    W[w_idx(4,0)][col] += 1; W[w_idx(4,1)][col] += 1
    col += 1

    U[u_idx(0,3)][col] += 1; U[u_idx(1,3)][col] += 1
    V[v_idx(3,4)][col] += 1
    W[w_idx(4,0)][col] += 1; W[w_idx(4,1)][col] += 1
    col += 1

    assert col == 32, col

    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)

    def col_tensor(Uc, Vc, Wc):
        uc_r = Uc.reshape(n, m)
        vc_r = Vc.reshape(m, p)
        wc_r = Wc.reshape(p, n)
        return np.einsum('ij,jk,ki->ijk', uc_r, vc_r, wc_r, optimize=True)

    def verify_fast(Ua2, Va2, Wa2, R):
        acc = np.zeros((n, m, p), dtype=np.float64)
        for c in range(R):
            acc += col_tensor(Ua2[:, c], Va2[:, c], Wa2[:, c])
        return np.allclose(acc, 1.0, atol=1e-6, rtol=0.0)

    if not verify_fast(Ua, Va, Wa, 32):
        # Fallback to schoolbook rank 40
        rank = 40
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[u_idx(i,j)][i*m+j*p+k] = 1
                    V[v_idx(j,k)][i*m+j*p+k] = 1
                    W[w_idx(k,i)][i*m+j*p+k] = 1
        return {"U": U, "V": V, "W": W}

    best_rank = 32
    best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()
    cur_U, cur_V, cur_W = Ua.copy(), Va.copy(), Wa.copy()
    cur_R = 32
    active_cols = list(range(32))

    def get_tensors(Uc, Vc, Wc, cols):
        return [col_tensor(Uc[:, c], Vc[:, c], Wc[:, c]) for c in cols]

    cur_tensors = get_tensors(cur_U, cur_V, cur_W, active_cols)

    found_improvement = True
    while found_improvement and time.time() < deadline - 10:
        found_improvement = False
        R = len(active_cols)
        if R < 2:
            break

        for i_c in range(R):
            if time.time() > deadline - 10:
                break
            c_orig = active_cols[i_c]
            T_c = cur_tensors[i_c]

            if np.allclose(T_c, 0, atol=1e-9):
                active_cols.pop(i_c)
                cur_tensors.pop(i_c)
                cur_R -= 1
                found_improvement = True
                continue

            eliminated = False
            for i_a in range(R):
                if i_a == i_c:
                    continue
                for i_b in range(i_a + 1, R):
                    if i_b == i_c:
                        continue

                    T_a = cur_tensors[i_a]
                    T_b = cur_tensors[i_b]
                    v_a = T_a.flatten()
                    v_b = T_b.flatten()
                    v_c = T_c.flatten()

                    M = np.column_stack((v_a, v_b))
                    try:
                        coeffs, residuals, rank_M, s = np.linalg.lstsq(M, v_c, rcond=None)
                        alpha, beta = coeffs
                        if len(residuals) > 0 and residuals[0] < 1e-9:
                            a_orig = active_cols[i_a]
                            b_orig = active_cols[i_b]

                            cur_U[:, a_orig] += alpha * cur_U[:, c_orig]
                            cur_V[:, a_orig] += alpha * cur_V[:, c_orig]
                            cur_W[:, a_orig] += alpha * cur_W[:, c_orig]

                            cur_U[:, b_orig] += beta * cur_U[:, c_orig]
                            cur_V[:, b_orig] += beta * cur_V[:, c_orig]
                            cur_W[:, b_orig] += beta * cur_W[:, c_orig]

                            active_cols.pop(i_c)
                            cur_tensors.pop(i_c)
                            cur_R -= 1
                            cur_tensors = get_tensors(cur_U, cur_V, cur_W, active_cols)

                            if verify_fast(cur_U, cur_V, cur_W, cur_R):
                                if cur_R < best_rank:
                                    best_rank = cur_R
                                    best_U = cur_U.copy()
                                    best_V = cur_V.copy()
                                    best_W = cur_W.copy()
                                found_improvement = True
                                eliminated = True
                                break
                    except np.linalg.LinAlgError:
                        pass
                if eliminated:
                    break
            if eliminated:
                break

    if not verify_fast(best_U, best_V, best_W, best_rank):
        best_U, best_V, best_W = Ua, Va, Wa
        best_rank = 32

    def to_lists(A):
        result = []
        for row in A:
            new_row = []
            for x in row:
                rx = round(x)
                if abs(x - rx) < 1e-9:
                    new_row.append(int(rx))
                else:
                    rh = round(2 * x) / 2.0
                    if abs(x - rh) < 1e-9:
                        new_row.append(rh)
                    else:
                        new_row.append(float(x))
            result.append(new_row)
        return result

    return {
        "U": to_lists(best_U),
        "V": to_lists(best_V),
        "W": to_lists(best_W),
    }
# EVOLVE-BLOCK-END
