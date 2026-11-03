import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _round_to_half_integer(x):
    """Round x to the nearest half-integer."""
    return round(x * 2) / 2.0

def _round_matrix_to_half_int(M):
    """Round all entries of matrix M to nearest half-integer."""
    return np.array([[ _round_to_half_integer(x) for x in row] for row in M])

def _strassen_2x2x2_terms():
    """Return the 7 rank-1 terms for 2x2x2 Strassen multiplication.
    Each term is (u, v, w) where u is length 4, v is length 4, w is length 4.
    Indexing: u[i*2+j], v[j*2+k], w[k*2+i]
    """
    terms = []
    # M1 = (a+d)(e+h) -> contributes to c00, c11
    u = [1, 0, 1, 0]  # a, d
    v = [1, 0, 1, 0]  # e, h
    w = [1, 0, 0, 1]  # c00, c11
    terms.append((u, v, w))
    # M2 = (c+d)e -> contributes to c10, c11
    u = [0, 0, 1, 1]  # c, d
    v = [1, 0, 0, 0]  # e
    w = [0, 1, 0, -1] # c10, -c11
    terms.append((u, v, w))
    # M3 = a(f-h) -> contributes to c01
    u = [1, 0, 0, 0]  # a
    v = [0, 1, 0, -1] # f, -h
    w = [0, 0, 1, 1]  # c01
    terms.append((u, v, w))
    # M4 = d(g-e) -> contributes to c10
    u = [0, 0, 0, 1]  # d
    v = [0, -1, 1, 0] # -e, g
    w = [0, 1, 1, 0]  # c10
    terms.append((u, v, w))
    # M5 = (a+b)h -> contributes to c00, c01
    u = [1, 1, 0, 0]  # a, b
    v = [0, 0, 0, 1]  # h
    w = [-1, 0, 1, 0] # -c00, c01
    terms.append((u, v, w))
    # M6 = (c-a)(e+f) -> contributes to c11
    u = [-1, 0, 1, 0] # -a, c
    v = [1, 1, 0, 0]  # e, f
    w = [0, 0, 0, 1]  # c11
    terms.append((u, v, w))
    # M7 = (b-d)(g+h) -> contributes to c10
    u = [0, 1, 0, -1] # b, -d
    v = [0, 0, 1, 1]  # g, h
    w = [0, 1, 0, 0]  # c10
    terms.append((u, v, w))
    return terms

def _kron_strassen_construct(n, m, p):
    """Construct a rank-65 decomposition using Kronecker-embedded Strassen.
    
    For 4x4 * 4x5:
    - Split A into 2x2 blocks of 2x2: A = [[A00, A01], [A10, A11]]
    - Split B into 2x3 blocks: B = [[B00, B01, B02], [B10, B11, B12]]
    - C = A*B has 2x3 blocks of 2x2
    
    For each 2x2 block C_ij = sum over valid k of A_ik * B_kj
    Use Strassen for the 2x2x2 multiplications where possible.
    
    The key: use Kronecker product structure. If we have a rank-r decomposition
    for 2x2x2, we can embed it into larger blocks.
    
    Actually, let's think more carefully:
    C is 4x5. We can view it as 2x2 blocks in rows, 2x3 blocks in cols.
    C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    C[0:2, 2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]
    C[0:2, 4:5] = A[0:2,0:2]*B[0:2,4:5] + A[0:2,2:4]*B[2:4,4:5]
    C[2:4, 0:2] = A[2:4,0:2]*B[0:2,0:2] + A[2:4,2:4]*B[2:4,0:2]
    C[2:4, 2:4] = A[2:4,0:2]*B[0:2,2:4] + A[2:4,2:4]*B[2:4,2:4]
    C[2:4, 4:5] = A[2:4,0:2]*B[0:2,4:5] + A[2:4,2:4]*B[2:4,4:5]
    
    For 2x2 * 2x2: use Strassen (rank 7)
    For 2x2 * 2x1: use naive (rank 4) or a 2x2x2 subproblem
    
    Total: 4 blocks of 2x2x2 (rank 7 each = 28) + 4 blocks of 2x2x1 (rank 4 each = 16)
    Wait, that's not right either. Let me recount.
    
    C has 6 blocks of size 2x2 (for cols 0-3) and 2 blocks of size 2x1 (for col 4).
    Actually C is 4x5, so blocks are:
    Row blocks: 2 (rows 0-1, rows 2-3)
    Col blocks: 3 (cols 0-1, cols 2-3, col 4)
    So 2*3 = 6 blocks. First 4 are 2x2, last 2 are 2x1.
    
    Each 2x2 block C_ij = A_i0 * B_0j + A_i1 * B_1j
    Each 2x1 block C_i2 = A_i0 * B_02 + A_i1 * B_12
    
    For 2x2 blocks: 4 of them, each needs rank-7 Strassen for two 2x2x2 products
    But we can't just add two Strassen decompositions because the intermediate
    products are different.
    
    Alternative approach: Use the fact that 4x4x5 can be decomposed by
    embedding 2x2x2 Strassen into a Kronecker structure.
    
    Let me use a simpler approach: schoolbook for the last column, Strassen for
    the 4x4 part.
    
    4x4 * 4x4: Use 2x2 block Strassen. Each 2x2 block product uses rank 7.
    There are 4 block positions in C, each is sum of 2 block products.
    But we need to be careful about sharing.
    
    Actually, the standard way: 4x4x4 has known rank 49 (using 2x2x2 Strassen
    embedded via Kronecker: 7^2 = 49).
    
    For 4x4x5: 4x4x4 (rank 49) + 4x4x1 (rank 16) = rank 65.
    """
    n, m, p = 4, 4, 5
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(np.array(u, dtype=float))
        V_cols.append(np.array(v, dtype=float))
        W_cols.append(np.array(w, dtype=float))
    
    # Part 1: 4x4 * 4x4 (first 4 columns of B)
    # Use Kronecker-embedded Strassen: rank 49
    # Split A into 2x2 blocks of 2x2, B into 2x2 blocks of 2x2
    # C_block = A_block * B_block uses Strassen
    # There are 4 output blocks, each is sum of 2 block products
    
    # For 4x4x4 via 2x2 block Strassen:
    # C00 = A00*B00 + A01*B10  (two 2x2x2 products)
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Each 2x2x2 product uses 7 terms. Total: 8 products * 7 = 56? No...
    # Actually, the Kronecker embedding gives rank 7*7 = 49 for the whole 4x4x4.
    
    # Let me construct it properly using the tensor product of Strassen terms.
    # If S = (s1, s2, ..., s7) is a rank-7 decomposition of 2x2x2,
    # then the 4x4x4 decomposition has rank 7*7 = 49.
    # The (a,b) term (a=0..6, b=0..6) has:
    #   U[i*4+j, ab] = U2[a, i*2+ia] * U2[b, j*2+jb]  ... this is getting complex.
    
    # Let me just do it explicitly.
    # 2x2x2 Strassen terms (u, v, w) with indices:
    # u[i*2+j], v[j*2+k], w[k*2+i]
    
    strassen = _strassen_2x2x2_terms()
    
    # For 4x4x4, decompose as Kronecker product of two 2x2x2 problems.
    # A is 4x4, viewed as 2x2 blocks each 2x2.
    # B is 4x4, viewed as 2x2 blocks each 2x2.
    # C is 4x4, viewed as 2x2 blocks each 2x2.
    # C[bi,bj] = sum over di,dj of A[bi,di] * B[di,bj]
    # Each A[bi,di] is 2x2, B[di,bj] is 2x2, product is 2x2.
    # Use Strassen for each 2x2x2 sub-multiplication.
    
    # But we can share the Strassen intermediates!
    # The key insight: C = sum_{di,dj} A[bi,di] * B[di,bj]
    # For fixed bi,bj: C[bi,bj] = A[bi,0]*B[0,bj] + A[bi,1]*B[1,bj]
    # Each term A[bi,d]*B[d,bj] is a 2x2x2 multiplication.
    
    # So we have 4*2 = 8 separate 2x2x2 multiplications, each with rank 7.
    # Total rank: 8*7 = 56. That's worse than 49.
    
    # The Kronecker product approach: treat the 4x4x4 tensor as a tensor
    # product of two 2x2x2 tensors. The rank of a tensor product is at most
    # the product of ranks. So rank <= 7*7 = 49.
    
    # Let me construct it:
    # The 4x4x4 tensor T[i,j,k] = delta_{i,k} (for matrix mult: C[i,k] = sum_j A[i,j]*B[j,k])
    # Wait, I'm confusing myself. Let me restart the construction.
    
    # Matrix mult 4x4x4: T[i,j,k] where C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 if we're computing the (i,k) entry from (i,j) and (j,k).
    # The tensor is T[i,j,k] = delta_{i,?} * delta_{?,k}... no.
    # T[i,j,k] is the coefficient of A[i,j]*B[j,k] in C[i,k]. It's 1 for all valid i,j,k.
    # So T[i,j,k] = 1 for all i,j,k in 0..3.
    
    # For 2x2x2: T2[i,j,k] = 1 for all i,j,k in 0..1.
    # For 4x4x4: T4[i,j,k] = 1 for all i,j,k in 0..3.
    # T4 = T2 ⊗ T2 (Kronecker product of tensors).
    # Rank(T4) <= Rank(T2)^2 = 49.
    
    # Construction: If T2 = sum_{r=1}^{7} u_r ⊗ v_r ⊗ w_r,
    # then T4 = sum_{r,s=1}^{7} (u_r ⊗ u_s) ⊗ (v_r ⊗ v_s) ⊗ (w_r ⊗ w_s)
    # where ⊗ for vectors is Kronecker product.
    
    # So for 4x4x4, we have 49 terms.
    # Term (r,s): 
    #   U4[i*4+j, r*7+s] = U2[i*2+ia, r] * U2[j*2+jb, s]  for ia,jb in 0..1
    #   Wait, that's not right either.
    
    # Let me think again. T4[i,j,k] = sum_{r,s} U4[i,j,r*7+s] * V4[j,k,r*7+s] * W4[k,i,r*7+s]
    # We want T4[i,j,k] = 1 for all i,j,k.
    # T4[i,j,k] = T2[i2,j2,k2] * T2[i1,j1,k1] where i = i2*2+i1, etc.
    # And T2[i2,j2,k2] = sum_r U2[i2,j2,r]*V2[j2,k2,r]*W2[k2,i2,r]
    # So T4[i,j,k] = sum_{r,s} U2[i2,j2,r]*V2[j2,k2,r]*W2[k2,i2,r] * U2[i1,j1,s]*V2[j1,k1,s]*W2[k1,i1,s]
    # = sum_{r,s} [U2[i2,j2,r]*U2[i1,j1,s]] * [V2[j2,k2,r]*V2[j1,k1,s]] * [W2[k2,i2,r]*W2[k1,i1,s]]
    
    # So U4[i,j,r*7+s] = U2[i2,j2,r] * U2[i1,j1,s]
    #    where i = i2*2+i1, j = j2*2+j1
    # V4[j,k,r*7+s] = V2[j2,k2,r] * V2[j1,k1,s]
    #    where j = j2*2+j1, k = k2*2+k1
    # W4[k,i,r*7+s] = W2[k2,i2,r] * W2[k1,i1,s]
    #    where k = k2*2+k1, i = i2*2+i1
    
    # Let me implement this:
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for r in range(7):
        for s in range(7):
            idx = r * 7 + s
            for i2 in range(2):
                for j2 in range(2):
                    for i1 in range(2):
                        for j1 in range(2):
                            i = i2 * 2 + i1
                            j = j2 * 2 + j1
                            U4[i*4+j, idx] = strassen[r][0][i2*2+j2] * strassen[s][0][i1*2+j1]
            
            for j2 in range(2):
                for k2 in range(2):
                    for j1 in range(2):
                        for k1 in range(2):
                            j = j2 * 2 + j1
                            k = k2 * 2 + k1
                            V4[j*4+k, idx] = strassen[r][1][j2*2+k2] * strassen[s][1][j1*2+k1]
            
            for k2 in range(2):
                for i2 in range(2):
                    for k1 in range(2):
                        for i1 in range(2):
                            k = k2 * 2 + k1
                            i = i2 * 2 + i1
                            W4[k*4+i, idx] = strassen[r][2][k2*2+i2] * strassen[s][2][k1*2+i1]
    
    # Add these 49 terms for the 4x4x4 part (first 4 columns)
    for r in range(49):
        u = np.zeros(16)
        v = np.zeros(16)
        w = np.zeros(20)  # p*n = 5*4 = 20
        u[:] = U4[:, r]
        v[:] = V4[:, r]
        # W only has 4x4 = 16 entries for the first 4 columns
        w[:16] = W4[:, r]
        add_term(u, v, w)
    
    # Part 2: 4x4 * 4x1 (last column of B)
    # C[:,4] = A * B[:,4]
    # This is just matrix-vector multiplication, rank = 16 (schoolbook)
    for i in range(4):
        for j in range(4):
            u = np.zeros(16)
            v = np.zeros(16)
            w = np.zeros(20)
            u[i*4+j] = 1
            v[j*4+4] = 1  # j*4 + k where k=4
            w[4*4+i] = 1  # k*n + i where k=4
            add_term(u, v, w)
    
    U_mat = np.array(U_cols).T
    V_mat = np.array(V_cols).T
    W_mat = np.array(W_cols).T
    return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()

def _prune_redundant_terms(U, V, W, n, m, p, tol=1e-9):
    """Try to remove linearly dependent rank-1 terms.
    
    A term r is redundant if there exist coefficients c_s (s != r) such that
    for all (i,j,k): sum_s c_s * U[i,j,s]*V[j,k,s]*W[k,i,s] = U[i,j,r]*V[j,k,r]*W[k,i,r]
    
    This is a nonlinear condition, but we can try a simpler approach:
    Check if the vectorized tensor contribution of term r is in the span
    of the contributions of other terms.
    """
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Vectorize each term's contribution to the tensor
    # Term r contributes: U[:,r] * V[:,r] * W[:,r] element-wise after proper indexing
    # Actually, the tensor entry T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # So term r's contribution to T[i,j,k] is U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    # Let's build the tensor contribution matrix: rows = (i,j,k) triples, cols = terms
    n_entries = n * m * p
    contrib = np.zeros((n_entries, rank))
    entry_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(rank):
                    contrib[entry_idx, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                entry_idx += 1
    
    # Now check if any column is in the span of the others
    kept = list(range(rank))
    changed = True
    while changed:
        changed = False
        for r in list(kept):
            if len(kept) <= 1:
                break
            others = [s for s in kept if s != r]
            if len(others) == 0:
                continue
            # Check if contrib[:,r] is in the span of contrib[:,others]
            A_others = contrib[:, others]
            b_r = contrib[:, r]
            try:
                coeffs, residuals, rank_sub, sv = np.linalg.lstsq(A_others, b_r, rcond=tol)
                residual = np.linalg.norm(A_others @ coeffs - b_r)
                norm_b = np.linalg.norm(b_r)
                if norm_b < 1e-12:
                    # Term r contributes nothing, remove it
                    kept.remove(r)
                    changed = True
                    continue
                if residual / norm_b < tol:
                    # Term r is in the span of others, remove it
                    kept.remove(r)
                    changed = True
            except:
                pass
    
    if len(kept) == rank:
        return U.tolist(), V.tolist(), W.tolist()
    
    U_pruned = U[:, kept]
    V_pruned = V[:, kept]
    W_pruned = W[:, kept]
    return U_pruned.tolist(), V_pruned.tolist(), W_pruned.tolist()

def _als_decompose(n, m, p, rank, seed=42, max_iters=200):
    """Try to find a rank-r decomposition using alternating least squares."""
    np.random.seed(seed)
    
    U = np.random.randn(n*m, rank) * 0.1
    V = np.random.randn(m*p, rank) * 0.1
    W = np.random.randn(p*n, rank) * 0.1
    
    for it in range(max_iters):
        for i in range(n):
            for j in range(m):
                A_mat = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                b_vec = np.ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        for j in range(m):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                b_vec = np.ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        for i in range(n):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                b_vec = np.ones(m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
    
    U_r = _round_matrix_to_half_int(U)
    V_r = _round_matrix_to_half_int(V)
    W_r = _round_matrix_to_half_int(W)
    
    if _verify_decomposition(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Get the Kronecker-Strassen decomposition (rank 65)
    U_base, V_base, W_base = _kron_strassen_construct(n, m, p)
    best_result = {"U": U_base, "V": V_base, "W": W_base}
    best_rank = len(U_base[0])
    
    # Try to prune redundant terms
    U_pruned, V_pruned, W_pruned = _prune_redundant_terms(
        np.array(U_base, dtype=float), np.array(V_base, dtype=float), 
        np.array(W_base, dtype=float), n, m, p
    )
    pruned_rank = len(U_pruned[0]) if U_pruned else best_rank
    if pruned_rank < best_rank:
        best_result = {"U": U_pruned, "V": V_pruned, "W": W_pruned}
        best_rank = pruned_rank
    
    # Try ALS to find lower-rank decompositions
    for rank in range(best_rank - 1, max(50, best_rank - 15), -1):
        if time.time() > deadline - 20:
            break
        
        found = False
        for seed in range(3):
            if time.time() > deadline - 20:
                break
            
            result = _als_decompose(n, m, p, rank, seed=seed, max_iters=150)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                found = True
                break
        
        if found:
            continue
    
    return best_result
# EVOLVE-BLOCK-END
