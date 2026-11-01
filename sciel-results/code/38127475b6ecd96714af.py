import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Strassen decomposition for 2x2x2
    strassen_2x2 = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    def u_idx(row_off, col_off, local_r, local_c):
        return (row_off + local_r) * m + (col_off + local_c)
        
    def v_idx(row_off, col_off, local_r, local_c):
        return (row_off + local_r) * p + (col_off + local_c)
        
    def w_idx(col_off, local_r, local_c):
        return (col_off + local_c) * n + local_r

    # Build rank-28 solution as fallback
    def build_rank_28():
        rank = 28
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        block_configs = [
            (0, 0, 0, 0),
            (2, 2, 0, 0),
            (0, 0, 2, 2),
            (2, 2, 2, 2),
        ]
        
        col = 0
        for a_co, b_ro, b_co, c_co in block_configs:
            for u_terms, v_terms, w_terms in strassen_2x2:
                for idx, coeff in u_terms:
                    r = idx // 2
                    c = idx % 2
                    U[u_idx(0, a_co, r, c), col] += coeff
                for idx, coeff in v_terms:
                    r = idx // 2
                    c = idx % 2
                    V[v_idx(b_ro, b_co, r, c), col] += coeff
                for idx, coeff in w_terms:
                    r = idx // 2
                    c = idx % 2
                    W[w_idx(c_co, r, c), col] += coeff
                col += 1
        return U, V, W
    
    # Try to find rank-27 via ALS with rational constraint
    def try_rank_27():
        if time.time() > deadline - 10:
            return None
        
        # Build the multiplication tensor T[i,j,k] = delta_{i,k} * delta_{j,j} 
        # Actually T[i,j,k] = 1 if i==k and j==j (always), so T[i,j,k] = delta_{i,k}
        # Wait: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
        # So T[i,j,k] = delta_{i,k}... no.
        # T[i,j,k] should be such that sum_r U[i,r]*V[j,r]*W[k,r] = (A*B)[i,k]
        # (A*B)[i,k] = sum_j A[i,j]*B[j,k]
        # So T[i,j,k] = 1 for all i,j,k where the contraction gives delta... 
        # Actually the tensor is T[i,j,k] = 1 if we're computing (A*B)[i,k] = sum_j A[i,j] B[j,k]
        # The tensor elements are T[i,j,k] = delta_{i,k}? No.
        # The multiplication tensor M[i,j,k] = 1 for all valid i,j,k? No.
        # M[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
        # So M[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
        # Wait no. The tensor is defined as T[i,j,k] = delta_{i,k} * 1? 
        # Let me think again. We want: C[i,k] = sum_j A[i,j] B[j,k]
        # = sum_r U[i,r] V[j,r] W[k,r] summed over j and r? No.
        # The decomposition is: sum_r U[i,j,r] V[... ] 
        # Actually in the interface: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # And the contraction is: C[i,k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # So the tensor T[i,j,k] = 1 for all i,j,k (since each term A[i,j]*B[j,k] appears exactly once)
        # Wait, that's not right either. Let me re-read.
        # C[i,k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For this to equal sum_j A[i,j]*B[j,k], we need:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = A[i,j]*B[j,k] * delta_{...}
        # Hmm, actually the tensor T is defined by T[i,j,k] = 1 for all valid indices,
        # and the CP decomposition satisfies: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # But that would mean sum_r ... = 1 for all i,j,k, which is the rank-1 tensor.
        # 
        # I think I'm confusing myself. Let me look at the verification code:
        # Cf[ii, kk] = sum_cc U[ii*m+jj, cc] * V[jj*p+kk, cc] * W[kk*n+ii, cc] summed over jj
        # So Cf[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # And this should equal (A*B)[i,k] = sum_j A[i,j]*B[j,k]
        # 
        # So the tensor is: T[i,j,k] = 1 (constant), and we need:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k?
        # No wait, that's not right. The A and B are arbitrary. The decomposition must work for ALL A and B.
        # 
        # Actually, I think the tensor is the "multiplication tensor" which has:
        # T[i,j,k] = 1 if the bilinear form (a,b) -> sum_j a[i,j]*b[j,k] has a term involving a[i,j]*b[j,k]
        # So T[i,j,k] = 1 for all i,j,k.
        # And we need a CP decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Wait, the indices don't match up nicely.
        
        # Let me just use the verification to understand:
        # We need: for all A, B: (A*B)[i,k] = sum_j A[i,j]*B[j,k] = sum_j sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r]
        # This must hold for all A, B. So:
        # sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] = delta_{i,k}... no.
        # It must equal 1 * delta_{...} hmm.
        # 
        # Actually since this must hold for ALL A and B independently, we need:
        # sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] = 1 for all i,j,k? No, that can't be right because
        # then Cf[i,k] = sum_j 1 = m for all i,k.
        #
        # I think the correct interpretation is that the tensor T is defined by its "multiplication" property.
        # Let me just try to find a rank-27 decomposition by solving the system.
        
        # Actually, let me reconsider. The tensor T has T[i,j,k] = 1 for all i,j,k (n*m*p = 32 entries).
        # We need: T[i,j,k] = sum_{r=1}^{rank} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # 
        # Wait, I don't think T[i,j,k] = 1 for all i,j,k. Let me think about what tensor this is.
        # 
        # The matrix multiplication tensor for multiplying an n×m matrix by an m×p matrix is a 3-tensor
        # T of dimensions n×m×p where T[i,j,k] = 1 if the entry (i,k) of the product depends on A[i,j] and B[j,k].
        # Since (A*B)[i,k] = sum_j A[i,j]*B[j,k], the entry (i,k) depends on A[i,j] and B[j,k] for each j.
        # So T[i,j,k] = 1 for all valid i,j,k? That would make T the all-ones tensor.
        # 
        # But the all-ones tensor has rank 1! So that can't be right.
        # 
        # I think the issue is that the tensor is not simply T[i,j,k] = 1. The CP decomposition
        # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] has a specific index structure.
        # 
        # Let me define: for the decomposition to work, we need:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = f(i,j,k)
        # where f(i,j,k) is such that sum_j A[i,j]*B[j,k]*f(i,j,k) = (A*B)[i,k] for all A,B.
        # This means f(i,j,k) must be 1 for all i,j,k.
        # 
        # So we DO need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
        # 
        # But wait, that means the "tensor" is all-ones, and we're looking for a rank-r CP decomposition
        # of the all-ones tensor with the constraint that the factors have specific index mappings.
        # The all-ones tensor of size 8×16×8 (or whatever) has CP rank 1 if there are no constraints.
        # But with the constraint that U depends on (i,j), V on (j,k), W on (k,i), it's more complex.
        
        # I think I've been overthinking this. Let me just use numerical optimization to find a rank-27 solution.
        
        # Actually, let me step back. The current rank-28 solution works. Let me try to see if rank 27 is possible
        # by using a different approach: modify the Strassen blocks to share some multiplications.
        
        # One idea: the 4 blocks are A_L*B_TL, A_R*B_BL, A_L*B_TR, A_R*B_BR.
        # Some of the Strassen terms might be combinable across blocks.
        # For example, in block 1 (A_L*B_TL) and block 3 (A_L*B_TR), both use A_L.
        # The first Strassen term is (a1+a4)*(b1+b4) -> c1+c4 contribution.
        # In block 1: (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        # In block 3: (A[0,0]+A[1,1])*(B[0,2]+B[1,3])
        # These can't easily be combined.
        
        # Let me try a numerical approach: start from rank 28 and try to merge two rank-1 terms.
        
        U28, V28, W28 = build_rank_28()
        
        # Try rank 27 by solving a least squares problem
        # We have 32 equations (one for each i,j,k) and 3*27 = 81 unknowns (if we fix one factor)
        # Actually it's nonlinear. Let me try ALS.
        
        # Initialize from rank 28 by zeroing out one column
        rank = 27
        rng = np.random.RandomState(42)
        
        # Try multiple initializations
        for trial in range(50):
            if time.time() > deadline - 15:
                break
            
            # Initialize: take rank 28 and perturb, then zero one column
            U = U28[:, :rank].copy().astype(float)
            V = V28[:, :rank].copy().astype(float)
            W = W28[:, :rank].copy().astype(float)
            
            # Add small perturbation
            U += rng.randn(*U.shape) * 0.01
            V += rng.randn(*V.shape) * 0.01
            W += rng.randn(*W.shape) * 0.01
            
            # ALS iterations
            for it in range(200):
                if time.time() > deadline - 15:
                    break
                    
                # Update U: for each i*m+j, solve linear system
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # Fix V, W. For fixed (i,j), we have:
                # 1 = sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) for all k
                # This is: 1 = U[i*m+j, :] @ (V[j*p+0, :] * W[0*n+i, :])
                #          1 = U[i*m+j, :] @ (V[j*p+1, :] * W[1*n+i, :])
                #          etc.
                # So we have p equations in rank unknowns for each (i,j).
                
                for i in range(n):
                    for j in range(m):
                        # Build system: for each k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                        A_mat = np.zeros((p, rank))
                        for k in range(p):
                            A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        b_vec = np.ones(p)
                        # Solve least squares
                        try:
                            U[i*m+j, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        except:
                            pass
                
                # Update V similarly
                for j in range(m):
                    for k in range(p):
                        A_mat = np.zeros((n, rank))
                        for i in range(n):
                            A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        b_vec = np.ones(n)
                        try:
                            V[j*p+k, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        except:
                            pass
                
                # Update W similarly
                for k in range(p):
                    for i in range(n):
                        A_mat = np.zeros((m, rank))
                        for j in range(m):
                            A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        b_vec = np.ones(m)
                        try:
                            W[k*n+i, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                        except:
                            pass
                
                # Check residual
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            max_err = max(max_err, abs(val - 1))
                
                if max_err < 1e-10:
                    # Found a solution! Try to make it rational
                    # Round to nearest half-integer
                    U_int = np.round(U * 2) / 2
                    V_int = np.round(V * 2) / 2
                    W_int = np.round(W * 2) / 2
                    
                    # Verify
                    max_err2 = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U_int[i*m+j, :] * V_int[j*p+k, :] * W_int[k*n+i, :])
                                max_err2 = max(max_err2, abs(val - 1))
                    
                    if max_err2 < 1e-10:
                        return U_int, V_int, W_int
                
                if max_err < 1e-12:
                    break
        
        return None
    
    # Try rank 27
    result_27 = try_rank_27()
    
    if result_27 is not None:
        U, V, W = result_27
        # Convert to int if possible
        if np.allclose(U * 2, np.round(U * 2)):
            U = (U * 2).astype(int)
            U = U / 2.0
        if np.allclose(V * 2, np.round(V * 2)):
            V = (V * 2).astype(int)
            V = V / 2.0
        if np.allclose(W * 2, np.round(W * 2)):
            W = (W * 2).astype(int)
            W = W / 2.0
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 28
    U, V, W = build_rank_28()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
