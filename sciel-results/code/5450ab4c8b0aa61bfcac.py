"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import nnls

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Build the rank-21 Strassen decomposition first
    rank_base = 21
    U = np.zeros((n * m, rank_base), dtype=float)
    V = np.zeros((m * p, rank_base), dtype=float)
    W = np.zeros((p * n, rank_base), dtype=float)
    
    strassen = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
        ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
        ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
        ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
        ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1})
    ]
    
    r = 0
    for b in range(3):
        for t in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen[t]
            for idx, val in u_coeffs.items():
                U[idx][r] = val
            for l, val in v_coeffs.items():
                if l == 0: v_idx = 2 * b
                elif l == 1: v_idx = 2 * b + 1
                elif l == 2: v_idx = 6 + 2 * b
                else: v_idx = 7 + 2 * b
                V[v_idx][r] = val
            for (i, k_local), val in w_coeffs.items():
                k_global = 2 * b + k_local
                w_idx = k_global * n + i
                W[w_idx][r] = val
            r += 1
    
    # Verify rank-21 solution
    def verify(U, V, W, tol=1e-9):
        # Check tensor equality
        # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # A[i,j] = U[i,:] dot V[j,:], B[j,k] = V[j,:] dot W[k,:]
        # C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_r U[i,r]V[j,r] sum_s V[j,s]W[k,s]
        # For rank-r decomposition: C[i,k] = sum_r U[i,r] * (sum_j V[j,r]^2) * W[k,r]... no
        # Actually: C = A*B, A[i,j]=sum_r U[i,r]V[j,r], B[j,k]=sum_r V[j,r]W[k,r]
        # C[i,k] = sum_j A[i,j]B[j,k] = sum_j sum_r sum_s U[i,r]V[j,r]V[j,s]W[k,s]
        # For this to equal sum_r U[i,r]V[j,r]W[k,r] summed properly...
        # The tensor is T[i,j,k] = C[i,k] which depends only on i and k, not j.
        # So we need: sum_j sum_r U[i,r]V[j,r] * (something involving W)
        # Standard: T[i,j,k] where T is the multiplication tensor.
        # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r] should equal delta-like structure
        # For matrix mult: (A*B)[i,k] = sum_j A[i,j]B[j,k]
        # So the tensor T has T[i,j,k] such that contracting with A[i,j] and B[j,k] gives C[i,k]
        # T[i,j,k] = 1 if this is the "middle" index... no.
        # The tensor for matrix multiplication M_{n,m,p} has T[i,j,k] where:
        # C[i,k] = sum_{j=1}^m sum_{r=1}^rank U[i,r] V[j,r] W[k,r]
        # This must equal sum_{j=1}^m A[i,j] B[j,k] for all A, B
        # So: sum_j sum_r U[i,r] V[j,r] W[k,r] A... no, U and V are fixed.
        # The decomposition says: the bilinear map (A,B) -> A*B has rank r
        # i.e., (A*B)[i,k] = sum_r U_r(A) V_r(B) W_r(...) -- no, W is part of the output
        # Correct: (A*B)[i,k] = sum_r alpha_r(A) * beta_r(B) where alpha_r, beta_r are linear forms
        # In the U,V,W formulation:
        # Let A be n x m, B be m x p
        # alpha_r(A) = sum_{i,j} U[i,r] * A[i,j]... no, U has shape (n*m, rank)
        # Actually U[i_global, r] where i_global = i*m + j? or i*n + j?
        # From the code: U has shape (n*m, rank), indices 0..3 for n=2,m=2
        # Index mapping: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> i*m+j
        # V has shape (m*p, rank), indices 0..11 for m=2,p=6
        # Index mapping: j*p + k
        # W has shape (p*n, rank), indices 0..11
        # Index mapping: k*n + i
        # Then: (A*B)[i,k] = sum_r U[i*m + ? ...] 
        # Hmm, let me re-read. The decomposition is:
        # (A*B)[i,k] = sum_r (sum_{j} U[i*m+j, r] A[i,j]) * ... no that's not right either
        # 
        # Standard tensor decomposition for matrix mult:
        # The multiplication tensor T in R^{n x m} ⊗ R^{m x p} ⊗ R^{p x n} (with trace convention)
        # T_{(i,j),(k,l),(r,s)} = delta_{j,k} delta_{l,s} * ... no
        # For (A*B)[i,k] = sum_j A[i,j] B[j,k]:
        # T is a 3-tensor with indices (i,j), (j,k), (i,k) but that's not right
        # 
        # Let me just verify numerically with random matrices
        def check():
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_ref = A @ B
            # Compute C from decomposition
            # (A*B)[i,k] = sum_r sum_j U[i*m+j, r] * A[i,j] ... no
            # The correct formula: 
            # C[i,k] = sum_r (sum_{j} U_flat[i*m+j, r] * A[i,j]) -- this doesn't make sense
            # 
            # Let me look at it differently. In the standard formulation:
            # Each rank-1 term contributes: outer product of three vectors
            # u in R^{n*m}, v in R^{m*p}, w in R^{p*n}
            # The term is u ⊗ v ⊗ w, and C = sum_r (u_r ⊗ v_r ⊗ w_r)
            # Contracting: C[i,k] = sum_j sum_r u[i*m+j, r] * v[j*p+k, r] * ... no
            # Actually the tensor T[i,j; j',k'; i'',k''] = delta[j,j'] * delta[... ]
            # I think the correct interpretation is:
            # (A*B)[i,k] = sum_r U_r(A) * V_r(B) where
            # U_r(A) = sum_{i,j} U[i*m+j, r] A[i,j] -- no, this gives a scalar but then what's W?
            #
            # I think W is used as: the output coefficient
            # Let me just trust the original code and verify it works
            
            # Based on the structure, I believe:
            # (A*B)[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * W[k*n+i, r] * ... no
            
            # Let me try: the rank-1 decomposition of the tensor means
            # T = sum_r u_r ⊗ v_r ⊗ w_r
            # where T has indices (i,j), (j,k), (i,k) with T[i,j; j',k'; i'',k''] = 
            # delta_{j,j'} delta_{i,i''} delta_{k,k''} (this is the multiplication tensor)
            
            # So: (A*B)[i,k] = <A, T[:, :, (i,k)]> = sum_{j,j',k'} A[i,j] B[j',k'] T[i,j; j',k'; i,k]
            # = sum_{j,j',k'} A[i,j] B[j',k'] delta_{j,j'} delta_{i,i} delta_{k,k}
            # = sum_j A[i,j] B[j,k] ✓
            
            # And T[i,j; j',k'; i,k] = delta_{j,j'} delta_{i,i} delta_{k,k}
            # So for fixed (i,k): T_{(i,j),(j',k')} = delta_{j,j'} delta_{k,k}
            # The rank-1 decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r means
            # T[a,b,c] = sum_r u_r[a] v_r[b] w_r[c]
            # where a indexes (i,j) in R^{n*m}, b indexes (j',k') in R^{m*p}, c indexes (i'',k'') in R^{p*n}
            
            # So: delta_{j,j'} delta_{i,i''} delta_{k,k''} = sum_r U[i*m+j, r] V[j'*p+k', r] W[k''*n+i'', r]
            
            # To compute C = A*B from the decomposition:
            # C[i,k] = sum_{j,j',k'} A[i,j] B[j',k'] T[(i,j),(j',k'),(i,k)]
            # = sum_{j,j',k'} A[i,j] B[j',k'] sum_r U[i*m+j, r] V[j'*p+k', r] W[k*n+i, r]
            # = sum_r W[k*n+i, r] * (sum_j A[i,j] U[i*m+j, r]) * (sum_{j',k'} B[j',k'] V[j'*p+k', r])
            
            C_comp = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for r in range(rank_base):
                        w_val = W[k * n + i, r]
                        if w_val != 0:
                            u_sum = 0.0
                            for j in range(m):
                                u_sum += A[i, j] * U[i * m + j, r]
                            v_sum = 0.0
                            for jp in range(m * p):
                                jj = jp // p
                                kk = jp % p
                                v_sum += B[jj, kk] * V[jp, r]
                            s += w_val * u_sum * v_sum
                    C_comp[i, k] = s
            return np.allclose(C_comp, C_ref, atol=tol)
        
        return check()
    
    if not verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Now attempt to reduce rank by finding linear dependencies
    # The 21 rank-1 tensors might have dependencies
    # We want to find if rank-20 suffices
    
    # Build the full tensor T of shape (n*m, m*p, p*n) = (4, 12, 12)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Verify the rank-21 decomposition reproduces T
    def verify_tensor(U, V, W, T, tol=1e-9):
        T_recon = np.tensordot(U, V, axes=(1, 1))  # shape (n*m, m*p) -- no
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_recon, T, atol=tol)
    
    if not verify_tensor(U, V, W, T):
        # Fallback to the working version
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank by eliminating one column
    # Check if any column of U, V, W is linearly dependent
    # For rank reduction from 21 to 20, we need to find a linear relation
    
    # Strategy: try to set one of the rank-1 terms to be a combination of others
    # This is equivalent to finding a vector (alpha_1, ..., alpha_21) such that
    # sum_r alpha_r U[:,r] = 0, sum_r alpha_r V[:,r] = 0, sum_r alpha_r W[:,r] = 0
    # But that would mean the tensor is in the span of fewer terms only if
    # the alpha_r are all zero... actually for rank reduction we need something else.
    
    # Better approach: try rank-20 directly with a constrained optimization
    # or try to find if we can merge two Strassen terms from different blocks
    
    # The key insight: the three blocks use the same Strassen pattern.
    # Perhaps we can share some terms across blocks.
    
    # Let's try a different approach: use the structure of the problem
    # The tensor is a "block diagonal" tensor with 3 identical 2x2x2 blocks
    # but embedded in a 2x2x6 structure
    
    # For now, let's try to verify if rank 20 is achievable by checking
    # if we can find a rank-20 decomposition using a search
    
    # Given time constraints, let's return the verified rank-21 solution
    # but try a quick heuristic first
    
    if time.time() < deadline:
        # Try to find if removing one term still works
        # Try removing each term and see if the remaining 20 span the tensor
        for remove_idx in range(rank_base):
            if time.time() > deadline - 5:
                break
            U_sub = np.delete(U, remove_idx, axis=1)
            V_sub = np.delete(V, remove_idx, axis=1)
            W_sub = np.delete(W, remove_idx, axis=1)
            
            # Check if T is in the span of the 20 remaining rank-1 tensors
            # This is a hard problem, so let's just check a few random points
            # Actually, let's check if the tensor can be reconstructed
            # by solving a least-squares problem
            
            # We need to find coefficients c_1..c_20 such that
            # T = sum_r c_r U_sub[:,r] ⊗ V_sub[:,r] ⊗ W_sub[:,r]
            # This is equivalent to: for each (a,b,c) index:
            # T[a,b,c] = sum_r U_sub[a,r] V_sub[b,r] W_sub[c,r] * c_r
            # Let x_r = c_r, then we have a linear system in x
            # But this is only solvable if the 20 tensors span T
            
            # Flatten: T_flat[a*m*p + b*p*n + c] = sum_r (U_sub[a,r] V_sub[b,r] W_sub[c,r]) x_r
            # This is a linear system: M x = T_flat
            # where M has shape (4*12*12, 20) = (576, 20)
            
            M = np.zeros((n * m * m * p * p * n, rank_base - 1))
            for r in range(rank_base - 1):
                # M[:, r] = U_sub[:,r] ⊗ V_sub[:,r] ⊗ W_sub[:,r] flattened
                M[:, r] = np.kron(np.kron(U_sub[:, r], V_sub[:, r]), W_sub[:, r])
            
            T_flat = T.flatten()
            
            # Check if T_flat is in the column space of M
            # Use least squares
            try:
                x, residuals, rank, sv = np.linalg.lstsq(M, T_flat, rcond=None)
                T_recon = M @ x
                if np.allclose(T_recon, T_flat, atol=1e-9):
                    # Found a rank-20 decomposition!
                    U_new = U_sub @ np.diag(x)  # no, x is just a scaling
                    # Actually the rank-1 terms are U_sub[:,r] ⊗ V_sub[:,r] ⊗ W_sub[:,r]
                    # and we found T = sum_r x_r * (those terms)
                    # So the new decomposition has:
                    # U_new[:,r] = x_r * U_sub[:,r] -- but that changes the scale
                    # We can absorb x_r into U:
                    U_final = U_sub * x[np.newaxis, :]
                    V_final = V_sub
                    W_final = W_sub
                    
                    if verify_tensor(U_final, V_final, W_final, T):
                        # Convert to list and return
                        # Round to nearest half-integer or integer if possible
                        U_out = np.round(U_final * 2) / 2
                        V_out = np.round(V_final * 2) / 2
                        W_out = np.round(W_final * 2) / 2
                        if verify_tensor(U_out, V_out, W_out, T):
                            return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
                        else:
                            return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
            except Exception:
                pass
    
    # Return the rank-21 solution
    U_out = np.round(U * 2) / 2
    V_out = np.round(V * 2) / 2
    W_out = np.round(W * 2) / 2
    if verify_tensor(U_out, V_out, W_out, T):
        return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
