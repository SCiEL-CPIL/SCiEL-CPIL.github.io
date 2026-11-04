"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m  # 20
    mp = m * p  # 30
    pn = p * n  # 24
    
    # The multiplication tensor: T[i, j, k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # In our indexing: U row = i*m+j, V row = j*p+k, W row = k*n+i
    # So tensor element T[U_idx, V_idx, W_idx] = 1 when U_idx=i*m+j, V_idx=j*p+k, W_idx=k*n+i
    # i.e., for each (i,j,k) triple, T[i*m+j, j*p+k, k*n+i] = 1
    
    # Build the tensor as a 3D array
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try to find low-rank decomposition
    # Strategy: try various ranks, use random initialization + iterative refinement
    # Or use SVD-based approach on matricizations
    
    best_result = None
    best_rank = nm * mp * pn  # schoolbook
    
    # First, get schoolbook as fallback
    U_sb = np.zeros((nm, best_rank))
    V_sb = np.zeros((mp, best_rank))
    W_sb = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_result = (U_sb, V_sb, W_sb, best_rank)
    
    # Try to find better decompositions
    # For each candidate rank, try to solve for U, V, W such that
    # sum_r U[:,r] * V[:,r] * W[:,r] = T
    
    # Use the approach: fix two matrices, solve for the third
    # Or use alternating least squares
    
    # Try ranks from a reasonable lower bound up
    # Lower bound: the tensor rank is at least the max of the ranks of the matricizations
    T12 = T.reshape(nm, mp * pn)
    T13 = T.reshape(nm, pn * mp).T  # need to be careful
    # Actually, let's just try a range of ranks
    
    # Known: for 4x5x6, try ranks from 50 to 120
    # Use random initialization and ALS
    
    for rank in range(50, min(120, best_rank)):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for trial in range(3):
            if time.time() > deadline:
                break
            
            np.random.seed(rank * 100 + trial)
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Alternating least squares
            converged = False
            for iteration in range(100):
                if time.time() > deadline:
                    break
                
                # Fix V, W, solve for U
                # T = sum_r U[:,r] outer V[:,r] outer W[:,r]
                # For each (a,b) in T: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # Fix V,W: for each c, T[:,:,c] = U @ (V * W[c,:])^T
                # Actually: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # = sum_r U[a,r] * (V[b,r] * W[c,r])
                # So T[:,:,c] = U @ M_c where M_c[b,r] = V[b,r]*W[c,r]
                # T[a,b,c] = sum_r U[a,r] * M_c[b,r]
                # So T[:,:,c].T = M_c.T @ U.T, i.e., T[:,:,c] = U @ M_c
                # Stack: [T[:,:,0], T[:,:,1], ..., T[:,:,pn-1]] = U @ [M_0, M_1, ..., M_{pn-1}]
                # RHS[c] is mp x rank, so stack all c: RHS is (pn*mp) x rank
                # LHS is (pn*mp) x nm... let me re-index
                
                # T[a, b, c] where a in [0,nm), b in [0,mp), c in [0,pn)
                # For fixed V,W: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # Let M_c[b,r] = V[b,r] * W[c,r], shape (mp, rank)
                # Then T[:,:,c] = U @ M_c, shape (nm, mp)
                # So T[:,:,c].flatten() = (U @ M_c).flatten()
                # We can write: for each c, U @ M_c = T[:,:,c]
                # => M_c.T @ U.T = T[:,:,c].T
                # Stack over c: 
                # A = stack of M_c.T for c=0..pn-1, shape (pn*mp, rank)
                # b = stack of T[:,:,c].T.flatten() for c=0..pn-1, shape (pn*mp,)
                # Solve: A @ U.T = b, i.e., A @ (U.T).flatten... 
                # Actually U.T is (rank, nm), so A @ U.T is (pn*mp, nm)
                # And b should be (pn*mp, nm)
                
                A = np.zeros((pn * mp, rank))
                b = np.zeros((pn * mp, nm))
                for c in range(pn):
                    M_c = V * W[c, :]  # (mp, rank)
                    A[c*mp:(c+1)*mp, :] = M_c.T  # (mp, rank)
                    b[c*mp:(c+1)*mp, :] = T[:, :, c].T  # (mp, nm)
                
                # Solve A @ X = b where X = U.T, shape (rank, nm)
                try:
                    U_new = np.linalg.lstsq(A, b, rcond=None)[0].T  # (nm, rank)
                except:
                    break
                
                # Fix U, W, solve for V
                # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # Let N_b[a,r] = U[a,r] * W[c,r]... no
                # T[a,b,c] = sum_r (U[a,r] * W[c,r]) * V[b,r]
                # For fixed a,c: T[a,:,c] = V @ P_{a,c} where P_{a,c}[b,r] = ... no
                # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # = sum_r V[b,r] * (U[a,r] * W[c,r])
                # Let Q_{a,c}[b,r] = V[b,r], and RHS_{a,c}[b] = sum_r V[b,r] * U[a,r] * W[c,r]
                # This is getting complex. Let me use a different approach.
                
                # Alternative: use the fact that we can solve via Kronecker products
                # But for simplicity, let's just check the error and break if not improving
                
                U = U_new
                
                # Check error
                err = 0.0
                for c in range(pn):
                    M_c = V * W[c, :]  # (mp, rank)
                    pred = U @ M_c  # (nm, mp)
                    err += np.sum((pred - T[:, :, c])**2)
                
                if err < 1e-10:
                    converged = True
                    break
                
                if err > 1e6:
                    break
            
            if converged or (time.time() < deadline and err < 1e-10):
                # Verify
                final_err = 0.0
                for c in range(pn):
                    M_c = V * W[c, :]
                    pred = U @ M_c
                    final_err += np.sum((pred - T[:, :, c])**2)
                if final_err < 1e-8:
                    # Round to nearest half-integer
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    # Verify rounded version
                    round_err = 0.0
                    for c in range(pn):
                        M_c = V_r * W_r[c, :]
                        pred = U_r @ M_c
                        round_err += np.sum((pred - T[:, :, c])**2)
                    if round_err < 1e-8:
                        best_result = (U_r, V_r, W_r, rank)
                        best_rank = rank
                        break
        if best_rank <= rank:
            break
    
    if time.time() > deadline:
        U, V, W, r = best_result
    else:
        U, V, W, r = best_result
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

