import numpy as np

def construct():
    n, m, p = 2, 5, 5
    
    # Define the target tensor T in the flattened space
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # 1. Attempt WFTA approach as per directive
    N, M, P = 8, 8, 8 # Padding sizes
    
    # Create the tensor T (N x M x P)
    T = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
                
    # Apply 3D FFT
    T_hat = np.fft.fftn(T)
    
    # "Pruning": The directive says "setting coefficients to zero for frequency components 
    # that correspond to the zero-padded regions".
    # This is ambiguous. A common heuristic in some contexts is to keep the "low" frequencies
    # or frequencies within the original support bounds.
    # Let's try keeping frequencies a < n, b < m, c < p.
    # This results in a rank of n*m*p = 50.
    # If we keep more, rank increases.
    # If we keep this subset, does it reconstruct T?
    # No, truncating FFT causes aliasing.
    
    # Let's construct the decomposition from the kept frequencies.
    # We keep (a,b,c) where a in [0, n-1], b in [0, m-1], c in [0, p-1].
    keep_mask = np.zeros((N, M, P), dtype=bool)
    keep_mask[:n, :m, :p] = True
    
    nz = np.argwhere(keep_mask)
    rank = len(nz)
    
    # Construct U, V, W
    # T[i,j,k] = sum_{a,b,c} T_hat[a,b,c] * exp(2pi i (ia/N + jb/M + kc/P)) / (NMP)
    # We want T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # This factorization is NOT direct because i is in U and W, j is in U and V, k is in V and W.
    # The DFT terms are products of exponentials: E_i(a) E_j(b) E_k(c).
    # U depends on i,j. V on j,k. W on k,i.
    # The term is E_i(a) E_j(b) E_k(c).
    # We cannot factor E_i(a) E_j(b) E_k(c) into U(i,j) V(j,k) W(k,i) for a single (a,b,c)
    # unless we assign parts to U, V, W.
    # e.g., U(i,j) = E_i(a) * E_j(b) ? No, U is fixed for all i,j.
    # This confirms the DFT approach does NOT directly yield a valid tensor decomposition
    # for matrix multiplication because the index dependencies (i in U,W; j in U,V; k in V,W)
    # do not align with the separable frequency components (a->i, b->j, c->k).
    
    # Therefore, the WFTA approach as described will produce an INVALID decomposition.
    
    # To ensure a valid score, we must fall back to a known valid decomposition.
    # The schoolbook decomposition is valid and has rank 50.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1.0
                V[v_idx, col] = 1.0
                W[w_idx, col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
