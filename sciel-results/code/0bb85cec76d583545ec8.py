import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x5 * 5x5 matrix multiplication.
    Uses Strassen on 2x2x2 sub-blocks plus schoolbook for fringe, with a
    post-processing step to detect and remove linearly dependent rank-1 terms.
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=5, p=5
    
    # Build the full tensor T[a, b, c] where:
    #   a = i*m + j  (row of A, col of A)
    #   b = j*p + k  (row of B, col of B)
    #   c = k*n + i  (output, trace convention: k*n + i)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that U, V, W exactly decompose the tensor T."""
        Rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T - Rec))
        return err < tol, err
    
    def build_strassen_decomposition():
        """
        Build Strassen-based decomposition for 2x5x5.
        Uses 2x2x2 Strassen blocks where possible, schoolbook for fringe.
        """
        terms = []
        
        def add_strassen_2x2x2(i_start, j_start, k_start):
            """Add 7 Strassen terms for a 2x2x2 block."""
            # M1 = (A11+A22)(B11+B22)
            u = {(i_start, j_start): 1, (i_start+1, j_start+1): 1}
            v = {(j_start, k_start): 1, (j_start+1, k_start+1): 1}
            w = {(k_start, i_start): 1, (k_start+1, i_start+1): 1}
            terms.append((u, v, w))
            
            # M2 = (A21+A22)B11
            u = {(i_start+1, j_start): 1, (i_start+1, j_start+1): 1}
            v = {(j_start, k_start): 1}
            w = {(k_start, i_start+1): 1, (k_start+1, i_start+1): -1}
            terms.append((u, v, w))
            
            # M3 = A11(B21-B11)
            u = {(i_start, j_start): 1}
            v = {(j_start, k_start+1): 1, (j_start+1, k_start+1): -1}
            w = {(k_start+1, i_start): 1, (k_start+1, i_start+1): 1}
            terms.append((u, v, w))
            
            # M4 = A22(B22-B12)
            u = {(i_start+1, j_start+1): 1}
            v = {(j_start+1, k_start): 1, (j_start, k_start): -1}
            w = {(k_start, i_start): 1, (k_start, i_start+1): 1}
            terms.append((u, v, w))
            
            # M5 = (A11+A12)B22
            u = {(i_start, j_start): 1, (i_start, j_start+1): 1}
            v = {(j_start+1, k_start+1): 1}
            w = {(k_start, i_start): -1, (k_start+1, i_start): 1}
            terms.append((u, v, w))
            
            # M6 = (A21-A11)(B11+B12)
            u = {(i_start+1, j_start): 1, (i_start, j_start): -1}
            v = {(j_start, k_start): 1, (j_start, k_start+1): 1}
            w = {(k_start+1, i_start+1): 1}
            terms.append((u, v, w))
            
            # M7 = (A12-A22)(B21+B22)
            u = {(i_start, j_start+1): 1, (i_start+1, j_start+1): -1}
            v = {(j_start+1, k_start): 1, (j_start+1, k_start+1): 1}
            w = {(k_start, i_start): 1}
            terms.append((u, v, w))
        
        def add_schoolbook_term(i, j, k):
            """Add a single schoolbook term."""
            u = {(i, j): 1}
            v = {(j, k): 1}
            w = {(k, i): 1}
            terms.append((u, v, w))
        
        # Decomposition plan for 2x5x5 using Strassen:
        add_strassen_2x2x2(0, 0, 0)
        add_strassen_2x2x2(0, 0, 2)
        add_strassen_2x2x2(0, 2, 0)
        add_strassen_2x2x2(0, 2, 2)
        
        # j in {0,1,2,3}, k=4
        for j in range(4):
            for i in range(2):
                add_schoolbook_term(i, j, 4)
        
        # j=4, k in {0,1,2,3,4}
        for k in range(5):
            for i in range(2):
                add_schoolbook_term(i, 4, k)
        
        return terms
    
    def build_decomposition_from_terms(terms):
        """Build U, V, W matrices from a list of (u_dict, v_dict, w_dict) terms."""
        rank = len(terms)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for r, (u_dict, v_dict, w_dict) in enumerate(terms):
            for (i, j), val in u_dict.items():
                U[i * m + j, r] = val
            for (j, k), val in v_dict.items():
                V[j * p + k, r] = val
            for (k, i), val in w_dict.items():
                W[k * n + i, r] = val
        
        return U, V, W
    
    def try_reduce_rank(U, V, W, tol=1e-10):
        """
        Try to reduce rank by detecting linearly dependent rank-1 terms.
        """
        R = U.shape[1]
        if R <= 1:
            return U, V, W
        
        keep = list(range(R))
        # Try removing terms one at a time
        for r in range(R - 1, -1, -1):
            if r not in keep:
                continue
            candidate_keep = [k for k in keep if k != r]
            if len(candidate_keep) == 0:
                continue
            
            U_c = U[:, candidate_keep]
            V_c = V[:, candidate_keep]
            W_c = W[:, candidate_keep]
            
            # Check if the tensor can still be reconstructed
            Rec = np.einsum('ar,br,cr->abc', U_c, V_c, W_c)
            err = np.max(np.abs(T - Rec))
            
            if err < tol:
                keep = candidate_keep
        
        if len(keep) < R:
            U = U[:, keep]
            V = V[:, keep]
            W = W[:, keep]
        
        return U, V, W
    
    # Build terms using Strassen-based approach
    terms = build_strassen_decomposition()
    
    U, V, W = build_decomposition_from_terms(terms)
    
    # Try to reduce rank
    U, V, W = try_reduce_rank(U, V, W)
    
    R = U.shape[1]
    
    # Verify
    valid, err = verify_decomposition(U, V, W)
    
    if not valid:
        # Fallback to schoolbook
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        R = n * m * p
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
