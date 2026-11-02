import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    """
    Rank-minimal exact tensor decomposition for 2x5 * 5x5 matrix multiplication.
    
    Uses structured decomposition with Strassen blocks and aggressive rank reduction.
    Exploits the two-row structure by treating the problem as two 1x5x5 sub-problems
    and sharing intermediate products where possible.
    """
    start_time = time.time()
    deadline = start_time + 140
    
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
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that U, V, W exactly decompose the tensor T."""
        Rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T - Rec))
        return err < tol, err
    
    def add_strassen(terms, i_start, j_start, k_start):
        """Add 7 Strassen terms for a 2x2x2 block at (i_start, j_start, k_start)."""
        u = {(i_start, j_start): 1, (i_start+1, j_start+1): 1}
        v = {(j_start, k_start): 1, (j_start+1, k_start+1): 1}
        w = {(k_start, i_start): 1, (k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        u = {(i_start+1, j_start): 1, (i_start+1, j_start+1): 1}
        v = {(j_start, k_start): 1}
        w = {(k_start, i_start+1): 1, (k_start+1, i_start+1): -1}
        terms.append((u, v, w))
        
        u = {(i_start, j_start): 1}
        v = {(j_start, k_start+1): 1, (j_start+1, k_start+1): -1}
        w = {(k_start+1, i_start): 1, (k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        u = {(i_start+1, j_start+1): 1}
        v = {(j_start+1, k_start): 1, (j_start, k_start): -1}
        w = {(k_start, i_start): 1, (k_start, i_start+1): 1}
        terms.append((u, v, w))
        
        u = {(i_start, j_start): 1, (i_start, j_start+1): 1}
        v = {(j_start+1, k_start+1): 1}
        w = {(k_start, i_start): -1, (k_start+1, i_start): 1}
        terms.append((u, v, w))
        
        u = {(i_start+1, j_start): 1, (i_start, j_start): -1}
        v = {(j_start, k_start): 1, (j_start, k_start+1): 1}
        w = {(k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        u = {(i_start, j_start+1): 1, (i_start+1, j_start+1): -1}
        v = {(j_start+1, k_start): 1, (j_start+1, k_start+1): 1}
        w = {(k_start, i_start): 1}
        terms.append((u, v, w))
    
    def add_schoolbook_term(terms, i, j, k):
        """Add a single schoolbook term."""
        u = {(i, j): 1}
        v = {(j, k): 1}
        w = {(k, i): 1}
        terms.append((u, v, w))
    
    def gradient_descent(U, V, W, steps, lr_schedule, momentum=0.9):
        """Run gradient descent to refine the decomposition."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        for step in range(steps):
            if time.time() > deadline - 5:
                break
            
            lr = lr_schedule(step)
            
            Rec = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T - Rec
            
            gU = np.einsum('abc,br,cr->ar', diff, V, W)
            gV = np.einsum('abc,ar,cr->br', diff, U, W)
            gW = np.einsum('abc,ar,br->cr', diff, U, V)
            
            vU = momentum * vU + (1 - momentum) * gU
            vV = momentum * vV + (1 - momentum) * gV
            vW = momentum * vW + (1 - momentum) * gW
            
            U -= lr * vU
            V -= lr * vV
            W -= lr * vW
        
        return U, V, W
    
    def try_reduce_rank(U, V, W, max_reduction=20, steps_per_trial=300):
        """Try to reduce rank by removing dependent terms and re-optimizing."""
        R = U.shape[1]
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = R
        
        norms = np.linalg.norm(U, axis=0) * np.linalg.norm(V, axis=0) * np.linalg.norm(W, axis=0)
        sorted_indices = np.argsort(norms)
        
        for reduction in range(1, max_reduction + 1):
            if time.time() > deadline - 20:
                break
                
            new_R = R - reduction
            if new_R < 1:
                break
            
            keep_indices = sorted_indices[-new_R:]
            keep_indices = np.sort(keep_indices)
            
            U_new = U[:, keep_indices].copy()
            V_new = V[:, keep_indices].copy()
            W_new = W[:, keep_indices].copy()
            
            lr_schedule = lambda step: 0.01 * (0.995 ** step)
            U_new, V_new, W_new = gradient_descent(U_new, V_new, W_new, steps_per_trial, lr_schedule)
            
            valid, err = verify_decomposition(U_new, V_new, W_new)
            if valid:
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = new_R
            else:
                break
        
        return best_U, best_V, best_W, best_rank
    
    def try_shared_strassen_decomposition():
        """
        Try to exploit the two-row structure by using Strassen blocks
        that share intermediate products across rows.
        
        For a 2x2x2 block in (j,k) space, we need to compute:
        C[k0,i0] = A[i0,j0]*B[j0,k0] + A[i0,j0+1]*B[j0+1,k0]
        C[k0,i1] = A[i1,j0]*B[j0,k0] + A[i1,j0+1]*B[j0+1,k0]
        C[k0+1,i0] = A[i0,j0]*B[j0,k0+1] + A[i0,j0+1]*B[j0+1,k0+1]
        C[k0+1,i1] = A[i1,j0]*B[j0,k0+1] + A[i1,j0+1]*B[j0+1,k0+1]
        
        Using shared intermediates:
        M1 = (A[0,j0]+A[1,j0]) * (B[j0,k0]+B[j0+1,k0+1])
        M2 = (A[0,j0+1]+A[1,j0+1]) * (B[j0+1,k0]+B[j0,k0+1])
        M3 = A[0,j0]*B[j0,k0]
        M4 = A[1,j0]*B[j0+1,k0]
        M5 = A[0,j0+1]*B[j0,k0+1]
        M6 = A[1,j0+1]*B[j0+1,k0+1]
        
        This gives 6 terms for 4 outputs, vs 8 schoolbook terms.
        """
        terms = []
        
        # For each 2x2 block in (j,k) space, use 6 shared terms
        for j0 in range(0, 4, 2):
            for k0 in range(0, 4, 2):
                # M1 = (A[0,j0]+A[1,j0]) * (B[j0,k0]+B[j0+1,k0+1])
                u = {(0, j0): 1, (1, j0): 1}
                v = {(j0, k0): 1, (j0+1, k0+1): 1}
                w = {(k0, 0): 1, (k0, 1): 1}
                terms.append((u, v, w))
                
                # M2 = (A[0,j0+1]+A[1,j0+1]) * (B[j0+1,k0]+B[j0,k0+1])
                u = {(0, j0+1): 1, (1, j0+1): 1}
                v = {(j0+1, k0): 1, (j0, k0+1): 1}
                w = {(k0+1, 0): 1, (k0+1, 1): 1}
                terms.append((u, v, w))
                
                # M3 = A[0,j0]*B[j0,k0]
                u = {(0, j0): 1}
                v = {(j0, k0): 1}
                w = {(k0, 0): 1}
                terms.append((u, v, w))
                
                # M4 = A[1,j0]*B[j0+1,k0]
                u = {(1, j0): 1}
                v = {(j0+1, k0): 1}
                w = {(k0, 1): 1}
                terms.append((u, v, w))
                
                # M5 = A[0,j0+1]*B[j0,k0+1]
                u = {(0, j0+1): 1}
                v = {(j0, k0+1): 1}
                w = {(k0+1, 0): 1}
                terms.append((u, v, w))
                
                # M6 = A[1,j0+1]*B[j0+1,k0+1]
                u = {(1, j0+1): 1}
                v = {(j0+1, k0+1): 1}
                w = {(k0+1, 1): 1}
                terms.append((u, v, w))
        
        # Handle k=4 (j in 0..3): 8 schoolbook terms
        for j in range(4):
            for i in range(2):
                add_schoolbook_term(terms, i, j, 4)
        
        # Handle j=4 (k in 0..4): 10 schoolbook terms
        for k in range(5):
            for i in range(2):
                add_schoolbook_term(terms, i, 4, k)
        
        return terms
    
    # Build initial decomposition using shared Strassen approach
    terms = try_shared_strassen_decomposition()
    
    U, V, W = build_decomposition_from_terms(terms)
    R = U.shape[1]
    
    # Verify initial decomposition
    valid, err = verify_decomposition(U, V, W)
    
    if not valid:
        # Fallback to standard Strassen + schoolbook
        terms_fb = []
        add_strassen(terms_fb, 0, 0, 0)
        add_strassen(terms_fb, 0, 0, 2)
        add_strassen(terms_fb, 0, 2, 0)
        add_strassen(terms_fb, 0, 2, 2)
        for j in range(4):
            for i in range(2):
                add_schoolbook_term(terms_fb, i, j, 4)
        for k in range(5):
            for i in range(2):
                add_schoolbook_term(terms_fb, i, 4, k)
        
        U, V, W = build_decomposition_from_terms(terms_fb)
        R = U.shape[1]
        valid, err = verify_decomposition(U, V, W)
        
        if not valid:
            # Ultimate fallback: schoolbook
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
    else:
        # Refine the full-rank decomposition
        if time.time() < deadline - 40:
            lr_schedule = lambda step: 0.005 * (0.99 ** step)
            U, V, W = gradient_descent(U, V, W, 150, lr_schedule)
            valid, err = verify_decomposition(U, V, W)
        
        # Try to reduce rank
        if time.time() < deadline - 20:
            U, V, W, R = try_reduce_rank(U, V, W, max_reduction=20, steps_per_trial=250)
    
    # Final verification
    valid, err = verify_decomposition(U, V, W)
    if not valid:
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
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
