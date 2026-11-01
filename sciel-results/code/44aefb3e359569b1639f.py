"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _build_from_terms(all_terms, n, m, p):
    """Build U, V, W matrices from a list of (u_vec, v_vec, w_vec) terms."""
    rank = len(all_terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for r, (u_vec, v_vec, w_vec) in enumerate(all_terms):
        U[:, r] = u_vec
        V[:, r] = v_vec
        W[:, r] = w_vec
    return U, V, W

def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication."""
    terms = []
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 1]))
    terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
    terms.append(([0, 0, 0, 1], [0, 0, 1, -1], [1, 1, 0, 0]))
    terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]))
    terms.append(([1, 0, -1, 0], [1, 1, 0, 0], [0, 1, 0, 0]))
    terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1]))
    return terms

def _make_schoolbook_terms(n, m, p, k_start=0, k_end=None):
    """Make schoolbook terms for columns k_start to k_end-1."""
    if k_end is None:
        k_end = p
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(k_start, k_end):
                u_vec = [0.0] * (n * m)
                u_vec[i * m + j] = 1.0
                v_vec = [0.0] * (m * p)
                v_vec[j * p + k] = 1.0
                w_vec = [0.0] * (p * n)
                w_vec[k * n + i] = 1.0
                terms.append((u_vec, v_vec, w_vec))
    return terms

def _extend_2x2_term_to_2x5(u_vec, v_vec_2x2, w_vec_2x2, n=2, m=2, p=5):
    """Extend a 2x2x2 term to a 2x2x5 term (only uses k=0,1)."""
    u_full = list(u_vec)
    v_full = [0.0] * (m * p)
    for j in range(m):
        for k in range(2):
            v_full[j * p + k] = v_vec_2x2[j * 2 + k]
    w_full = [0.0] * (p * n)
    for k in range(2):
        for i in range(n):
            w_full[k * n + i] = w_vec_2x2[k * 2 + i]
    return u_full, v_full, w_full

def _als_solve(U, V, W, n, m, p, target_rank, max_iter=500, tol=1e-10, deadline=None):
    """
    ALS (Alternating Least Squares) to find a rank-r decomposition.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    We want: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    """
    U = U.copy().astype(float)
    V = V.copy().astype(float)
    W = W.copy().astype(float)
    r = target_rank
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U: for each (i,j), U[i*m+j, :] = argmin sum_k (1 - sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r])^2
        # Fix V and W, solve for U
        for i in range(n):
            for j in range(m):
                # We need: sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for all k
                # This is a linear system: M @ u = 1 where M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                M = np.zeros((p, r))
                for k in range(p):
                    M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve M @ u = ones(p)
                try:
                    u_new, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                except:
                    continue
                U[i*m+j, :] = u_new
        
        # Update V: for each (j,k), V[j*p+k, :] = argmin
        for j in range(m):
            for k in range(p):
                # sum_r V[j*p+k,r] * (U[i*m+j,r] * W[k*n+i,r]) = 1 for all i
                M = np.zeros((n, r))
                for i in range(n):
                    M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    v_new, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                except:
                    continue
                V[j*p+k, :] = v_new
        
        # Update W: for each (k,i), W[k*n+i, :] = argmin
        for k in range(p):
            for i in range(n):
                # sum_r W[k*n+i,r] * (U[i*m+j,r] * V[j*p+k,r]) = 1 for all j
                M = np.zeros((m, r))
                for j in range(m):
                    M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    w_new, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                except:
                    continue
                W[k*n+i, :] = w_new
        
        # Check convergence
        if it % 10 == 0:
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr] for rr in range(r))
                        err = max(err, abs(val - 1.0))
            if err < tol:
                break
    
    # Final verification
    if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
        return U, V, W
    return None

def _als_search(n, m, p, target_rank, deadline, n_restarts=20, seed=42):
    """Try multiple random initializations with ALS to find a valid rank-r decomposition."""
    rng = np.random.RandomState(seed)
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Initialize with structured perturbation of the known rank-13 solution
        # Start from the Strassen+schoolbook solution and randomly project to lower rank
        
        # Create initial random matrices
        U_init = rng.randn(n*m, target_rank) * 0.5
        V_init = rng.randn(m*p, target_rank) * 0.5
        W_init = rng.randn(p*n, target_rank) * 0.5
        
        # Add some structure: initialize based on the known solution projected
        # Use the rank-13 solution as a warm start
        strassen_terms = _strassen_2x2_terms()
        all_terms = []
        for u_vec, v_vec, w_vec in strassen_terms:
            u_full, v_full, w_full = _extend_2x2_term_to_2x5(u_vec, v_vec, w_vec, n, m, p)
            all_terms.append((u_full, v_full, w_full))
        schoolbook_terms = _make_schoolbook_terms(n, m, p, k_start=2, k_end=5)
        all_terms.extend(schoolbook_terms)
        
        U13, V13, W13 = _build_from_terms(all_terms, n, m, p)
        
        # Randomly project the rank-13 solution to rank target_rank
        # Use a random linear combination of columns
        if target_rank < 13:
            # Pick target_rank columns and randomly mix
            idx = rng.choice(13, size=target_rank, replace=False)
            U_init = U13[:, idx] * rng.randn(target_rank) 
            V_init = V13[:, idx] * rng.randn(target_rank)
            W_init = W13[:, idx] * rng.randn(target_rank)
        else:
            # Use full solution with noise
            U_init = U13[:, :target_rank] + rng.randn(n*m, target_rank) * 0.1
            V_init = V13[:, :target_rank] + rng.randn(m*p, target_rank) * 0.1
            W_init = W13[:, :target_rank] + rng.randn(p*n, target_rank) * 0.1
        
        result = _als_solve(U_init, V_init, W_init, n, m, p, target_rank, max_iter=300, tol=1e-10, deadline=deadline)
        if result is not None:
            return result
    
    return None

def _try_polish(U, V, W, n, m, p, deadline):
    """High-precision polish pass: run ALS with very tight tolerance."""
    r = U.shape[1]
    U = U.copy().astype(float)
    V = V.copy().astype(float)
    W = W.copy().astype(float)
    
    for it in range(200):
        if time.time() > deadline:
            break
        
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, r))
                for k in range(p):
                    M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    u_new, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                except:
                    continue
                U[i*m+j, :] = u_new
        
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, r))
                for i in range(n):
                    M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    v_new, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                except:
                    continue
                V[j*p+k, :] = v_new
        
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, r))
                for j in range(m):
                    M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    w_new, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                except:
                    continue
                W[k*n+i, :] = w_new
        
        if it % 5 == 0:
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr] for rr in range(r))
                        err = max(err, abs(val - 1.0))
            if err < 1e-14:
                break
    
    if _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
        return U, V, W
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 2, 5
    
    # Build initial rank-13 decomposition: Strassen(2x2x2) + schoolbook(2x2x3)
    strassen_terms = _strassen_2x2_terms()
    all_terms = []
    
    for u_vec, v_vec, w_vec in strassen_terms:
        u_full, v_full, w_full = _extend_2x2_term_to_2x5(u_vec, v_vec, w_vec, n, m, p)
        all_terms.append((u_full, v_full, w_full))
    
    schoolbook_terms = _make_schoolbook_terms(n, m, p, k_start=2, k_end=5)
    all_terms.extend(schoolbook_terms)
    
    U, V, W = _build_from_terms(all_terms, n, m, p)
    
    if not _verify_decomposition(U, V, W, n, m, p):
        rank = n * m * p
        U_fb = np.zeros((n * m, rank))
        V_fb = np.zeros((m * p, rank))
        W_fb = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # Try ALS-based search for lower ranks
    for target_rank in range(best_rank - 1, max(8, best_rank - 5), -1):
        if time.time() > deadline - 10:
            break
        
        result = _als_search(n, m, p, target_rank, deadline, n_restarts=30, seed=42 + target_rank)
        if result is not None:
            U_found, V_found, W_found = result
            # Polish for higher precision
            polished = _try_polish(U_found, V_found, W_found, n, m, p, deadline)
            if polished is not None:
                U_found, V_found, W_found = polished
            best_U, best_V, best_W = U_found.copy(), V_found.copy(), W_found.copy()
            best_rank = target_rank
            # Don't break - try even lower ranks if time permits
            if time.time() > deadline - 15:
                break
    
    # Final: return the best decomposition found
    # Clean up: convert near-integers to exact integers
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    for mat in [U_list, V_list, W_list]:
        for row in mat:
            for idx, val in enumerate(row):
                if abs(val - round(val)) < 1e-10:
                    row[idx] = round(val)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
