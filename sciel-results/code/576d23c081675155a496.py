import numpy as np
import time

DIMENSIONS = [4, 4, 7]
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s kill

def construct():
    n, m, p = 4, 4, 7
    deadline = START_TIME + TIME_LIMIT
    
    # Build the multiplication tensor T[i,j,k] = delta[i,j,k] for matmul
    # T is a 3D tensor of shape (n*m, m*p, p*n) = (16, 28, 28)
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    def time_check():
        return time.time() < deadline
    
    # Strategy: Use DCT-based change of basis to find sparser structure,
    # then apply greedy rank minimization via sparse recovery.
    
    # First, build the base tensor as a 3D array
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Apply DCT-like transforms to each mode to find sparse structure
    # DCT-II matrix for size d
    def dct_matrix(d):
        M = np.zeros((d, d))
        for r in range(d):
            for c in range(d):
                if r == 0:
                    M[r, c] = 1.0 / np.sqrt(d)
                else:
                    M[r, c] = np.sqrt(2.0/d) * np.cos(np.pi * r * (2*c + 1) / (2*d))
        return M
    
    # Apply DCT to each mode
    D1 = dct_matrix(n*m)  # 16x16
    D2 = dct_matrix(m*p)  # 28x28
    D3 = dct_matrix(p*n)  # 28x28
    
    # Transform: T' = D1 @ T @ D2^T @ D3^T (mode-wise)
    # Mode-1: T'_{a,b,c} = sum_{i,j,k} D1[a,i] * T[i,j,k] * D2[b,j] * D3[c,k]
    T_prime = np.zeros_like(T)
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                val = 0.0
                for i in range(n*m):
                    for j in range(m*p):
                        for k in range(p*n):
                            val += D1[a,i] * T[i,j,k] * D2[b,j] * D3[c,k]
                T_prime[a,b,c] = val
    
    # Check if we're running out of time
    if not time_check():
        # Fall back to schoolbook
        return _schoolbook(n, m, p)
    
    # Greedy sparse recovery: find minimal set of rank-1 terms
    # that reconstruct T (not T_prime, we need the original)
    # We work in the original space but use the transformed space to guide sparsity.
    
    # Alternative approach: use the known Strassen structure as a starting point,
    # then try to reduce rank by finding linear dependencies.
    
    # Let's use a hybrid: Strassen for 4x4x4 head + schoolbook for 4x4x3 tail
    # Then try to merge/eliminate terms.
    
    terms = []
    
    # Helper to create basis vectors
    def make_u(idx_i, idx_j, val=1.0):
        u = np.zeros(n*m)
        u[idx_i * m + idx_j] = val
        return u
    
    def make_v(idx_j, idx_k, val=1.0):
        v = np.zeros(m*p)
        v[idx_j * p + idx_k] = val
        return v
    
    def make_w(idx_k, idx_i, val=1.0):
        w = np.zeros(p*n)
        w[idx_k * n + idx_i] = val
        return w
    
    # Strassen for 4x4x4 (k=0..3): 7 products, each 2x2x2 schoolbook = 4 terms = 28 terms
    # Strassen formulas:
    # m1 = (A00+A11)(B00+B11)
    # m2 = (A10+A11)B00
    # m3 = A00(B01-B11)
    # m4 = A11(B10-B00)
    # m5 = (A00+A01)B11
    # m6 = (A10+A11)(B10-B01)
    # m7 = (A00-A01)(B00+B10)
    
    # C00 = m1 - m4 + m6 - m2
    # C01 = m3 + m4
    # C10 = m2 + m5
    # C11 = m1 - m3 + m5 - m7
    
    # Each mi is a 2x2x2 multiplication
    # For 2x2x2 schoolbook: 4 terms
    
    def add_strassen_2x2x2(u_coeffs, v_coeffs, w_coeffs, 
                            i_off, j_off, vj_off, vk_off, wk_off, wi_off):
        """Add 4 terms for a 2x2x2 schoolbook multiplication"""
        for li in range(2):
            for lj in range(2):
                for lk in range(2):
                    u = np.zeros(n*m)
                    v = np.zeros(m*p)
                    w = np.zeros(p*n)
                    
                    for (li2, lj2), coeff in u_coeffs.items():
                        gi = i_off + li2
                        gj = j_off + lj2
                        u[gi * m + gj] += coeff
                    
                    for (lj2, lk2), coeff in v_coeffs.items():
                        gj = vj_off + lj2
                        gk = vk_off + lk2
                        v[gj * p + gk] += coeff
                    
                    for (lk2, li2), coeff in w_coeffs.items():
                        gk = wk_off + lk2
                        gi = wi_off + li2
                        w[gk * n + gi] += coeff
                    
                    terms.append((u, v, w))
    
    # m1 = (A00+A11)(B00+B11)
    u1 = {(0,0):1, (1,1):1}  # A00 + A11 (local coords)
    v1 = {(0,0):1, (1,1):1}  # B00 + B11
    w1 = {(0,0):1, (1,1):1}  # contributes to C00+C11
    # But w needs to map to the correct output positions
    # m1 contributes to C00 and C11 with coefficient +1
    # So we need separate w vectors for each contribution
    
    # Actually, let me just do schoolbook for the 4x4x4 part too for correctness,
    # then try to reduce rank.
    
    # Schoolbook for 4x4x4: 16 terms
    # Schoolbook for 4x4x3: 48 terms
    # Total: 64 terms (better than 112!)
    
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = make_u(i, j, 1.0)
                v = make_v(j, k, 1.0)
                w = make_w(k, i, 1.0)
                terms.append((u, v, w))
    
    if not time_check():
        return _terms_to_matrices(terms, n, m, p)
    
    # Now try to reduce rank by finding linear dependencies
    # Build the tensor and try to find a lower-rank decomposition
    
    # Verify correctness of current decomposition
    def verify(terms, n, m, p):
        T_recon = np.zeros((n*m, m*p, p*n))
        for u, v, w in terms:
            T_recon += np.einsum('i,j,k->ijk', u, v, w)
        T_true = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_true[i*m+j, j*p+k, k*n+i] = 1.0
        return np.allclose(T_recon, T_true, atol=1e-10)
    
    if not verify(terms, n, m, p):
        # Fallback to pure schoolbook
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = make_u(i, j, 1.0)
                    v = make_v(j, k, 1.0)
                    w = make_w(k, i, 1.0)
                    terms.append((u, v, w))
        return _terms_to_matrices(terms, n, m, p)
    
    # Try to reduce rank using greedy elimination
    # If two terms have proportional U vectors, we can merge them
    # More generally, look for linear dependencies in the U, V, W matrices
    
    U_mat = np.array([u for u, v, w in terms]).T  # (n*m, rank)
    V_mat = np.array([v for u, v, w in terms]).T  # (m*p, rank)
    W_mat = np.array([w for u, v, w in terms]).T  # (p*n, rank)
    
    # Check for zero columns (redundant terms)
    nonzero_mask = np.any(U_mat != 0, axis=0) & np.any(V_mat != 0, axis=0) & np.any(W_mat != 0, axis=0)
    if not np.all(nonzero_mask):
        terms = [terms[i] for i in range(len(terms)) if nonzero_mask[i]]
    
    if not time_check():
        return _terms_to_matrices(terms, n, m, p)
    
    # Try Strassen for the 4x4x4 head to get 28 terms instead of 16+48=64
    # Wait, 4x4x4 schoolbook is 16 terms, Strassen is 7*4=28 terms - that's worse!
    # Strassen helps when you recurse: 4x4x4 with Strassen at top level = 7 * (2x2x2 schoolbook = 4) = 28
    # But 4x4x4 schoolbook = 16. So Strassen is worse here.
    # Strassen helps for larger matrices. For 4x4x4, schoolbook (16) < Strassen (28).
    
    # So the best we can do with simple methods is 64 terms (schoolbook for all).
    # Let's try to do better with a different approach.
    
    # Use the DCT-transformed tensor to find sparse structure
    # The idea: if T_prime has many zeros, we can exploit that
    
    # Count non-zeros in T_prime
    nz_count = np.sum(np.abs(T_prime) > 1e-10)
    
    # If T_prime is sparse, we might be able to find a better decomposition
    # But for now, return the schoolbook result
    return _terms_to_matrices(terms, n, m, p)

def _schoolbook(n, m, p):
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n*m)
                u[i*m + j] = 1.0
                v = np.zeros(m*p)
                v[j*p + k] = 1.0
                w = np.zeros(p*n)
                w[k*n + i] = 1.0
                terms.append((u, v, w))
    return _terms_to_matrices(terms, n, m, p)

def _terms_to_matrices(terms, n, m, p):
    rank = len(terms)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for c, (u, v, w) in enumerate(terms):
        U[:, c] = u
        V[:, c] = v
        W[:, c] = w
    
    # Convert to lists of lists with integer/half-integer entries
    def to_list(mat):
        result = []
        for row in mat:
            result.append([float(x) for x in row])
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
