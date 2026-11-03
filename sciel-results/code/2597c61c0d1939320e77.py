import time
import numpy as np

def construct():
    n, m, p = 2, 4, 6
    deadline = time.time() + 135.0

    # Build the full multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def check_decomposition(U, V, W):
        """Check if sum_r U[:,r] * V[:,r] * W[:,r] == T (up to tolerance)"""
        reconstructed = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(reconstructed, T, atol=1e-9)

    def to_lists(U, V, W):
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    def make_rank42():
        """Construct rank-42 decomposition using 6 blocks of 2x2*2x2 Strassen."""
        U = np.zeros((n*m, 42))
        V = np.zeros((m*p, 42))
        W = np.zeros((p*n, 42))
        
        # 6 blocks: (a_rows, a_cols, b_rows, b_cols)
        # a_rows: rows of A (2x4), a_cols: cols of A
        # b_rows: rows of B (4x6), b_cols: cols of B
        blocks = [
            ([0, 1], [0, 1], [0, 1], [0, 1]),
            ([0, 1], [0, 1], [0, 1], [2, 3]),
            ([0, 1], [0, 1], [0, 1], [4, 5]),
            ([0, 1], [2, 3], [2, 3], [0, 1]),
            ([0, 1], [2, 3], [2, 3], [2, 3]),
            ([0, 1], [2, 3], [2, 3], [4, 5]),
        ]
        
        # Strassen 7 terms for 2x2*2x2
        # Each term: (u_coeffs, v_coeffs, w_coeffs)
        # where coeffs are lists of (local_index, coefficient)
        strassen = [
            # C00 = A00*(B00+B11) - (A10+A11)*B10
            ([(0,1)], [(0,1),(3,1)], [(0,1)]),
            # C01 = A01*(B00+B11)
            ([(1,1)], [(0,1),(3,1)], [(1,1)]),
            # C10 = A10*(B00+B11) - (A00+A10)*B11
            ([(2,1)], [(0,1),(3,1)], [(2,1)]),
            # C11 = A11*(B00+B11)
            ([(3,1)], [(0,1),(3,1)], [(3,1)]),
            # C00 += A11*(B10-B11)
            ([(3,1)], [(2,1),(3,-1)], [(0,1)]),
            # C10 += A00*(B11-B10)
            ([(0,1)], [(3,1),(2,-1)], [(2,1)]),
            # C01 += (A00+A11)*B11
            ([(0,1),(3,1)], [(3,1)], [(1,1)]),
        ]
        
        col = 0
        for a_rows, a_cols, b_rows, b_cols in blocks:
            # Map local 2x2 indices to global indices
            # Local: 0=(r0,c0), 1=(r0,c1), 2=(r1,c0), 3=(r1,c1)
            a_map = {0: a_rows[0]*m+a_cols[0], 1: a_rows[0]*m+a_cols[1],
                     2: a_rows[1]*m+a_cols[0], 3: a_rows[1]*m+a_cols[1]}
            b_map = {0: b_rows[0]*p+b_cols[0], 1: b_rows[0]*p+b_cols[1],
                     2: b_rows[1]*p+b_cols[0], 3: b_rows[1]*p+b_cols[1]}
            w_map = {0: b_cols[0]*n+a_rows[0], 1: b_cols[1]*n+a_rows[0],
                     2: b_cols[0]*n+a_rows[1], 3: b_cols[1]*n+a_rows[1]}
            
            for u_c, v_c, w_c in strassen:
                for li, cf in u_c:
                    U[a_map[li], col] += cf
                for li, cf in v_c:
                    V[b_map[li], col] += cf
                for li, cf in w_c:
                    W[w_map[li], col] += cf
                col += 1
        
        return U, V, W

    def make_rank48_schoolbook():
        """Fallback: schoolbook decomposition (rank = n*m*p = 48)."""
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, idx] = 1.0
                    V[j*p+k, idx] = 1.0
                    W[k*n+i, idx] = 1.0
                    idx += 1
        return U, V, W

    # Try rank-42 first
    U, V, W = make_rank42()
    
    if check_decomposition(U, V, W):
        # Try to reduce rank by finding linear dependencies
        rank = U.shape[1]
        
        # Check for zero columns
        for r in range(rank-1, -1, -1):
            if time.time() > deadline - 15:
                break
            if np.allclose(U[:, r], 0) and np.allclose(V[:, r], 0) and np.allclose(W[:, r], 0):
                U = np.delete(U, r, axis=1)
                V = np.delete(V, r, axis=1)
                W = np.delete(W, r, axis=1)
                rank -= 1
        
        # Check for identical columns
        for r in range(rank-1, -1, -1):
            if time.time() > deadline - 15:
                break
            for s in range(r):
                if np.allclose(U[:, r], U[:, s]) and np.allclose(V[:, r], V[:, s]) and np.allclose(W[:, r], W[:, s]):
                    U = np.delete(U, r, axis=1)
                    V = np.delete(V, r, axis=1)
                    W = np.delete(W, r, axis=1)
                    rank -= 1
                    break
        
        if check_decomposition(U, V, W):
            return to_lists(U, V, W)
    
    # Fallback to rank-48
    U, V, W = make_rank48_schoolbook()
    return to_lists(U, V, W)
