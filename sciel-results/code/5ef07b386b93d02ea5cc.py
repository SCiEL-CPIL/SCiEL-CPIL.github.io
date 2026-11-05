"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Rank-19 decomposition for 2x3 x 3x4 matrix multiplication.
    Uses iterative alternating least squares to find a valid decomposition.
    """
    n, m, p = DIMENSIONS  # 2, 3, 4
    nm = n * m  # 6
    mp = m * p  # 12
    pn = p * n  # 8
    
    # Build target tensor T_flat: shape (nm*mp, pn)
    # T_flat[(a,b),(c,d) -> x, (k,i) -> y] = 1 if a=i, b=c, d=k else 0
    T_flat = np.zeros((nm * mp, pn), dtype=float)
    for i in range(n):
        for k in range(p):
            for j in range(m):
                x = (i * m + j) * mp + (j * p + k)
                y = k * n + i
                T_flat[x, y] = 1.0
    
    R = 19
    np.random.seed(123)
    
    # Initialize
    U = np.random.randn(nm, R)
    V = np.random.randn(mp, R)
    W = np.random.randn(pn, R)
    
    for iteration in range(500):
        # Fix U, V: solve for W
        # T_flat[x,y] = sum_r U[x//mp, r] * V[x%mp, r] * W[y, r]
        A = np.zeros((nm * mp, R))
        for x in range(nm * mp):
            a = x // mp
            b = x % mp
            A[x, :] = U[a, :] * V[b, :]
        W = np.linalg.lstsq(A, T_flat, rcond=None)[0].T
        
        # Fix V, W: solve for U
        # T_flat[x,y] = sum_r U[x//mp, r] * V[x%mp, r] * W[y, r]
        # For fixed a: T_flat[a*mp:(a+1)*mp, y] = sum_r U[a,r] * (V[:,r] * W[y,r])
        # Let M[b,y] = sum_r V[b,r] * W[y,r] = (V @ W.T)[b,y]
        VW = V @ W.T
        U_new = np.zeros((nm, R))
        for a in range(nm):
            block = T_flat[a*mp:(a+1)*mp, :]  # (mp, pn)
            # block = U[a,:] @ VW  => U[a,:] = lstsq(VW.T, block.T)
            U[a, :] = np.linalg.lstsq(VW.T, block.T, rcond=None)[0]
        U = U_new
        
        # Fix U, W: solve for V
        # T_flat[a*mp+b, y] = sum_r U[a,r] * V[b,r] * W[y,r]
        # For fixed b: T_flat[a*mp+b, y] for all a,y = sum_r V[b,r] * (U[a,r]*W[y,r])
        # C[(a*pn+y), r] = U[a,r] * W[y,r]
        C_mat = np.zeros((nm * pn, R))
        for r in range(R):
            C_mat[:, r] = np.outer(U[:, r], W[:, r]).flatten()
        
        V_new = np.zeros((mp, R))
        for b in range(mp):
            T_b = np.zeros(nm * pn)
            for a in range(nm):
                for y in range(pn):
                    T_b[a * pn + y] = T_flat[a * mp + b, y]
            V[b, :] = np.linalg.lstsq(C_mat, T_b, rcond=None)[0]
        V = V_new
    
    # Verify
    max_err = 0.0
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for i in range(n):
                        for k in range(p):
                            x = (a * m + b) * mp + (c * p + d)
                            y = k * n + i
                            val = np.sum(U[a*m+b, :] * V[c*p+d, :] * W[k*n+i, :])
                            expected = 1.0 if (a == i and b == c and d == k) else 0.0
                            max_err = max(max_err, abs(val - expected))
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    def round_matrix(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                if abs(val - round(val)) < 1e-4:
                    new_row.append(round(val))
                elif abs(val * 2 - round(val * 2)) < 1e-4:
                    new_row.append(round(val * 2) / 2)
                else:
                    new_row.append(val)
            result.append(new_row)
        return result
    
    return {"U": round_matrix(U_list), "V": round_matrix(V_list), "W": round_matrix(W_list)}
# EVOLVE-BLOCK-END
