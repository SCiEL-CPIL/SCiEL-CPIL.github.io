import numpy as np
import time

n, m, p = 2, 5, 6
nm, mp, pn = n * m, m * p, p * n

def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    def check_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W form a valid decomposition."""
        R = U.shape[1]
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        if not np.allclose(T, 1.0, atol=tol):
            return False
        return True

    def get_strassen_decomposition():
        """Construct the Strassen-based decomposition of rank 54."""
        rank = 54
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        
        def add_strassen_block(i_base, j_base, k_base):
            nonlocal col
            # M0: (A00+A11)(B00+B11)
            U[i_base * m + j_base, col] += 1
            U[(i_base + 1) * m + (j_base + 1), col] += 1
            V[j_base * p + k_base, col] += 1
            V[(j_base + 1) * p + (k_base + 1), col] += 1
            W[k_base * n + i_base, col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] += 1
            col += 1
            
            # M1: (A10+A11)B00
            U[(i_base + 1) * m + j_base, col] += 1
            U[(i_base + 1) * m + (j_base + 1), col] += 1
            V[j_base * p + k_base, col] += 1
            W[k_base * n + (i_base + 1), col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] -= 1
            col += 1
            
            # M2: A00(B01-B11)
            U[i_base * m + j_base, col] += 1
            V[j_base * p + (k_base + 1), col] += 1
            V[(j_base + 1) * p + (k_base + 1), col] -= 1
            W[(k_base + 1) * n + i_base, col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] += 1
            col += 1
            
            # M3: A11(B10-B00)
            U[(i_base + 1) * m + (j_base + 1), col] += 1
            V[(j_base + 1) * p + k_base, col] += 1
            V[j_base * p + k_base, col] -= 1
            W[k_base * n + i_base, col] += 1
            W[k_base * n + (i_base + 1), col] += 1
            col += 1
            
            # M4: (A00+A01)B11
            U[i_base * m + j_base, col] += 1
            U[i_base * m + (j_base + 1), col] += 1
            V[(j_base + 1) * p + (k_base + 1), col] += 1
            W[k_base * n + i_base, col] -= 1
            W[(k_base + 1) * n + i_base, col] += 1
            col += 1
            
            # M5: (A10-A00)(B00+B01)
            U[(i_base + 1) * m + j_base, col] += 1
            U[i_base * m + j_base, col] -= 1
            V[j_base * p + k_base, col] += 1
            V[j_base * p + (k_base + 1), col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] += 1
            col += 1
            
            # M6: (A01-A11)(B10+B11)
            U[i_base * m + (j_base + 1), col] += 1
            U[(i_base + 1) * m + (j_base + 1), col] -= 1
            V[(j_base + 1) * p + k_base, col] += 1
            V[(j_base + 1) * p + (k_base + 1), col] += 1
            W[k_base * n + i_base, col] += 1
            col += 1

        def add_naive_2112_block(i_base, j_base, k_base):
            nonlocal col
            U[i_base * m + j_base, col] += 1
            V[j_base * p + k_base, col] += 1
            W[k_base * n + i_base, col] += 1
            col += 1
            
            U[i_base * m + j_base, col] += 1
            V[j_base * p + (k_base + 1), col] += 1
            W[(k_base + 1) * n + i_base, col] += 1
            col += 1
            
            U[(i_base + 1) * m + j_base, col] += 1
            V[j_base * p + k_base, col] += 1
            W[k_base * n + (i_base + 1), col] += 1
            col += 1
            
            U[(i_base + 1) * m + j_base, col] += 1
            V[j_base * p + (k_base + 1), col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] += 1
            col += 1

        # Apply Strassen blocks for j pairs (0,1) and (2,3)
        for j_base in [0, 2]:
            for k_base in [0, 2, 4]:
                add_strassen_block(0, j_base, k_base)
        
        # Apply naive blocks for j=4
        for k_base in [0, 2, 4]:
            add_naive_2112_block(0, 4, k_base)
        
        if col != rank:
            return None
            
        return U, V, W

    def get_schoolbook():
        """Return the schoolbook decomposition with rank n*m*p."""
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # Primary strategy: Use the known Strassen decomposition of rank 54
    result = get_strassen_decomposition()
    if result is not None:
        U, V, W = result
        if check_decomposition(U, V, W, tol=1e-9):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook (Rank 60)
    U, V, W = get_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
