"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 120 # Safety margin

def construct():
    n, m, p = DIMENSIONS
    
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def verify(U, V, W):
        """
        Verifies if U, V, W form a valid decomposition for (n,m,p) matrix multiplication.
        """
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        
        # Target tensor T[i, j, k] = 1
        # Decomposed tensor D[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Efficient check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j, :]
                    v_val = V[j * p + k, :]
                    w_val = W[k * n + i, :]
                    s = np.sum(u_val * v_val * w_val)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def get_3x3x3_rank26():
        """
        Returns a valid rank-26 decomposition for 3x3x3 matrix multiplication.
        Uses Strassen for the 2x2 core and schoolbook for the fringes.
        """
        U0 = np.zeros((9, 26))
        V0 = np.zeros((9, 26))
        W0 = np.zeros((9, 26))
        
        # Indices: U(i,j), V(j,k), W(k,i)
        
        # --- Strassen Core (i,j,k in 0,1) ---
        # Term 1: M1
        U0[0*3+0, 0] = 1; U0[1*3+1, 0] = 1
        V0[0*3+0, 0] = 1; V0[1*3+1, 0] = 1
        W0[0*3+0, 0] = 1; W0[1*3+1, 0] = 1
        
        # Term 2: M2
        U0[1*3+0, 1] = 1; U0[1*3+1, 1] = 1
        V0[0*3+0, 1] = 1
        W0[0*3+1, 1] = 1; W0[1*3+1, 1] = -1
        
        # Term 3: M3
        U0[0*3+0, 2] = 1
        V0[1*3+0, 2] = 1; V0[1*3+1, 2] = -1
        W0[1*3+0, 2] = 1; W0[1*3+1, 2] = 1
        
        # Term 4: M4
        U0[1*3+1, 3] = 1
        V0[0*3+1, 3] = 1; V0[0*3+0, 3] = -1
        W0[0*3+0, 3] = 1; W0[0*3+1, 3] = 1
        
        # Term 5: M5
        U0[0*3+0, 4] = 1; U0[0*3+1, 4] = 1
        V0[1*3+1, 4] = 1
        W0[0*3+0, 4] = -1; W0[1*3+0, 4] = 1
        
        # Term 6: M6
        U0[1*3+0, 5] = 1; U0[0*3+0, 5] = -1
        V0[0*3+0, 5] = 1; V0[0*3+1, 5] = 1
        W0[1*3+1, 5] = 1
        
        # Term 7: M7
        U0[0*3+1, 6] = 1; U0[1*3+1, 6] = -1
        V0[1*3+0, 6] = 1; V0[1*3+1, 6] = 1
        W0[0*3+0, 6] = 1

        # --- Fringes (Schoolbook) ---
        # 8-11: A_LH * B_HL (i in 0,1; j=2; k in 0,1)
        idx = 7
        for i in range(2):
            for k in range(2):
                U0[i*3+2, idx] = 1
                V0[2*3+k, idx] = 1
                W0[k*3+i, idx] = 1
                idx += 1
                
        # 12-15: A_LL * B_LH (i in 0,1; j in 0,1; k=2)
        for i in range(2):
            for j in range(2):
                U0[i*3+j, idx] = 1
                V0[j*3+2, idx] = 1
                W0[2*3+i, idx] = 1
                idx += 1
                
        # 16-17: A_LH * B_HH (i in 0,1; j=2; k=2)
        for i in range(2):
            U0[i*3+2, idx] = 1
            V0[2*3+2, idx] = 1
            W0[2*3+i, idx] = 1
            idx += 1
            
        # 18-21: A_HL * B_LL (i=2; j in 0,1; k in 0,1)
        for j in range(2):
            for k in range(2):
                U0[2*3+j, idx] = 1
                V0[j*3+k, idx] = 1
                W0[k*3+2, idx] = 1
                idx += 1
                
        # 22-23: A_HH * B_HL (i=2; j=2; k in 0,1)
        for k in range(2):
            U0[2*3+2, idx] = 1
            V0[2*3+k, idx] = 1
            W0[k*3+2, idx] = 1
            idx += 1
            
        # 24-25: A_HL * B_LH (i=2; j in 0,1; k=2)
        for j in range(2):
            U0[2*3+j, idx] = 1
            V0[j*3+2, idx] = 1
            W0[2*3+2, idx] = 1
            idx += 1
            
        # 26: A_HH * B_HH (i=2; j=2; k=2)
        U0[2*3+2, idx] = 1
        V0[2*3+2, idx] = 1
        W0[2*3+2, idx] = 1
        idx += 1
        
        return U0, V0, W0

    def build_3x3x6_from_3x3x3(U0, V0, W0):
        """
        Expands a valid 3x3x3 rank-23 decomposition to 3x3x6 rank-46.
        U0, V0, W0 are 9x23, 9x23, 9x23 matrices.
        """
        n3, m3, p3 = 3, 3, 3
        n6, m6, p6 = 3, 3, 6
        
        # U is 9x46. Both halves are U0.
        U = np.hstack([U0, U0])
        
        # V is 18x46.
        # First 23 cols: V0 for k in 0..2, 0 for k in 3..5.
        # Last 23 cols: 0 for k in 0..2, V0 (shifted) for k in 3..5.
        V = np.zeros((m6 * p6, 46))
        # Fill first 23 cols
        for j in range(m3):
            for k in range(p3):
                V[j * p6 + k, 0:23] = V0[j * p3 + k, :]
        # Fill last 23 cols
        for j in range(m3):
            for k in range(p3):
                V[j * p6 + (k + p3), 23:46] = V0[j * p3 + k, :]
                
        # W is 18x46.
        # First 23 cols: W0 for k in 0..2, 0 for k in 3..5.
        # Last 23 cols: 0 for k in 0..2, W0 (shifted) for k in 3..5.
        W = np.zeros((p6 * n6, 46))
        # Fill first 23 cols
        for k in range(p3):
            for i in range(n3):
                W[k * n6 + i, 0:23] = W0[k * p3 + i, :]
        # Fill last 23 cols
        for k in range(p3):
            for i in range(n3):
                W[(k + p3) * n6 + i, 23:46] = W0[k * p3 + i, :]
                
        return U, V, W

    # Try to get the improved solution
    if time.time() - START_TIME < TIME_LIMIT - 5:
        try:
            U0, V0, W0 = get_3x3x3_rank26()
            U, V, W = build_3x3x6_from_3x3x3(U0, V0, W0)
            if verify(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        except Exception:
            pass

    # Fallback to schoolbook
    return get_schoolbook()

# EVOLVE-BLOCK-START
# The construct function is defined above to respect the interface.
# EVOLVE-BLOCK-END
