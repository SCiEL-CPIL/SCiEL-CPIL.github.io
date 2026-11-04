import time
import numpy as np

START = time.time()
DEADLINE = START + 115.0  # Safety margin before 120s

def _strassen_2x2x2():
    """Strassen 2x2x2 decomposition, rank 7."""
    U = np.zeros((4, 7), dtype=np.int16)
    V = np.zeros((4, 7), dtype=np.int16)
    W = np.zeros((4, 7), dtype=np.int16)
    # M1 = (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2 = (A10+A11)B00
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
    # M3 = A00(B01-B11)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
    # M4 = A11(B10-B00)
    U[3,3]=1; V[0,3]=-1; V[2,3]=1; W[0,3]=1; W[1,3]=1
    # M5 = (A00+A01)B11
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    # M6 = (A10-A00)(B00+B01)
    U[0,5]=-1; U[2,5]=1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7 = (A01-A11)(B10+B11)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _build_4x4x4_tensors():
    """Build 4x4x4 via Kronecker product of two 2x2x2 Strassen."""
    U2, V2, W2 = _strassen_2x2x2()
    rank = 49
    U4 = np.zeros((16, rank), dtype=np.int16)
    V4 = np.zeros((16, rank), dtype=np.int16)
    W4 = np.zeros((16, rank), dtype=np.int16)
    for c0 in range(7):
        for c1 in range(7):
            c = c0*7 + c1
            for i in range(4):
                for j in range(4):
                    U4[i*4+j, c] = U2[(i//2)*2+(j//2), c0] * U2[(i%2)*2+(j%2), c1]
            for j in range(4):
                for k in range(4):
                    V4[j*4+k, c] = V2[(j//2)*2+(k//2), c0] * V2[(j%2)*2+(k%2), c1]
            for k in range(4):
                for i in range(4):
                    W4[k*4+i, c] = W2[(k//2)*2+(i//2), c0] * W2[(k%2)*2+(i//2), c1]
    return U4, V4, W4

def _build_rank98():
    """
    Construct rank-98 decomposition for 4x4 * 4x8 matrix multiplication.
    """
    n, m, p = 4, 4, 8
    U4, V4, W4 = _build_4x4x4_tensors()
    
    total_rank = 98
    U = np.zeros((n * m, total_rank), dtype=np.int16)
    V = np.zeros((m * p, total_rank), dtype=np.int16)
    W = np.zeros((p * n, total_rank), dtype=np.int16)
    
    # Block 0: k in [0, 4)
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*8 + k, 0:49] = V4[j*4 + k, :]
    for k in range(4):
        for i in range(4):
            W[k*4 + i, 0:49] = W4[k*4 + i, :]
            
    # Block 1: k in [4, 8)
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4, 8):
            V[j*8 + k, 49:98] = V4[j*4 + (k-4), :]
    for k in range(4, 8):
        for i in range(4):
            W[k*4 + i, 49:98] = W4[(k-4)*4 + i, :]
            
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = U.shape[1]
    # Check shapes
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    # Verify: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 if i,j,k consistent, else 0
    # Actually the tensor T[i,j,k] = 1 for all valid i,j,k in matrix mult
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] should be 1 for all i,j,k
    Uf = U.astype(np.float64)
    Vf = V.astype(np.float64)
    Wf = W.astype(np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(Uf[i*m+j, :] * Vf[j*p+k, :] * Wf[k*n+i, :])
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, max_iter=200):
    """Try to reduce rank by finding and eliminating linear dependencies."""
    rank = U.shape[1]
    Uf = U.astype(np.float64).copy()
    Vf = V.astype(np.float64).copy()
    Wf = W.astype(np.float64).copy()
    
    for iteration in range(max_iter):
        if time.time() > DEADLINE - 5:
            break
        # Try to find a column that can be eliminated
        # A column c can be eliminated if there exist columns c1..cr such that
        # U[:,c] = sum a_i U[:,c_i], V[:,c] = sum a_i V[:,c_i], W[:,c] = sum a_i W[:,c_i]
        # This is a system of equations. Let's try a simpler approach:
        # Check if any column is a linear combination of others in all three matrices simultaneously.
        
        # For each pair of columns, check if one is redundant
        found_reduction = False
        for c in range(rank):
            if time.time() > DEADLINE - 5:
                break
            # Try to express column c as combination of others
            # Set up: find a such that U[:,c] = U[:,others] @ a
            # and V[:,c] = V[:,others] @ a  
            # and W[:,c] = W[:,others] @ a
            others = [i for i in range(rank) if i != c]
            if len(others) == 0:
                continue
            
            Uo = Uf[:, others]
            Vo = Vf[:, others]
            Wo = Wf[:, others]
            Uc = Uf[:, c]
            Vc = Vf[:, c]
            Wc = Wf[:, c]
            
            # Solve least squares for a from U
            a_U, _, _, _ = np.linalg.lstsq(Uo, Uc, rcond=None)
            # Check if this a also works for V and W
            V_pred = Vo @ a_U
            W_pred = Wo @ a_U
            if np.allclose(V_pred, Vc, atol=1e-10) and np.allclose(W_pred, Wc, atol=1e-10):
                # Remove column c
                Uf = np.delete(Uf, c, axis=1)
                Vf = np.delete(Vf, c, axis=1)
                Wf = np.delete(Wf, c, axis=1)
                rank -= 1
                found_reduction = True
                break
        
        if not found_reduction:
            break
    
    return Uf, Vf, Wf, rank

def construct():
    n, m, p = 4, 4, 8
    
    # Construct the known rank-98 solution
    U, V, W = _build_rank98()
    
    # Try to reduce rank
    Uf, Vf, Wf, new_rank = _try_reduce_rank(U, V, W, n, m, p, max_iter=100)
    
    if new_rank < 98:
        # Convert back to integer/half-integer if possible
        # For now, return as float if reduction was found
        U_out = Uf
        V_out = Vf
        W_out = Wf
    else:
        U_out = U
        V_out = V
        W_out = W
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
