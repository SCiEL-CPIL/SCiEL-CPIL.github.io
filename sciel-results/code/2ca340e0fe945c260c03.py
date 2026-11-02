"""Matrix multiplication 2x3x3 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]

def _verify_rank17():
    """Verify the rank-17 Strassen+schoolbook decomposition is correct."""
    n, m, p = 2, 3, 3
    # Build U, V, W for rank 17
    rank = 17
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # m1: (a00+a11)*(b00+b11)
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[3][0] = 1
    
    # m2: (a10+a11)*b00
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # m3: a00*(b11-b00)
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # m4: a11*(b10-b00)
    U[4][3] = 1
    V[3][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # m5: (a00+a01)*b11
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # m6: (a10-a00)*(b00+b01)
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # m7: (a01-a11)*(b10+b11)
    U[1][6] = 1; U[4][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[0][6] = 1
    
    # m8: a00*b02
    U[0][7] = 1
    V[2][7] = 1
    W[4][7] = 1
    
    # m9: a01*b12
    U[1][8] = 1
    V[5][8] = 1
    W[4][8] = 1
    
    # m10: a10*b02
    U[3][9] = 1
    V[2][9] = 1
    W[5][9] = 1
    
    # m11: a11*b12
    U[4][10] = 1
    V[5][10] = 1
    W[5][10] = 1
    
    # m12: a02*b20
    U[2][11] = 1
    V[6][11] = 1
    W[0][11] = 1
    
    # m13: a12*b20
    U[5][12] = 1
    V[6][12] = 1
    W[1][12] = 1
    
    # m14: a02*b21
    U[2][13] = 1
    V[7][13] = 1
    W[2][13] = 1
    
    # m15: a12*b21
    U[5][14] = 1
    V[7][14] = 1
    W[3][14] = 1
    
    # m16: a02*b22
    U[2][15] = 1
    V[8][15] = 1
    W[4][15] = 1
    
    # m17: a12*b22
    U[5][16] = 1
    V[8][16] = 1
    W[5][16] = 1
    
    # Verify: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should be 1 if (i,j,k) is in the support (all indices where a_{i,j} * b_{j,k} contributes to c_{k,i})
    # Actually, the tensor is T[i,j,k] = 1 for all valid (i,j,k) since it's just the multiplication tensor
    # Wait, no. The tensor T_{i,j,k} has T_{i,j,k}=1 if and only if the product a_{i,j} * b_{j,k} contributes to c_{k,i}.
    # In standard matrix multiplication C = A * B where A is n x m and B is m x p, C is n x p.
    # C_{i,k} = sum_j A_{i,j} * B_{j,k}
    # So the tensor T_{i,j,k} = 1 for all (i,j,k) in the valid range.
    # Actually no, T_{i,j,k} is the trilinear form: T(A,B,C) = sum_{i,j,k} T_{i,j,k} * A_{i,j} * B_{j,k} * C_{k,i}
    # For matrix multiplication, T_{i,j,k} = 1 for all valid (i,j,k).
    
    # Let me just verify by computing the product and checking it matches standard multiplication.
    # Actually, let me just verify the tensor decomposition by checking all (i,j,k).
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # For the multiplication tensor, T[i,j,k] should be 1
                if abs(s - 1.0) > 1e-10:
                    return False, U, V, W
    return True, U, V, W


def _build_schoolbook():
    """Build a rank-18 schoolbook decomposition (definitely correct)."""
    n, m, p = 2, 3, 3
    rank = n * m * p  # 18
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, r] = 1
                V[j*p+k, r] = 1
                W[k*n+i, r] = 1
                r += 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Try rank-17 first
    try:
        valid, U, V, W = _verify_rank17()
        if valid:
            # Convert to list of lists for the expected interface
            U_list = [[int(round(x)) if abs(x - round(x)) < 1e-10 else float(x) for x in row] for row in U]
            V_list = [[int(round(x)) if abs(x - round(x)) < 1e-10 else float(x) for x in row] for row in V]
            W_list = [[int(round(x)) if abs(x - round(x)) < 1e-10 else float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    except Exception:
        pass
    
    # Fallback to schoolbook rank 18
    try:
        if time.time() > deadline - 5.0:
            # Emergency: return a simple valid result
            U, V, W = _build_schoolbook()
            U_list = [[int(round(x)) if abs(x - round(x)) < 1e-10 else float(x) for x in row] for row in U]
            V_list = [[int(round(x)) if abs(x - round(x)) < 1e-10 else float(x) for x in row] for row in V]
            W_list = [[int(round(x)) if abs(x - round(x)) < 1e-10 else float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
        
        U, V, W = _build_schoolbook()
        U_list = [[int(round(x)) if abs(x - round(x)) < 1e-10 else float(x) for x in row] for row in U]
        V_list = [[int(round(x)) if abs(x - round(x)) < 1e-10 else float(x) for x in row] for row in V]
        W_list = [[int(round(x)) if abs(x - round(x)) < 1e-10 else float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    except Exception:
        # Last resort: minimal valid structure
        n, m, p = 2, 3, 3
        rank = 18
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
