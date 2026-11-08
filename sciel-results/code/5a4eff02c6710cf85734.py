import time
import numpy as np
from scipy.linalg import svd

DIMENSIONS = [4, 4, 8]

def _strassen_2x2():
    """Strassen 2x2x2: U(4,7), V(4,7), W(4,7) with trace convention W[k*2+i]."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # M1=(A11-A10)(B00+B01)
    U[3][0]=1; U[2][0]=-1; V[0][0]=1; V[1][0]=1; W[0][0]=1; W[2][0]=-1; W[3][0]=1
    # M2=(A10+A11)(B10+B11)
    U[2][1]=1; U[3][1]=1; V[2][1]=1; V[3][1]=1; W[2][1]=1; W[3][1]=1
    # M3=(A00+A10)(B00)
    U[0][2]=1; U[2][2]=1; V[0][2]=1; W[2][2]=1; W[3][2]=1
    # M4=(A11+A01)(B11)
    U[3][3]=1; U[1][3]=1; V[3][3]=1; W[0][3]=1; W[2][3]=-1
    # M5=(A00-A10)(B01+B11)
    U[0][4]=1; U[2][4]=-1; V[1][4]=1; V[3][4]=1; W[0][4]=-1; W[2][4]=1
    # M6=(A01+A11)(B10-B00)
    U[1][5]=1; U[3][5]=1; V[2][5]=1; V[0][5]=-1; W[2][5]=-1; W[3][5]=-1
    # M7=(A00+A01)(B00-B10)
    U[0][6]=1; U[1][6]=1; V[0][6]=1; V[2][6]=-1; W[0][6]=1; W[2][6]=-1
    return U, V, W

def _build_strassen_4x4x4():
    """Build 4x4x4 decomposition using Strassen Kronecker (rank 49)."""
    U2, V2, W2 = _strassen_2x2()
    r2 = 7
    r4 = r2 * r2  # 49
    U4 = [[0]*r4 for _ in range(16)]
    V4 = [[0]*r4 for _ in range(16)]
    W4 = [[0]*r4 for _ in range(16)]
    
    for rb in range(r2):
        for rs in range(r2):
            col = rb * r2 + rs
            # U factor
            for ib in range(2):
                for i0 in range(2):
                    for jb in range(2):
                        for j0 in range(2):
                            a = ib * 8 + i0 * 4 + jb * 2 + j0
                            ab = ib * 2 + jb
                            as_idx = i0 * 2 + j0
                            U4[a][col] = U2[ab][rb] * U2[as_idx][rs]
            # V factor
            for jb2 in range(2):
                for j02 in range(2):
                    for kb in range(2):
                        for k0 in range(2):
                            b = jb2 * 8 + j02 * 4 + kb * 2 + k0
                            bb = jb2 * 2 + kb
                            bs = j02 * 2 + k0
                            V4[b][col] = V2[bb][rb] * V2[bs][rs]
            # W factor
            for kb2 in range(2):
                for k02 in range(2):
                    for ib2 in range(2):
                        for i02 in range(2):
                            c = kb2 * 8 + k02 * 4 + ib2 * 2 + i02
                            cb = kb2 * 2 + ib2
                            cs = k02 * 2 + i02
                            W4[c][col] = W2[cb][rb] * W2[cs][rs]
    return U4, V4, W4

def _build_parallel_4x4x8():
    """
    Construct 4x4x8 as two parallel 4x4x4 multiplications.
    Rank = 49 + 49 = 98.
    U: (16, 98)
    V: (32, 98)
    W: (32, 98)
    """
    U4, V4, W4 = _build_strassen_4x4x4()
    r4 = 49
    r8 = 2 * r4
    
    U8 = [[0]*r8 for _ in range(16)]
    V8 = [[0]*r8 for _ in range(32)]
    W8 = [[0]*r8 for _ in range(32)]
    
    # Block 1: Columns 0-3 of B (indices 0-15 in V), Columns 0-3 of W (indices 0-15)
    # Block 2: Columns 4-7 of B (indices 16-31 in V), Columns 4-7 of W (indices 16-31)
    
    for r in range(r4):
        for i in range(16):
            # U is shared across both blocks (A is the same matrix)
            U8[i][r] = U4[i][r]
            U8[i][r + r4] = U4[i][r]
            
        for j in range(4):
            for k in range(4):
                # V index for block 1: j*4 + k
                # V index for block 2: 16 + j*4 + k
                V8[j * 4 + k][r] = V4[j * 4 + k][r]
                V8[16 + j * 4 + k][r + r4] = V4[j * 4 + k][r]
                
        for k in range(4):
            for i in range(4):
                # W index for block 1: k*4 + i
                # W index for block 2: 16 + k*4 + i
                W8[k * 4 + i][r] = W4[k * 4 + i][r]
                W8[16 + k * 4 + i][r + r4] = W4[k * 4 + i][r]
                
    return U8, V8, W8

def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify tensor decomposition: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] = delta."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for k2 in range(p):
                            a = i * m + j
                            b = j2 * p + k
                            c = k2 * n + i2
                            expected = 1.0 if (j == j2 and i == i2 and k == k2) else 0.0
                            actual = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                            if abs(actual - expected) > 1e-9:
                                return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, current_rank, time_budget=10):
    """
    Attempt to reduce rank by finding linear dependencies in the tensor decomposition.
    This is a heuristic: we try to project the factor matrices onto a lower-dimensional subspace
    that still preserves the tensor structure.
    
    Note: This is a complex problem. A simpler approach is to check if any columns are linearly
    dependent across the three matrices simultaneously.
    """
    start = time.time()
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check for obvious linear dependencies in individual matrices
    # If a column in U is zero, we can drop it (if V and W columns are also zero or compatible)
    # This is a very basic reduction.
    
    # A more sophisticated method would be to use the "tensor rank reduction" algorithms,
    # but they are computationally expensive and complex.
    
    # For now, we just return the original if no simple reduction is found.
    # In a real implementation, we might try to merge columns that are identical up to scaling.
    
    # Check for identical columns in U, V, W
    # This is O(R^2 * N) which is 98^2 * 16 ~ 150k ops, fast enough.
    
    # Let's just return None to indicate no reduction found for now,
    # as implementing a robust rank reduction is beyond the scope of a quick heuristic.
    
    return None

def schoolbook_decomposition(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook (rank 128)
    best_result = schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try parallel 4x4x4 construction (rank 98)
    try:
        U8, V8, W8 = _build_parallel_4x4x8()
        if U8 is not None:
            rank = len(U8[0])
            # Verify correctness
            if _verify_decomposition(U8, V8, W8, n, m, p, rank):
                if rank < best_rank:
                    best_result = {"U": U8, "V": V8, "W": U8} # Wait, W8
                    best_result = {"U": U8, "V": V8, "W": W8}
                    best_rank = rank
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_result
    
    # Try rank reduction (heuristic)
    try:
        U_red, V_red, W_red = _try_reduce_rank(
            best_result['U'], best_result['V'], best_result['W'],
            n, m, p, best_rank, max_time=10
        )
        if U_red is not None:
            rank = len(U_red[0])
            if rank < best_rank:
                if _verify_decomposition(U_red, V_red, W_red, n, m, p, rank):
                    best_result = {"U": U_red, "V": V_red, "W": W_red}
                    best_rank = rank
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_result
    
    return best_result

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
