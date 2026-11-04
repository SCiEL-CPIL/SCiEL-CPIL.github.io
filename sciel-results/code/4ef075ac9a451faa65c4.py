import time
import numpy as np

START = time.time()
DEADLINE = START + 130.0

def _strassen_2x2x2():
    """Strassen 2x2x2 decomposition, rank 7."""
    U = np.zeros((4, 7), dtype=np.int16)
    V = np.zeros((4, 7), dtype=np.int16)
    W = np.zeros((4, 7), dtype=np.int16)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
    U[3,3]=1; V[0,3]=-1; V[2,3]=1; W[0,3]=1; W[1,3]=1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    U[0,5]=-1; U[2,5]=1; V[0,5]=1; V[1,5]=1; W[3,5]=1
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
    """Standard rank-98 decomposition: two independent 4x4x4 blocks."""
    U4, V4, W4 = _build_4x4x4_tensors()
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)
    
    for block in range(2):
        col_start = block * 49
        col_end = (block + 1) * 49
        U[:, col_start:col_end] = U4
        for j in range(4):
            for k_local in range(4):
                k_global = block * 4 + k_local
                V[j*8 + k_global, col_start:col_end] = V4[j*4 + k_local, :]
        for k_local in range(4):
            k_global = block * 4 + k_local
            for i in range(4):
                W[k_global*4 + i, col_start:col_end] = W4[k_local*4 + i, :]
    
    # Verify
    n, m, p = 4, 4, 8
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(total_rank):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(total - 1.0) > 1e-9:
                    return None
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook(n=4, m=4, p=8):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank_reduction(U, V, W, n=4, m=4, p=8, target_rank=80, max_iters=200):
    """Try to reduce rank by zeroing out columns and adjusting others."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    # Build the tensor from current decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want to try removing columns one at a time
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for _iter in range(max_iters):
        if time.time() > DEADLINE - 10:
            break
        
        # Try removing each column
        removed = False
        for col in range(rank):
            # Temporarily remove column
            U_new = np.delete(U, col, axis=1)
            V_new = np.delete(V, col, axis=1)
            W_new = np.delete(W, col, axis=1)
            
            # Check if the tensor is still satisfied
            # For each (i,j,k), check if sum over remaining cols equals 1
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank - 1):
                            s += U_new[i*m+j, r] * V_new[j*p+k, r] * W_new[k*n+i, r]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                U, V, W = U_new, V_new, W_new
                rank -= 1
                removed = True
                if rank <= target_rank:
                    break
                continue
        
        if not removed:
            break
    
    if rank < best_rank:
        best_rank = rank
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Convert back to integer/half-integer if possible
    # Check if all entries are close to integers or half-integers
    all_valid = True
    for arr in [best_U, best_V, best_W]:
        for val in arr.flatten():
            # Check if val is close to an integer or half-integer
            nearest_int = round(val * 2) / 2.0
            if abs(val - nearest_int) > 1e-6:
                all_valid = False
                break
        if not all_valid:
            break
    
    if all_valid:
        U_out = np.round(best_U * 2) / 2.0
        V_out = np.round(best_V * 2) / 2.0
        W_out = np.round(best_W * 2) / 2.0
        
        # Final verification
        n, m, p = 4, 4, 8
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(best_rank):
                        s += U_out[i*m+j, r] * V_out[j*p+k, r] * W_out[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return None
        
        return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
    
    return None

def construct():
    # Try rank-98
    result_98 = None
    if time.time() < DEADLINE - 5:
        try:
            result_98 = _build_rank98()
            if result_98 is not None:
                pass
        except Exception:
            pass
    
    # Try to reduce rank from 98
    if result_98 is not None and time.time() < DEADLINE - 20:
        try:
            U98 = np.array(result_98['U'], dtype=np.float64)
            V98 = np.array(result_98['V'], dtype=np.float64)
            W98 = np.array(result_98['W'], dtype=np.float64)
            reduced = _try_rank_reduction(U98, V98, W98, target_rank=80, max_iters=100)
            if reduced is not None:
                return reduced
        except Exception:
            pass
    
    # Return rank-98 if available
    if result_98 is not None:
        return result_98
    
    # Final fallback: schoolbook
    return _schoolbook()

if __name__ == "__main__":
    res = construct()
    rank = len(res['U'][0])
    print(f"Rank: {rank}")
    n, m, p = 4, 4, 8
    U, V, W = res['U'], res['V'], res['W']
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    print(f"Valid: {valid}")
