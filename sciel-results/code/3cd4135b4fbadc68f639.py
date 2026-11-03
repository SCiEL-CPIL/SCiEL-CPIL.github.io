import time
import numpy as np

def _verify_fast(U, V, W):
    """Fast verification using numpy."""
    n, m, p = 2, 3, 8
    R = len(U[0]) if U else 0
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build target tensor
    # T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(k,i) with same (i,j,k)
    for a in range(n * m):
        i = a // m
        j = a % m
        for b in range(m * p):
            j2 = b // p
            k = b % p
            for c in range(p * n):
                i2 = c % n
                k2 = c // n
                expected = 1 if (i == i2 and j == j2 and k == k2) else 0
                
                val = np.sum(U_np[a, :] * V_np[b, :] * W_np[c, :])
                
                if abs(val - expected) > 1e-9:
                    return False
    return True

def _build_rank_44():
    """Build the known rank-44 decomposition."""
    n, m, p = 2, 3, 8
    dim_U = n * m  # 6
    dim_V = m * p  # 24
    dim_W = p * n  # 16
    rank = 44
    
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]
    
    col = 0
    
    # Part 1: 2x1x8 part (j=0)
    for k in range(p):
        U[0 * m + 0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        U[1 * m + 0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1
    
    # Part 2: 2x2x8 part (j=1,2)
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # P1: (A[0,1] + A[1,2]) * (B[1,k0] + B[2,k1])
        U[0 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # P2: (A[1,1] + A[1,2]) * B[1,k0]
        U[1 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1
        
        # P3: A[0,1] * (B[1,k1] - B[2,k1])
        U[0 * m + 1][col] = 1
        V[1 * p + k1][col] = 1
        V[2 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # P4: A[1,2] * (B[2,k0] - B[1,k0])
        U[1 * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        V[1 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # P5: (A[0,1] + A[0,2]) * B[2,k1]
        U[0 * m + 1][col] = 1
        U[0 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # P6: (A[1,1] - A[0,1]) * (B[1,k0] + B[1,k1])
        U[1 * m + 1][col] = 1
        U[0 * m + 1][col] = -1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # P7: (A[0,2] - A[1,2]) * (B[2,k0] + B[2,k1])
        U[0 * m + 2][col] = 1
        U[1 * m + 2][col] = -1
        V[2 * p + k0][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}

def _try_rank_reduction(base_solution, target_rank, deadline):
    """Try to reduce the rank of the solution by zeroing out columns."""
    n, m, p = 2, 3, 8
    U = np.array(base_solution["U"], dtype=float)
    V = np.array(base_solution["V"], dtype=float)
    W = np.array(base_solution["W"], dtype=float)
    
    current_rank = len(U[0])
    
    # We will try to remove columns one by one
    # This is a heuristic approach
    for r in range(current_rank - 1, target_rank - 1, -1):
        if time.time() > deadline:
            break
        
        # Try removing column r
        U_new = np.delete(U, r, axis=1)
        V_new = np.delete(V, r, axis=1)
        W_new = np.delete(W, r, axis=1)
        
        if _verify_fast(U_new.tolist(), V_new.tolist(), W_new.tolist()):
            U = U_new
            V = V_new
            W = W_new
            current_rank = r
        else:
            # Try to adjust other columns to compensate? Too complex.
            pass
            
    if current_rank < len(base_solution["U"][0]):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

def construct():
    n, m, p = 2, 3, 8
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill
    
    # Start with known rank-44 solution
    result = _build_rank_44()
    
    # Try to find a better (lower rank) solution
    if time.time() < deadline - 10:
        better = _try_rank_reduction(result, 32, deadline)
        if better is not None:
            if _verify_fast(better["U"], better["V"], better["W"]):
                result = better
    
    return result

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
    if _verify_fast(res["U"], res["V"], res["W"]):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
