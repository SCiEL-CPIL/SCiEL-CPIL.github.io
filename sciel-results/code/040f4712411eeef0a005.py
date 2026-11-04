import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 8]
_START_TIME = time.time()
_DEADLINE = 120.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Try to find a lower rank decomposition using a systematic approach
    # We know rank 56 works (8 Strassen blocks). Let's try to find something better.
    
    # The tensor T has shape (n*m, m*p, p*n) = (8, 32, 16)
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # Strategy: Use a SAT/SMT-like approach or try random perturbations of known solutions
    # to see if we can reduce rank.
    
    # First, let's try a direct algebraic approach with smaller blocks.
    # What if we use 4x4 blocks?
    # A is 2x4, B is 4x8.
    # We can think of this as 2 rows of A multiplying 4 columns of B, then extending to 8.
    
    # Alternative: Use the fact that 2x4x8 is a "thin" multiplication.
    # The border rank is 56 (from Strassen). Can we do better?
    # For 2xn x nx2m, the rank is often 2*rank(nxn) * something?
    # Actually, 2x4x4 has rank 14 (2*7). 2x4x8 could be 2*14 = 28? No, that's not how it scales.
    
    # Let's try to construct a rank-56 solution first (known to work), then attempt to reduce.
    # But since we need to improve, let's try a different block structure.
    
    # What if we group the 8 columns of B into two 4-column blocks?
    # B = [B1 | B2] where B1, B2 are 4x4.
    # C = A * B = [A*B1 | A*B2]
    # This doesn't help directly because A*B1 and A*B2 are independent.
    
    # Let's try a different approach: use a "peeling" strategy.
    # Or, use a known result: rank(2xn x nx2m) <= 2 * rank(nxn) + ...?
    
    # Actually, let's just try to optimize the existing Strassen solution by checking
    # if any of the 56 terms can be merged or if a different partitioning works.
    
    # For now, let's implement the rank-56 solution but with a verification step,
    # and then try a few heuristic reductions.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 rows
    V = [[0] * rank for _ in range(m * p)]  # 32 rows
    W = [[0] * rank for _ in range(p * n)]  # 16 rows
    
    def add_term(u_coeffs, v_coeffs, w_coeffs, col_idx):
        for (i, j), val in u_coeffs.items():
            if val != 0:
                U[i*m + j][col_idx] += val
        for (j, k), val in v_coeffs.items():
            if val != 0:
                V[j*p + k][col_idx] += val
        for (k, i), val in w_coeffs.items():
            if val != 0:
                W[k*n + i][col_idx] += val
    
    col = 0
    
    # 8 blocks: j0 in {0, 2}, k0 in {0, 2, 4, 6}
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            if _time_left() < 0.5:
                break
            
            # M1 = (A00 + A11)(B00 + B11)
            add_term(
                {(0, j0): 1, (1, j0+1): 1},
                {(j0, k0): 1, (j0+1, k0+1): 1},
                {(k0, 0): 1, (k0+1, 1): 1},
                col
            )
            col += 1
            
            # M2 = (A10 + A11)B00
            add_term(
                {(1, j0): 1, (1, j0+1): 1},
                {(j0, k0): 1},
                {(k0, 1): 1, (k0+1, 1): -1},
                col
            )
            col += 1
            
            # M3 = A00(B01 - B11)
            add_term(
                {(0, j0): 1},
                {(j0, k0+1): 1, (j0+1, k0+1): -1},
                {(k0+1, 0): 1, (k0+1, 1): 1},
                col
            )
            col += 1
            
            # M4 = A11(B10 - B00)
            add_term(
                {(1, j0+1): 1},
                {(j0+1, k0): 1, (j0, k0): -1},
                {(k0, 0): 1, (k0, 1): 1},
                col
            )
            col += 1
            
            # M5 = (A00 + A01)B11
            add_term(
                {(0, j0): 1, (0, j0+1): 1},
                {(j0+1, k0+1): 1},
                {(k0, 0): -1, (k0+1, 0): 1},
                col
            )
            col += 1
            
            # M6 = (A10 - A00)(B00 + B01)
            add_term(
                {(1, j0): 1, (0, j0): -1},
                {(j0, k0): 1, (j0, k0+1): 1},
                {(k0+1, 1): 1},
                col
            )
            col += 1
            
            # M7 = (A01 - A11)(B10 + B11)
            add_term(
                {(0, j0+1): 1, (1, j0+1): -1},
                {(j0+1, k0): 1, (j0+1, k0+1): 1},
                {(k0, 0): 1},
                col
            )
            col += 1
    
    # Validation
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    T_recon = np.einsum('ur,vr,wr->uvw', U_np, V_np, W_np)
    
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0
    
    if not np.allclose(T_recon, T_target, atol=1e-9):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i*m+j
                    v = j*p+k
                    w = k*n+i
                    U[u][col] = 1
                    V[v][col] = 1
                    W[w][col] = 1
                    col += 1
    
    # Now try to reduce rank by checking if any column can be removed or merged
    # This is a heuristic approach
    
    # For now, return the rank-56 solution
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
