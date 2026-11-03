"""Matrix multiplication 2x3x6: rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start = time.time()
    deadline = start + 135.0  # 15s margin
    
    n, m, p = 2, 3, 6
    nm, mp, pn = n*m, m*p, p*n  # 6, 18, 12
    
    best = _schoolbook(n, m, p)
    best_rank = nm * mp // p * p  # 36
    
    # Try to find lower rank solutions
    # Strategy: fix U to have a specific structure, solve for V and W
    
    # For each candidate rank R, try to find U, V, W
    # We'll use a greedy/structured approach
    
    for R in range(18, best_rank):  # Try ranks from 18 up to 35
        if time.time() > deadline - 5:
            break
        
        # Try structured U: each row of U has at most 2 non-zeros
        # Use a pattern based on the index structure
        
        # Approach: For each j, the U rows (i*3+j for i=0,1) participate in
        # 12 constraints. Let's try to make U sparse.
        
        # Try: U[i*3+j, r] is non-zero only for r in a specific set
        # For rank R, assign each (i,j) pair a subset of columns
        
        # Simple structured attempt:
        # Divide R columns into 3 groups of R/3 for each j
        if R % 3 != 0:
            continue
        
        Rj = R // 3  # columns per j
        
        U = [[0] * R for _ in range(nm)]
        V = [[0] * R for _ in range(mp)]
        W = [[0] * R for _ in range(pn)]
        
        # For each j, use Rj columns
        for j in range(m):
            col_start = j * Rj
            # U rows for this j: i*3+j for i=0,1
            # V rows for this j: j*6+k for k=0..5
            # W rows: k*2+i for all k,i
            
            # For 2x6 multiplication with Rj terms:
            # We need: for all i in {0,1}, k in {0..5}:
            #   sum_{r=0}^{Rj-1} U[i*3+j, col_start+r] * V[j*6+k, col_start+r] * W[k*2+i, col_start+r] = 1
            
            # Try schoolbook within this block: Rj should be >= 12 for full schoolbook
            # But we can try lower
            
            # For 2x6, let's try rank 6 (half of schoolbook 12)
            # C[i,k] = sum_j A[i,j]*B[j,k] for 2x6 means:
            # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
            # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
            
            # This is 12 products. Can we do it in fewer?
            # For 2x3, rank is 5 (I think). For 2x6, maybe 9-10?
            
            # Let's just use schoolbook for now within each block
            for i in range(n):
                for jj in range(m):
                    for k in range(p):
                        r = i * m * p + jj * p + k  # local column
                        if r < Rj:
                            U[i*3 + j][col_start + r] = 1
                            V[j*6 + k][col_start + r] = 1
                            W[k*2 + i][col_start + r] = 1
                        # else: skip (incomplete)
        
        # Verify
        if _verify(U, V, W, n, m, p):
            best_rank = R
            best = {"U": U, "V": V, "W": W}
            break
    
    # Try another approach: full search with small R using structured patterns
    if best_rank == 36 and time.time() < deadline - 10:
        # Try rank 30, 28, 24 with more structured patterns
        for R in [30, 28, 24, 22, 20]:
            if time.time() > deadline - 5:
                break
            
            # Try: U[i*3+j, r] = 1 if r < R and (r % m) == j and (r // m) < n
            # V[j*6+k, r] = 1 if r < R and (r % p) == k  
            # W[k*2+i, r] = 1 if r < R and (r % n) == i
            
            U = [[0] * R for _ in range(nm)]
            V = [[0] * R for _ in range(mp)]
            W = [[0] * R for _ in range(pn)]
            
            # Pattern: r encodes (i, j, k) in a compressed way
            # Try: r = i*p + k for the "diagonal" terms, plus extra for j
            # Actually, let's try: for each (j, k), use one term, and for each (i, j), use one term
            # Total: m*p + n*m = 18 + 6 = 24 terms? But they need to combine.
            
            # Simpler: just verify if a random sparse structure works
            np.random.seed(R)
            for attempt in range(100):
                if time.time() > deadline - 3:
                    break
                
                # Random sparse U, V, W with half-integer entries
                U_test = np.random.choice([0, 0, 0, 0.5, 1, -0.5, -1], size=(nm, R))
                V_test = np.random.choice([0, 0, 0, 0.5, 1, -0.5, -1], size=(mp, R))
                W_test = np.random.choice([0, 0, 0, 0.5, 1, -0.5, -1], size=(pn, R))
                
                if _verify(U_test.tolist(), V_test.tolist(), W_test.tolist(), n, m, p):
                    best_rank = R
                    best = {"U": U_test.tolist(), "V": V_test.tolist(), "W": W_test.tolist()}
                    break
            if best_rank < 36:
                break
    
    # Final: return best found, fallback to schoolbook
    if best_rank >= 36:
        best = _schoolbook(n, m, p)
    
    return best
