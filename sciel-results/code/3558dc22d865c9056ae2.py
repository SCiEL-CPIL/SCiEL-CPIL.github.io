import numpy as np
import time
import random

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    def verify_solution(U, V, W, R):
        """Verify that U, V, W with rank R correctly decompose the matrix multiplication tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for l in range(m):
                        val = 0.0
                        for r in range(R):
                            val += W[k * n + i, r] * U[i * m + j, r] * V[l * p + k, r]
                        expected = 1.0 if j == l else 0.0
                        if abs(val - expected) > 1e-9:
                            return False
        return True

    def search_decomposition(n, m, p, target_rank, time_limit=10):
        """
        Search for a decomposition of M_{n,m,p} with rank target_rank.
        Returns U, V, W or None.
        """
        start = time.time()
        U_rows = n * m
        V_rows = m * p
        W_rows = p * n
        
        # We will search for U, V with entries in {-1, 0, 1} and solve for W.
        # Then snap W to nearest 0.5.
        
        for _ in range(1000):
            if time.time() - start > time_limit:
                break
                
            # Random U, V
            U = np.random.choice([-1, 0, 1], size=(U_rows, target_rank))
            V = np.random.choice([-1, 0, 1], size=(V_rows, target_rank))
            
            # Build the linear system for W
            # We need W such that for all i, k, j, l:
            # sum_r W[k*n+i, r] U[i*m+j, r] V[l*p+k, r] = delta(j,l)
            
            # This is a linear system in W.
            # Let's flatten the constraints.
            # There are n * p * m * m constraints? No, only n * p * m constraints are independent?
            # Actually, the tensor is defined by T[i, j, k, l] = delta(j,l).
            # We need sum_r W_{ki} U_{ij} V_{lk} = delta(j,l).
            
            # We can set up a system A * w = b.
            # w is vectorized W.
            # Each constraint (i, k, j, l) gives an equation.
            # But many are redundant.
            # Let's just use all n*p*m*m equations? That might be too many (4*6*4*4 = 384).
            # Target rank 22. 22 * 24 = 528 variables.
            # 384 equations. Underdetermined?
            # Wait, W has p*n*R = 6*4*22 = 528 variables.
            # Equations: n*p*m*m = 4*6*4*4 = 384.
            # So we have more variables than equations.
            # We can use least squares.
            
            A = np.zeros((n * p * m * m, W_rows * target_rank))
            b = np.zeros(n * p * m * m)
            
            idx = 0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for l in range(m):
                            expected = 1.0 if j == l else 0.0
                            b[idx] = expected
                            for r in range(target_rank):
                                A[idx, (k * n + i) * target_rank + r] = U[i * m + j, r] * V[l * p + k, r]
                            idx += 1
            
            # Solve A w = b
            try:
                w, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            W = w.reshape(W_rows, target_rank)
            
            # Snap to nearest 0.5
            W_snapped = np.round(W * 2) / 2.0
            
            # Verify
            if verify_solution(U.tolist(), V.tolist(), W_snapped.tolist(), target_rank):
                return U, V, W_snapped
                
        return None

    # First, try to find a decomposition for 4x4x4 with rank 22
    # Then use schoolbook for 4x4x2 (rank 32)
    # Total rank 54
    
    U_444 = None
    V_444 = None
    W_444 = None
    rank_444 = None
    
    # Try ranks 22, 23, 24, 25, 26, 27, 28, 29, 30
    for r in range(22, 31):
        if time.time() > deadline - 20:
            break
        res = search_decomposition(4, 4, 4, r, time_limit=5)
        if res:
            U_444, V_444, W_444 = res
            rank_444 = r
            break
            
    if U_444 is None:
        # Fallback to schoolbook for 4x4x4 (rank 64)
        rank_444 = 64
        U_444 = np.zeros((16, 64))
        V_444 = np.zeros((16, 64))
        W_444 = np.zeros((16, 64))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_444[i*4+j, col] = 1
                    V_444[j*4+k, col] = 1
                    W_444[k*4+i, col] = 1
                    col += 1
    else:
        # Convert to lists for verification/return later, but keep as arrays for now
        pass

    # Schoolbook for 4x4x2 (cols 4,5 of B)
    # B is 4x6. Cols 4,5 correspond to k=4,5.
    # We need to decompose the part where k is in {4,5}.
    # This is equivalent to M_{4,4,2} with indices shifted.
    # U_442: 16 x 32
    # V_442: 8 x 32 (rows j*p+k for j in 0..3, k in 4..5)
    # W_442: 8 x 32 (rows k*n+i for k in 4..5, i in 0..3)
    
    rank_442 = 32
    U_442 = np.zeros((16, 32))
    V_442 = np.zeros((8, 32))
    W_442 = np.zeros((8, 32))
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in [4, 5]:
                U_442[i*4+j, col] = 1
                V_442[j*4 + (k-4), col] = 1 # Map k=4,5 to 0,1
                W_442[(k-4)*4 + i, col] = 1 # Map k=4,5 to 0,1
                col += 1

    # Combine
    total_rank = rank_444 + rank_442
    U_full = np.zeros((16, total_rank))
    V_full = np.zeros((24, total_rank))
    W_full = np.zeros((24, total_rank))
    
    # Part 1: 4x4x4
    U_full[:, :rank_444] = U_444
    # V_444 has 16 rows (j*4+k for k in 0..3).
    # V_full has 24 rows (j*6+k for k in 0..5).
    # We need to map V_444 rows to V_full rows for k=0..3.
    # V_444 row index: j*4+k.
    # V_full row index: j*6+k.
    for j in range(4):
        for k in range(4):
            V_full[j*6+k, :rank_444] = V_444[j*4+k, :]
    
    # W_444 has 16 rows (k*4+i for k in 0..3).
    # W_full has 24 rows (k*4+i for k in 0..5).
    # W_444 row index: k*4+i.
    # W_full row index: k*4+i.
    for k in range(4):
        for i in range(4):
            W_full[k*4+i, :rank_444] = W_444[k*4+i, :]

    # Part 2: 4x4x2
    U_full[:, rank_444:] = U_442
    # V_442 has 8 rows (j*2+k' for k' in 0..1, corresponding to k=4,5).
    # V_full rows for k=4,5 are j*6+4 and j*6+5.
    for j in range(4):
        for k_prime in range(2):
            k = 4 + k_prime
            V_full[j*6+k, rank_444:] = V_442[j*2+k_prime, :]
            
    # W_442 has 8 rows (k'*4+i for k' in 0..1, corresponding to k=4,5).
    # W_full rows for k=4,5 are 4*4+i and 5*4+i.
    for k_prime in range(2):
        k = 4 + k_prime
        for i in range(4):
            W_full[k*4+i, rank_444:] = W_442[k_prime*4+i, :]

    # Verify full solution
    if verify_solution(U_full.tolist(), V_full.tolist(), W_full.tolist(), total_rank):
        return {
            "U": [[float(v) for v in row] for row in U_full],
            "V": [[float(v) for v in row] for row in V_full],
            "W": [[float(v) for v in row] for row in W_full]
        }
    
    # Fallback: Schoolbook for 4x4x6
    rank_sb = 96
    U_sb = np.zeros((16, 96))
    V_sb = np.zeros((24, 96))
    W_sb = np.zeros((24, 96))
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(6):
                U_sb[i*4+j, col] = 1
                V_sb[j*6+k, col] = 1
                W_sb[k*4+i, col] = 1
                col += 1
                
    return {
        "U": [[float(v) for v in row] for row in U_sb],
        "V": [[float(v) for v in row] for row in V_sb],
        "W": [[float(v) for v in row] for row in W_sb]
    }
