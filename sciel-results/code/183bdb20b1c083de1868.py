"""Rewritten program for 3x4x4 matrix multiplication tensor decomposition."""
import time
import random
import itertools
import numpy as np

DIMENSIONS = [3, 4, 4]

def _verify(U, V, W, n, m, p, rank, tol=1e-8):
    """Check if U, V, W form a valid decomposition of the matmul tensor."""
    # Build the matmul tensor T[i,j,k] = 1 if the corresponding entry exists
    # Actually, the matmul tensor is T[i,j,k] = 1 for all i,j,k (it's the "multiplication" tensor)
    # The decomposition requires: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # Wait, let me re-read the task.
    
    # The matrix multiplication tensor for (n x m) * (m x p) -> (n x p)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has entries T[i,j,k] = 1 (it's the standard matmul tensor)
    # A rank-r decomposition gives: T[i,j,k] = sum_{r=1}^r U[i,j,r] * V[j,k,r] * W[i,k,r]
    # Wait, the interface says:
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # W[k*n+i, r] - so W is indexed by (k,i) where k is from p and i is from n
    
    # So the decomposition is:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Let's verify this.
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape to 3D tensors
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    # Compute sum over r for each (i,j,k)
    # result[i,j,k] = sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
    result = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    # The matmul tensor should be all 1s
    target = np.ones((n, m, p))
    
    if not np.allclose(result, target, atol=tol):
        return False
    return True

def _verify_half_integer(U, V, W):
    """Check if all entries are integers or half-integers."""
    for matrix in [U, V, W]:
        for row in matrix:
            for val in row:
                # Check if val is integer or half-integer
                v = float(val)
                if abs(v * 2 - round(v * 2)) > 1e-8:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    best = {"U": U_sb, "V": V_sb, "W": W_sb, "rank": rank_schoolbook}
    
    # Try to find lower rank decompositions using random search
    # The minimum rank for 3x4x4 is known to be at most 48 (schoolbook) 
    # but potentially lower. Let's try random rank from 47 down.
    
    # For each candidate rank, try random initialization and verify
    for rank in range(rank_schoolbook - 1, 20, -1):
        if time.time() > deadline:
            break
        
        # Try several random initializations
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random half-integer initialization
            U_rand = [[random.choice([0, 0.5, 1, -0.5, -1]) for _ in range(rank)] for _ in range(n * m)]
            V_rand = [[random.choice([0, 0.5, 1, -0.5, -1]) for _ in range(rank)] for _ in range(m * p)]
            W_rand = [[random.choice([0, 0.5, 1, -0.5, -1]) for _ in range(rank)] for _ in range(p * n)]
            
            if _verify(U_rand, V_rand, W_rand, n, m, p, rank):
                if rank < best["rank"]:
                    best = {"U": U_rand, "V": V_rand, "W": W_rand, "rank": rank}
                break  # Found a valid decomposition for this rank
    
    # Also try a structured approach: use the schoolbook but try to merge terms
    # This is more complex, so we'll rely on random search for now.
    
    # Return the best found
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
