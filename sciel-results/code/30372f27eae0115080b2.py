"""Tensor decomposition for 3x3 by 3x6 matrix multiplication."""
import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [3, 3, 6]

def _verify(U, V, W, n, m, p):
    """Check if decomposition is valid."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the multiplication tensor
    # T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
    # Actually, the tensor for matrix multiplication C = A*B has:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor T has T[i,j,k] = 1 for all i,j,k
    
    # The decomposition says: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Wait, the convention is W[k*n+i, c], so W is indexed by (k,i)
    # Let me re-read: "The output factor follows the official trace convention W[k*n+i,r]"
    # So W is a matrix of shape (p*n, rank), where row index is k*n + i
    
    # The decomposition: T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                max_err = max(max_err, abs(val - 1.0))
    
    return max_err

def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication."""
    n, m, p = 3, 3, 6
    
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    
    # Start with schoolbook
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
    
    best_result = {"U": U, "V": W, "W": W}  # placeholder
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Try to find a lower rank decomposition using a randomized/structured search
    # Strategy: try small ranks and see if we can find integer/half-integer solutions
    
    # For 3x3 * 3x6, the tensor has 3*3*6 = 54 entries.
    # A rank-r decomposition requires finding U (9 x r), V (18 x r), W (18 x r)
    # such that for all (i,j,k): sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
    
    # This is a system of 54 equations in 9r + 18r + 18r = 45r unknowns.
    # For r >= 2, we have more unknowns than equations, so solutions should exist.
    # But we need exact integer/half-integer solutions.
    
    # Let's try a different approach: use known results or structured decompositions.
    # For 2x2x2, Strassen gives rank 7.
    # For 3x3x3, best known is rank 23.
    # For 3x3x6, we can potentially use a block structure.
    
    # Let me try to construct a decomposition using a partition approach.
    # Split the 3x6 matrix into three 3x2 blocks.
    # Then C = A * [B1, B2, B3] = [A*B1, A*B2, A*B3]
    # Each A*B_i is a 3x2 multiplication.
    # If we can do 3x2 in some rank, total rank would be 3 * rank(3x2).
    # 3x2 matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k] for k=0,1
    # This is equivalent to two 3x1 multiplications, but we can share some computations.
    
    # Actually, let me try a more direct approach: search for a low-rank solution
    # using the fact that we need exact arithmetic.
    
    # Let me try rank values from low to high and use a simple iterative method.
    # For each rank r, we try to find U, V, W with entries in {0, 1/2, 1, -1/2, -1}
    # such that the tensor equation is satisfied.
    
    # This is a constraint satisfaction problem. Let's try small ranks first.
    
    # Actually, let me try a smarter approach: use the structure of the problem.
    # The tensor T[i,j,k] = 1 for all i,j,k in the 3x3x6 case.
    
    # Let me try rank = 2, 3, ..., up to some limit, using a randomized search
    # with exact arithmetic verification.
    
    import random
    random.seed(42)
    
    def try_rank(r, max_iters=5000):
        """Try to find a decomposition of rank r."""
        if time.time() > deadline:
            return None
        
        # U: 9 x r, V: 18 x r, W: 18 x r
        # Entries from {0, 1/2, 1, -1/2, -1}
        vals = [0, 0.5, 1, -0.5, -1]
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            U = [[random.choice(vals) for _ in range(r)] for _ in range(9)]
            V = [[random.choice(vals) for _ in range(r)] for _ in range(18)]
            W = [[random.choice(vals) for _ in range(r)] for _ in range(18)]
            
            # Verify
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                        if abs(s - 1.0) > 1e-9:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                return U, V, W
        
        return None
    
    # Try ranks from low to high
    for r in range(2, 20):
        if time.time() > deadline:
            break
        result = try_rank(r, max_iters=2000)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_result = {"U": U, "V": V, "W": W}
            break
    
    if best_rank < rank:
        return best_result
    
    # If no lower rank found, return schoolbook
    return {"U": U, "V": V, "W": W}
