"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 5]
MAX_STEPS = 8

def _schoolbook(n, m, p):
    """
    Standard schoolbook matrix multiplication tensor decomposition.
    Rank = n * m * p.
    U shape: (n*m, rank)
    V shape: (m*p, rank)
    W shape: (p*n, rank)
    Indexing:
    U[i*m+j, c]
    V[j*p+k, c]
    W[k*n+i, c]
    """
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

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill switch
    
    n, m, p = DIMENSIONS
    base_rank = n * m * p
    
    # 1. Establish the safe baseline (Schoolbook)
    best = _schoolbook(n, m, p)
    best_rank = base_rank
    
    def _verify_decomposition(U, V, W, n, m, p):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        rank = len(U[0]) if U else 0
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        for c in range(rank):
            if len(U[c]) != rank or len(V[c]) != rank or len(W[c]) != rank:
                return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    def _try_partial_strassen(n, m, p, deadline):
        """
        Try to find a lower-rank decomposition using a partial Strassen approach.
        
        Strategy: Partition the 3x4 and 4x5 matrices into blocks and use
        known low-rank decompositions for the block multiplications.
        
        For 3x4 x 4x5, we can try:
        - Partition A (3x4) into rows: A_top (1x4), A_mid (1x4), A_bot (1x4)
        - Partition B (4x5) into columns: B_left (4x2), B_right (4x3)
        
        Then C = A*B is 3x5, partitioned as:
        C_top_left (1x2) = A_top * B_left
        C_top_right (1x3) = A_top * B_right
        C_mid_left (1x2) = A_mid * B_left
        C_mid_right (1x3) = A_mid * B_right
        C_bot_left (1x2) = A_bot * B_left
        C_bot_right (1x3) = A_bot * B_right
        
        Each of these is a 1x4 * 4xk multiplication, which is just a dot product.
        The dot product of a 1x4 and 4xk matrix has rank 4 (schoolbook) for each k.
        
        Total rank = 3 * 2 * 4 + 3 * 3 * 4 = 24 + 36 = 60 (same as schoolbook).
        
        Let me try a different partitioning. What if I use 2x2 blocks?
        
        Actually, let me try to use the fact that 4 = 2+2 and 5 = 2+3.
        
        A (3x4) = [A11 A12; A21 A22; A31 A32] where Aij are 1x2
        B (4x5) = [B11 B12; B21 B22] where B11, B21 are 2x2 and B12, B22 are 2x3
        
        Then C = A*B:
        C[i,k] = sum_j A[i,j] B[j,k]
        
        This gets complicated. Let me just try a random local search.
        """
        import random
        random.seed(42)
        
        # Try to find a rank-59 decomposition by starting with schoolbook
        # and trying to merge two terms.
        
        # The schoolbook has 60 terms. Each term c corresponds to (i_c, j_c, k_c).
        # Can we find two terms that can be merged into one?
        
        # For a rank-59 decomposition, we need to express the sum of all 60 schoolbook
        # rank-1 tensors as a sum of 59 rank-1 tensors.
        
        # This is equivalent to finding a rank-1 tensor T such that
        # T = sum_{c=1}^{60} T_c - sum_{c=1}^{59} T'_c
        # for some rank-1 tensors T'_c.
        
        # This is a hard problem. Let me try a heuristic:
        # Pick a random subset of 59 schoolbook terms and see if we can adjust
        # their coefficients to match the full tensor.
        
        # Actually, the coefficients are all 1 in the schoolbook, so we can't just
        # adjust them. We need to find new rank-1 tensors.
        
        # Let me try a different approach: use the "tensor decomposition" algorithm.
        # We want to find U, V, W such that:
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
        
        # This is a system of 60 equations. Let's try to solve it for rank 59.
        
        # For rank 59, we have 59*(12+20+15) = 59*47 = 2773 unknowns and 60 equations.
        # This is highly underdetermined, but the equations are nonlinear.
        
        # Let me try a simple heuristic: start with the schoolbook and try to
        # eliminate one term by expressing it as a combination of the others.
        
        # Term (i,j,k) can be expressed as:
        # e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
        
        # Can we write this as a sum of other rank-1 tensors? In general, no,
        # because the schoolbook terms are linearly independent.
        
        # So we need to find genuinely new rank-1 tensors. Let me try a random search.
        
        # Generate random rank-1 tensors and try to find a linear combination
        # that matches the target tensor.
        
        # For rank 59, we need to solve:
        # sum_{c=1}^{59} u_c ⊗ v_c ⊗ w_c = T
        
        # where T is the matrix multiplication tensor.
        
        # This is a system of 60 equations (one for each (i,j,k) triple) with
        # 59*47 = 2773 unknowns.
        
        # Let me try to use a simple iterative method: start with random initial
        # values and use gradient descent to minimize the error.
        
        # Define the error function:
        # E = sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
        
        # We want to minimize E. If E = 0, we have a valid decomposition.
        
        # Let me try this with a few iterations.
        
        def _random_init(rank, n, m, p):
            U = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(n*m)]
            V = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(m*p)]
            W = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(p*n)]
            return U, V, W
        
        def _compute_error(U, V, W, n, m, p):
            rank = len(U[0])
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                        error += (s - 1.0) ** 2
            return error
        
        # Try a few random initializations and see if we can get close to zero error.
        # This is unlikely to work for rank 59, but let's try.
        
        for attempt in range(10):
            if time.time() > deadline - 30:
                break
            
            U, V, W = _random_init(59, n, m, p)
            err = _compute_error(U, V, W, n, m, p)
            
            # If error is very small, we might have a solution
            if err < 1e-6:
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try the partial Strassen / random search
    result = _try_partial_strassen(n, m, p, deadline)
    if result is not None:
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            best = result
            best_rank = 59
    
    # Check time to ensure we don't accidentally timeout
    if time.time() > deadline - 5.0:
        return best

    return best
# EVOLVE-BLOCK-END
