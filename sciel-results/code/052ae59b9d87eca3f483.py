import time
import numpy as np
from itertools import product

def construct():
    n, m, p = 2, 3, 5
    nm, mp, pn = n * m, m * p, p * n  # 6, 15, 10
    
    start_time = time.time()
    deadline = start_time + 115.0

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        r = len(U[0]) if U else 0
        if not U or not V or not W:
            return False
        if len(V[0]) != r or len(W[0]) != r:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def solve_w(U, V):
        """
        Given U and V (lists of lists), solve for W such that
        sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] == 1 for all i,j,k.
        Returns W as a list of lists if a valid integer/half-integer solution exists, else None.
        """
        r = len(U[0])
        # We have 30 equations (i,j,k) and 10*rank unknowns (W indices).
        # Actually, W is (p*n, rank) = (10, rank).
        # Equation for (i,j,k): sum_c (U[i*m+j][c] * V[j*p+k][c]) * W[k*n+i][c] = 1
        # This is a linear system A * W_flat = b, where b is ones(30).
        
        A = np.zeros((30, 10 * r))
        b = np.ones(30)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    w_row_idx = k * n + i
                    for c in range(r):
                        coeff = U[i * m + j][c] * V[j * p + k][c]
                        A[idx, w_row_idx * r + c] = coeff
                    idx += 1
        
        try:
            # Use lstsq to find the minimum norm solution
            x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        except:
            return None
            
        W_sol = x.reshape(10, r)
        
        # Check if W_sol is close to integers or half-integers
        # The problem allows integer/half-integer entries.
        
        # Try integers
        W_int = np.round(W_sol)
        if np.allclose(W_int, W_sol, atol=1e-6):
            # Verify exactness with the integer values
            W_list = W_int.tolist()
            if verify(U, V, W_list):
                return W_list
                
        # Try half-integers
        W_half = np.round(W_sol * 2) / 2.0
        if np.allclose(W_half, W_sol, atol=1e-6):
            W_list = W_half.tolist()
            if verify(U, V, W_list):
                return W_list
                
        return None

    def generate_strassen_candidates(target_rank):
        """
        Generates candidates for U and V based on structured splits.
        We split the middle dimension m=3 into 1+2.
        A = [A0 | A12] where A0 is 2x1, A12 is 2x2.
        B = [B0; B12] where B0 is 1x5, B12 is 2x5.
        
        C = A0*B0 + A12*B12
        
        We can handle A0*B0 (rank 2 outer product part) and A12*B12 (Strassen rank 7 part).
        However, the standard Strassen for 2x2 * 2x5 is not directly rank 7.
        Strassen is for 2x2 * 2x2.
        
        Let's look at the structure more carefully.
        The tensor is T_{i,j,k} = sum_l A_{i,l} B_{l,k}.
        
        A structured approach:
        1. Use the 2x2 Strassen identity for the sub-blocks where possible.
        2. The "1" dimension in the middle (l=0) couples with all i and k.
        
        Let's try a specific rank 28 structure.
        Rank 28 = 30 - 2.
        
        Alternative: Try to find a rank 26 or 27 solution.
        Known bounds:
        For 2x3 * 3x5, the rank is likely around 24-28.
        
        Let's construct U and V using a "block" approach.
        We will define U and V columns explicitly based on linear combinations.
        
        Strategy:
        Iterate over possible "sign patterns" for the Strassen-like terms.
        """
        
        candidates = []
        
        # We will try to build U and V such that they are "sparse" in a structured way.
        # Let's assume the rank R is composed of:
        # - 7 terms for the 2x2 * 2x2 part (using Strassen on cols 1,2 of A and rows 1,2 of B)
        # - 5 terms for the 1x1 * 1x5 part? No, that's 5.
        # - Cross terms?
        
        # Actually, let's just generate random sparse U, V but with a bias towards
        # "Strassen-like" sparsity patterns in the 2x2 blocks.
        
        # For now, let's stick to a more robust randomized search but with
        # "structured" initialization: 
        # U and V entries are chosen from {-1, 0, 1} but with a constraint that
        # the "effective" rank of sub-blocks is low.
        
        # Let's try a different tactic: 
        # We know that if we fix U and V, W is determined (if the system is consistent).
        # We can search for U and V.
        
        # Let's try to find a solution of rank 28 first, then try to reduce.
        # The current program tries 28. Let's try 26, 27, 28.
        
        return candidates

    def try_rank(rank):
        """
        Attempt to find a decomposition of the given rank.
        Uses a randomized local search on U and V with structured priors.
        """
        if time.time() > deadline:
            return None
            
        # We will use a "simulated annealing" style search on the discrete space of U and V entries in {-1, 0, 1}.
        # Objective: Minimize the error of the linear system A W = b, but we want exact integer/half-integer W.
        # Better: Just solve for W and check if it's integer/half-integer.
        
        # To increase the chance of W being integer/half-integer, U and V should be "well-conditioned" or "orthogonal-ish".
        # Random {-1, 0, 1} matrices are often okay.
        
        # Let's try a few random initializations.
        for attempt in range(200):
            if time.time() > deadline:
                break
                
            # Generate random sparse U and V
            # Density: ~0.3 for U, ~0.3 for V to keep them sparse
            U = np.zeros((nm, rank), dtype=int)
            V = np.zeros((mp, rank), dtype=int)
            
            mask_u = np.random.rand(nm, rank) < 0.3
            mask_v = np.random.rand(mp, rank) < 0.3
            
            U[mask_u] = np.random.choice([-1, 1], size=(nm, rank))[mask_u]
            V[mask_v] = np.random.choice([-1, 1], size=(mp, rank))[mask_v]
            
            # Ensure no zero columns (which would be wasted rank)
            # And no zero rows in U or V if possible? Not strictly necessary.
            
            W = solve_w(U.tolist(), V.tolist())
            if W is not None:
                return {"U": U.tolist(), "V": V.tolist(), "W": W}
                
        return None

    # Try ranks from 26 upwards
    # 30 is schoolbook.
    # 28 is a likely candidate.
    # 26 is possible.
    
    for rank in [26, 27, 28]:
        if time.time() > deadline:
            break
        res = try_rank(rank)
        if res:
            return res

    return schoolbook()
