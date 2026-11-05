import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s

    def time_left():
        return deadline - time.time()

    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def check_decomposition(U, V, W, T, tol=1e-6):
        T_rec = np.einsum('ir,vr,wr->ivw', U, V, W, optimize=True)
        return np.max(np.abs(T_rec - T)) < tol

    # Rank 28 decomposition (Strassen + Schoolbook) - known valid solution
    def build_rank_28():
        rank = 28
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0

        def add_strassen(i1, i2, j1, j2, k1, k2):
            nonlocal col
            U[i1*m+j1, col] = 1; U[i2*m+j2, col] = 1
            V[j1*p+k1, col] = 1; V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = 1; W[k2*n+i2, col] = 1
            col += 1
            U[i2*m+j1, col] = 1; U[i2*m+j2, col] = 1
            V[j1*p+k1, col] = 1
            W[k1*n+i2, col] = 1; W[k2*n+i2, col] = -1
            col += 1
            U[i1*m+j1, col] = 1
            V[j1*p+k2, col] = 1; V[j2*p+k2, col] = -1
            W[k2*n+i1, col] = 1; W[k2*n+i2, col] = 1
            col += 1
            U[i2*m+j2, col] = 1
            V[j2*p+k1, col] = 1; V[j1*p+k1, col] = -1
            W[k1*n+i1, col] = 1; W[k1*n+i2, col] = 1
            col += 1
            U[i1*m+j1, col] = 1; U[i1*m+j2, col] = 1
            V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = -1; W[k2*n+i1, col] = 1
            col += 1
            U[i2*m+j1, col] = 1; U[i1*m+j1, col] = -1
            V[j1*p+k1, col] = 1; V[j1*p+k2, col] = 1
            W[k2*n+i2, col] = 1
            col += 1
            U[i1*m+j2, col] = 1; U[i2*m+j2, col] = -1
            V[j2*p+k1, col] = 1; V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = 1
            col += 1

        def add_outer(i, j, k):
            nonlocal col
            U[i*m+j, col] = 1
            V[j*p+k, col] = 1
            W[k*n+i, col] = 1
            col += 1

        add_strassen(0, 1, 0, 1, 0, 1)
        add_strassen(0, 1, 0, 1, 2, 3)
        
        for i in range(n):
            for k in range(2):
                add_outer(i, 2, k)
        
        for j in range(2):
            for i in range(n):
                add_outer(i, j, 4)
        
        for i in range(n):
            for k in range(2, 5):
                add_outer(i, 2, k)

        assert col == rank
        return U, V, W

    # Try to find a lower rank decomposition using MPS-inspired search
    # The idea: model the tensor as having a matrix product state structure
    # with bond dimension d, and try to minimize the number of terms
    
    def try_mps_decomposition(max_rank=27, max_bond_dim=3):
        """
        Try to find a decomposition with rank < 28 using MPS-inspired approach.
        We model the tensor as a sum of rank-1 terms where each term has
        a structure that can be decomposed into local interactions.
        """
        if time_left() < 5:
            return None
        
        # For MPS approach, we try to find a decomposition where the rank-1 terms
        # have a structured pattern that exploits the small dimensions
        
        # The tensor T[i*m+j, j*p+k, k*n+i] = 1 has a specific structure.
        # Let's try to find a decomposition with rank r < 28.
        
        # Strategy: Use the fact that the tensor has a "banded" structure in some sense.
        # The indices (i, j, k) range over (0..1, 0..2, 0..4).
        
        # Try different rank values and see if we can find a valid decomposition
        for rank in range(27, 20, -1):
            if time_left() < 3:
                break
            
            # Build a system of equations for the decomposition
            # T_rec[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            # This is a nonlinear system. We'll use a heuristic approach:
            # Start with the rank-28 solution and try to merge terms.
            
            U28, V28, W28 = build_rank_28()
            
            # Try to merge pairs of terms that are similar
            merged_U = U28.copy()
            merged_V = V28.copy()
            merged_W = W28.copy()
            
            # Try merging each pair of columns
            for r1 in range(28):
                if time_left() < 2:
                    break
                for r2 in range(r1 + 1, 28):
                    # Check if columns r1 and r2 can be merged
                    # They can be merged if their supports don't overlap in a way
                    # that prevents a single rank-1 term from representing both
                    
                    # Compute what the merged term would need to be
                    # This is complex, so we use a simpler heuristic:
                    # Try to find a linear combination that cancels out
                    
                    # Create a test: can we replace columns r1 and r2 with one column?
                    # We need: U[:,r1]*V[:,r1]*W[:,r1] + U[:,r2]*V[:,r2]*W[:,r2] 
                    #         = U[:,new]*V[:,new]*W[:,new]
                    
                    # This is generally not possible, so we skip this approach
                    pass
            
            # Alternative: try to find a decomposition from scratch using the MPS structure
            # For bond dimension d, the number of terms is related to the bond dimension
            # For a 2x3x5 tensor with MPS bond dimension d, the rank is at most d^2 * something
            
            # Let's try a more direct approach: use the known structure to build a lower-rank decomposition
            # The key insight is that for 2x3x5, we can potentially use a rank-27 decomposition
            # by exploiting the asymmetry in dimensions
            
            # Try to build a rank-27 decomposition
            if rank == 27:
                U27, V27, W27 = build_rank_27_attempt()
                if U27 is not None and check_decomposition(U27, V27, W27, T):
                    return U27, V27, W27
        
        return None
    
    def build_rank_27_attempt():
        """
        Attempt to build a rank-27 decomposition by modifying the rank-28 solution.
        The idea is to find a way to save one term by exploiting the structure.
        """
        if time_left() < 2:
            return None, None, None
        
        # Start with the rank-28 solution
        U28, V28, W28 = build_rank_28()
        
        # The rank-28 solution uses:
        # - 7 terms for Strassen(0,1,0,1,0,1)
        # - 7 terms for Strassen(0,1,0,1,2,3)
        # - 4 terms for (i,2,k) with k in {0,1}
        # - 4 terms for (i,j,4) with j in {0,1}
        # - 6 terms for (i,2,k) with k in {2,3,4}
        # Total: 7 + 7 + 4 + 4 + 6 = 28
        
        # To get rank 27, we need to save one term.
        # Let's try to modify the schoolbook parts to use fewer terms.
        
        # The terms (i,2,k) for k in {0,1} and k in {2,3,4} are all schoolbook terms.
        # Can we combine some of them?
        
        # For example, can we combine (0,2,0) and (1,2,0)?
        # U[0*3+0, r] = 1, U[1*3+0, r] = 1
        # V[0*5+0, r] = 1
        # W[0*2+0, r] = 1, W[0*2+1, r] = 1
        # This would give T[0,0,0] += 1, T[1,0,0] += 1, T[0,0,1] += 1, T[1,0,1] += 1
        # But we only need T[0,0,0] += 1 and T[1,0,0] += 1, not the others.
        
        # So we can't simply combine them.
        
        # Let's try a different approach: use a more sophisticated decomposition
        # that exploits the full structure of the 2x3x5 tensor.
        
        # For now, return None to indicate we couldn't find a rank-27 decomposition
        return None, None, None
    
    # First, try the known rank-28 solution
    U28, V28, W28 = build_rank_28()
    if check_decomposition(U28, V28, W28, T):
        best_result = (U28, V28, W28)
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_result = (U, V, W)
    
    # Try to find a better solution using MPS-inspired search
    if time_left() > 10:
        mps_result = try_mps_decomposition()
        if mps_result is not None:
            U_mps, V_mps, W_mps = mps_result
            if check_decomposition(U_mps, V_mps, W_mps, T):
                best_result = (U_mps, V_mps, W_mps)
    
    U_best, V_best, W_best = best_result
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
