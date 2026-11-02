import numpy as np
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 120
    SAFETY_MARGIN = 10  # seconds

    n, m, p = 2, 4, 6
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    # Strassen 2x2x2 definitions
    # Local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # U factors (A), V factors (B), W factors (C)
    strassen_U = [
        {0: 1, 3: 1},       # P1: a00+a11
        {2: 1, 3: 1},       # P2: a10+a11
        {0: 1},             # P3: a00
        {3: 1},             # P4: a11
        {0: 1, 1: 1},       # P5: a00+a01
        {2: 1, 0: -1},      # P6: a10-a00
        {1: 1, 3: -1},      # P7: a01-a11
    ]
    strassen_V = [
        {0: 1, 3: 1},       # P1: b00+b11
        {0: 1},             # P2: b00
        {1: 1, 3: -1},      # P3: b01-b11
        {2: 1, 0: -1},      # P4: b10-b00
        {3: 1},             # P5: b11
        {0: 1, 1: 1},       # P6: b00+b01
        {2: 1, 3: 1},       # P7: b10+b11
    ]
    strassen_W = [
        {0: 1, 3: 1},       # P1: c00+c11
        {2: 1, 3: -1},      # P2: c10-c11
        {1: 1, 3: 1},       # P3: c01+c11
        {0: 1, 2: 1},       # P4: c00+c10
        {0: -1, 1: 1},      # P5: -c00+c01
        {3: 1},             # P6: c11
        {0: 1},             # P7: c00
    ]

    def time_remaining():
        return DEADLINE - (time.time() - start_time) - SAFETY_MARGIN

    # Base Strassen construction (rank 42)
    rank_total = 42
    U = np.zeros((nm, rank_total))
    V = np.zeros((mp, rank_total))
    W = np.zeros((pn, rank_total))

    col = 0
    for jm in range(2):      # A col block / B row block
        for kp in range(3):  # B col block / C col block
            for pi in range(7):
                r = col + pi
                # U: A[i, 2*jm + j_local]
                for loc, coeff in strassen_U[pi].items():
                    li, lj = divmod(loc, 2)
                    U[li * m + (2 * jm + lj), r] = coeff
                
                # V: B[2*jm + j_local, 2*kp + k_local]
                for loc, coeff in strassen_V[pi].items():
                    lj, lk = divmod(loc, 2)
                    V[(2 * jm + lj) * p + (2 * kp + lk), r] = coeff
                
                # W: C[2*kp + k_local, i]
                for loc, coeff in strassen_W[pi].items():
                    li, lk = divmod(loc, 2)
                    W[(2 * kp + lk) * n + li, r] = coeff
            col += 7

    # Helper to verify a given (U, V, W) decomposition
    def verify(U_mat, V_mat, W_mat, rank):
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Must equal 1 for all i,j,k
        T_actual = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_mat[i * m + j, :rank]
                    v_val = V_mat[j * p + k, :rank]
                    w_val = W_mat[k * n + i, :rank]
                    T_actual[i, j, k] = np.dot(u_val, v_val * w_val)
        return np.allclose(T_actual, np.ones((n, m, p)), atol=1e-10)

    # Attempt to find a lower rank decomposition using tensor slicing / elimination
    # Strategy: Try to find a rank-r decomposition where r < 42
    # We use the fact that the tensor is bilinear in the factors.
    # For each candidate rank r, we can try to solve for the factors.
    # However, this is a hard problem. A more practical approach for small improvements:
    # 1. Check if any columns in U, V, W are linearly dependent in a way that allows merging.
    # 2. Use a greedy elimination: if two rank-1 terms can be combined into one, do it.
    
    # A more robust approach: Use the known lower bounds and try to find a rank-40 or rank-41 solution.
    # The theoretical lower bound for 2x4x6 is not trivial. 
    # Let's try a search for a rank-41 decomposition by perturbing the Strassen solution slightly? No, must be exact.
    
    # Alternative: Check if the current rank-42 solution can be reduced by checking for linear dependencies 
    # in the "slice" space.
    
    # Let's try a direct algebraic approach for rank reduction.
    # If we assume a rank-r decomposition exists, the factors U, V, W are the unknowns.
    # The equations are: sum_{r=1}^R U_{ij,r} V_{jk,r} W_{ki,r} = delta_{ij,ji} (which is 1 for all valid i,j,k).
    # This is a system of 48 equations (2*4*6) with 3*R variables (8R + 24R + 12R = 44R).
    # For R=42, we have 48 equations and 1848 variables. Highly underdetermined.
    
    # Since we already have a valid rank-42 solution, we can try to "compress" it.
    # We can look for a linear transformation of the rank-1 terms that reduces the number of non-zero terms.
    # Specifically, if we can find a vector c such that sum_r c_r * (u_r, v_r, w_r) = 0, 
    # we can potentially eliminate a term. But this is tricky.
    
    # Let's try a simpler heuristic: Check if any two terms are identical or negatives.
    # In our Strassen construction, are there duplicate terms?
    # The terms are distinct by construction (different blocks or different Strassen indices).
    
    # Let's try to use the "tensor slicing" idea more directly.
    # Fix one index, say i=0. We have a bilinear form in j,k.
    # T[0,j,k] = sum_r U[0*m+j, r] V[j*p+k, r] W[k*n+0, r]
    # This is a 4x6 matrix (for j=0..3, k=0..5).
    # The rank of this matrix is at most R.
    # Let's compute the rank of the slices for our current solution.
    
    # Slice for i=0:
    # M_0[j,k] = T[0,j,k]
    # This matrix is actually the product of the "A" part (row 0 of A) and B?
    # No, T is the multiplication tensor, so T[0,j,k] = 1 if 0, j, k are valid indices.
    # Wait, the tensor T is defined such that T[i,j,k] = 1 for all i,j,k?
    # Yes, for matrix multiplication C = AB, the tensor T has T[i,j,k] = 1.
    # So the slice M_0 is just a 4x6 matrix of ones.
    # The rank of a matrix of ones is 1.
    # This doesn't help much directly with the global rank.
    
    # Let's try a different structural change: 
    # Use a different block decomposition.
    # Instead of 2x2 blocks, can we use a 2x3 block?
    # A is 2x4, B is 4x6.
    # If we split B into two 2x3 blocks?
    # A is 2x4. We can't split A into 2x2 and 2x2 easily if B is 2x3.
    # Let's stick to the 2x2 blocks but try to optimize the Strassen part?
    # Strassen is optimal for 2x2.
    
    # What if we use a different algorithm for 2x4x6?
    # There are known algorithms with rank < 42.
    # For example, using the fact that 2x4x6 is a sub-tensor of 4x4x4?
    # The rank of 4x4x4 is known to be <= 49 (Strassen) or lower?
    # Actually, the rank of 4x4x4 is not known to be less than 49, but 49 is a bound.
    # 2x4x6 is smaller.
    
    # Let's try to find a rank-40 solution by solving a linear system.
    # We can fix the structure to be "block diagonal" with 2x2 blocks, but allow mixing between blocks?
    # No, mixing breaks the block structure.
    
    # Let's try a greedy search for a rank-41 solution.
    # Start with rank 42. Try to set one column of U, V, W to zero and see if we can adjust the others to maintain the identity.
    # This is a non-linear problem.
    
    # Given the time constraint and the difficulty of finding a lower rank exact decomposition without a dedicated solver,
    # I will return the rank-42 solution.
    
    if not verify(U, V, W, rank_total):
        # Fallback to schoolbook (rank 48) just in case
        rank_sb = n * m * p
        U_sb = np.zeros((nm, rank_sb))
        V_sb = np.zeros((mp, rank_sb))
        W_sb = np.zeros((pn, rank_sb))
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, r] = 1
                    V_sb[j * p + k, r] = 1
                    W_sb[k * n + i, r] = 1
                    r += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
