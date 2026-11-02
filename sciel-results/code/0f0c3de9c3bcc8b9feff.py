import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit

    n, m, p = 2, 2, 11

    # The tensor for 2x2 * 2x11 matrix multiplication
    # T[a, b, c] where:
    #   a = i*m + j (i in 0..1, j in 0..1) -> 4 entries (A indices)
    #   b = j*p + k (j in 0..1, k in 0..10) -> 22 entries (B indices)
    #   c = k*n + i (k in 0..10, i in 0..1) -> 22 entries (C indices)
    # T[i*m+j, j*p+k, k*n+i] = 1

    # Rank 39: 5 Strassen blocks (rank 7 each) + 1 schoolbook column (rank 4)
    rank = 39
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    r = 0

    # Add 5 Strassen blocks for column pairs (0,1), (2,3), (4,5), (6,7), (8,9)
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1

        # B indices: B[j, k] -> V[j*p + k]
        v_e = k0         # B[0, k0]
        v_f = k1         # B[0, k1]
        v_g = p + k0     # B[1, k0]
        v_h = p + k1     # B[1, k1]

        # W indices: C[i, k] -> W[k*n + i]
        w_c00 = k0 * n    # C[0, k0]
        w_c10 = k0 * n + 1  # C[1, k0]
        w_c01 = k1 * n    # C[0, k1]
        w_c11 = k1 * n + 1  # C[1, k1]

        # A indices: A[i,j] -> U[i*m + j]
        # u00=0, u01=1, u10=2, u11=3

        # m1 = (a+d)(e+h)
        U[0, r] += 1; U[3, r] += 1
        V[v_e, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1; W[w_c11, r] += 1
        r += 1

        # m2 = (c+d)e
        U[2, r] += 1; U[3, r] += 1
        V[v_e, r] += 1
        W[w_c10, r] += 1; W[w_c11, r] += -1
        r += 1

        # m3 = a(f-h)
        U[0, r] += 1
        V[v_f, r] += 1; V[v_h, r] += -1
        W[w_c01, r] += 1; W[w_c11, r] += 1
        r += 1

        # m4 = d(g-e)
        U[3, r] += 1
        V[v_g, r] += 1; V[v_e, r] += -1
        W[w_c00, r] += 1; W[w_c10, r] += 1
        r += 1

        # m5 = (a+b)h
        U[0, r] += 1; U[1, r] += 1
        V[v_h, r] += 1
        W[w_c00, r] += -1; W[w_c01, r] += 1
        r += 1

        # m6 = (c-a)(e+f)
        U[2, r] += 1; U[0, r] += -1
        V[v_e, r] += 1; V[v_f, r] += 1
        W[w_c11, r] += 1
        r += 1

        # m7 = (b-d)(g+h)
        U[1, r] += 1; U[3, r] += -1
        V[v_g, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1
        r += 1

        if time.time() > deadline:
            break

    # Last column k=10: schoolbook (4 multiplications)
    if r < rank:
        k2 = 10
        v_e2 = k2          # B[0, 10]
        v_g2 = p + k2      # B[1, 10]
        w_c02 = k2 * n       # C[0, 10]
        w_c12 = k2 * n + 1   # C[1, 10]

        # a * B[0, 10] -> C[0,10]
        U[0, r] += 1
        V[v_e2, r] += 1
        W[w_c02, r] += 1
        r += 1

        # b * B[1, 10] -> C[0,10]
        U[1, r] += 1
        V[v_g2, r] += 1
        W[w_c02, r] += 1
        r += 1

        # c * B[0, 10] -> C[1,10]
        U[2, r] += 1
        V[v_e2, r] += 1
        W[w_c12, r] += 1
        r += 1

        # d * B[1, 10] -> C[1,10]
        U[3, r] += 1
        V[v_g2, r] += 1
        W[w_c12, r] += 1
        r += 1

    # Verify the decomposition
    def verify(U_arr, V_arr, W_arr):
        """Check if U, V, W exactly decompose the 2x2*2x11 multiplication tensor."""
        try:
            # Build target tensor: T[i*m+j, j*p+k, k*n+i] = 1
            T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i * m + j, j * p + k, k * n + i] = 1

            # Reconstruct via decomposition
            T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
            return np.array_equal(T_recon, T)
        except Exception:
            return False

    # Try to improve: attempt rank 38 by merging the last schoolbook column with Strassen
    # Idea: Can we do the last column (k=10) with fewer than 4 multiplications by sharing
    # with adjacent Strassen blocks? This is unlikely to work directly, but let's try
    # a structured search for rank 38.

    if time.time() < deadline - 30:
        # Try to find rank 38 by attempting to merge operations
        # One approach: use a modified Strassen-like scheme for the last 3 columns
        # that shares some multiplications
        
        # For now, try a targeted search: can we replace the 4 schoolbook terms for k=10
        # with 3 terms that also help with k=9?
        
        # Actually, let's try a different approach: use the fact that the tensor has
        # structure to try a rank 38 decomposition via solving a system of equations.
        
        # The tensor T has 4*22*22 = 1936 entries, but only 44 are non-zero (one for each
        # (i,j,k) triple). A rank-r decomposition means T = sum_{r'} u_r' ⊗ v_r' ⊗ w_r'
        
        # For rank 38, we need to find U (4x38), V (22x38), W (22x38) such that
        # sum_{r'} U[a,r'] V[b,r'] W[c,r'] = T[a,b,c]
        
        # This is a system of 1936 equations in 38*(4+22+22) = 2280 unknowns.
        # But the equations are nonlinear (bilinear in the entries).
        
        # A practical approach: fix some structure and solve for the rest.
        
        # Let's try: keep the first 5 Strassen blocks (35 terms) and try to do the last
        # column with 3 terms instead of 4.
        
        # The last column k=10 requires:
        # C[0,10] = a*B[0,10] + b*B[1,10]
        # C[1,10] = c*B[0,10] + d*B[1,10]
        
        # With 3 terms, we need to represent these 4 products with 3 multiplications.
        # This is impossible in general (you need at least 4 for 2x2 schoolbook), but
        # maybe we can share with the k=9 column?
        
        # Let's try a different strategy: use a 3x3 Strassen-like scheme for columns 8,9,10
        # that uses fewer than 7+4=11 terms.
        
        # Actually, the theoretical lower bound for 2x2 * 2x11 is not well-known, but
        # rank 39 is likely optimal or near-optimal. Let's focus on robust verification
        # and try a few structured modifications.
        
        # Try: modify the last Strassen block (columns 8,9) to also contribute to column 10
        # by adjusting the W matrix entries.
        
        if time.time() < deadline - 25:
            # Create a modified version where the last Strassen block also contributes to k=10
            U_mod = U.copy()
            V_mod = V.copy()
            W_mod = W.copy()
            
            # The last Strassen block (block 4) handles columns 8,9
            # Its 7 terms start at r=28
            # Let's see if we can add contributions to k=10 in the W matrix
            # and adjust U, V accordingly
            
            # This is a heuristic approach; let's try random perturbations
            # of the last 4 terms (schoolbook for k=10) to see if we can merge them.
            
            # Actually, let's try a more principled approach:
            # Solve for a rank 38 decomposition by fixing the first 35 terms (5 Strassen blocks)
            # and trying to find 3 more terms that complete the decomposition.
            
            U_fixed = U[:, :35].copy()
            V_fixed = V[:, :35].copy()
            W_fixed = W[:, :35].copy()
            
            # The residual tensor is T - sum_{r'=0}^{34} u_r' ⊗ v_r' ⊗ w_r'
            T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i * m + j, j * p + k, k * n + i] = 1
            
            T_partial = np.einsum('ar,br,cr->abc', U_fixed, V_fixed, W_fixed)
            T_residual = T - T_partial
            
            # Check if T_residual can be decomposed with rank 3
            # T_residual[a,b,c] = sum_{r'=0}^{2} U[a,r'] V[b,r'] W[c,r']
            # This is a 4x22x22 tensor with rank <= 3
            
            # The non-zero entries of T_residual should be only in the k=10 column
            # Let's check
            if np.all(T_residual[:, :, :20] == 0):
                # Good, the residual is only in the last column
                # T_residual[a, b, c] is non-zero only when c is in {20, 21} (k=10, i=0,1)
                # and b is in {10, 21} (j=0,k=10 and j=1,k=10)
                
                # We need to find U (4x3), V (22x3), W (22x3) such that
                # sum_{r'} U[a,r'] V[b,r'] W[c,r'] = T_residual[a,b,c]
                
                # This is equivalent to finding a rank-3 decomposition of the 4x22x22 tensor
                # which is zero except for specific entries.
                
                # The non-zero entries are:
                # T_residual[0, 10, 20] = 1  (a=0, b=10, c=20)
                # T_residual[1, 21, 20] = 1  (a=1, b=21, c=20)
                # T_residual[2, 10, 21] = 1  (a=2, b=10, c=21)
                # T_residual[3, 21, 21] = 1  (a=3, b=21, c=21)
                
                # This is exactly the schoolbook 2x2 matrix multiplication, which has rank 7
                # (Strassen) or 4 (schoolbook). It cannot be done with rank 3.
                # So rank 38 is not achievable with this structure.
                pass
            
            # Since rank 38 is not achievable with this simple approach, let's try
            # a global search for rank 38 with a different structure.
            
            # For now, return the rank 39 solution if it verifies.
            if verify(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Final verification and return
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook rank 44
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb), dtype=np.int64)
    V_sb = np.zeros((m * p, rank_sb), dtype=np.int64)
    W_sb = np.zeros((p * n, rank_sb), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
