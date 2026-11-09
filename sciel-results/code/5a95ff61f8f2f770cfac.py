import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 3]

def construct():
    start_time = time.time()
    deadline = 120.0

    n, m, p = DIMENSIONS  # A: n x m, B: m x p, C: n x p

    def verify_decomp(U, V, W, rank, tol=1e-8):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m + j, :] * V[j*p + k, :] * W[k*n + i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def round_to_half(arr):
        return np.round(arr * 2) / 2.0

    def get_rank_17_solution():
        rank = 17
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Term 1: Strassen-like for 2x2x2 core (terms 0-6)
        # C[0,0] = A00*B00 + A01*B10 + A02*B20
        # C[0,1] = A00*B01 + A01*B11 + A02*B21
        # C[1,0] = A10*B00 + A11*B10 + A12*B20
        # C[1,1] = A10*B01 + A11*B11 + A12*B21
        # C[0,2] = A00*B02 + A01*B12 + A02*B22
        # C[1,2] = A10*B02 + A11*B12 + A12*B22
        
        # Using the verified rank-17 from the original program:
        # Term 1
        U[0][0] += 1; U[4][0] += 1
        V[0][0] += 1; V[4][0] += 1
        W[0][0] += 1; W[3][0] += 1
        
        # Term 2
        U[3][1] += 1; U[4][1] += 1
        V[0][1] += 1
        W[1][1] += 1; W[3][1] -= 1
        
        # Term 3
        U[0][2] += 1
        V[1][2] += 1; V[4][2] -= 1
        W[2][2] += 1; W[3][2] += 1
        
        # Term 4
        U[4][3] += 1
        V[3][3] += 1; V[0][3] -= 1
        W[0][3] += 1; W[1][3] += 1
        
        # Term 5
        U[0][4] += 1; U[1][4] += 1
        V[4][4] += 1
        W[0][4] -= 1; W[2][4] += 1
        
        # Term 6
        U[3][5] += 1; U[0][5] -= 1
        V[0][5] += 1; V[1][5] += 1
        W[3][5] += 1
        
        # Term 7
        U[1][6] += 1; U[4][6] -= 1
        V[3][6] += 1; V[4][6] += 1
        W[0][6] += 1
        
        # Term 8: A00B02
        U[0][7] += 1
        V[2][7] += 1
        W[4][7] += 1
        
        # Term 9: A01B12
        U[1][8] += 1
        V[5][8] += 1
        W[4][8] += 1
        
        # Term 10: A10B02
        U[3][9] += 1
        V[2][9] += 1
        W[5][9] += 1
        
        # Term 11: A11B12
        U[4][10] += 1
        V[5][10] += 1
        W[5][10] += 1
        
        # Term 12: A02B20
        U[2][11] += 1
        V[6][11] += 1
        W[0][11] += 1
        
        # Term 13: A02B21
        U[2][12] += 1
        V[7][12] += 1
        W[2][12] += 1
        
        # Term 14: A02B22
        U[2][13] += 1
        V[8][13] += 1
        W[4][13] += 1
        
        # Term 15: A12B20
        U[5][14] += 1
        V[6][14] += 1
        W[1][14] += 1
        
        # Term 16: A12B21
        U[5][15] += 1
        V[7][15] += 1
        W[3][15] += 1
        
        # Term 17: A12B22
        U[5][16] += 1
        V[8][16] += 1
        W[5][16] += 1
        
        return U, V, W

    def try_rank_16_structured():
        """
        Try to find a rank-16 decomposition by exploiting the structure.
        
        Key observation: The 6 terms involving A02/A12 and B20/B21/B22 form
        a 2x3x3 sub-tensor. We can try to decompose this with fewer than 6 terms
        by combining with the Strassen terms for the 2x2x2 core.
        
        Strategy: Use Strassen rank-7 for the 2x2x2 core (A00,A01,A10,A11 x B00,B01,B10,B11),
        then find a rank-9 decomposition for the remaining 6 C entries that involve
        A02, A12, B02, B12, B20, B21, B22.
        
        Actually, let's try a different structural approach:
        - The full tensor has 18 rank-1 terms (schoolbook).
        - We need to reduce to 16, so we need to save 2 terms.
        - Look for pairs of terms that can be merged using (x+y)(z+w) = xz + xw + yz + yw
          type identities, but in a way that saves exactly 2 total terms.
        """
        if time.time() > deadline - 60:
            return None
        
        rank = 16
        
        # Approach: Try all pairs of "merge patterns" on the rank-17 solution
        # The rank-17 solution has 7 Strassen terms + 10 simple terms.
        # Try to merge 2 of the simple terms with 1 Strassen term to save 1 term,
        # and find another such merge to save another term.
        
        # Actually, let's try a more direct algebraic search.
        # Represent the tensor as a 3D array and try to find rank-16 decomposition
        # using a structured ansatz.
        
        # Build the target tensor T[i,j,k] = 1 if (i,j,k) corresponds to a valid product term
        # T[A_idx, B_idx, C_idx] where:
        #   A_idx = i*m + j (i in {0,1}, j in {0,1,2})
        #   B_idx = j*p + k (j in {0,1,2}, k in {0,1,2})  
        #   C_idx = k*n + i (k in {0,1,2}, i in {0,1})
        # T[A_idx, B_idx, C_idx] = 1 iff the product A[i,j]*B[j,k] contributes to C[i,k]
        
        # The valid (i,j,k) triples for A[i,j]*B[j,k] -> C[i,k]:
        valid_terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    A_idx = i * m + j
                    B_idx = j * p + k
                    C_idx = k * n + i
                    valid_terms.append((A_idx, B_idx, C_idx))
        
        # Now try a structured rank-16 ansatz:
        # Use 7 terms for the 2x2x2 core (Strassen) and 9 terms for the rest.
        # The "rest" involves A02, A12, B02, B12, B20, B21, B22.
        
        # Let's define the 9 terms for the extended part.
        # We need to cover:
        # C[0,2] = A00*B02 + A01*B12 + A02*B22
        # C[1,2] = A10*B02 + A11*B12 + A12*B22
        # C[0,0] += A02*B20
        # C[0,1] += A02*B21
        # C[1,0] += A12*B20
        # C[1,1] += A12*B21
        
        # Try a rank-9 ansatz for these 6 contributions:
        # Term A: (A00+A02)*(B02+B20) -> contributes to C[0,2] and C[0,0]
        #   A00*B02 -> C[0,2] (good)
        #   A00*B20 -> not valid (A00 is row 0, B20 is col 0, but they don't multiply)
        #   This doesn't work because A and B indices must match in the middle.
        
        # The constraint is that for a rank-1 term U[a,t]*V[b,t]*W[c,t],
        # we need b = j*p+k and a = i*m+j for some consistent j.
        # This means the middle index j must be the same in both A and B.
        
        # So we can only combine terms that share the same middle index j.
        # For j=0: A00,A01,A02 can combine with B00,B01,B02
        # For j=1: A01,A11,A12 can combine with B10,B11,B12  
        # For j=2: A02,A12 can combine with B20,B21,B22
        
        # Wait, that's not right. Let me re-read the indexing.
        # A is n x m = 2 x 3, so A[i,j] with i in {0,1}, j in {0,1,2}
        # B is m x p = 3 x 3, so B[j,k] with j in {0,1,2}, k in {0,1,2}
        # C is n x p = 2 x 3, so C[i,k] with i in {0,1}, k in {0,1,2}
        # C[i,k] = sum_j A[i,j] * B[j,k]
        
        # So for a rank-1 term, we pick:
        # - A indices: subset of {(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)}
        # - B indices: subset of {(0,0),...,(2,2)}
        # - C indices: subset of {(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)}
        # And the product must exactly match: for each (i,j) in A and (j,k) in B with same j,
        # the term contributes to C[i,k].
        
        # The key constraint: if A contains (i1,j1) and (i2,j2) with j1 != j2,
        # and B contains (j1,k1) and (j2,k2), then we get cross terms that may not be wanted.
        
        # Let's try a systematic search over structured rank-16 decompositions.
        # Use the rank-17 solution as a starting point and try to merge terms.
        
        U17, V17, W17 = get_rank_17_solution()
        
        # Try dropping each pair of columns and see if the remaining 15 columns
        # can be adjusted to form a valid rank-15... no, we want rank-16.
        
        # Try dropping 1 column from rank-17 and adding a new column that compensates.
        for drop_idx in range(17):
            if time.time() > deadline - 30:
                break
            
            mask = [i for i in range(17) if i != drop_idx]
            U0 = U17[:, mask].copy()
            V0 = V17[:, mask].copy()
            W0 = W17[:, mask].copy()
            
            # The residual tensor after removing column drop_idx
            # We need to find a rank-1 term that exactly cancels the residual.
            # Residual R[a,b,c] = sum_{t in mask} U0[a,t]*V0[b,t]*W0[c,t] - T[a,b,c]
            # We need R to be a rank-1 tensor: R[a,b,c] = U_new[a]*V_new[b]*W_new[c]
            
            # Compute residual
            residual = np.zeros((n*m, m*p, p*n))
            for a in range(n*m):
                for b in range(m*p):
                    for c in range(p*n):
                        s = 0.0
                        for t in range(16):
                            s += U0[a,t] * V0[b,t] * W0[c,t]
                        # Target: T[a,b,c] = 1 if (a,b,c) is a valid term, else 0
                        i_a = a // m
                        j_a = a % m
                        j_b = b // p
                        k_b = b % p
                        k_c = c // n
                        i_c = c % n
                        target = 1.0 if (j_a == j_b and i_a == i_c and k_b == k_c) else 0.0
                        residual[a,b,c] = s - target
            
            # Check if residual is rank-1: residual[a,b,c] = u[a]*v[b]*w[c]
            # For a non-zero residual, pick a,b,c with residual != 0
            nonzero = np.argwhere(np.abs(residual) > 1e-10)
            if len(nonzero) == 0:
                # Already rank-16!
                return [U0.tolist(), V0.tolist(), W0.tolist()]
            
            # Try to fit rank-1
            a0, b0, c0 = nonzero[0]
            r0 = residual[a0,b0,c0]
            if abs(r0) < 1e-10:
                continue
            
            # u[a]*v[b]*w[c] = residual
            # Set u[a0] = 1, then v[b]*w[c] = residual[a0,b,c]/r0 * r0... 
            # Let's set u[a0] = r0, v[b0] = 1, w[c0] = 1
            # Then for all a: u[a] = residual[a,b0,c0]
            # For all b: v[b] = residual[a0,b,c0]
            # For all c: w[c] = residual[a0,b0,c]
            
            u_new = residual[:, b0, c0] / r0
            v_new = residual[a0, :, c0] / r0
            w_new = residual[a0, b0, :] / r0
            
            # Verify: u_new[a]*v_new[b]*w_new[c] should equal residual[a,b,c]
            # u_new[a]*v_new[b]*w_new[c] = residual[a,b0,c0]*residual[a0,b,c0]*residual[a0,b0,c] / r0^2
            # This equals residual[a,b,c] only if the tensor is rank-1.
            
            max_err = 0.0
            for a in range(n*m):
                for b in range(m*p):
                    for c in range(p*n):
                        predicted = u_new[a] * v_new[b] * w_new[c]
                        err = abs(predicted - residual[a,b,c])
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-8:
                # We have a rank-16 decomposition!
                U16 = np.column_stack([U0, u_new])
                V16 = np.column_stack([V0, v_new])
                W16 = np.column_stack([W0, w_new])
                
                # Verify
                if verify_decomp(U16, V16, W16, 16):
                    # Try rounding
                    for round_fn in [round_to_half, lambda x: np.round(x)]:
                        U_r = round_fn(U16)
                        V_r = round_fn(V16)
                        W_r = round_fn(W16)
                        if verify_decomp(U_r, V_r, W_r, 16):
                            return [U_r.tolist(), V_r.tolist(), W_r.tolist()]
                    return [U16.tolist(), V16.tolist(), W16.tolist()]
        
        return None

    # Try rank 16
    result_16 = try_rank_16_structured()
    
    if result_16:
        U, V, W = result_16
        return {"U": U, "V": V, "W": W}
        
    # Fall back to rank 17
    U17, V17, W17 = get_rank_17_solution()
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
