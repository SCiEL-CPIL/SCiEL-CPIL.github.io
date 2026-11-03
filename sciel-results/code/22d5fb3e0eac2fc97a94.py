import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 4, 4, 6
    
    # Build schoolbook decomposition first (rank 96)
    # T[i,j,k] = 1 for all i,j,k
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    # where r = i*m + j*p + k... no, that's not right.
    # For schoolbook: each term (i,j,k) contributes A[i,j]*B[j,k] to C[i,k]
    # So U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    # with r indexing over all (i,j,k) triples
    
    # Actually for schoolbook, we have n*m*p = 4*4*6 = 96 terms
    # Each term r corresponds to a unique (i,j,k)
    # U[i*m+j, r] = 1 if the term involves A[i,j]
    # V[j*p+k, r] = 1 if the term involves B[j,k]  
    # W[k*n+i, r] = 1 if the term contributes to C[i,k]
    
    # For schoolbook, each term is a single (i,j,k) triple:
    # U has 1 at position (i*m+j), V has 1 at (j*p+k), W has 1 at (k*n+i)
    
    # Let's build the 77-term Strassen-based decomposition
    # Outer: 2x2 blocks of A (4x4), B (4x6), C (4x6)
    # Inner: 2x2 x 2x3 -> 2x3 with rank 11
    
    # Define inner terms for 2x2 x 2x3 -> 2x3
    # A_local: 2x2 (indices r,s in {0,1})
    # B_local: 2x3 (indices s in {0,1}, k in {0,1,2})
    # C_local: 2x3 (indices r in {0,1}, k in {0,1,2})
    
    # 7 Strassen terms for the 2x2 x 2x2 part (columns 0,1)
    # + 4 schoolbook terms for column 2
    # Total: 11 terms
    
    inner_terms = []
    
    # Strassen for 2x2 x 2x2 (columns 0 and 1)
    # M1 = (A00+A11)(B00+B11)
    inner_terms.append((
        [(0,0,1), (1,1,1)],
        [(0,0,1), (1,1,1)],
        [(0,0,1), (1,1,1)]
    ))
    # M2 = (A10+A11)B00
    inner_terms.append((
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,0,1), (1,1,-1)]
    ))
    # M3 = A00(B10-B00)
    inner_terms.append((
        [(0,0,1)],
        [(1,0,1), (0,0,-1)],
        [(0,1,1), (1,1,1)]
    ))
    # M4 = A11(B01-B10)
    inner_terms.append((
        [(1,1,1)],
        [(0,1,1), (1,0,-1)],
        [(0,0,1), (1,0,1)]
    ))
    # M5 = (A00+A01)B11
    inner_terms.append((
        [(0,0,1), (0,1,1)],
        [(1,1,1)],
        [(0,0,-1), (0,1,1)]
    ))
    # M6 = (A10-A00)(B00+B01)
    inner_terms.append((
        [(1,0,1), (0,0,-1)],
        [(0,0,1), (0,1,1)],
        [(1,1,1)]
    ))
    # M7 = (A01-A11)(B10+B11)
    inner_terms.append((
        [(0,1,1), (1,1,-1)],
        [(1,0,1), (1,1,1)],
        [(0,0,1)]
    ))
    
    # Schoolbook for column 2
    # A00*B02 -> C02
    inner_terms.append((
        [(0,0,1)],
        [(0,2,1)],
        [(0,2,1)]
    ))
    # A01*B12 -> C02
    inner_terms.append((
        [(0,1,1)],
        [(1,2,1)],
        [(0,2,1)]
    ))
    # A10*B02 -> C12
    inner_terms.append((
        [(1,0,1)],
        [(0,2,1)],
        [(1,2,1)]
    ))
    # A11*B12 -> C12
    inner_terms.append((
        [(1,1,1)],
        [(1,2,1)],
        [(1,2,1)]
    ))
    
    # Outer Strassen for 4x4 x 4x6
    # A = [[A00, A01], [A10, A11]] each 2x2
    # B = [[B00, B01], [B10, B11]] each 2x3
    # C = [[C00, C01], [C10, C11]] each 2x3
    #
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # Block indexing: block b has row_block = b//2, col_block = b%2
    # For A (4x4): global (I,J) = (2*(b//2)+r, 2*(b%2)+s)
    # For B (4x6): global (J,K) = (2*(b//2)+s, 3*(b%2)+k)
    # For C (4x6): global (I,K) = (2*(b//2)+r, 3*(b%2)+k)
    
    outer_terms = [
        # M1: A={0,3}, B={0,3}, C={0,3}
        ([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]),
        # M2: A={2,3}, B={0}, C={2,3} with C3=-1
        ([(2,1), (3,1)], [(0,1)], [(2,1), (3,-1)]),
        # M3: A={0}, B={1,3} with B3=-1, C={1,3} with C3=1... 
        # Wait, let me re-derive.
        # M3 = A00*(B10-B00)
        # A00 is block 0, B10 is block 2, B00 is block 0
        # So A={0}, B={2,0} with coeffs (1,-1)
        # C01 = M3 + M5, so M3 contributes to C01 (block 1)
        # But M3 is a 2x3 matrix, and it goes into C01
        # Actually each M is a full 2x3 result
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        #
        # So M3 appears in C01 and C11
        # M3 = A00 * (B10 - B00)
        # A00 = block 0
        # B10 = block 2, B00 = block 0
        # Result is a 2x3 matrix that goes into C01 and C11
        # C01 is block 1, C11 is block 3
    ]
    
    # Let me redefine more carefully
    # Each outer term M_oi is a 2x3 matrix computed as (A_comb) * (B_comb)
    # where A_comb is a 2x2 matrix (sum of A blocks with coeffs)
    # and B_comb is a 2x3 matrix (sum of B blocks with coeffs)
    # Then M_oi contributes to various C blocks with coefficients
    
    # M1 = (A00+A11) * (B00+B11)
    #   A blocks: 0(+1), 3(+1)
    #   B blocks: 0(+1), 3(+1)
    #   C contributions: C00(+1), C11(+1)
    # M2 = (A10+A11) * B00
    #   A blocks: 2(+1), 3(+1)
    #   B blocks: 0(+1)
    #   C contributions: C10(+1), C11(-1)
    # M3 = A00 * (B10-B00)
    #   A blocks: 0(+1)
    #   B blocks: 2(+1), 0(-1)
    #   C contributions: C01(+1), C11(+1)
    # M4 = A11 * (B01-B10)
    #   A blocks: 3(+1)
    #   B blocks: 1(+1), 2(-1)
    #   C contributions: C00(+1), C10(+1)
    # M5 = (A00+A01) * B11
    #   A blocks: 0(+1), 1(+1)
    #   B blocks: 3(+1)
    #   C contributions: C00(-1), C01(+1)
    # M6 = (A10-A00) * (B00+B01)
    #   A blocks: 2(+1), 0(-1)
    #   B blocks: 0(+1), 1(+1)
    #   C contributions: C11(+1)
    # M7 = (A01-A11) * (B10+B11)
    #   A blocks: 1(+1), 3(-1)
    #   B blocks: 2(+1), 3(+1)
    #   C contributions: C00(-1), C11(+1)
    
    outer_terms = [
        # (A_blocks, B_blocks, C_blocks)
        # C_blocks: list of (block_id, coeff)
        ([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]),       # M1
        ([(2,1), (3,1)], [(0,1)], [(2,1), (3,-1)]),              # M2
        ([(0,1)], [(2,1), (0,-1)], [(1,1), (3,1)]),              # M3
        ([(3,1)], [(1,1), (2,-1)], [(0,1), (2,1)]),              # M4
        ([(0,1), (1,1)], [(3,1)], [(0,-1), (1,1)]),              # M5
        ([(2,1), (0,-1)], [(0,1), (1,1)], [(3,1)]),              # M6
        ([(1,1), (3,-1)], [(2,1), (3,1)], [(0,-1), (3,1)]),      # M7
    ]
    
    rank_base = 7 * 11  # 77
    U = np.zeros((n*m, rank_base))
    V = np.zeros((m*p, rank_base))
    W = np.zeros((p*n, rank_base))
    
    def get_A_global(b, r, s):
        """Get global (I, J) for A block b, local (r, s)"""
        I = 2*(b//2) + r
        J = 2*(b%2) + s
        return I, J
    
    def get_B_global(b, s, k):
        """Get global (J, K) for B block b, local (s, k)"""
        J = 2*(b//2) + s
        K = 3*(b%2) + k
        return J, K
    
    def get_C_global(b, r, k):
        """Get global (I, K) for C block b, local (r, k)"""
        I = 2*(b//2) + r
        K = 3*(b%2) + k
        return I, K
    
    for oi, (A_blocks, B_blocks, C_blocks) in enumerate(outer_terms):
        for ii, (A_inner, B_inner, C_inner) in enumerate(inner_terms):
            r_idx = oi * 11 + ii
            
            # A contributions: U[I*m+J, r_idx]
            for (r_loc, s_loc, ca_inner) in A_inner:
                for (b, cb_outer) in A_blocks:
                    I, J = get_A_global(b, r_loc, s_loc)
                    U[I*m + J, r_idx] += ca_inner * cb_outer
            
            # B contributions: V[J*p+K, r_idx]
            for (s_loc, k_loc, cb_inner) in B_inner:
                for (b, cb_outer) in B_blocks:
                    J, K = get_B_global(b, s_loc, k_loc)
                    V[J*p + K, r_idx] += cb_inner * cb_outer
            
            # W contributions: W[K*n+I, r_idx]
            for (r_loc, k_loc, cc_inner) in C_inner:
                for (b, cb_outer) in C_blocks:
                    I, K = get_C_global(b, r_loc, k_loc)
                    W[K*n + I, r_idx] += cc_inner * cb_outer
    
    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat, n, m, p):
        """Check if sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1 for all i,j,k"""
        R = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(R):
                        u_val = U_mat[i*m+j, r]
                        v_val = V_mat[j*p+k, r]
                        w_val = W_mat[k*n+i, r]
                        total += u_val * v_val * w_val
                    if abs(total - 1.0) > 1e-9:
                        return False, (i, j, k, total)
        return True, None
    
    if time.time() < deadline - 20:
        valid, info = verify(U, V, W, n, m, p)
        if not valid:
            # Fall back to schoolbook
            R_school = n * m * p
            U_school = np.zeros((n*m, R_school))
            V_school = np.zeros((m*p, R_school))
            W_school = np.zeros((p*n, R_school))
            r = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_school[i*m+j, r] = 1
                        V_school[j*p+k, r] = 1
                        W_school[k*n+i, r] = 1
                        r += 1
            return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Try to reduce rank by finding linear dependencies
    # This is complex, so for now return the 77-term decomposition
    # If we have time, we could try more advanced techniques
    
    if time.time() < deadline - 5:
        # Check for zero columns
        nonzero_cols = []
        for r in range(rank_base):
            if np.any(U[:, r] != 0) and np.any(V[:, r] != 0) and np.any(W[:, r] != 0):
                nonzero_cols.append(r)
        
        if len(nonzero_cols) < rank_base:
            U = U[:, nonzero_cols]
            V = V[:, nonzero_cols]
            W = W[:, nonzero_cols]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
