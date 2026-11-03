import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_school = n * m * p  # 40

    start_time = time.time()
    deadline = start_time + 110.0

    # Build schoolbook decomposition as fallback
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_school, "V": V_school, "W": U_school[:pn]}
    best_rank = rank_school

    def verify_decomposition(U_mat, V_mat, W_mat):
        rank = U_mat.shape[1]
        # Check all 1-entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_mat[i * m + j]
                    v_val = V_mat[j * p + k]
                    w_val = W_mat[k * n + i]
                    s = np.sum(u_val * v_val * w_val)
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Check all 0-entries (sample for speed, but do all for correctness)
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        continue
                    s = np.sum(U_mat[a] * V_mat[b] * W_mat[c])
                    if abs(s) > 1e-9:
                        return False
        return True

    def to_half_int_list(mat):
        return [[int(round(mat[i, r] * 2)) / 2 for r in range(mat.shape[1])] for i in range(mat.shape[0])]

    # Strassen 2x2x2 patterns
    strassen_patterns = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    # 2x2x1 patterns (rank 4)
    patterns_221 = [
        ([1, 0, 0, 0], [1, 0], [1, 0]),
        ([0, 1, 0, 0], [0, 1], [1, 0]),
        ([0, 0, 1, 0], [1, 0], [0, 1]),
        ([0, 0, 0, 1], [0, 1], [0, 1]),
    ]

    # Build rank-36 solution (verified working)
    total_rank_36 = 36
    U36 = np.zeros((nm, total_rank_36), dtype=np.float64)
    V36 = np.zeros((mp, total_rank_36), dtype=np.float64)
    W36 = np.zeros((pn, total_rank_36), dtype=np.float64)

    col = 0
    for k_block in range(2):  # k=0,1 and k=2,3
        for j_block in range(2):  # j=0,1 and j=2,3
            for u_pat, v_pat, w_pat in strassen_patterns:
                for idx, val in enumerate(u_pat):
                    if val != 0:
                        i_local = idx // 2
                        j_local = idx % 2
                        U36[(i_local) * m + j_block * 2 + j_local, col] = val
                for idx, val in enumerate(v_pat):
                    if val != 0:
                        j_local = idx // 2
                        k_local = idx % 2
                        V36[(j_block * 2 + j_local) * p + k_block * 2 + k_local, col] = val
                for idx, val in enumerate(w_pat):
                    if val != 0:
                        i_local = idx // 2
                        k_local = idx % 2
                        W36[(k_block * 2 + k_local) * n + i_local, col] = val
                col += 1
    # Add 2x2x1 for k=4
    for j_block in range(2):
        for u_pat, v_pat, w_pat in patterns_221:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    U36[i_local * m + j_block * 2 + j_local, col] = val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx
                    V36[(j_block * 2 + j_local) * p + 4, col] = val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_local = idx
                    W36[4 * n + i_local, col] = val
            col += 1

    assert col == total_rank_36, f"Expected 36, got {col}"

    if verify_decomposition(U36, V36, W36):
        best_result = {"U": to_half_int_list(U36), "V": to_half_int_list(V36), "W": to_half_int_list(W36)}
        best_rank = 36

    if time.time() > deadline - 20:
        return best_result

    # Now try to find rank-34 using rank-10 2x2x3 decomposition
    # Strategy: split p=5 into p_left=3 and p_right=2
    # For each (i,j) pair in 2x2, we need 2x2x3 (rank 10) + 2x2x2 (rank 7)
    # But we can share structure across j blocks.
    # Total: 2 j-blocks * (10 + 7) = 34
    
    # We need a rank-10 decomposition of 2x2x3.
    # Let's try to find one by solving a system of equations.
    # The 2x2x3 tensor has 2*2*3 = 12 entries, rank 10 means 10 rank-1 terms.
    # Each rank-1 term is u ⊗ v ⊗ w where u is 2-dim, v is 2-dim, w is 3-dim.
    # We need sum of 10 such terms = the multiplication tensor.
    
    # The 2x2x3 multiplication tensor T[i,j,k] = 1 if (i,j,k) is a valid triple
    # T has 12 ones.
    
    # Let's parameterize and solve.
    # Actually, let's try a known construction.
    # The 2x2x3 tensor can be decomposed using 10 rank-1 terms.
    # One approach: use the 2x2x2 Strassen (7 terms) for k=0,1 and 2x2x1 (4 terms) for k=2
    # That gives 11, not 10. We need to save 1.
    
    # Alternative: use a different decomposition.
    # Let me try to find rank-10 by optimization.
    
    # Actually, let me try a specific construction:
    # For 2x2x3, the tensor is:
    # T[i,j,k] = delta (all 1s for valid (i,j,k))
    # We want: T = sum_{r=1}^{10} U_r[i] * V_r[j] * W_r[k]
    
    # Let's set up the equations and solve.
    # Variables: 10 * (2+2+3) = 70 unknowns
    # Equations: 12 (one per tensor entry)
    # This is underdetermined, so we need additional constraints.
    # We can fix some values to reduce symmetry.
    
    # Let me try a different approach: use the fact that 2x2x3 has a known rank-10.
    # From literature, the rank of 2x2x3 is 10.
    # Let me construct it explicitly.
    
    # The 2x2x3 tensor T where T[i,j,k]=1 for all i in {0,1}, j in {0,1}, k in {0,1,2}
    # We can write T = sum of 10 rank-1 tensors.
    
    # One construction:
    # Use 7 Strassen terms for the 2x2x2 part (k=0,1)
    # Plus 3 terms for k=2 (instead of 4)
    # The 3 terms for k=2: we need to cover 4 entries (i=0,1; j=0,1) at k=2
    # With 3 rank-1 terms, each contributing to k=2.
    # Term r: U_r[i] * V_r[j] * W_r[2]
    # We need: sum_r U_r[i]*V_r[j]*W_r[2] = 1 for all (i,j)
    # This is a 2x2 matrix of ones, which has rank 1.
    # So we can do it with 1 term: U=[1,1], V=[1,1], W[2]=1
    # But we also need the k=0,1 parts to work.
    
    # Let me try: 7 Strassen terms for k=0,1 (with W_r[2]=0 for these 7)
    # Plus 3 additional terms that cover k=2 and also adjust k=0,1 if needed.
    
    # Actually, the 7 Strassen terms already give the correct 2x2x2 for k=0,1.
    # For k=2, we just need 1 term: U=[1,1], V=[1,1], W=[0,0,1]
    # Total: 8 terms. But rank of 2x2x3 is 10, not 8!
    # So 8 is not sufficient. The issue is that the 7 Strassen terms have specific W values
    # that may not allow a simple extension.
    
    # Let me reconsider. The rank of 2x2x3 is indeed 10.
    # Let me just use 10 terms and solve.
    
    # I'll use a numerical approach: set up the 12 equations with 10 rank-1 terms.
    # Fix gauge: set W_r[0] = 1 for r=1..10 (or some other normalization)
    
    # Actually, let me just try to find a rank-10 decomposition by trial.
    # The 2x2x3 tensor can be written as:
    # T = T_2x2x2 (k=0,1) + T_2x2x1 (k=2)
    # T_2x2x2 has rank 7 (Strassen)
    # T_2x2x1 has rank 4
    # But the sum can have rank less than 7+4=11.
    # In fact, rank(T) = 10.
    
    # Let me try to find it by combining.
    # Write T = sum_{r=1}^{10} u_r ⊗ v_r ⊗ w_r
    # where w_r is a 3-vector.
    
    # Let me parameterize with 10 terms and solve the 12 equations.
    # I'll use a simple iterative method or just try random initializations.
    
    def find_2x2x3_rank10():
        """Find a rank-10 decomposition of the 2x2x3 multiplication tensor."""
        # T[i,j,k] = 1 for all i,j,k
        # We want sum_r U_r[i]*V_r[j]*W_r[k] = 1
        
        # Use 10 terms. Set up equations.
        # Unknowns: 10*(2+2+3) = 70
        # Equations: 12
        
        # Fix gauge: for each r, set W_r[0] = 1 (if possible)
        # Actually, let's just solve directly.
        
        # Let's try a specific construction based on known results.
        # The 2x2x3 tensor of rank 10.
        
        # One known construction:
        # Use the 7 Strassen terms for k=0,1, with W_r[2] = 0
        # Plus 3 more terms to handle k=2 and fix any issues.
        
        # Strassen terms for 2x2x2:
        s = [
            ([1, 0], [1, 0], [1, 0]),
            ([1, 1], [0, 1], [0, 1]),
            ([0, 1], [1, -1], [1, 0]),
            ([1, 0], [1, 1], [0, 1]),
            ([0, 1], [0, 1], [1, 1]),
            ([1, 0], [-1, 1], [1, 0]),
            ([1, 1], [1, 0], [0, 1]),
        ]
        # Wait, let me use the standard Strassen patterns properly.
        # For 2x2x2: C = A*B where A,B,C are 2x2.
        # M1 = (A00+A11)(B00+B11)
        # M2 = A10*(B00+B01)
        # M3 = A00*(B10-B11)
        # M4 = A11*(B01-B10)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B10)
        # M7 = (A01-A11)*(B01+B11)
        
        # In our indexing: A[i,j], B[j,k], C[i,k]
        # i,j in {0,1}, k in {0,1}
        
        # Let me just use the patterns from the current code.
        # strassen_patterns from above.
        
        # For 2x2x3, let me try:
        # 7 terms with W_r = [w_r0, w_r1, 0] (Strassen-like for k=0,1)
        # 3 terms with W_r = [0, 0, 1] (for k=2)
        # But the 3 terms for k=2 need to give C[i,2] = sum_j A[i,j]*B[j,2]
        # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
        # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
        
        # With 3 rank-1 terms for k=2:
        # sum_{r=8}^{10} U_r[i] * V_r[j] * 1 = C[i,2]
        # This is a 2x2 matrix that must equal A[:, :2] @ B[0:2, 2]
        # The matrix [[A00*B02 + A01*B12], [A10*B02 + A11*B12]]
        # This is a sum of 2 rank-1 matrices: A[:,0]*B[0,2] + A[:,1]*B[1,2]
        # So we need at least 2 terms. With 3 terms, we have freedom.
        
        # Let's try:
        # Term 8: U=[1,0], V=[1,0], W=[0,0,1]  -> contributes A00*B02 to C02
        # Term 9: U=[0,1], V=[1,0], W=[0,0,1]  -> contributes A10*B02 to C12
        # Term 10: U=[1,1], V=[0,1], W=[0,0,1] -> contributes A01*B12 + A11*B12
        
        # Check C02: Term8 gives 1*1*1=1 (A00*B02), Term10 gives 1*0*1=0. Total=1. Good.
        # Check C12: Term9 gives 1*1*1=1 (A10*B02), Term10 gives 1*1*1=1 (A11*B12). 
        # But we need A10*B02 + A11*B12. Term9 gives A10*B02, Term10 gives A11*B12. Good!
        # But wait, Term10 also contributes to C02: U10[0]*V10[0]*1 = 1*0*1 = 0. Good.
        
        # But we also need to check that these terms don't interfere with k=0,1.
        # Since W_r[0]=W_r[1]=0 for r=8,9,10, they don't affect k=0,1. Good.
        
        # Now check k=0,1 with the 7 Strassen terms.
        # The 7 Strassen terms give the correct 2x2x2.
        
        # So total: 7 + 3 = 10 terms. Let's verify.
        
        # Let me write out all 10 terms explicitly.
        # Using the strassen_patterns from the code:
        # Pattern format: (u_pat, v_pat, w_pat) where u_pat is 4-element (i0j0, i0j1, i1j0, i1j1)
        
        sp = strassen_patterns  # 7 patterns
        
        terms = []
        for u_pat, v_pat, w_pat in sp:
            # u_pat: [u00, u01, u10, u11]
            # v_pat: [v00, v01, v10, v11]
            # w_pat: [w00, w01, w10, w11]
            U = [u_pat[0], u_pat[1], u_pat[2], u_pat[3]]
            V = [v_pat[0], v_pat[1], v_pat[2], v_pat[3]]
            W = [w_pat[0], w_pat[1], w_pat[2], w_pat[3], 0]  # extend to 3
            terms.append((U, V, W))
        
        # Add 3 terms for k=2
        terms.append(([1, 0], [1, 0], [0, 0, 1]))
        terms.append(([0, 1], [1, 0], [0, 0, 1]))
        terms.append(([1, 1], [0, 1], [0, 0, 1]))
        
        # Verify
        # Build the tensor
        T = np.zeros((2, 2, 3))
        for U, V, W in terms:
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        T[i, j, k] += U[i] * V[j] * W[k]
        
        # Check if T[i,j,k] = 1 for all
        if np.allclose(T, 1.0):
            return terms
        else:
            # Try different combinations
            # The issue might be with the Strassen patterns.
            # Let me try to find the correct 10-term decomposition by solving.
            return None
    
    terms_223 = find_2x2x3_rank10()
    
    if terms_223 is not None and time.time() < deadline - 10:
        # Build the full 2x4x5 decomposition using 2x2x3 blocks
        # For each j-block (0-1 and 2-3), use the 2x2x3 decomposition
        # But we need to handle the k-indexing carefully.
        
        # The 2x2x3 decomposition covers k=0,1,2.
        # We need k=0,1,2,3,4.
        # So we can use:
        # j-block 0 (j=0,1): 2x2x3 for k=0,1,2 (10 terms) + 2x2x2 for k=3,4 (7 terms) = 17
        # j-block 1 (j=2,3): 2x2x3 for k=0,1,2 (10 terms) + 2x2x2 for k=3,4 (7 terms) = 17
        # Total: 34
        
        # But wait, the 2x2x2 for k=3,4 needs a Strassen decomposition.
        # We have that.
        
        total_rank_34 = 34
        U34 = np.zeros((nm, total_rank_34), dtype=np.float64)
        V34 = np.zeros((mp, total_rank_34), dtype=np.float64)
        W34 = np.zeros((pn, total_rank_34), dtype=np.float64)
        
        col = 0
        for j_block in range(2):
            # Add 2x2x3 terms for k=0,1,2
            for U_t, V_t, W_t in terms_223:
                for i in range(2):
                    if U_t[i] != 0:
                        U34[i * m + j_block * 2, col] = U_t[i]
                        U34[i * m + j_block * 2 + 1, col] = 0  # This is wrong
                # Let me redo this properly.
                # U_t is [U0, U1] for i=0,1
                # V_t is [V0, V1] for j=0,1 (local)
                # W_t is [W0, W1, W2] for k=0,1,2
                for i in range(2):
                    for j in range(2):
                        pass  # U is per (i,j), not per i
                # Actually, U_t[i] is the coefficient for A[i, j_local]
                # But in our tensor, U is indexed by (i, j) where j is global.
                # For j_block, j_global = j_block*2 + j_local
                for i in range(2):
                    U34[i * m + j_block * 2, col] = U_t[i]  # j_local=0
                    U34[i * m + j_block * 2 + 1, col] = U_t[i]  # j_local=1? No!
                
                # This is getting complicated. Let me restart with a cleaner approach.
                break
            break
        # The above is messy. Let me use a cleaner construction.
        
        # Actually, the 2x2x3 decomposition I found has U_t[i] as the coefficient for A[i, *]
        # But in the full tensor, U is indexed by (i, j) where j is the column of A.
        # For a 2x2 block (i=0,1; j=j0,j1), U[i, j] = U_t[i] * (something for j)
        
        # I think the issue is that the 2x2x3 decomposition I constructed is for
        # A being 2x2, B being 2x3, C being 2x3.
        # In the full problem, for j-block 0, A is 2x2 (columns 0,1), B is 2x5 (rows 0,1), C is 2x5.
        # So I need to map the 2x2x3 decomposition to this.
        
        # Let me just use the 36-rank solution since it's verified.
        return best_result
    else:
        return best_result
# EVOLVE-BLOCK-END
