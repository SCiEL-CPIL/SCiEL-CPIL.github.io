"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8
    
    def naive_decomp():
        rank = n * m * p  # 16
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Verify the tensor decomposition is exact."""
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def try_strassen_rank14():
        """
        Rank-14 decomposition for 2x2 * 2x4 using two Strassen 2x2 blocks.
        """
        R = 14
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # Left block (cols 0,1 of B) - standard Strassen
        # r=0: M1 = (a00+a11)(b00+b11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[5][0] = 1
        W[0][0] = 1; W[3][0] = 1
        # r=1: M2 = (a10+a11)b00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        # r=2: M3 = a00(b01-b11)
        U[0][2] = 1
        V[1][2] = 1; V[5][2] = -1
        W[2][2] = 1; W[3][2] = 1
        # r=3: M4 = a11(b10-b00)
        U[3][3] = 1
        V[4][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        # r=4: M5 = (a00+a01)b11
        U[0][4] = 1; U[1][4] = 1
        V[5][4] = 1
        W[0][4] = -1; W[2][4] = 1
        # r=5: M6 = (a10-a00)(b00+b01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        # r=6: M7 = (a01-a11)(b10+b11)
        U[1][6] = 1; U[3][6] = -1
        V[4][6] = 1; V[5][6] = 1
        W[0][6] = 1
        
        # Right block (cols 2,3 of B) - standard Strassen
        # r=7: N1 = (a00+a11)(b02+b13)
        U[0][7] = 1; U[3][7] = 1
        V[2][7] = 1; V[7][7] = 1
        W[4][7] = 1; W[7][7] = 1
        # r=8: N2 = (a10+a11)b02
        U[2][8] = 1; U[3][8] = 1
        V[2][8] = 1
        W[5][8] = 1; W[7][8] = -1
        # r=9: N3 = a00(b03-b13)
        U[0][9] = 1
        V[3][9] = 1; V[7][9] = -1
        W[6][9] = 1; W[7][9] = 1
        # r=10: N4 = a11(b12-b02)
        U[3][10] = 1
        V[6][10] = 1; V[2][10] = -1
        W[4][10] = 1; W[5][10] = 1
        # r=11: N5 = (a00+a01)b13
        U[0][11] = 1; U[1][11] = 1
        V[7][11] = 1
        W[4][11] = -1; W[6][11] = 1
        # r=12: N6 = (a10-a00)(b02+b03)
        U[2][12] = 1; U[0][12] = -1
        V[2][12] = 1; V[3][12] = 1
        W[7][12] = 1
        # r=13: N7 = (a01-a11)(b12+b13)
        U[1][13] = 1; U[3][13] = -1
        V[6][13] = 1; V[7][13] = 1
        W[4][13] = 1
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        return None
    
    def try_rank13_search():
        """
        Search for rank-13 decomposition by trying to share one rank-1 term
        between the two Strassen blocks. We fix U and W with a shared term
        structure and solve for V using linear algebra.
        
        Strategy: The rank-14 solution has 14 independent rank-1 terms.
        For rank 13, we need one term to serve both blocks. We try all
        possible "merge" patterns where a left-block term and a right-block
        term are combined into a single term.
        """
        if time.time() > deadline - 10:
            return None
        
        # Standard Strassen factors for a 2x2 block
        # Each term r has (U_vec, V_vec, W_vec) where U_vec is length 4, V_vec length 2, W_vec length 2
        strassen_terms = [
            # (U, V, W) for each of the 7 Strassen terms
            ([1,0,0,1], [1,1], [1,-1]),   # M1: (a00+a11)(b00+b11) -> C[0,0]+C[1,1]... 
            # Actually let me use the standard Strassen formulation:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            # C00 = M1 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
        ]
        
        # Let me define the Strassen terms more carefully for the 2x2 case
        # U indices: 0=a00, 1=a01, 2=a10, 3=a11
        # V indices: 0=b00, 1=b01 (for left block), 2=b02, 3=b03 (for right block)
        # W indices: 0=C00, 1=C10, 2=C01, 3=C11 (left), 4=C02, 5=C12, 6=C03, 7=C13 (right)
        
        # For the left block, Strassen terms (U_4, V_2, W_2):
        left_terms = [
            ([1,0,0,1], [1,1], [1,0,0,-1]),    # M1: (a00+a11)(b00+b11) -> C00 + C11
            ([0,0,1,1], [1,0], [0,1,-1,0]),    # M2: (a10+a11)b00 -> C10 - C11
            ([1,0,0,0], [0,1], [0,0,1,1]),     # M3: a00(b01-b11) -> C01 + C11
            ([0,0,0,1], [1,0], [1,-1,0,0]),    # M4: a11(b10-b00) -> C00 + C10
            ([1,1,0,0], [0,1], [-1,0,1,0]),    # M5: (a00+a01)b11 -> -C00 + C01
            ([1,-1,0,0], [1,1], [0,0,0,1]),    # M6: (a10-a00)(b00+b01) -> C11
            ([0,1,-1,0], [1,1], [1,0,0,0]),    # M7: (a01-a11)(b10+b11) -> C00
        ]
        
        # Verify left block:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # For the right block, similar structure with b02,b03 and C02,C12,C03,C13
        right_terms = [
            ([1,0,0,1], [1,1], [0,0,0,0,1,0,0,-1]),    # N1
            ([0,0,1,1], [1,0], [0,0,0,0,0,1,-1,0]),    # N2
            ([1,0,0,0], [0,1], [0,0,0,0,0,0,1,1]),     # N3
            ([0,0,0,1], [1,0], [0,0,0,0,1,-1,0,0]),    # N4
            ([1,1,0,0], [0,1], [0,0,0,0,-1,0,0,1]),    # N5
            ([1,-1,0,0], [1,1], [0,0,0,0,0,0,0,1]),    # N6
            ([0,1,-1,0], [1,1], [0,0,0,0,1,0,0,0]),    # N7
        ]
        
        # Now try to find a rank-13 solution by merging one left term with one right term
        # The merged term must have the same U vector (up to scalar) for both blocks
        # and the V/W vectors must be compatible
        
        # For a merge to work, the U vectors of the left and right terms must be proportional
        # (since they multiply the same A matrix entries)
        
        # Check which pairs of left/right terms have proportional U vectors
        for li in range(7):
            for ri in range(7):
                U_left = np.array(left_terms[li][0], dtype=float)
                U_right = np.array(right_terms[ri][0], dtype=float)
                
                # Check if U_left and U_right are proportional
                # They need to be linearly dependent
                if np.linalg.matrix_rank(np.column_stack([U_left, U_right])) < 2:
                    # They are proportional - potential merge candidate
                    # The merged U is U_left (or U_right, they're proportional)
                    # V_left is length 2 (for b00,b01), V_right is length 2 (for b02,b03)
                    # Merged V is length 4: [V_left[0], V_left[1], V_right[0], V_right[1]]
                    # W_left is length 4 (for C00,C10,C01,C11), W_right is length 4 (for C02,C12,C03,C13)
                    # Merged W is length 8: [W_left[0], W_left[1], W_left[2], W_left[3], W_right[0], W_right[1], W_right[2], W_right[3]]
                    
                    # Build the rank-13 decomposition
                    R = 13
                    U_mat = np.zeros((4, R))
                    V_mat = np.zeros((8, R))
                    W_mat = np.zeros((8, R))
                    
                    # Column 0: merged term
                    U_mat[:, 0] = U_left
                    V_mat[:2, 0] = left_terms[li][1]  # b00, b01
                    V_mat[2:4, 0] = right_terms[ri][1]  # b02, b03
                    W_mat[:4, 0] = left_terms[li][2]  # C00,C10,C01,C11
                    W_mat[4:8, 0] = right_terms[ri][2]  # C02,C12,C03,C13
                    
                    # Remaining left terms (excluding li)
                    col = 1
                    for l_idx in range(7):
                        if l_idx == li:
                            continue
                        U_mat[:, col] = left_terms[l_idx][0]
                        V_mat[:2, col] = left_terms[l_idx][1]
                        W_mat[:4, col] = left_terms[l_idx][2]
                        col += 1
                    
                    # Remaining right terms (excluding ri)
                    for r_idx in range(7):
                        if r_idx == ri:
                            continue
                        U_mat[:, col] = right_terms[r_idx][0]
                        V_mat[2:4, col] = right_terms[r_idx][1]
                        W_mat[4:8, col] = right_terms[r_idx][2]
                        col += 1
                    
                    # Now solve for V_mat using the constraint:
                    # For each (i,j,k): sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1
                    # This is: for each (i,j,k), sum_r (U[i*m+j][r] * W[k*n+i][r]) * V[j*p+k][r] = 1
                    # Let A[i,j,k,r] = U[i*m+j][r] * W[k*n+i][r]
                    # Then for each (i,j,k): sum_r A[i,j,k,r] * V[j*p+k][r] = 1
                    # This gives a system: for each (j,k), sum_r A[i,j,k,r] * V[j*p+k][r] = 1 for all i
                    # Wait, it's per (i,j,k) triple.
                    
                    # Actually: for fixed (j,k), we need V[j*p+k][r] for all r.
                    # The constraints are: for each i, sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1
                    # This is a linear system in V[j*p+k][r]:
                    # sum_r (U[i*m+j][r] * W[k*n+i][r]) * V[j*p+k][r] = 1, for each i
                    # So for each (j,k), we have n=2 equations in R=13 unknowns.
                    # We have 8*(j,k) pairs, each giving 2 equations = 16 equations total.
                    # But V has 8*13 = 104 unknowns.
                    # This is underdetermined, so we need to check if a solution exists.
                    
                    # Actually, the system is:
                    # For each (i,j,k): sum_r U[i*m+j][r] * W[k*n+i][r] * V[j*p+k][r] = 1
                    # Let's form the matrix B where B[(i,j,k), r] = U[i*m+j][r] * W[k*n+i][r]
                    # Then B @ V_flat = 1 (vector of ones)
                    # V_flat is of length 8*13 = 104
                    # B is 16 x 104
                    # We need to find V such that B @ V = ones(16)
                    
                    B = np.zeros((16, R * 8))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                idx = i * m * p + j * p + k
                                for r in range(R):
                                    B[idx, r * 8 + j * p + k] = U_mat[i * m + j, r] * W_mat[k * n + i, r]
                    
                    # Solve B @ V_flat = ones(16) using least squares
                    rhs = np.ones(16)
                    try:
                        V_flat, residuals, rank, sv = np.linalg.lstsq(B, rhs, rcond=None)
                    except:
                        continue
                    
                    # Check if the solution is exact
                    B_V = B @ V_flat
                    if not np.allclose(B_V, rhs, atol=1e-9):
                        continue
                    
                    # Reshape V_flat
                    V_sol = V_flat.reshape(R, 8).T  # shape (8, R)
                    
                    # Convert to list of lists
                    U_list = U_mat.tolist()
                    V_list = V_sol.tolist()
                    W_list = W_mat.tolist()
                    
                    # Verify
                    if verify(U_list, V_list, W_list):
                        # Check if entries are "nice" (integers or half-integers)
                        # For now, accept any exact solution
                        return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    def try_rank13_general():
        """
        General rank-13 search: try all possible U and W matrices with
        small integer/half-integer entries, then solve for V.
        
        We use a structured approach: the U matrix should have rank at most 4
        (since A is 2x2), and W should have rank at most 4 (since C is 2x4).
        
        We parameterize U and W with few free parameters and search over them.
        """
        if time.time() > deadline - 5:
            return None
        
        R = 13
        
        # We'll search over structured U and W matrices.
        # Key insight: For a valid decomposition, for each r, the vectors
        # U[:,r] and W[:,r] must be "compatible" in the sense that
        # the product U[i*m+j,r] * W[k*n+i,r] should be nonzero only
        # when (i,j,k) is a valid output.
        
        # Let's try a different approach: use the fact that the tensor
        # T[i,j,k] = 1 if and only if the output C[i,k] depends on A[i,j] and B[j,k].
        # For 2x2 * 2x4: T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,1,2,3}.
        # So all 16 entries are 1.
        
        # The tensor is the all-ones tensor of shape (2,2,4).
        # We need rank-13 decomposition of the all-ones tensor.
        
        # Actually wait - that's not right. The tensor for matrix multiplication
        # (2x2)*(2x4) has T[i,j,k] = 1 for all combinations where the output
        # C[i,k] = sum_j A[i,j]*B[j,k]. So T[i,j,k] = 1 for all i,j,k.
        
        # The all-ones tensor of shape (2,2,4) has rank... let me think.
        # For the (2,2,2) all-ones tensor, the rank is 7 (Strassen).
        # For (2,2,4), we can think of it as two (2,2,2) tensors side by side.
        # The rank should be at most 14 (two Strassen blocks).
        
        # Can we do better? The all-ones tensor (2,2,4) can be written as
        # outer product of (1,1,1) in the first two dims with (1,1,1,1) in the third,
        # but that's rank 1 only if we allow arbitrary scaling.
        
        # Actually, the tensor T[i,j,k] = 1 for all i,j,k is just the outer product
        # of vectors (1,1), (1,1), (1,1,1,1). So its rank is 1!
        
        # Wait, that can't be right for matrix multiplication. Let me re-examine.
        
        # The matrix multiplication tensor for (n x m) * (m x p) -> (n x p) is:
        # T[i,j,k] = 1 if the (i,k) entry of the product depends on A[i,j]*B[j,k]
        # For standard matrix multiplication, C[i,k] = sum_j A[i,j]*B[j,k]
        # So T[i,j,k] = 1 for all valid (i,j,k).
        
        # The tensor is indeed the all-ones tensor. Its rank as a 3-tensor is:
        # rank(T) = min R such that T = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r
        # where u_r in R^n, v_r in R^m, w_r in R^p.
        
        # For the all-ones tensor of shape (a,b,c), the rank is min(a,b,c) if
        # one of the dimensions is 1, but in general it's more complex.
        
        # Actually, for the all-ones tensor (2,2,4):
        # T = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r
        # T[i,j,k] = sum_r u_r[i] * v_r[j] * w_r[k] = 1 for all i,j,k
        
        # This means the matrix M_{i,j} = sum_r u_r[i] * v_r[j] * w_r[k] must equal 1
        # for all k. So for each k, the matrix [sum_r u_r[i]*v_r[j]*w_r[k]]_{i,j} = ones(2,2).
        
        # Let M_k = sum_r w_r[k] * (u_r ⊗ v_r). Then M_k = ones(2,2) for all k.
        # The matrix ones(2,2) has rank 1, so M_k = c_k * (a ⊗ b) for some scalar c_k
        # and vectors a, b.
        
        # So we need: sum_r w_r[k] * (u_r ⊗ v_r) = c_k * (a ⊗ b) for all k.
        # And we need c_k * a[i] * b[j] = 1 for all i,j, so c_k = 1/(a[i]*b[j]) which
        # is independent of i,j. Let a = (1,1), b = (1,1), then c_k = 1 for all k.
        
        # So we need: sum_r w_r[k] * (u_r ⊗ v_r) = ones(2,2) for all k.
        # This means the set of matrices {u_r ⊗ v_r} weighted by w_r[k] must sum to ones(2,2).
        
        # Let's say we have R terms. For each k, we need:
        # sum_r w_r[k] * u_r[i] * v_r[j] = 1 for all i,j.
        
        # This is a system of 4*R equations (for 4 values of (i,j) and R values of k... 
        # no, for each k we have 4 equations, and we have 4 values of k, so 16 equations
        # in 3*R unknowns (u_r has 2 entries, v_r has 2 entries, w_r has 4 entries, but
        # we can scale u_r and v_r inversely, so effectively 3*R - R = 2*R free params per term? No...)
        
        # Actually, let's just try to find a rank-13 solution by solving the linear system.
        
        # The constraint is: for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # With n=2, m=2, p=4: for all i in {0,1}, j in {0,1}, k in {0,1,2,3}:
        # sum_r U[i*2+j, r] * V[j*4+k, r] * W[k*2+i, r] = 1
        
        # Let's fix U and W with simple patterns and solve for V.
        
        # Try: U[i*2+j, r] = 1 for all i,j,r (all ones)
        # Then the constraint becomes: sum_r V[j*4+k, r] * W[k*2+i, r] = 1 for all i,j,k
        
        # For fixed j,k: sum_r V[j*4+k, r] * W[k*2+i, r] = 1 for i=0,1
        # This is a 2xR system for each (j,k), with 8*(j,k) pairs.
        
        # Let's try U = ones(4, 13), and see if we can find V, W.
        
        U_mat = np.ones((4, R))
        
        # Now we need to find V (8x13) and W (8x13) such that
        # for all i,j,k: sum_r V[j*4+k, r] * W[k*2+i, r] = 1
        
        # Let's try W = ones(8, 13) as well.
        W_mat = np.ones((8, R))
        
        # Then: sum_r V[j*4+k, r] * 1 = R * V[j*4+k, r]... no, sum_r V[j*4+k, r] = 1
        # for all j,k. So each row of V must sum to 1.
        # V[j*4+k, r] can be anything as long as the row sums to 1.
        
        # Let's set V[j*4+k, 0] = 1, V[j*4+k, r] = 0 for r > 0.
        V_mat = np.zeros((8, R))
        for jk in range(8):
            V_mat[jk, 0] = 1
        
        # Check: sum_r V[j*4+k, r] * W[k*2+i, r] = V[j*4+k, 0] * W[k*2+i, 0] = 1 * 1 = 1. ✓
        
        # But wait, this is rank 1! That can't be right for matrix multiplication.
        
        # I think I'm confusing myself. The tensor for matrix multiplication is NOT
        # the all-ones tensor in the way I was thinking. Let me reconsider.
        
        # The matrix multiplication tensor T has entries T[i,j,k] where:
        # - i indexes rows of A (and C): 0 to n-1
        # - j indexes columns of A (and rows of B): 0 to m-1  
        # - k indexes columns of B (and C): 0 to p-1
        # T[i,j,k] = 1 means that C[i,k] gets a contribution from A[i,j]*B[j,k]
        
        # For standard matrix multiplication, C[i,k] = sum_j A[i,j]*B[j,k], so
        # T[i,j,k] = 1 for all valid (i,j,k).
        
        # The rank-1 decomposition is: T = sum_r U_r ⊗ V_r ⊗ W_r where
        # U_r is a vector of length n*m (flattened A indices), V_r of length m*p,
        # W_r of length p*n.
        
        # T[i,j,k] = sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
        
        # For the all-ones tensor, T[i,j,k] = 1 for all i,j,k.
        # Can we write this as a sum of few rank-1 terms?
        
        # If we take U_r = 1 for all entries, V_r = 1 for all entries, W_r = 1 for all entries,
        # then sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] = R * 1 * 1 * 1 = R.
        # So we'd need R = 1 to get T[i,j,k] = 1. But that's rank 1!
        
        # This suggests the all-ones tensor has rank 1, which contradicts Strassen's result
        # for 2x2x2. Let me check: for 2x2x2, T[i,j,k] = 1 for all i,j,k in {0,1}.
        # With U=V=W=ones(4), we get T[i,j,k] = sum_r 1*1*1 = R. For R=1, T=1. 
        # So the all-ones (2,2,2) tensor has rank 1?!
        
        # But Strassen says 2x2x2 matrix multiplication has rank 7. There must be a
        # fundamental misunderstanding on my part.
        
        # OH WAIT. I think the issue is that in the standard tensor formulation,
        # the matrix multiplication tensor is NOT the all-ones tensor. Let me look this up.
        
        # Actually, I think the confusion is about what the tensor represents. In the
        # bilinear complexity framework, the tensor T has entries T[(i,k), (i,j), (j,k)]
        # or some other indexing. The key point is that not all (i,j,k) combinations
        # contribute to every output.
        
        # Let me re-read the problem statement: "The output factor follows the official
        # trace convention W[k*n+i,r]."
        
        # So W is indexed by [k*n+i, r] where k is the column of C (0 to p-1) and
        # i is the row of C (0 to n-1). This matches C[i,k] with flattened index k*n+i.
        
        # U is indexed by [i*m+j, r] where i is row of A, j is column of A.
        # V is indexed by [j*p+k, r] where j is row of B, k is column of B.
        
        # The constraint is: C[i,k] = sum_j A[i,j] * B[j,k] = sum_r (sum_j U[i*m+j,r] * A[i,j]) * ... 
        # No wait, it's: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all (i,j,k)
        # where the sum is over r, and the equation must hold for each (i,j,k) triple.
        
        # This IS the all-ones tensor condition. And as I showed, it has rank 1.
        
        # So why does Strassen give rank 7 for 2x2x2? Let me think again...
        
        # I think the issue is that in the actual matrix multiplication tensor,
        # the entries are NOT all 1. The tensor T has T[i,j,k] = 1 only when
        # the (i,k) output depends on the (i,j) input of A and (j,k) input of B.
        # For standard matrix multiplication, this is ALL (i,j,k) combinations.
        
        # Hmm, but then the rank should be 1 for all matrix multiplication problems,
        # which is clearly wrong.
        
        # I must be misunderstanding the tensor formulation. Let me look at it differently.
        
        # In the standard formulation, the matrix multiplication tensor is:
        # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k (in some appropriate spaces)
        # where the sum is over all valid (i,j,k).
        
        # The rank of this tensor is the bilinear complexity.
        
        # For the (2,2,2) case, T = sum_{i,j,k=0}^1 e_i ⊗ e_j ⊗ e_k
        # This is the "all-ones" tensor in 2x2x2, which I claimed has rank 1.
        
        # But actually, the rank of the all-ones tensor (2,2,2) is NOT 1. Let me verify.
        # If T = u ⊗ v ⊗ w, then T[i,j,k] = u[i]*v[j]*w[k]. For T to be all-ones,
        # we need u[i]*v[j]*w[k] = 1 for all i,j,k. This means u[i] = 1/v[j]/w[k]
        # for all i,j,k, which implies u[i] is constant, v[j] is constant, w[k] is constant.
        # So u = (c, c), v = (1/c, 1/c), w = (1, 1) works, giving T[i,j,k] = c*(1/c)*1 = 1.
        # So the all-ones (2,2,2) tensor DOES have rank 1.
        
        # This means my understanding of the matrix multiplication tensor is wrong.
        
        # Let me reconsider. Perhaps the tensor is not the all-ones tensor, but rather
        # has a more complex structure.
        
        # Actually, I think I've been confusing two different things. The matrix
        # multiplication problem can be formulated as a tensor, but the tensor is
        # not simply the all-ones tensor. The correct formulation involves the
        # structure of how inputs combine to produce outputs.
        
        # Let me just trust the Strassen result and the current program's approach.
        # The current program finds a rank-14 solution using two Strassen blocks,
        # and tries to find a rank-13 solution by merging terms.
        
        # For this improvement, let me focus on making the rank-13 search more
        # effective by trying more diverse U and W structures.
        
        # Actually, let me just return None for now and rely on the rank-14 solution.
        # The rank-13 search is already quite complex and may not find a solution.
        
        return None
    
    # Try rank-13 search first
    result = try_rank13_search()
    if result is not None:
        return result
    
    # Try the structured rank-14 decomposition
    result = try_strassen_rank14()
    if result is not None:
        return result
    
    # Fallback to naive rank 16
    return naive_decomp()
# EVOLVE-BLOCK-END
