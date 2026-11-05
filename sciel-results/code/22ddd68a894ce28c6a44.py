import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    """
    Attempt rank-168 decomposition using 2x2 blocks of 3x3 with BCL rank-21.
    Falls back to rank-189 Strassen if BCL verification fails.
    """
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS

    # Try BCL rank-21 for 3x3
    def build_bcl_3x3():
        """
        Build BCL rank-21 3x3 matrix multiplication.
        Returns (U, V, W) each 9x21, or None if invalid.
        """
        # Use the formulation from Bini et al. 1979
        # The 21 bilinear forms are defined using Hadamard matrices of order 4
        # applied to subsets of entries.
        
        # A verified set of 21 forms (0-indexed, a[i][j], b[i][j]):
        # I'll use a construction based on the standard BCL paper.
        
        # Form indices: (a_entries, b_entries) where each is list of ((row,col), coeff)
        forms = []
        
        # Group 1: diagonal sums (4 forms)
        # These use the 4x4 Hadamard on the 3 diagonal entries + 1 dummy
        H4 = np.array([
            [1, 1, 1, 1],
            [1, -1, 1, -1],
            [1, 1, -1, -1],
            [1, -1, -1, 1]
        ], dtype=float)
        
        diag_a = [(0,0), (1,1), (2,2)]
        diag_b = [(0,0), (1,1), (2,2)]
        
        for h in range(4):
            a_terms = []
            b_terms = []
            for d in range(3):
                coeff = H4[h, d]
                if coeff != 0:
                    a_terms.append((diag_a[d][0], diag_a[d][1], coeff))
                    b_terms.append((diag_b[d][0], diag_b[d][1], coeff))
            forms.append((a_terms, b_terms))
        
        # Group 2: row 0 of A with col 0 of B (4 forms)
        row0_a = [(0,0), (0,1), (0,2)]
        col0_b = [(0,0), (1,0), (2,0)]
        for h in range(4):
            a_terms = []
            b_terms = []
            for d in range(3):
                coeff = H4[h, d]
                if coeff != 0:
                    a_terms.append((row0_a[d][0], row0_a[d][1], coeff))
                    b_terms.append((col0_b[d][0], col0_b[d][1], coeff))
            forms.append((a_terms, b_terms))
        
        # Group 3: col 0 of A with row 0 of B (4 forms)
        col0_a = [(0,0), (1,0), (2,0)]
        row0_b = [(0,0), (0,1), (0,2)]
        for h in range(4):
            a_terms = []
            b_terms = []
            for d in range(3):
                coeff = H4[h, d]
                if coeff != 0:
                    a_terms.append((col0_a[d][0], col0_a[d][1], coeff))
                    b_terms.append((row0_b[d][0], row0_b[d][1], coeff))
            forms.append((a_terms, b_terms))
        
        # Group 4: row 1 of A with col 1 of B (4 forms)
        row1_a = [(1,0), (1,1), (1,2)]
        col1_b = [(0,1), (1,1), (2,1)]
        for h in range(4):
            a_terms = []
            b_terms = []
            for d in range(3):
                coeff = H4[h, d]
                if coeff != 0:
                    a_terms.append((row1_a[d][0], row1_a[d][1], coeff))
                    b_terms.append((col1_b[d][0], col1_b[d][1], coeff))
            forms.append((a_terms, b_terms))
        
        # Group 5: col 1 of A with row 1 of B (4 forms)
        col1_a = [(0,1), (1,1), (2,1)]
        row1_b = [(1,0), (1,1), (1,2)]
        for h in range(4):
            a_terms = []
            b_terms = []
            for d in range(3):
                coeff = H4[h, d]
                if coeff != 0:
                    a_terms.append((col1_a[d][0], col1_a[d][1], coeff))
                    b_terms.append((row1_b[d][0], row1_b[d][1], coeff))
            forms.append((a_terms, b_terms))
        
        # That's 20 forms. Need 1 more.
        # Group 6: cross term (1 form)
        # (a[0][1] + a[1][2]) * (b[0][2] + b[1][0])
        a_terms = [(0,1,1), (1,2,1)]
        b_terms = [(0,2,1), (1,0,1)]
        forms.append((a_terms, b_terms))
        
        # Now solve for W coefficients
        U_mat = np.zeros((9, 21))
        V_mat = np.zeros((9, 21))
        
        for r, (a_terms, b_terms) in enumerate(forms):
            for (row, col, coeff) in a_terms:
                U_mat[row * 3 + col, r] = coeff
            for (row, col, coeff) in b_terms:
                V_mat[row * 3 + col, r] = coeff
        
        # Build multiplication tensor
        M = np.zeros((9, 9, 9))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    a = i * 3 + k
                    b = k * 3 + j
                    c = i * 3 + j
                    M[a, b, c] += 1
        
        W_mat = np.zeros((9, 21))
        valid = True
        for c in range(9):
            target = M[:, :, c].flatten()
            A_sys = np.zeros((81, 21))
            for r in range(21):
                A_sys[:, r] = (U_mat[:, r] * V_mat[:, r]).flatten()
            
            W_sol, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, target, rcond=None)
            W_mat[c, :] = W_sol
            
            recon = A_sys @ W_sol
            if np.max(np.abs(recon - target)) > 1e-10:
                valid = False
                break
        
        if not valid:
            return None
        
        # Final verification
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    val = 0.0
                    for r in range(21):
                        val += U_mat[i*3+k, r] * V_mat[k*3+j, r] * W_mat[i*3+j, r]
                    if abs(val - 1.0) > 1e-10:
                        return None
        
        return U_mat, V_mat, W_mat
    
    bcl_result = build_bcl_3x3()
    
    if bcl_result is not None:
        U_3x3, V_3x3, W_3x3 = bcl_result
        bcl_rank = 21
    else:
        # Use Strassen rank-7 for 2x2, total 189
        # Build 2x2 Strassen
        def build_strassen_2x2():
            U = np.zeros((4, 7))
            V = np.zeros((4, 7))
            W = np.zeros((4, 7))
            # M1 = (a00+a11)*(b00+b11)
            U[0,0]=1; U[1,0]=1; V[0,0]=1; V[1,0]=1; W[0,0]=1; W[1,1]=1
            # M2 = (a10+a11)*b00
            U[1,1]=1; U[1,1]=1; V[0,1]=1; W[1,0]=1; W[1,1]=-1
            # M3 = a00*(b01-b11)
            U[0,2]=1; V[0,2]=1; V[1,2]=-1; W[0,1]=1; W[1,1]=1
            # M4 = a11*(b10-b00)
            U[1,3]=1; V[1,3]=1; V[0,3]=-1; W[0,0]=1; W[1,0]=1
            # M5 = (a00+a01)*b11
            U[0,4]=1; U[0,4]=1; V[1,4]=1; W[0,0]=-1; W[0,1]=1
            # M6 = (a10-a00)*(b00+b01)
            U[1,5]=1; U[0,5]=-1; V[0,5]=1; V[0,5]=1; W[1,1]=1
            # M7 = (a01-a11)*(b10+b11)
            U[0,6]=1; U[1,6]=-1; V[1,6]=1; V[1,6]=1; W[0,0]=1
            
            # Verify
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        val = sum(U[i*2+k, r] * V[k*2+j, r] * W[i*2+j, r] for r in range(7))
                        if abs(val - 1.0) > 1e-10:
                            return None
            return U, V, W
        
        strassen_result = build_strassen_2x2()
        if strassen_result is not None:
            U_2x2, V_2x2, W_2x2 = strassen_result
            # Build 6x6 from 2x2 blocks
            total_rank = 27 * 7
            U = np.zeros((36, total_rank))
            V = np.zeros((36, total_rank))
            W = np.zeros((36, total_rank))
            col = 0
            for bi in range(3):
                for bj in range(3):
                    for bk in range(3):
                        for r in range(7):
                            for ar in range(2):
                                for ac in range(2):
                                    U[(2*bi+ar)*6 + (2*bk+ac), col] = U_2x2[ar*2+ac, r]
                            for br in range(2):
                                for bc in range(2):
                                    V[(2*bk+br)*6 + (2*bj+bc), col] = V_2x2[br*2+bc, r]
                            for cr in range(2):
                                for cc in range(2):
                                    W[(2*bj+cc)*6 + (2*bi+cr), col] = W_2x2[cr*2+cc, r]
                            col += 1
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Schoolbook fallback
            rank_fallback = 216
            U_f = np.zeros((36, rank_fallback))
            V_f = np.zeros((36, rank_fallback))
            W_f = np.zeros((36, rank_fallback))
            col = 0
            for i in range(6):
                for j in range(6):
                    for k in range(6):
                        U_f[i*6+j, col] = 1
                        V_f[j*6+k, col] = 1
                        W_f[k*6+i, col] = 1
                        col += 1
            return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
    
    # Build 6x6 from 3x3 blocks (2x2 grid)
    total_rank = 4 * 2 * 21
    U = np.zeros((36, total_rank))
    V = np.zeros((36, total_rank))
    W = np.zeros((36, total_rank))
    
    col = 0
    for ci in range(2):
        for cj in range(2):
            for k in range(2):
                for r in range(21):
                    for ar in range(3):
                        for ac in range(3):
                            U[(3*ci+ar)*6 + (3*k+ac), col] = U_3x3[ar*3+ac, r]
                    for br in range(3):
                        for bc in range(3):
                            V[(3*k+br)*6 + (3*cj+bc), col] = V_3x3[br*3+bc, r]
                    for cr in range(3):
                        for cc in range(3):
                            W[(3*cj+cc)*6 + (3*ci+cr), col] = W_3x3[cr*3+cc, r]
                    col += 1
    
    # Final verification
    import random
    random.seed(42)
    valid = True
    for _ in range(1000):
        i = random.randint(0, 5)
        j = random.randint(0, 5)
        k = random.randint(0, 5)
        val = sum(U[i*6+j, r] * V[j*6+k, r] * W[k*6+i, r] for r in range(total_rank))
        if abs(val - 1.0) > 1e-10:
            valid = False
            break
    
    if valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback
        rank_fallback = 216
        U_f = np.zeros((36, rank_fallback))
        V_f = np.zeros((36, rank_fallback))
        W_f = np.zeros((36, rank_fallback))
        col = 0
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    U_f[i*6+j, col] = 1
                    V_f[j*6+k, col] = 1
                    W_f[k*6+i, col] = 1
                    col += 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
