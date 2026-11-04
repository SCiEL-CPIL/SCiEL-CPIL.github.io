import numpy as np

def construct():
    n, m, p = 3, 5, 6
    
    # We'll build the decomposition by splitting:
    # m=5 into m1=3, m2=2
    # p=6 into p1=3, p2=3
    # This gives 4 sub-problems:
    # 1. 3x3 x 3x3 -> rank 26
    # 2. 3x3 x 3x3 -> rank 26  
    # 3. 3x2 x 2x3 -> rank 17
    # 4. 3x2 x 2x3 -> rank 17
    # Total: 86
    
    rank = 86
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col_offset = 0
    
    # Helper to add a sub-decomposition
    def add_decomp(sub_U, sub_V, sub_W, off_i, off_j, off_k, col_off):
        """
        sub_U: (n_sub * m_sub, rank_sub)
        sub_V: (m_sub * p_sub, rank_sub)  
        sub_W: (p_sub * n_sub, rank_sub)
        off_i: starting row in A (and C)
        off_j: starting col in A (and row in B)
        off_k: starting col in B (and row in C)
        col_off: starting column in global U,V,W
        """
        n_sub = sub_U.shape[0] // sub_U.shape[1] if sub_U.shape[1] > 0 else 0
        # Actually, we need to pass dimensions explicitly
        # Let's refactor
        pass
    
    # This is getting complex. Let me just build it directly.
    
    # For 3x3x3 with rank 26:
    # Split n=3 into n1=1, n2=2
    # Part 1: 1x3x3, schoolbook rank 9
    # Part 2: 2x3x3, split m=3 into m1=1, m2=2
    #   Part 2a: 2x1x3, schoolbook rank 6
    #   Part 2b: 2x2x3, split p=3 into p1=1, p2=2
    #     Part 2b1: 2x2x1, schoolbook rank 4
    #     Part 2b2: 2x2x2, Strassen rank 7
    #   Total part 2: 6+4+7 = 17
    # Total 3x3x3: 9+17 = 26
    
    # For 3x2x3 with rank 17:
    # Split n=3 into n1=1, n2=2
    # Part 1: 1x2x3, schoolbook rank 6
    # Part 2: 2x2x3, split p=3 into p1=1, p2=2
    #   Part 2a: 2x2x1, schoolbook rank 4
    #   Part 2b: 2x2x2, Strassen rank 7
    #   Total part 2: 4+7 = 11
    # Total 3x2x3: 6+11 = 17
    
    # Now let's implement each piece
    
    # Strassen 2x2x2 (rank 7)
    def strassen_222():
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))
        
        # Term 0: M1 = (A00+A11)(B00+B11)
        U_s[0, 0] = 1; U_s[3, 0] = 1
        V_s[0, 0] = 1; V_s[3, 0] = 1
        W_s[0, 0] = 1; W_s[3, 0] = 1
        
        # Term 1: M2 = (A10+A11)B00
        U_s[2, 1] = 1; U_s[3, 1] = 1
        V_s[0, 1] = 1
        W_s[2, 1] = 1
        
        # Term 2: M3 = A00(B10-B11)
        U_s[0, 2] = 1
        V_s[2, 2] = 1; V_s[3, 2] = -1
        W_s[1, 2] = 1
        
        # Term 3: M4 = A11(B01-B00)
        U_s[3, 3] = 1
        V_s[1, 3] = 1; V_s[0, 3] = -1
        W_s[0, 3] = 1; W_s[2, 3] = 1
        
        # Term 4: M5 = (A00+A01)B11
        U_s[0, 4] = 1; U_s[1, 4] = 1
        V_s[3, 4] = 1
        W_s[0, 4] = -1; W_s[1, 4] = 1
        
        # Term 5: M6 = (A10-A00)(B00+B10)
        U_s[2, 5] = 1; U_s[0, 5] = -1
        V_s[0, 5] = 1; V_s[2, 5] = 1
        W_s[3, 5] = 1
        
        # Term 6: M7 = (A01-A11)(B01+B11)
        U_s[1, 6] = 1; U_s[3, 6] = -1
        V_s[1, 6] = 1; V_s[3, 6] = 1
        W_s[0, 6] = 1; W_s[3, 6] = 1
        
        return U_s, V_s, W_s
    
    # Schoolbook for nxm x mxp
    def schoolbook(n_sub, m_sub, p_sub):
        rank_sub = n_sub * m_sub * p_sub
        U_s = np.zeros((n_sub * m_sub, rank_sub))
        V_s = np.zeros((m_sub * p_sub, rank_sub))
        W_s = np.zeros((p_sub * n_sub, rank_sub))
        col = 0
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    U_s[i * m_sub + j, col] = 1
                    V_s[j * p_sub + k, col] = 1
                    W_s[k * n_sub + i, col] = 1
                    col += 1
        return U_s, V_s, W_s
    
    # 2x2x1: schoolbook rank 4
    # 2x2x2: Strassen rank 7
    # 2x2x3: 2x2x1 + 2x2x2 = 4+7 = 11
    def decomp_223():
        # n=2, m=2, p=3
        # Split p into p1=1, p2=2
        U1, V1, W1 = schoolbook(2, 2, 1)  # rank 4
        U2, V2, W2 = strassen_222()       # rank 7
        
        U = np.zeros((4, 11))
        V = np.zeros((4, 11))
        W = np.zeros((6, 11))  # p*n = 3*2 = 6
        
        # Place 2x2x1 part (p indices 0)
        U[:, 0:4] = U1
        V[:, 0:4] = V1
        W[0:2, 0:4] = W1  # p=1, n=2, so W is 2x4, placed at rows 0-1
        
        # Place 2x2x2 part (p indices 1,2)
        U[:, 4:11] = U2
        V[:, 4:11] = V2
        W[2:6, 4:11] = W2  # p indices 1,2 -> rows 2-5 (since p*n=6, p=1 starts at row 2*2=4? No)
        
        # Wait, W has shape (p*n, rank) = (3*2, 11) = (6, 11)
        # Index in W: k*n + i where k is p-index, i is n-index
        # For 2x2x1: p_sub=1, n_sub=2, W1 shape (2,4), indices k=0, i=0,1 -> rows 0,1
        # For 2x2x2: p_sub=2, n_sub=2, W2 shape (4,7), indices k=0,1, i=0,1 -> rows 0-3 in local, global rows 2*2+0=4 to 2*2+3=7? No.
        
        # Global W index: k_global * n_sub + i_global
        # For the 2x2x2 part, p_local indices 0,1 correspond to p_global indices 1,2
        # So k_global = 1 + k_local
        # W_global index = (1 + k_local) * 2 + i_local
        # For k_local=0: rows 2+0=2, 2+1=3
        # For k_local=1: rows 4+0=4, 4+1=5
        
        W[2:4, 4:11] = W2[0:2, :]  # k_local=0
        W[4:6, 4:11] = W2[2:4, :]  # k_local=1
        
        return U, V, W
    
    # 2x3x3: split m into m1=1, m2=2
    # 2x1x3: schoolbook rank 6
    # 2x2x3: rank 11
    def decomp_233():
        # n=2, m=3, p=3
        U1, V1, W1 = schoolbook(2, 1, 3)  # rank 6
        U2, V2, W2 = decomp_223()         # rank 11
        
        U = np.zeros((6, 17))  # n*m = 2*3 = 6
        V = np.zeros((9, 17))  # m*p = 3*3 = 9
        W = np.zeros((6, 17))  # p*n = 3*2 = 6
        
        # Place 2x1x3 (m index 0)
        U[0:2, 0:6] = U1  # n*m_sub = 2*1 = 2 rows
        V[0:3, 0:6] = V1  # m_sub*p = 1*3 = 3 rows
        W[:, 0:6] = W1
        
        # Place 2x2x3 (m indices 1,2)
        U[2:6, 6:17] = U2  # n*m_sub = 2*2 = 4 rows
        V[3:9, 6:17] = V2  # m_sub*p = 2*3 = 6 rows
        W[:, 6:17] = W2
        
        return U, V, W
    
    # 3x3x3: split n into n1=1, n2=2
    # 1x3x3: schoolbook rank 9
    # 2x3x3: rank 17
    def decomp_333():
        # n=3, m=3, p=3
        U1, V1, W1 = schoolbook(1, 3, 3)  # rank 9
        U2, V2, W2 = decomp_233()         # rank 17
        
        U = np.zeros((9, 26))  # n*m = 9
        V = np.zeros((9, 26))  # m*p = 9
        W = np.zeros((9, 26))  # p*n = 9
        
        # Place 1x3x3 (n index 0)
        U[0:3, 0:9] = U1
        V[:, 0:9] = V1
        W[0:3, 0:9] = W1  # p*n_sub = 3*1 = 3 rows
        
        # Place 2x3x3 (n indices 1,2)
        U[3:9, 9:26] = U2
        V[:, 9:26] = V2
        W[3:9, 9:26] = W2  # p*n_sub = 3*2 = 6 rows, starting at 3
        
        return U, V, W
    
    # 3x2x3: split n into n1=1, n2=2
    # 1x2x3: schoolbook rank 6
    # 2x2x3: rank 11
    def decomp_323():
        # n=3, m=2, p=3
        U1, V1, W1 = schoolbook(1, 2, 3)  # rank 6
        U2, V2, W2 = decomp_223()         # rank 11
        
        U = np.zeros((6, 17))  # n*m = 6
        V = np.zeros((6, 17))  # m*p = 6
        W = np.zeros((9, 17))  # p*n = 9
        
        # Place 1x2x3 (n index 0)
        U[0:2, 0:6] = U1
        V[:, 0:6] = V1
        W[0:3, 0:6] = W1  # p*n_sub = 3*1 = 3 rows
        
        # Place 2x2x3 (n indices 1,2)
        U[2:6, 6:17] = U2
        V[:, 6:17] = V2
        W[3:9, 6:17] = W2  # p*n_sub = 3*2 = 6 rows, starting at 3
        
        return U, V, W
    
    # Now assemble the full 3x5x6 decomposition
    # Split m=5 into m1=3, m2=2
    # Split p=6 into p1=3, p2=3
    
    # Sub-problem 1: A[0:3,0:3] x B[0:3,0:3] -> 3x3x3, rank 26
    U1, V1, W1 = decomp_333()
    
    # Sub-problem 2: A[0:3,0:3] x B[0:3,3:6] -> 3x3x3, rank 26
    U2, V2, W2 = decomp_333()
    
    # Sub-problem 3: A[0:3,3:5] x B[3:5,0:3] -> 3x2x3, rank 17
    U3, V3, W3 = decomp_323()
    
    # Sub-problem 4: A[0:3,3:5] x B[3:5,3:6] -> 3x2x3, rank 17
    U4, V4, W4 = decomp_323()
    
    # Assemble
    U_full = np.zeros((15, 86))
    V_full = np.zeros((30, 86))
    W_full = np.zeros((18, 86))
    
    # Place sub-problem 1: A rows 0-2, cols 0-2; B rows 0-2, cols 0-2; C rows 0-2, cols 0-2
    U_full[0:9, 0:26] = U1
    V_full[0:9, 0:26] = V1
    W_full[0:9, 0:26] = W1
    
    # Place sub-problem 2: A rows 0-2, cols 0-2; B rows 0-2, cols 3-5; C rows 0-2, cols 3-5
    U_full[0:9, 26:52] = U2
    V_full[0:9, 26:52] = V2
    # C: p*n = 6*3 = 18, index = k*n + i, k=3,4,5 -> rows 9-17
    W_full[9:18, 26:52] = W2
    
    # Place sub-problem 3: A rows 0-2, cols 3-4; B rows 3-4, cols 0-2; C rows 0-2, cols 0-2
    U_full[9:15, 52:69] = U3
    V_full[9:15, 52:69] = V3
    W_full[0:9, 52:69] = W3
    
    # Place sub-problem 4: A rows 0-2, cols 3-4; B rows 3-4, cols 3-5; C rows 0-2, cols 3-5
    U_full[9:15, 69:86] = U4
    V_full[9:15, 69:86] = V4
    W_full[9:18, 69:86] = W4
    
    return {"U": U_full, "V": V_full, "W": W_full}

if __name__ == "__main__":
    construct()
