def construct():
    n, m, p = 5, 5, 5
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_vals, v_vals, w_vals):
        U = np.zeros((n*m, 1), dtype=float)
        V = np.zeros((m*p, 1), dtype=float)
        W = np.zeros((p*n, 1), dtype=float)
        for (i, j), val in u_vals.items():
            U[i*m+j, 0] = val
        for (j, k), val in v_vals.items():
            V[j*p+k, 0] = val
        for (k, i), val in w_vals.items():
            W[k*n+i, 0] = val
        U_list.append(U.flatten())
        V_list.append(V.flatten())
        W_list.append(W.flatten())
    
    # C[0,0] = A[0,0]*B[0,0]
    add_term({(0,0):1}, {(0,0):1}, {(0,0):1})
    
    # C[0,k] = A[0,0]*B[0,k] for k=1,2,3,4
    for k in [1,2,3,4]:
        add_term({(0,0):1}, {(0,k):1}, {(k,0):1})
    
    # C[i,0] = A[i,0]*B[0,0] for i=1,2,3,4
    for i in [1,2,3,4]:
        add_term({(i,0):1}, {(0,0):1}, {(0,i):1})
    
    # C[i,k] for i=1..4, k=1..4: A[i,0]*B[0,k]
    for i in [1,2,3,4]:
        for k in [1,2,3,4]:
            add_term({(i,0):1}, {(0,k):1}, {(k,i):1})
    
    # C[i,k] for i=1..4, k=1..4: A44*B44 via Strassen
    # Map indices 1..4 to 0..3 for Strassen
    # Strassen for 4x4: split into 2+2, use Strassen on each 2x2
    
    # Actually, let me use Strassen for 4x4 directly
    # 4x4 Strassen: 7 multiplications of 2x2, each with rank 7, total 49? No.
    # Strassen for 4x4: split 4x4 into four 2x2 blocks, use 7 multiplications of 2x2.
    # Each 2x2 multiplication has rank 7 (Strassen), so total rank = 7 * 7 = 49? 
    # No, that's not right. Strassen for 4x4 uses 7 multiplications, each of which is a 2x2 multiplication with rank 7. So total rank = 7 * 7 = 49.
    # But we can do better: use Strassen for 4x4 with rank 16 (by padding to 4x4 and using the 4x4 Strassen algorithm).
    
    # Actually, the standard Strassen for 4x4 has rank 16 (7 multiplications of 2x2, but each 2x2 uses 7 scalar multiplications, so 7*7=49? No...)
    
    # I'm confused. Let me just use the trivial 4x4 decomposition (rank 16) for now, or better, use Strassen.
    
    # For 4x4, use Strassen: rank 16? No, Strassen for 4x4 is 7 * (rank of 2x2) = 7 * 7 = 49? That's worse than trivial 64? No, 49 < 64.
    # Wait, trivial 4x4 is 4*4*4 = 64. Strassen 4x4 is 7 * (2x2 trivial = 8) = 56? Or 7 * (2x2 Strassen = 7) = 49?
    # I think Strassen 4x4 has rank 49 (7 multiplications, each a 2x2 with Strassen rank 7).
    
    # Hmm, but I recall that Strassen for n=4 has rank 16. Let me check: Strassen's algorithm for 4x4 uses 7 multiplications of 2x2 matrices. If each 2x2 multiplication is done with rank 7 (Strassen), then total rank is 7*7 = 49. If each 2x2 is done with rank 8 (trivial), total is 7*8 = 56.
    
    # But I've also seen claims that Strassen for 4x4 has rank 16. I think that's wrong. Let me just use the trivial 4x4 (rank 16) for the 4x4 block.
    
    # Actually, for the 4x4 block (indices 1..4), the trivial decomposition has rank 4*4*4 = 64. That's too many.
    
    # Let me use Strassen for the 4x4 block. I'll implement it recursively.
    
    def strassen_4x4(i_base, j_base, k_base):
        """Strassen for 4x4 block with base indices"""
        # Split into 2x2 blocks
        # Block 00: i_base..i_base+1, j_base..j_base+1, k_base..k_base+1
        # Block 01: i_base..i_base+1, j_base+2..j_base+3, k_base..k_base+1
        # etc.
        
        # M1 = (A00+A11)(B00+B11)
        add_term({(i_base,j_base):1, (i_base+1,j_base+1):1, (i_base+2,j_base+2):1, (i_base+3,j_base+3):1},
                 {(j_base,k_base):1, (j_base+1,k_base+1):1, (j_base+2,k_base+2):1, (j_base+3,k_base+3):1},
                 {(k_base,i_base):1, (k_base+1,i_base+1):1, (k_base+2,i_base+2):1, (k_base+3,i_base+3):1})
        # This is getting too complex. Let me just use the trivial 4x4 for now.
        pass
    
    # For now, use trivial 4x4 for the A44*B44 part
    for i in [1,2,3,4]:
        for j in [1,2,3,4]:
            for k in [1,2,3,4]:
                add_term({(i,j):1}, {(j,k):1}, {(k,i):1})
    
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
