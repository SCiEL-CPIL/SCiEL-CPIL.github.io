def construct():
    n, m, p = 2, 4, 6
    rank = 36
    U = np.zeros((n*m, rank))  # (8, 36)
    V = np.zeros((m*p, rank))  # (24, 36)
    W = np.zeros((p*n, rank))  # (12, 36)
    
    col = 0
    # Part 1: 2x4 x 4x4 (columns 0-3 of C), rank 28
    # Split into A0*B0 + A1*B1, each (2x2)*(2x4) done as two Strassen 2x2x2's
    
    for source in range(2):  # 0: A0 with B0, 1: A1 with B1
        for col_block in range(2):  # 0: cols 0-1 of B, 1: cols 2-3 of B
            for t in range(7):
                # P = A[:, 2*source : 2*source+2], Q = B[2*source:2*source+2, 2*col_block:2*col_block+2]
                # P[i,j] = A[i, 2*source+j]
                # Q[i,j] = B[2*source+i, 2*col_block+j]
                # Output C[i, 2*col_block+j]
                
                ps = 2 * source  # P column offset
                qs = 2 * source  # Q row offset
                cb = 2 * col_block  # output column offset
                
                if t == 0:
                    # M1 = (P00+P11)*(Q00+Q11)
                    U[0*4 + ps+0, col] += 1
                    U[1*4 + ps+1, col] += 1
                    V[(qs+0)*6 + cb+0, col] += 1
                    V[(qs+1)*6 + cb+1, col] += 1
                    W[(cb+0)*2 + 0, col] += 1   # R00
                    W[(cb+1)*2 + 1, col] += 1   # R11
                elif t == 1:
                    # M2 = (P10+P11)*Q00
                    U[1*4 + ps+0, col] += 1
                    U[1*4 + ps+1, col] += 1
                    V[(qs+0)*6 + cb+0, col] += 1
                    W[(cb+0)*2 + 1, col] += 1   # R10
                    W[(cb+1)*2 + 1, col] += -1  # -R11
                elif t == 2:
                    # M3 = P00*(Q01-Q11)
                    U[0*4 + ps+0, col] += 1
                    V[(qs+0)*6 + cb+1, col] += 1
                    V[(qs+1)*6 + cb+1, col] += -1
                    W[(cb+1)*2 + 0, col] += 1   # R01
                    W[(cb+1)*2 + 1, col] += 1   # R11
                elif t == 3:
                    # M4 = P11*(Q10-Q00)
                    U[1*4 + ps+1, col] += 1
                    V[(qs+1)*6 + cb+0, col] += 1
                    V[(qs+0)*6 + cb+0, col] += -1
                    W[(cb+0)*2 + 0, col] += 1   # R00
                    W[(cb+0)*2 + 1, col] += 1   # R10
                elif t == 4:
                    # M5 = (P00+P01)*Q11
                    U[0*4 + ps+0, col] += 1
                    U[0*4 + ps+1, col] += 1
                    V[(qs+1)*6 + cb+1, col] += 1
                    W[(cb+0)*2 + 0, col] += -1  # -R00
                    W[(cb+1)*2 + 0, col] += 1   # R01
                elif t == 5:
                    # M6 = (P10-P00)*(Q00+Q01)
                    U[1*4 + ps+0, col] += 1
                    U[0*4 + ps+0, col] += -1
                    V[(qs+0)*6 + cb+0, col] += 1
                    V[(qs+0)*6 + cb+1, col] += 1
                    W[(cb+1)*2 + 1, col] += 1   # R11
                elif t == 6:
                    # M7 = (P01-P11)*(Q10+Q11)
                    U[0*4 + ps+1, col] += 1
                    U[1*4 + ps+1, col] += -1
                    V[(qs+1)*6 + cb+0, col] += 1
                    V[(qs+1)*6 + cb+1, col] += 1
                    W[(cb+0)*2 + 0, col] += 1   # R00
                
                col += 1
    
    # Part 2: 2x4 x 4x2 (columns 4-5 of C), rank 8 (schoolbook)
    for i in range(2):
        for k in range(4):
            for j in range(2):
                U[i*4 + k, col] = 1
                V[k*6 + (4+j), col] = 1
                W[(4+j)*2 + i, col] = 1
                col += 1
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
