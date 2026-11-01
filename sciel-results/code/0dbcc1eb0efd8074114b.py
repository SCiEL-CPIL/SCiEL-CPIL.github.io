"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition of rank n*m*p
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Try to improve using Strassen-like decomposition for the 2x2 part
    # The idea: decompose 2x2 x 2x13 as (2x2 x 2x2) + (2x2 x 2x11)
    # Use rank-7 for the 2x2x2 part and schoolbook for the rest
    # Total rank = 7 + 4*(p-2) = 7 + 44 = 51 < 52

    if p >= 2:
        strassen_rank = 7
        schoolbook_cols = p - 2
        schoolbook_rank = 4 * schoolbook_cols
        total_rank = strassen_rank + schoolbook_rank

        if total_rank < best_rank:
            U2 = [[0] * total_rank for _ in range(n * m)]  # 4 rows
            V2 = [[0] * total_rank for _ in range(m * p)]  # 26 rows
            W2 = [[0] * total_rank for _ in range(p * n)]  # 26 rows

            col = 0
            # Strassen decomposition for columns 0,1 of B
            # A = [[a,b],[c,d]], B_cols01 = [[e,f],[g,h]], C_cols01 = [[ae+bg, af+bh],[ce+dg, cf+dh]]
            #
            # V index mapping for cols 0,1: V[j*p+k]
            #   V[0*13+0]=e (j=0,k=0), V[1*13+0]=g (j=1,k=0)
            #   V[0*13+1]=f (j=0,k=1), V[1*13+1]=h (j=1,k=1)
            #
            # W index mapping: W[k*n+i]
            #   W[0*2+0]=C00 (k=0,i=0), W[1*2+0]=C10 (k=1,i=0)
            #   W[0*2+1]=C01 (k=0,i=1), W[1*2+1]=C11 (k=1,i=1)
            #
            # U index: 0=a, 1=b, 2=c, 3=d

            # Standard Strassen terms for 2x2 x 2x2:
            # We need 7 rank-1 tensors (u, v, w) such that
            # sum_t u_t[i,j] * v_t[j',k] * w_t[k',i'] = delta for the multiplication.
            
            # Using the standard Strassen decomposition:
            # M1 = (a+b)(e+f)
            # M2 = (c+d)e
            # M3 = a(f-h)
            # M4 = d(g-e)
            # M5 = (a+d)(e+h)
            # M6 = (b+c)(g-e)
            # M7 = (b+d)(f-h)
            
            # C00 = M1 - M3 - M4 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M6 - M7

            # Let's define the 7 terms as (U_vec, V_vec, W_vec)
            # U_vec length 4 (a,b,c,d)
            # V_vec length 26 (e,g,f,h, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
            #   Indices: 0=e, 1=g, 2=f, 3=h, 4..25=0
            # W_vec length 26 (C00,C10,C01,C11, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
            #   Indices: 0=C00, 1=C10, 2=C01, 3=C11, 4..25=0
            
            # Term 1: M1 = (a+b)(e+f)
            # U: [1, 1, 0, 0]
            # V: [1, 0, 1, 0, ...] (e+f)
            # W: contributes to C00 and C11 with sign +1?
            # C00 = M1 - M3 - M4 + M7
            # C11 = M1 - M2 + M6 - M7
            # So M1 contributes +1 to C00 and +1 to C11.
            # W: [1, 0, 0, 1, ...]
            
            strassen_terms = [
                # (U, V, W)
                ([1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]), # M1
                ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]), # M2 (c+d)e -> C10
                ([1, 0, 0, 0], [0, 0, 1, -1], [-1, 0, 1, 0]), # M3 a(f-h) -> -C00, +C01
                ([0, 0, 0, 1], [0, 1, -1, 0], [-1, 1, 0, 0]), # M4 d(g-e) -> -C00, +C10
                ([1, 0, 0, 1], [1, 0, 0, 1], [0, 0, 1, 0]), # M5 (a+d)(e+h) -> C01
                ([0, 1, 1, 0], [0, 1, -1, 0], [0, 0, 0, 1]), # M6 (b+c)(g-e) -> C11
                ([0, 1, 0, 1], [0, 0, 1, -1], [1, 0, 0, -1]), # M7 (b+d)(f-h) -> C00, -C11
            ]
            
            # Verify M7 contribution:
            # C00 = ... + M7
            # C11 = ... - M7
            # So W for M7 should be [1, 0, 0, -1].
            # My term 7 W is [1, 0, 0, -1]. Correct.
            
            # Verify M3 contribution:
            # C00 = M1 - M3 ...
            # C01 = M3 + M5 ...
            # So W for M3 should be [-1, 0, 1, 0].
            # My term 3 W is [-1, 0, 1, 0]. Correct.
            
            # Verify M4 contribution:
            # C00 = ... - M4 ...
            # C10 = M2 + M4 ...
            # So W for M4 should be [-1, 1, 0, 0].
            # My term 4 W is [-1, 1, 0, 0]. Correct.
            
            # Verify M2 contribution:
            # C10 = M2 + M4
            # So W for M2 should be [0, 1, 0, 0].
            # My term 2 W is [0, 1, 0, 0]. Correct.
            
            # Verify M5 contribution:
            # C01 = M3 + M5
            # So W for M5 should be [0, 0, 1, 0].
            # My term 5 W is [0, 0, 1, 0]. Correct.
            
            # Verify M6 contribution:
            # C11 = ... + M6 ...
            # So W for M6 should be [0, 0, 0, 1].
            # My term 6 W is [0, 0, 0, 1]. Correct.

            for t in strassen_terms:
                u, v, w = t
                # Expand v and w to full size
                v_full = [0] * (m * p)
                w_full = [0] * (p * n)
                
                # v indices for cols 0,1: 0=e, 1=g, 2=f, 3=h
                # v_full[0] = v[0] (e)
                # v_full[13] = v[1] (g) -- wait, V index is j*p+k.
                # j=0, k=0 -> 0 (e)
                # j=1, k=0 -> 13 (g)
                # j=0, k=1 -> 1 (f)
                # j=1, k=1 -> 14 (h)
                
                v_full[0] = v[0] # e
                v_full[13] = v[1] # g
                v_full[1] = v[2] # f
                v_full[14] = v[3] # h
                
                # w indices: k*n+i
                # k=0, i=0 -> 0 (C00)
                # k=1, i=0 -> 2 (C10)
                # k=0, i=1 -> 1 (C01)
                # k=1, i=1 -> 3 (C11)
                
                w_full[0] = w[0] # C00
                w_full[2] = w[1] # C10
                w_full[1] = w[2] # C01
                w_full[3] = w[3] # C11
                
                for idx in range(n * m):
                    U2[idx][col] = u[idx]
                for idx in range(m * p):
                    V2[idx][col] = v_full[idx]
                for idx in range(p * n):
                    W2[idx][col] = w_full[idx]
                col += 1

            # Schoolbook for remaining columns
            for i in range(n):
                for j in range(m):
                    for k in range(2, p):
                        U2[i * m + j][col] = 1
                        V2[j * p + k][col] = 1
                        W2[k * n + i][col] = 1
                        col += 1

            best_result = {"U": U2, "V": V2, "W": W2}
            best_rank = total_rank

    return best_result
# EVOLVE-BLOCK-END
