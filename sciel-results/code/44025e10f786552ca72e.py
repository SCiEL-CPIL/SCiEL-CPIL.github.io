        def construct():
            n, m, p = 2, 3, 7
            rank = 39
            U = [[0.0] * rank for _ in range(n * m)]
            V = [[0.0] * rank for _ in range(m * p)]
            W = [[0.0] * rank for _ in range(p * n)]
            
            col = 0
            
            # Strassen blocks for k in {0,1}, {2,3}, {4,5}
            for k_start in [0, 2, 4]:
                # M1
                U[0*m + 0][col] += 1
                V[0*p + k_start][col] += 1
                V[1*p + k_start + 1][col] += 1
                W[(k_start + 1)*n + 0][col] += 1
                col += 1
                
                # M2
                U[0*m + 0][col] += 1
                U[1*m + 0][col] += 1
                V[1*p + k_start + 1][col] += 1
                W[(k_start + 1)*n + 0][col] += 1
                col += 1
                
                # M3
                U[1*m + 1][col] += 1
                V[0*p + k_start][col] += 1
                V[0*p + k_start + 1][col] += 1
                W[(k_start)*n + 1][col] += 1
                col += 1
                
                # M4
                U[0*m + 1][col] += 1
                V[1*p + k_start][col] += 1
                V[0*p + k_start][col] -= 1
                W[(k_start)*n + 0][col] += 1 # C_00 += M4
                W[(k_start)*n + 1][col] += 1 # C_10 += M4
                col += 1
                
                # M5
                U[0*m + 0][col] += 1
                U[1*m + 1][col] += 1
                V[1*p + k_start + 1][col] += 1
                V[0*p + k_start][col] += 1
                W[(k_start + 1)*n + 0][col] += 1 # C_00 += M5
                W[(k_start + 1)*n + 1][col] += 1 # C_11 += M5 (Wait, C_11 is i=1, k=1)
                # Check C indices:
                # C_00 (i=0, k=0): M5 + M4 - M2 + M6
                # C_01 (i=0, k=1): M1 + M2
                # C_10 (i=1, k=0): M3 + M4
                # C_11 (i=1, k=1): M1 - M3 + M5 + M7
                
                # Let's re-verify W assignments.
                # M5 contributes to C_00 and C_11.
                W[(k_start)*n + 0][col] += 1
                W[(k_start + 1)*n + 1][col] += 1
                col += 1
                
                # M6
                U[1*m + 0][col] += 1
                U[1*m + 1][col] += 1
                V[0*p + k_start + 1][col] += 1
                V[1*p + k_start + 1][col] -= 1
                W[(k_start)*n + 0][col] += 1 # C_00 += M6
                col += 1
                
                # M7
                U[0*m + 0][col] += 1
                U[0*m + 1][col] += 1
                V[1*p + k_start][col] += 1
                V[0*p + k_start][col] -= 1
                W[(k_start + 1)*n + 1][col] += 1 # C_11 += M7
                col += 1
            
            # Schoolbook for k=6 (index 6)
            # C_i,6 = A_i,0 B_0,6 + A_i,1 B_1,6
            for i in range(2):
                for j in range(2):
                    U[i*m + j][col] = 1
                    V[j*p + 6][col] = 1
                    W[6*n + i][col] = 1
                    col += 1
            
            # Schoolbook for j=2 (index 2)
            # C_i,k += A_i,2 B_2,k
            for i in range(2):
                for k in range(7):
                    U[i*m + 2][col] = 1
                    V[2*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
            
            return {"U": U, "V": V, "W": W}
        