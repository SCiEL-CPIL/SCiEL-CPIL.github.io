<<<<<<< SEARCH
                # W: C[ci,cj] elements
                for li in range(2):
                    for lj in range(2):
                        c_idx = (2*ci + li) * 6 + (2*cj + lj)
                        W[c_idx, col] += W_sub[li*2+lj, r]
            
            # Second Strassen: A[ci,1] * B[1,cj]
            # A[ci,1]: rows 2*ci to 2*ci+1, cols 2-3
            # B[1,cj]: rows 2-3, cols 2*cj to 2*cj+1
            
            for r in range(7):
                col = base + 7 + r
                # U: A[ci,1] elements
                for li in range(2):
                    for lj in range(2):
                        a_idx = (2*ci + li) * 4 + (2 + lj)
                        U[a_idx, col] += U_sub[li*2+lj, r]
                # V: B[1,cj] elements
                for li in range(2):
                    for lj in range(2):
                        b_idx = (2 + li) * 6 + (2*cj + lj)
                        V[b_idx, col] += V_sub[li*2+lj, r]
                # W: C[ci,cj] elements (same as first)
                for li in range(2):
                    for lj in range(2):
                        c_idx = (2*ci + li) * 6 + (2*cj + lj)
                        W[c_idx, col] += W_sub[li*2+lj, r]
=======
                # W: C[ci,cj] elements
                # Task convention: W[k*n+i, r] where k is col, i is row
                for li in range(2):
                    for lj in range(2):
                        # global row i = 2*ci + li
                        # global col k = 2*cj + lj
                        c_idx = (2*cj + lj) * 4 + (2*ci + li)
                        W[c_idx, col] += W_sub[li*2+lj, r]
            
            # Second Strassen: A[ci,1] * B[1,cj]
            # A[ci,1]: rows 2*ci to 2*ci+1, cols 2-3
            # B[1,cj]: rows 2-3, cols 2*cj to 2*cj+1
            
            for r in range(7):
                col = base + 7 + r
                # U: A[ci,1] elements
                for li in range(2):
                    for lj in range(2):
                        a_idx = (2*ci + li) * 4 + (2 + lj)
                        U[a_idx, col] += U_sub[li*2+lj, r]
                # V: B[1,cj] elements
                for li in range(2):
                    for lj in range(2):
                        b_idx = (2 + li) * 6 + (2*cj + lj)
                        V[b_idx, col] += V_sub[li*2+lj, r]
                # W: C[ci,cj] elements (same as first)
                for li in range(2):
                    for lj in range(2):
                        c_idx = (2*cj + lj) * 4 + (2*ci + li)
                        W[c_idx, col] += W_sub[li*2+lj, r]
>>>>>>> REPLACE
