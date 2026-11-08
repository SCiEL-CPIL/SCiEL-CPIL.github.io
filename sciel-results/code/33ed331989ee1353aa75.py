            V = [[0] * target_rank for _ in range(mp)]
            W = [[0] * target_rank for _ in range(pn)]
            
            for c in range(target_rank):
                for (j, k) in v_constraints[c]:
                    V[j*p+k][c] = 1
                for (k, i) in w_constraints[c]:
                    W[k*n+i][c] = 1
   