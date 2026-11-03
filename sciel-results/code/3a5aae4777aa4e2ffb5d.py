def _verify_3x3_decomp(U, V, W):
    # Check if sum U_c V_c W_c == T_matmul
    # T[i,j,k] = 1 if j==j (always) and i,i and k,k?
    # T_matmul[i,j,k] = 1 if the term is A[i,j]B[j,k] -> C[i,k].
    # So T[i,j,k] = 1 for all i,j,k? No.
    # The tensor T is defined by T[i,j,k] = 1 if the indices match the multiplication.
    # A[i,j] * B[j,k] -> C[i,k].
    # So T[i,j,k] = 1.
    # Wait, the tensor is $T = \sum_{i,j,k} e_i \otimes e_j \otimes e_k$.
    # So T[i,j,k] = 1 for all i,j,k in 0..2.
    
    # Check:
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # W index for C[i,k] is k*3+i
                w_idx = k*3+i
                # U index for A[i,j] is i*3+j
                u_idx = i*3+j
                # V index for B[j,k] is j*3+k
                v_idx = j*3+k
                
                sum_val = 0
                for c in range(len(U[0])):
                    sum_val += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                
                if abs(sum_val - 1.0) > 1e-9:
                    return False
    return True
