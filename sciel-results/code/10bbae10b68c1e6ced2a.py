import numpy as np

def construct():
    n, m, p = 6, 6, 6
    rank = 189  # 7 * 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 decomposition
    # u, v, w are 2x2 matrices
    strassen = [
        # u, v, w
        ([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]),
        ([[0, 0], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [1, -1]]),
        ([[1, 0], [0, 0]], [[0, 0], [1, -1]], [[0, 1], [0, 1]]),
        ([[0, 0], [0, 1]], [[0, 1], [-1, 0]], [[1, 0], [1, 0]]),
        ([[1, 1], [0, 0]], [[0, 0], [0, 1]], [[-1, 1], [0, 0]]),
        ([[0, 0], [1, -1]], [[1, 0], [1, 0]], [[0, 0], [0, 1]]),
        ([[0, 1], [0, -1]], [[0, 1], [0, 1]], [[1, 0], [0, 0]])
    ]
    
    # Schoolbook 3x3 decomposition
    # a, b, c are 3x3 matrices
    schoolbook = []
    for alpha in range(3):
        for beta in range(3):
            for gamma in range(3):
                a = [[0]*3 for _ in range(3)]
                b = [[0]*3 for _ in range(3)]
                c = [[0]*3 for _ in range(3)]
                a[alpha][beta] = 1
                b[beta][gamma] = 1
                c[alpha][gamma] = 1
                schoolbook.append((a, b, c))
                
    col = 0
    for s in range(7):
        u_s, v_s, w_s = strassen[s]
        for t in range(27):
            a_t, b_t, c_t = schoolbook[t]
            
            # Iterate over non-zero entries
            for i_blk in range(2):
                for j_blk in range(2):
                    if u_s[i_blk][j_blk] != 0:
                        for alpha in range(3):
                            for beta in range(3):
                                if a_t[alpha][beta] != 0:
                                    val_u = u_s[i_blk][j_blk] * a_t[alpha][beta]
                                    I = 3 * i_blk + alpha
                                    J = 3 * j_blk + beta
                                    idx_u = I * 6 + J
                                    U[idx_u][col] += val_u
                                    
                                    # V and W depend on beta, gamma and k_blk
                                    # We need to iterate over k_blk and gamma too for V and W
                                    # But U only depends on i_blk, j_blk, alpha, beta
                                    # So we can't just loop once.
                                    # We need to loop over all indices for V and W as well.
                                    pass
            # The above loop is incomplete for V and W.
            # Let's restructure to loop over all 4 indices for each matrix separately or together.
            # Since the support is a product of supports, we can iterate over the Cartesian product.
            
            # Reset col accumulation? No, we are building column 'col'.
            # We must clear U, V, W for this column first?
            # They are initialized to 0. We are adding to them.
            # But the loops above only filled U.
            
            # Let's do it properly:
            # For each (s, t), we have a rank-1 tensor.
            # U is supported on (i_blk, j_blk, alpha, beta)
            # V is supported on (j_blk, k_blk, beta, gamma)
            # W is supported on (k_blk, i_blk, gamma, alpha)
            
            # We can just iterate over all possible combinations of block and entry indices
            # and check if the product is non-zero.
            
            # Since 2*2*3*3 = 36 iterations per matrix, and 7*27=189 terms,
            # 189 * 3 * 36 is small.
            
            # Clear current column (though it's zero initially)
            # Actually, we are constructing the whole matrix.
            
            for i_blk in range(2):
                for j_blk in range(2):
                    for k_blk in range(2):
                        for alpha in range(3):
                            for beta in range(3):
                                for gamma in range(3):
                                    # Check U
                                    if u_s[i_blk][j_blk] != 0 and a_t[alpha][beta] != 0:
                                        val_u = u_s[i_blk][j_blk] * a_t[alpha][beta]
                                        I = 3 * i_blk + alpha
                                        J = 3 * j_blk + beta
                                        idx_u = I * 6 + J
                                        U[idx_u][col] += val_u
                                        
                                    # Check V
                                    if v_s[j_blk][k_blk] != 0 and b_t[beta][gamma] != 0:
                                        val_v = v_s[j_blk][k_blk] * b_t[beta][gamma]
                                        J = 3 * j_blk + beta
                                        K = 3 * k_blk + gamma
                                        idx_v = J * 6 + K
                                        V[idx_v][col] += val_v
                                        
                                    # Check W
                                    if w_s[i_blk][k_blk] != 0 and c_t[alpha][gamma] != 0:
                                        val_w = w_s[i_blk][k_blk] * c_t[alpha][gamma]
                                        K = 3 * k_blk + gamma
                                        I = 3 * i_blk + alpha
                                        idx_w = K * 6 + I
                                        W[idx_w][col] += val_w
                                        
            col += 1

    return {"U": U, "V": V, "W": W}
