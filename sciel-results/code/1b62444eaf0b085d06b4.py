"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-93 tensor decomposition for 4x4 * 4x7 matrix multiplication.
    
    Strategy:
      Part 1: 4x4 * 4x4 (columns 0-3) using recursive Strassen, rank 49
      Part 2: 4x4 * 4x3 (columns 4-6) using 4 blocks of (2x2)*(2x3), each rank 11, total 44
      Grand total: 49 + 44 = 93
    """
    n, m, p = DIMENSIONS  # 4, 4, 7
    start_time = time.time()
    deadline = start_time + 110  # safety margin under 150s wall clock

    # Strassen rank-7 decomposition of the 2x2x2 matmul tensor.
    s2x2 = [
        ((1,0,0,1), (1,0,0,1), (1,0,0,-1)),
        ((0,0,1,1), (1,0,0,0), (0,0,1,-1)),
        ((1,0,0,0), (0,1,0,-1), (0,1,0,1)),
        ((0,0,0,1), (0,0,1,-1), (-1,0,1,0)),
        ((1,1,0,0), (0,0,0,1), (0,-1,0,1)),
        ((-1,0,1,0), (1,1,0,0), (1,0,0,0)),
        ((0,1,0,-1), (0,0,1,1), (-1,0,0,0)),
    ]

    all_terms = []

    # Part 1: Recursive Strassen for 4x4*4x4 (columns 0-3)
    terms_part1 = []
    
    block_defs = [
        ([((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)]),
        ([((1,0),1), ((1,1),1)], [((0,0),1)]),
        ([((0,0),1)], [((0,1),1), ((1,1),-1)]),
        ([((1,1),1)], [((1,0),1), ((0,0),-1)]),
        ([((0,0),1), ((0,1),1)], [((1,1),1)]),
        ([((1,0),1), ((0,0),-1)], [((0,0),1), ((0,1),1)]),
        ([((0,1),1), ((1,1),-1)], [((1,0),1), ((1,1),1)]),
    ]

    out_recon = {
        (0,0): [(0,1), (3,-1), (5,1), (6,-1)],
        (1,0): [(1,1), (3,1)],
        (0,1): [(2,1), (4,1)],
        (1,1): [(0,1), (2,-1), (1,1), (4,-1)],
    }

    for k, (x_blocks, y_blocks) in enumerate(block_defs):
        X = {}
        for (br, bc), coeff in x_blocks:
            for lr in range(2):
                for lc in range(2):
                    ar, ac = br*2+lr, bc*2+lc
                    if (lr,lc) not in X:
                        X[(lr,lc)] = {}
                    X[(lr,lc)][(ar,ac)] = X[(lr,lc)].get((ar,ac), 0) + coeff

        Y = {}
        for (br, bc), coeff in y_blocks:
            for lr in range(2):
                for lc in range(2):
                    br_, bc_ = br*2+lr, bc*2+lc
                    if (lr,lc) not in Y:
                        Y[(lr,lc)] = {}
                    Y[(lr,lc)][(br_,bc_)] = Y[(lr,lc)].get((br_,bc_), 0) + coeff

        for t in range(7):
            xvec, yvec, zvec = s2x2[t]
            a_vec = {}
            for lr in range(2):
                for lc in range(2):
                    xc = xvec[lr*2+lc]
                    if xc == 0:
                        continue
                    if (lr,lc) not in X:
                        continue
                    for (ar,ac), ac2 in X[(lr,lc)].items():
                        val = xc * ac2
                        if val != 0:
                            a_vec[(ar,ac)] = a_vec.get((ar,ac), 0) + val

            b_vec = {}
            for lr in range(2):
                for lc in range(2):
                    yc = yvec[lr*2+lc]
                    if yc == 0:
                        continue
                    if (lr,lc) not in Y:
                        continue
                    for (br_,bc_), bc2 in Y[(lr,lc)].items():
                        val = yc * bc2
                        if val != 0:
                            b_vec[(br_,bc_)] = b_vec.get((br_,bc_), 0) + val

            c_vec = {}
            for (ob_r, ob_c), sign_list in out_recon.items():
                for (mk, sign) in sign_list:
                    if mk != k:
                        continue
                    for lr in range(2):
                        for lc in range(2):
                            zc = zvec[lr*2+lc]
                            if zc == 0:
                                continue
                            cr, cc = ob_r*2+lr, ob_c*2+lc
                            val = sign * zc
                            if val != 0:
                                c_vec[(cr,cc)] = c_vec.get((cr,cc), 0) + val

            if a_vec and b_vec and c_vec:
                terms_part1.append((a_vec, b_vec, c_vec))

    # Part 2: 4x4 * 4x3 (columns 4-6) using 4 blocks of (2x2)*(2x3)
    terms_part2 = []
    for br in range(2):
        for bc in range(2):
            # Strassen for columns 4-5
            for t in range(7):
                xvec, yvec, zvec = s2x2[t]
                a_vec = {}
                for lr in range(2):
                    for lc in range(2):
                        xc = xvec[lr*2+lc]
                        if xc == 0:
                            continue
                        ar, ac = br*2+lr, bc*2+lc
                        a_vec[(ar,ac)] = a_vec.get((ar,ac), 0) + xc

                b_vec = {}
                for lr in range(2):
                    for lc in range(2):
                        yc = yvec[lr*2+lc]
                        if yc == 0:
                            continue
                        br_, bc_ = bc*2+lr, 4+lc
                        b_vec[(br_,bc_)] = b_vec.get((br_,bc_), 0) + yc

                c_vec = {}
                for lr in range(2):
                    for lc in range(2):
                        zc = zvec[lr*2+lc]
                        if zc == 0:
                            continue
                        cr, cc = br*2+lr, 4+lc
                        c_vec[(cr,cc)] = c_vec.get((cr,cc), 0) + zc

                if a_vec and b_vec and c_vec:
                    terms_part2.append((a_vec, b_vec, c_vec))

            # Schoolbook for column 6
            for lr in range(2):
                for lc in range(2):
                    ar, ac = br*2+lr, bc*2+lc
                    br_, bc_ = bc*2+lr, 6
                    cr, cc = br*2+lr, 6
                    a_vec = {(ar,ac): 1}
                    b_vec = {(br_,bc_): 1}
                    c_vec = {(cr,cc): 1}
                    terms_part2.append((a_vec, b_vec, c_vec))

    all_terms = terms_part1 + terms_part2
    rank = len(all_terms)

    # Build U, V, W matrices
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    for r, (a_vec, b_vec, c_vec) in enumerate(all_terms):
        for (ar, ac), val in a_vec.items():
            U[ar * m + ac][r] = val
        for (br, bc), val in b_vec.items():
            V[br * p + bc][r] = val
        for (cr, cc), val in c_vec.items():
            W[cc * n + cr][r] = val

    # Verification and Fallback
    try:
        if time.time() < deadline:
            Ua = np.array(U, dtype=float)
            Va = np.array(V, dtype=float)
            Wa = np.array(W, dtype=float)
            
            np.random.seed(42)
            A_test = np.random.randint(-10, 10, size=(n, m))
            B_test = np.random.randint(-10, 10, size=(m, p))
            C_expected = A_test @ B_test
            
            C_comp = np.zeros((n, p), dtype=float)
            
            U_reshaped = Ua.reshape(n, m, rank)
            V_reshaped = Va.reshape(m, p, rank)
            W_reshaped = Wa.reshape(p, n, rank)
            
            for r in range(rank):
                u_r = U_reshaped[:, :, r]
                v_r = V_reshaped[:, :, r]
                w_r = W_reshaped[:, :, r]
                
                temp = u_r @ v_r
                C_comp += temp * w_r.T
                
            if not np.allclose(C_comp, C_expected, atol=1e-9):
                rank_fb = n * m * p
                U_fb = np.zeros((n * m, rank_fb))
                V_fb = np.zeros((m * p, rank_fb))
                W_fb = np.zeros((p * n, rank_fb))
                
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_fb[i * m + j, col] = 1.0
                            V_fb[j * p + k, col] = 1.0
                            W_fb[k * n + i, col] = 1.0
                            col += 1
                            
                return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
                
    except Exception:
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1.0
                    V_fb[j * p + k, col] = 1.0
                    W_fb[k * n + i, col] = 1.0
                    col += 1
                    
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
