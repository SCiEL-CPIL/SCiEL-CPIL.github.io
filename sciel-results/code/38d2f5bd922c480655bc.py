import numpy as np
import time

def construct():
    n, m, p = 2, 3, 8
    
    def get_rank_44():
        rank = 44
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        # Part 1: A1 (2x1) * B1 (1x8) -> 2x8
        for i in range(2):
            for k in range(8):
                U[i * 3 + 0][col] = 1
                V[0 * 8 + k][col] = 1
                W[k * 2 + i][col] = 1
                col += 1
            
        # Part 2: A2 (2x2) * B2 (2x8) -> 2x8
        for block in range(4):
            for r in range(7):
                # Strassen terms for 2x2
                # We need to map the 7 Strassen terms to the specific indices
                # This part of the original code was a bit magic, let's re-verify or just use the original logic if it works
                # Actually, the original code worked. I will keep it as fallback.
                pass 
        # Re-implementing the original logic for safety
        col = 0
        for i in range(2):
            for k in range(8):
                U[i * 3 + 0][col] = 1
                V[0 * 8 + k][col] = 1
                W[k * 2 + i][col] = 1
                col += 1
        for block in range(4):
            # 1. (a00+a11)(b00+b11)
            U[1][col] = 1
            U[5][col] = 1
            V[1*8 + block*2][col] = 1
            V[2*8 + block*2+1][col] = 1
            W[(block*2)*2 + 0][col] = 1
            W[(block*2+1)*2 + 1][col] = 1
            col += 1
            # 2. (a10+a11)b00
            U[4][col] = 1
            U[5][col] = 1
            V[1*8 + block*2][col] = 1
            W[(block*2)*2 + 1][col] = 1
            W[(block*2+1)*2 + 1][col] = -1
            col += 1
            # 3. a00(b01-b11)
            U[1][col] = 1
            V[1*8 + block*2+1][col] = 1
            V[2*8 + block*2+1][col] = -1
            W[(block*2+1)*2 + 0][col] = 1
            W[(block*2+1)*2 + 1][col] = 1
            col += 1
            # 4. a11(b10-b00)
            U[5][col] = 1
            V[2*8 + block*2][col] = 1
            V[1*8 + block*2][col] = -1
            W[(block*2)*2 + 0][col] = 1
            W[(block*2)*2 + 1][col] = 1
            col += 1
            # 5. (a00+a01)b11
            U[1][col] = 1
            U[2][col] = 1
            V[2*8 + block*2+1][col] = 1
            W[(block*2)*2 + 0][col] = -1
            W[(block*2+1)*2 + 0][col] = 1
            col += 1
            # 6. (a10-a00)(b00+b01)
            U[4][col] = 1
            U[1][col] = -1
            V[1*8 + block*2][col] = 1
            V[1*8 + block*2+1][col] = 1
            W[(block*2+1)*2 + 1][col] = 1
            col += 1
            # 7. (a01-a11)(b10+b11)
            U[2][col] = 1
            U[5][col] = -1
            V[2*8 + block*2][col] = 1
            V[2*8 + block*2+1][col] = 1
            W[(block*2)*2 + 0][col] = 1
            col += 1
        return {"U": U, "V": V, "W": W}

    # Attempt to find rank 10 for 2x3x2
    def find_rank_10_for_232():
        # n=2, m=3, p=2
        # U: 6x10, V: 6x10, W: 4x10
        # Target tensor
        target = np.zeros((6, 6, 4))
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    u = i * 3 + j
                    v = j * 2 + k
                    w = k * 2 + i
                    target[u, v, w] = 1.0

        # Initial guess: Rank 11 solution
        # Strassen for 2x2 part (j=0,1) + Outer for j=2
        # Actually, let's just use random or a simple heuristic if we don't have a good one.
        # But we can use the 11-rank structure as a start for 10.
        # Let's try to optimize from a rank 11 solution by dropping one term.
        # Construct rank 11 solution
        U11 = np.zeros((6, 11))
        V11 = np.zeros((6, 11))
        W11 = np.zeros((4, 11))
        
        # Part 1: j=0,1 (2x2x2 Strassen)
        # Indices: i in 0,1; j in 0,1; k in 0,1
        # u = 3i+j, v = 2j+k, w = 2k+i
        # Standard Strassen for 2x2:
        # A = [[a00, a01], [a10, a11]]
        # B = [[b00, b01], [b10, b11]]
        # Here A is 2x2 (rows i, cols j=0,1). B is 2x2 (rows j=0,1, cols k).
        # u_idx: 0,1 for i=0, j=0,1. 3,4 for i=1, j=0,1.
        # v_idx: 0,1 for j=0, k=0,1. 2,3 for j=1, k=0,1.
        # w_idx: 0,1 for k=0, i=0,1. 2,3 for k=1, i=0,1.
        
        # Term 1: (a00+a11)(b00+b11)
        U11[0, 0] = 1; U11[4, 0] = 1
        V11[0, 0] = 1; V11[3, 0] = 1
        W11[0, 0] = 1; W11[3, 0] = 1
        # Term 2: (a10+a11)b00
        U11[3, 1] = 1; U11[4, 1] = 1
        V11[0, 1] = 1
        W11[1, 1] = 1; W11[3, 1] = -1
        # Term 3: a00(b01-b11)
        U11[0, 2] = 1
        V11[1, 2] = 1; V11[3, 2] = -1
        W11[2, 2] = 1; W11[3, 2] = 1
        # Term 4: a11(b10-b00)
        U11[4, 3] = 1
        V11[2, 3] = 1; V11[0, 3] = -1
        W11[0, 3] = 1; W11[1, 3] = 1
        # Term 5: (a00+a01)b11
        U11[0, 4] = 1; U11[1, 4] = 1
        V11[3, 4] = 1
        W11[0, 4] = -1; W11[2, 4] = 1
        # Term 6: (a10-a00)(b00+b01)
        U11[3, 5] = 1; U11[0, 5] = -1
        V11[0, 5] = 1; V11[1, 5] = 1
        W11[1, 5] = 1
        # Term 7: (a01-a11)(b10+b11)
        U11[1, 6] = 1; U11[4, 6] = -1
        V11[2, 6] = 1; V11[3, 6] = 1
        W11[0, 6] = 1
        
        # Part 2: j=2 (2x1x2 Outer)
        # u = 3i+2, v = 2*2+k = 4+k, w = 2k+i
        # i=0, k=0: u=2, v=4, w=0
        # i=0, k=1: u=2, v=5, w=2
        # i=1, k=0: u=5, v=4, w=1
        # i=1, k=1: u=5, v=5, w=3
        U11[2, 7] = 1; V11[4, 7] = 1; W11[0, 7] = 1
        U11[2, 8] = 1; V11[5, 8] = 1; W11[2, 8] = 1
        U11[5, 9] = 1; V11[4, 9] = 1; W11[1, 9] = 1
        U11[5, 10] = 1; V11[5, 10] = 1; W11[3, 10] = 1

        # Drop last term and optimize
        U_init = U11[:, :10].flatten()
        V_init = V11[:, :10].flatten()
        W_init = W11[:, :10].flatten()
        x0 = np.concatenate([U_init, V_init, W_init])
        
        def residuals(params):
            U = params[:60].reshape(6, 10).T
            V = params[60:120].reshape(6, 10).T
            W = params[120:].reshape(4, 10).T
            T = np.einsum('ur,vr,wr->uvw', U, V, W)
            return (T - target).flatten()

        try:
            import scipy.optimize as opt
            result = opt.least_squares(residuals, x0, method='trf', max_nfev=1000)
            if result.success and np.linalg.norm(result.fun) < 1e-6:
                U = result.x[:60].reshape(6, 10).T
                V = result.x[60:120].reshape(6, 10).T
                W = result.x[120:].reshape(4, 10).T
                return {"U": U, "V": V, "W": W}
        except:
            pass
        return None

    rank_10_sol = find_rank_10_for_232()
    
    if rank_10_sol:
        # Build rank 40 solution for 2x3x8
        # Split B into 4 blocks of 3x2
        # Each block uses the rank 10 solution
        # Map indices
        U_final = np.zeros((6, 40))
        V_final = np.zeros((24, 40))
        W_final = np.zeros((16, 40))
        
        U10 = rank_10_sol['U']
        V10 = rank_10_sol['V']
        W10 = rank_10_sol['W']
        
        for block in range(4):
            # V indices for this block: rows 0..2, cols block*2 .. block*2+1
            # In V_final, these are at rows 0..2, cols block*2..
            # Wait, V is 3x8.
            # V_final[row, col]
            # For block b, the B matrix is B[:, b*2:b*2+2]
            # So V10 rows correspond to V_final rows 0,1,2.
            # V10 cols correspond to V_final cols b*2, b*2+1.
            
            # W indices for this block: cols b*2, b*2+1
            # W_final row = k*2 + i.
            # k is 0..7.
            # For block b, k is b*2, b*2+1.
            # So W10 rows (0..3) map to W_final rows:
            # w10=0 (k=0,i=0) -> w_final = (b*2)*2 + 0
            # w10=1 (k=0,i=1) -> w_final = (b*2)*2 + 1
            # w10=2 (k=1,i=0) -> w_final = (b*2+1)*2 + 0
            # w10=3 (k=1,i=1) -> w_final = (b*2+1)*2 + 1
            
            offset = block * 10
            for r in range(10):
                U_final[:, offset+r] = U10[:, r]
                
                # V mapping
                for v_row in range(3):
                    for v_col in range(2):
                        v10_idx = v_row * 2 + v_col
                        vf_idx = v_row * 8 + (block * 2 + v_col)
                        V_final[vf_idx, offset+r] = V10[v10_idx, r]
                        
                # W mapping
                w10_to_wf = {
                    0: (block * 2) * 2 + 0,
                    1: (block * 2) * 2 + 1,
                    2: (block * 2 + 1) * 2 + 0,
                    3: (block * 2 + 1) * 2 + 1
                }
                for w10_idx in range(4):
                    W_final[w10_to_wf[w10_idx], offset+r] = W10[w10_idx, r]
                    
        return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
    else:
        return get_rank_44()
