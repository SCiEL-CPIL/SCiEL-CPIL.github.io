import numpy as np
import time
import random

def construct():
    n, m, p = 2, 3, 8
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
    # Try to find rank 27 decomposition for 2x2x8
    # If not found, use rank 28 (Strassen x4)
    
    def get_rank_28_228():
        # Returns U_part, V_part, W_part for 2x2x8 with rank 28
        r = 28
        Up = [[0] * r for _ in range(4)] # A is 2x2, flattened 4
        Vp = [[0] * r for _ in range(16)] # B is 2x8, flattened 16
        Wp = [[0] * r for _ in range(16)] # C is 8x2, flattened 16
        
        c = 0
        for block in range(4):
            # Strassen 2x2x2
            # 1. (a00+a11)(b00+b11)
            Up[0][c] = 1
            Up[3][c] = 1
            Vp[0*8 + block*2][c] = 1
            Vp[1*8 + block*2+1][c] = 1
            Wp[(block*2)*2 + 0][c] = 1
            Wp[(block*2+1)*2 + 1][c] = 1
            c += 1
            # 2. (a10+a11)b00
            Up[2][c] = 1
            Up[3][c] = 1
            Vp[0*8 + block*2][c] = 1
            Wp[(block*2)*2 + 1][c] = 1
            Wp[(block*2+1)*2 + 1][c] = -1
            c += 1
            # 3. a00(b01-b11)
            Up[0][c] = 1
            Vp[0*8 + block*2+1][c] = 1
            Vp[1*8 + block*2+1][c] = -1
            Wp[(block*2+1)*2 + 0][c] = 1
            Wp[(block*2+1)*2 + 1][c] = 1
            c += 1
            # 4. a11(b10-b00)
            Up[3][c] = 1
            Vp[1*8 + block*2][c] = 1
            Vp[0*8 + block*2][c] = -1
            Wp[(block*2)*2 + 0][c] = 1
            Wp[(block*2)*2 + 1][c] = 1
            c += 1
            # 5. (a00+a01)b11
            Up[0][c] = 1
            Up[1][c] = 1
            Vp[1*8 + block*2+1][c] = 1
            Wp[(block*2)*2 + 0][c] = -1
            Wp[(block*2+1)*2 + 0][c] = 1
            c += 1
            # 6. (a10-a00)(b00+b01)
            Up[2][c] = 1
            Up[0][c] = -1
            Vp[0*8 + block*2][c] = 1
            Vp[0*8 + block*2+1][c] = 1
            Wp[(block*2+1)*2 + 1][c] = 1
            c += 1
            # 7. (a01-a11)(b10+b11)
            Up[1][c] = 1
            Up[3][c] = -1
            Vp[1*8 + block*2][c] = 1
            Vp[1*8 + block*2+1][c] = 1
            Wp[(block*2)*2 + 0][c] = 1
            c += 1
        return Up, Vp, Wp

    # Attempt numerical search for rank 27
    found_27 = False
    Up27 = None
    Vp27 = None
    Wp27 = None
    
    deadline = time.time() + 100
    if time.time() < deadline:
        try:
            import scipy.optimize as opt
            # Tensor T for 2x2x8
            # U: 4x27, V: 16x27, W: 16x27
            # T[i,j,k] = 1 if i,j,k correspond to A[i,j] B[j,k] C[k,i]?
            # A is 2x2. Indices 0..3.
            # B is 2x8. Indices 0..15.
            # C is 8x2. Indices 0..15.
            # T is 4x16x16? No.
            # Tensor T_{a, b, c} where a in A, b in B, c in C.
            # Non-zero if A_a * B_b contributes to C_c.
            # A_a is A[i, j]. B_b is B[j, k]. C_c is C[k, i].
            # So T_{(i,j), (j,k), (k,i)} = 1.
            
            T = np.zeros((4, 16, 16))
            for i in range(2):
                for j in range(2):
                    a_idx = i * 2 + j
                    for k in range(8):
                        b_idx = j * 8 + k
                        c_idx = k * 2 + i
                        T[a_idx, b_idx, c_idx] = 1

            def objective(x):
                # x is flattened U, V, W
                # U: 4*27, V: 16*27, W: 16*27
                r = 27
                U_flat = x[:4*r].reshape(4, r)
                V_flat = x[4*r:4*r+16*r].reshape(16, r)
                W_flat = x[4*r+16*r:].reshape(16, r)
                
                # Compute tensor
                # T_calc[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # Use einsum or loops
                # T_calc = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat)
                # This might be slow.
                # T_calc = np.zeros((4,16,16))
                # for a in range(4):
                #   for b in range(16):
                #     for c in range(16):
                #       T_calc[a,b,c] = np.dot(U_flat[a], V_flat[b] * W_flat[c]) # No
                # T_calc = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat)
                T_calc = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat)
                err = np.sum((T - T_calc)**2)
                return err

            # Initial guess
            r = 27
            n_vars = 4*r + 16*r + 16*r
            best_err = float('inf')
            best_x = None
            
            # Try a few restarts
            for restart in range(20):
                if time.time() > deadline: break
                x0 = np.random.randn(n_vars) * 0.1
                res = opt.minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
                if res.fun < best_err:
                    best_err = res.fun
                    best_x = res.x
                    if best_err < 1e-8:
                        break
            
            if best_err < 1e-6:
                # Extract U, V, W
                r = 27
                U_flat = best_x[:4*r].reshape(4, r)
                V_flat = best_x[4*r:4*r+16*r].reshape(16, r)
                W_flat = best_x[4*r+16*r:].reshape(16, r)
                
                # Round to nearest 0.5
                U_round = np.round(U_flat * 2) / 2
                V_round = np.round(V_flat * 2) / 2
                W_round = np.round(W_flat * 2) / 2
                
                # Verify
                T_calc = np.einsum('ar,br,cr->abc', U_round, V_round, W_round)
                if np.allclose(T, T_calc, atol=1e-6):
                    Up27 = U_round.tolist()
                    Vp27 = V_round.tolist()
                    Wp27 = W_round.tolist()
                    found_27 = True
        except Exception as e:
            pass

    if found_27:
        # Merge Part 1 and Part 2
        rank = 16 + 27
        U = [[0] * rank for _ in range(6)]
        V = [[0] * rank for _ in range(24)]
        W = [[0] * rank for _ in range(16)]
        
        # Part 1
        col = 0
        for i in range(2):
            for k in range(8):
                U[i * 3 + 0][col] = 1
                V[0 * 8 + k][col] = 1
                W[k * 2 + i][col] = 1
                col += 1
        
        # Part 2
        for r in range(27):
            for i in range(4):
                U[i * 3 + 1 + i//2][col] = Up27[i][r] # Mapping A2 indices to global
                # A2 local: 0->A[0,1], 1->A[0,2], 2->A[1,1], 3->A[1,2]
                # Global: 1, 2, 4, 5
            # Correct mapping:
            # local 0 (a00) -> global 1
            # local 1 (a01) -> global 2
            # local 2 (a10) -> global 4
            # local 3 (a11) -> global 5
            map_a = [1, 2, 4, 5]
            for i in range(4):
                U[map_a[i]][col] = Up27[i][r]
                
            for j in range(16):
                V[8 + j][col] = Vp27[j][r] # B2 local 0..15 -> global 8..23
                
            for k in range(16):
                W[k][col] = Wp27[k][r] # C local 0..15 -> global 0..15
                
            col += 1
    else:
        # Use rank 44
        Up28, Vp28, Wp28 = get_rank_28_228()
        rank = 44
        U = [[0] * rank for _ in range(6)]
        V = [[0] * rank for _ in range(24)]
        W = [[0] * rank for _ in range(16)]
        
        col = 0
        for i in range(2):
            for k in range(8):
                U[i * 3 + 0][col] = 1
                V[0 * 8 + k][col] = 1
                W[k * 2 + i][col] = 1
                col += 1
                
        map_a = [1, 2, 4, 5]
        for r in range(28):
            for i in range(4):
                U[map_a[i]][col] = Up28[i][r]
            for j in range(16):
                V[8 + j][col] = Vp28[j][r]
            for k in range(16):
                W[k][col] = Wp28[k][r]
            col += 1

    return {"U": U, "V": V, "W": W}
