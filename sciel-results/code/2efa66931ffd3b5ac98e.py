import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # 1. Construct Rank 28 Solution
    rank_28 = 28
    U28 = np.zeros((n * m, rank_28), dtype=int)
    V28 = np.zeros((m * p, rank_28), dtype=int)
    W28 = np.zeros((p * n, rank_28), dtype=int)

    # Strassen blocks
    # Block k uses 7 columns starting at k*7
    # We define the 4 blocks as in the current program
    blocks = [
        ([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]),
        ([0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1]),
        ([0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3]),
        ([0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3]),
    ]

    for blk in range(4):
        ar0, ar1 = blocks[blk][0]
        ac0, ac1 = blocks[blk][1]
        br0, br1 = blocks[blk][2]
        bc0, bc1 = blocks[blk][3]
        cr0, cr1 = blocks[blk][4]
        cc0, cc1 = blocks[blk][5]

        base = blk * 7

        aa00 = ar0 * m + ac0
        aa01 = ar0 * m + ac1
        aa10 = ar1 * m + ac0
        aa11 = ar1 * m + ac1

        bb00 = br0 * p + bc0
        bb01 = br0 * p + bc1
        bb10 = br1 * p + bc0
        bb11 = br1 * p + bc1

        cc00 = cc0 * n + cr0
        cc01 = cc1 * n + cr0
        cc10 = cc0 * n + cr1
        cc11 = cc1 * n + cr1

        # M1 = (a00+a11)(b00+b11)
        U28[aa00][base + 0] += 1
        U28[aa11][base + 0] += 1
        V28[bb00][base + 0] += 1
        V28[bb11][base + 0] += 1
        W28[cc00][base + 0] += 1
        W28[cc11][base + 0] += 1

        # M2 = (a10+a11)b00
        U28[aa10][base + 1] += 1
        U28[aa11][base + 1] += 1
        V28[bb00][base + 1] += 1
        W28[cc10][base + 1] += 1
        W28[cc11][base + 1] -= 1

        # M3 = a00(b01-b11)
        U28[aa00][base + 2] += 1
        V28[bb01][base + 2] += 1
        V28[bb11][base + 2] -= 1
        W28[cc01][base + 2] += 1
        W28[cc11][base + 2] += 1

        # M4 = a11(b10-b00)
        U28[aa11][base + 3] += 1
        V28[bb10][base + 3] += 1
        V28[bb00][base + 3] -= 1
        W28[cc00][base + 3] += 1
        W28[cc10][base + 3] += 1

        # M5 = (a00+a01)b11
        U28[aa00][base + 4] += 1
        U28[aa01][base + 4] += 1
        V28[bb11][base + 4] += 1
        W28[cc00][base + 4] -= 1
        W28[cc01][base + 4] += 1

        # M6 = (a10-a00)(b00+b01)
        U28[aa10][base + 5] += 1
        U28[aa00][base + 5] -= 1
        V28[bb00][base + 5] += 1
        V28[bb01][base + 5] += 1
        W28[cc11][base + 5] += 1

        # M7 = (a01-a11)(b10+b11)
        U28[aa01][base + 6] += 1
        U28[aa11][base + 6] -= 1
        V28[bb10][base + 6] += 1
        V28[bb11][base + 6] += 1
        W28[cc00][base + 6] += 1

    def verify_fast(U_mat, V_mat, W_mat):
        """
        Fast verification using vectorized operations.
        U: (n*m, R)
        V: (m*p, R)
        W: (p*n, R)
        """
        Uf = np.array(U_mat, dtype=float)
        Vf = np.array(V_mat, dtype=float)
        Wf = np.array(W_mat, dtype=float)
        
        # Reshape U to (n, m, R)
        U_reshaped = Uf.reshape(n, m, -1)
        # Reshape V to (m, p, R)
        V_reshaped = Vf.reshape(m, p, -1)
        # Reshape W to (p, n, R)
        W_reshaped = Wf.reshape(p, n, -1)
        
        # Compute M_ikr = sum_j U_ijk V_jkr
        # einsum: 'ijr,jkr->ikr'
        M = np.einsum('ijr,jkr->ikr', U_reshaped, V_reshaped)
        
        # Compute C_ik = sum_r W_kir M_ikr
        # einsum: 'kir,ikr->ik'
        C_approx = np.einsum('kir,ikr->ik', W_reshaped, M)
        
        # Compute exact C
        A = np.arange(n * m).reshape(n, m) + 1.0
        B = np.arange(m * p).reshape(m, p) + 2.0
        C_exact = A @ B
        
        return np.allclose(C_exact, C_approx, atol=1e-9)

    if not verify_fast(U28, V28, W28):
        return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}

    # 2. Attempt Rank 27 Search
    # Build target tensor T
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                T[a * m + b, b * p + c, c * n + a] = 1.0

    target_rank = 27
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None

    # We will try to optimize 27 terms.
    # Initialize by dropping one column from the 28-term solution.
    # We try dropping each of the 28 columns to see if any leads to a solution.
    
    # Use a simple gradient descent with momentum
    # Variables: U (8x27), V (16x27), W (8x27)
    
    lr = 0.05
    momentum = 0.9
    max_iters = 5000
    
    # Pre-compute einsums for speed?
    # Error E = T - sum_r U_r V_r W_r
    # Grad U_r = - sum_{j,k} E_{ijk} V_{jr} W_{kr}
    
    for drop_idx in range(rank_28):
        if time.time() > deadline - 20:
            break
            
        # Initialize from 28-solution minus one column
        mask = np.ones(rank_28, dtype=bool)
        mask[drop_idx] = False
        U_init = U28[:, mask].astype(float)
        V_init = V28[:, mask].astype(float)
        W_init = W28[:, mask].astype(float)
        
        # Add small noise to escape local minima / symmetry
        U_cur = U_init + np.random.randn(*U_init.shape) * 0.1
        V_cur = V_init + np.random.randn(*V_init.shape) * 0.1
        W_cur = W_init + np.random.randn(*W_init.shape) * 0.1
        
        vU = np.zeros_like(U_cur)
        vV = np.zeros_like(V_cur)
        vW = np.zeros_like(W_cur)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Compute T_est
            T_est = np.zeros_like(T)
            for r in range(target_rank):
                T_est += np.einsum('i,j,k->ijk', U_cur[:, r], V_cur[:, r], W_cur[:, r])
                
            E = T - T_est
            err_norm = np.linalg.norm(E)
            
            if err_norm < 1e-6:
                # Check if we can round to integers/half-integers
                U_round = np.round(U_cur * 2) / 2
                V_round = np.round(V_cur * 2) / 2
                W_round = np.round(W_cur * 2) / 2
                
                # Verify rounded solution
                if verify(U_round.astype(int), V_round.astype(int), W_round.astype(int)):
                    return {"U": U_round.astype(int).tolist(), "V": V_round.astype(int).tolist(), "W": W_round.astype(int).tolist()}
                
                # If not integer, check if the float solution is valid (though task asks for int/half-int)
                # The task says "integer/half-integer entries". So we must round.
                # If rounding fails, we continue.
                
            if err_norm < best_err:
                best_err = err_norm
                best_U = U_cur.copy()
                best_V = V_cur.copy()
                best_W = W_cur.copy()
                
            # Compute Gradients
            # gU[:, r] = - einsum('ijk,jr,kr->i', E, V_cur, W_cur)
            # gV[:, r] = - einsum('ijk,ir,kr->j', E, U_cur, W_cur)
            # gW[:, r] = - einsum('ijk,ir,jr->k', E, U_cur, V_cur)
            
            gU = -np.einsum('ijk,jr,kr->ir', E, V_cur, W_cur)
            gV = -np.einsum('ijk,ir,kr->jr', E, U_cur, W_cur)
            gW = -np.einsum('ijk,ir,jr->kr', E, U_cur, V_cur)
            
            # Update with momentum
            vU = momentum * vU - lr * gU
            vV = momentum * vV - lr * gV
            vW = momentum * vW - lr * gW
            
            U_cur += vU
            V_cur += vV
            W_cur += vW
            
            # Normalize to prevent divergence (optional, but helps)
            # for r in range(target_rank):
            #     norm_U = np.linalg.norm(U_cur[:, r])
            #     if norm_U > 0:
            #         U_cur[:, r] /= norm_U
            #     norm_V = np.linalg.norm(V_cur[:, r])
            #     if norm_V > 0:
            #         V_cur[:, r] /= norm_V
            #     norm_W = np.linalg.norm(W_cur[:, r])
            #     if norm_W > 0:
            #         W_cur[:, r] /= norm_W
            #     # Re-scale back? This is tricky. Better to just use small LR.

    # If rank 27 not found, return rank 28
    return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
