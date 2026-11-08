import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin

    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    # Define the target tensor
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[(a, b, c)] = 1

    def verify_fast(U_np, V_np, W_np, rank):
        """Fast vectorized verification using numpy."""
        # Compute all tensor entries: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # We need T[a,b,c] = 1 if (a,b,c) in target, else 0
        # Use einsum or manual broadcasting
        # U: (nm, rank), V: (mp, rank), W: (pn, rank)
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # This is a 3-tensor. We can compute it as:
        # For each r, outer product U[:,r] @ V[:,r] gives (nm, mp) matrix,
        # then multiply by W[:,r] and accumulate into (nm, mp, pn)
        
        T = np.zeros((n * m, m * p, p * n))
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U_np[:, r], V_np[:, r], W_np[:, r])
        
        # Check target entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    if abs(T[a, b, c] - 1) > 1e-9:
                        return False
        # Check off-diagonal
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    if (a, b, c) not in target:
                        if abs(T[a, b, c]) > 1e-9:
                            return False
        return True

    def verify(U, V, W, rank):
        """Check if U, V, W (as lists) form a valid decomposition."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        return verify_fast(U_np, V_np, W_np, rank)

    # Strassen-based rank 28 decomposition
    def strassen_decomposition():
        rank = 28
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]

        strassen_terms = [
            ("M1", [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            ("M2", [(1,0,1), (1,1,1)], [(0,0,1)], [(0,0,1), (1,0,1)]),
            ("M3", [(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
            ("M4", [(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
            ("M5", [(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (1,1,-1)]),
            ("M6", [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
            ("M7", [(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1), (1,1,-1)]),
        ]

        blocks = [
            (0, 0, 0, 0, 0, 0),
            (0, 2, 0, 2, 0, 2),
            (0, 0, 2, 0, 0, 0),
            (0, 2, 2, 2, 0, 2),
        ]

        col = 0
        for a_off, ac_off, b_off, bc_off, c_off, cc_off in blocks:
            for name, u_coeffs, v_coeffs, w_coeffs in strassen_terms:
                for (ar, ac, coeff_u) in u_coeffs:
                    global_ar = a_off + ar
                    global_ac = ac_off + ac
                    U[global_ar * m + global_ac][col] += coeff_u
                for (br, bc, coeff_v) in v_coeffs:
                    global_br = b_off + br
                    global_bc = bc_off + bc
                    V[global_br * p + global_bc][col] += coeff_v
                for (cr, cc, coeff_w) in w_coeffs:
                    global_cr = c_off + cr
                    global_cc = cc_off + cc
                    W[global_cc * n + global_cr][col] += coeff_w
                col += 1

        return {"U": U, "V": V, "W": W}

    # Try Strassen decomposition
    if time.time() < deadline - 10:
        strassen = strassen_decomposition()
        if verify(strassen["U"], strassen["V"], strassen["W"], 28):
            best = strassen
            best_rank = 28

    # Gradient-based optimization to find lower rank
    if time.time() < deadline - 30 and best_rank > 24:
        import random
        random.seed(42)
        np.random.seed(42)
        
        def tensor_loss(U_np, V_np, W_np, rank):
            """Compute loss: sum of (T[a,b,c] - target[a,b,c])^2 over all entries."""
            T = np.zeros((n * m, m * p, p * n))
            for r in range(rank):
                T += np.einsum('a,b,c->abc', U_np[:, r], V_np[:, r], W_np[:, r])
            
            loss = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        loss += (T[a, b, c] - 1) ** 2
            for a in range(n * m):
                for b in range(m * p):
                    for c in range(p * n):
                        if (a, b, c) not in target:
                            loss += T[a, b, c] ** 2
            return loss

        def gradient_descent(rank, max_iters=2000, lr=0.01):
            U_np = np.random.randn(n * m, rank) * 0.3
            V_np = np.random.randn(m * p, rank) * 0.3
            W_np = np.random.randn(p * n, rank) * 0.3
            
            best_loss = float('inf')
            best_state = (U_np.copy(), V_np.copy(), W_np.copy())
            
            for it in range(max_iters):
                if time.time() > deadline - 35:
                    break
                # Compute tensor
                T = np.zeros((n * m, m * p, p * n))
                for r in range(rank):
                    T += np.einsum('a,b,c->abc', U_np[:, r], V_np[:, r], W_np[:, r])
                
                # Compute gradient
                dU = np.zeros_like(U_np)
                dV = np.zeros_like(V_np)
                dW = np.zeros_like(W_np)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            err = T[a, b, c] - 1
                            for r in range(rank):
                                dU[a, r] += 2 * err * V_np[b, r] * W_np[c, r]
                                dV[b, r] += 2 * err * U_np[a, r] * W_np[c, r]
                                dW[c, r] += 2 * err * U_np[a, r] * V_np[b, r]
                
                for a in range(n * m):
                    for b in range(m * p):
                        for c in range(p * n):
                            if (a, b, c) not in target:
                                err = T[a, b, c]
                                for r in range(rank):
                                    dU[a, r] += 2 * err * V_np[b, r] * W_np[c, r]
                                    dV[b, r] += 2 * err * U_np[a, r] * W_np[c, r]
                                    dW[c, r] += 2 * err * U_np[a, r] * V_np[b, r]
                
                # Update
                U_np -= lr * dU
                V_np -= lr * dV
                W_np -= lr * dW
                
                # Check loss
                loss = tensor_loss(U_np, V_np, W_np, rank)
                if loss < best_loss:
                    best_loss = loss
                    best_state = (U_np.copy(), V_np.copy(), W_np.copy())
                
                if best_loss < 1e-8:
                    return best_state[0], best_state[1], best_state[2]
            
            return best_state[0], best_state[1], best_state[2]

        for target_rank in [27, 26, 25, 24]:
            if time.time() > deadline - 35:
                break
            found = False
            for attempt in range(10):
                if time.time() > deadline - 35:
                    break
                U_np, V_np, W_np = gradient_descent(target_rank, max_iters=1500, lr=0.005)
                U_list = U_np.tolist()
                V_list = V_np.tolist()
                W_list = W_np.tolist()
                if verify(U_list, V_list, W_list, target_rank):
                    best = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    found = True
                    break
            if found:
                break

    return best
# EVOLVE-BLOCK-END
