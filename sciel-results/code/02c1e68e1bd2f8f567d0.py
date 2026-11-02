"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    u_dim = n * m  # 4
    v_dim = m * p  # 8
    w_dim = p * n  # 8
    
    deadline = time.time() + 140
    
    def build_rank14():
        rank = 14
        U = [[0] * rank for _ in range(u_dim)]
        V = [[0] * rank for _ in range(v_dim)]
        W = [[0] * rank for _ in range(w_dim)]
        
        terms = []
        def add_term(u_entries, v_entries, w_entries):
            terms.append((u_entries, v_entries, w_entries))
        
        add_term([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])
        add_term([(1,0,1), (1,1,1)], [(0,0,1)], [(0,1,1), (1,1,-1)])
        add_term([(0,0,1)], [(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)])
        add_term([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)])
        add_term([(0,0,1), (0,1,1)], [(1,1,1)], [(1,0,1), (0,0,-1)])
        add_term([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)])
        add_term([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
        
        for r in range(7):
            u_entries, v_entries, w_entries = terms[r]
            col1 = r
            for i, j, val in u_entries:
                u_idx = i * m + j
                U[u_idx][col1] += val
            for j, k, val in v_entries:
                v_idx = j * p + k
                V[v_idx][col1] += val
            for k, i, val in w_entries:
                w_idx = k * n + i
                W[w_idx][col1] += val
            
            col2 = 7 + r
            for i, j, val in u_entries:
                u_idx = i * m + j
                U[u_idx][col2] += val
            for j, k, val in v_entries:
                k_new = k + 2
                v_idx = j * p + k_new
                V[v_idx][col2] += val
            for k, i, val in w_entries:
                k_new = k + 2
                w_idx = k_new * n + i
                W[w_idx][col2] += val
        return U, V, W
    
    U14, V14, W14 = build_rank14()
    U14_np = np.array(U14, dtype=float)
    V14_np = np.array(V14, dtype=float)
    W14_np = np.array(W14, dtype=float)
    
    def verify(U, V, W, tol=1e-6):
        rank = len(U[0])
        np.random.seed(123)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_comp = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    u_val = sum(U[i*m+j][r] * A[i,j] for j in range(m))
                    v_val = sum(V[j*p+k][r] * B[j,k] for j in range(m))
                    w_val = W[k*n+i][r]
                    s += u_val * v_val * w_val
                C_comp[i,k] = s
        return np.allclose(C_comp, C_expected, atol=tol)
    
    def try_rank13(seed=42, max_iters=2000):
        np.random.seed(seed)
        R = 13
        drop = np.random.randint(14)
        idx = [i for i in range(14) if i != drop]
        U = U14_np[:, idx].copy()
        V = V14_np[:, idx].copy()
        W = W14_np[:, idx].copy()
        U += 0.1 * np.random.randn(*U.shape)
        V += 0.1 * np.random.randn(*V.shape)
        W += 0.1 * np.random.randn(*W.shape)
        
        i_vals, j_vals, jp_vals, k_vals, targets = [], [], [], [], []
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        i_vals.append(i)
                        j_vals.append(j)
                        jp_vals.append(jp)
                        k_vals.append(k)
                        targets.append(1.0 if j == jp else 0.0)
        
        i_arr = np.array(i_vals)
        j_arr = np.array(j_vals)
        jp_arr = np.array(jp_vals)
        k_arr = np.array(k_vals)
        target_arr = np.array(targets)
        
        u_idx = i_arr * m + j_arr
        v_idx = jp_arr * p + k_arr
        w_idx = k_arr * n + i_arr
        r_arr = np.arange(R)
        
        def loss_and_grads(U, V, W):
            U_sel = U[u_idx[:, None], r_arr[None, :]]
            V_sel = V[v_idx[:, None], r_arr[None, :]]
            W_sel = W[w_idx[:, None], r_arr[None, :]]
            val = U_sel * V_sel * W_sel
            sums = val.sum(axis=1)
            err = sums - target_arr
            loss = np.sum(err ** 2)
            dU = 2 * err[:, None] * V_sel * W_sel
            dV = 2 * err[:, None] * U_sel * W_sel
            dW = 2 * err[:, None] * U_sel * V_sel
            gU = np.zeros_like(U)
            gV = np.zeros_like(V)
            gW = np.zeros_like(W)
            for c in range(len(u_idx)):
                gU[u_idx[c]] += dU[c]
                gV[v_idx[c]] += dV[c]
                gW[w_idx[c]] += dW[c]
            return loss, gU, gV, gW
        
        lr = 0.0005
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            loss, gU, gV, gW = loss_and_grads(U, V, W)
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if best_loss < 1e-10:
                    break
            U -= lr * gU
            V -= lr * gV
            W -= lr * gW
            if it % 50 == 0 and it > 0:
                new_loss, _, _, _ = loss_and_grads(U, V, W)
                if new_loss > loss * 1.3:
                    lr *= 0.7
                elif new_loss < loss * 0.7:
                    lr *= 1.3
        
        return best_U, best_V, best_W, best_loss
    
    best13 = None
    for attempt in range(10):
        if time.time() > deadline - 20:
            break
        U13, V13, W13, loss13 = try_rank13(seed=42 + attempt)
        if best13 is None or loss13 < best13[3]:
            best13 = (U13, V13, W13, loss13)
        if loss13 < 1e-8:
            break
    
    if best13 is not None and best13[3] < 1e-6:
        U13, V13, W13 = best13[0], best13[1], best13[2]
        U_r = np.round(U13 * 2) / 2
        V_r = np.round(V13 * 2) / 2
        W_r = np.round(W13 * 2) / 2
        U_out = U_r.tolist()
        V_out = V_r.tolist()
        W_out = W_r.tolist()
        if verify(U_out, V_out, W_out, tol=1e-5):
            return {"U": U_out, "V": V_out, "W": W_out}
    
    U_out = U14_np.tolist()
    V_out = V14_np.tolist()
    W_out = W14_np.tolist()
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
