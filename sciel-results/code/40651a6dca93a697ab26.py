import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    deadline = time.time() + 145
    
    # Schoolbook fallback: rank 80
    best_rank = n * m * p  # 80
    U_sb = np.zeros((n*m, best_rank))
    V_sb = np.zeros((m*p, best_rank))
    W_sb = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Precompute indices for non-zero entries
    # a = i*m + j, b = j*p + k, c = k*n + i
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                indices.append((a, b, c))
    
    # For efficient gradient calculation of <T, D>
    # d<T,D>/dU[ar] = sum_{k} V[b_{ijk}, r] W[c_{ijk}, r] where a_{ijk} = a
    # For fixed a (i,j), the sum is over k.
    # We can precompute for each a, the list of (b, c) pairs.
    U_grad_terms = [[] for _ in range(n*m)]
    V_grad_terms = [[] for _ in range(m*p)]
    W_grad_terms = [[] for _ in range(p*n)]
    
    for a, b, c in indices:
        U_grad_terms[a].append((b, c))
        V_grad_terms[b].append((a, c))
        W_grad_terms[c].append((a, b))
        
    def loss_and_grad(U, V, W):
        R = U.shape[1]
        # Compute S_U, S_V, S_W
        # S_U[r, s] = sum_a U[ar] U[as]
        S_U = U.T @ U
        S_V = V.T @ V
        S_W = W.T @ W
        
        # ||D||_F^2 = sum_{r,s} S_U[r,s] S_V[r,s] S_W[r,s]
        norm_sq = np.sum(S_U * S_V * S_W)
        
        # <T, D> = sum_{(a,b,c) in indices} sum_r U[ar] V[br] W[cr]
        # We can compute this efficiently?
        # sum_r (sum_{a in U_grad_terms?} No.
        # It's sum_{a,b,c in indices} (U[:, a? no U is row a)
        # Let's just use einsum for the sparse part, it's only 80 terms * R
        # Actually, we can vectorize:
        # For each term r: sum_{(a,b,c) in indices} U[a,r] V[b,r] W[c,r]
        # This is sum_r ( U[indices_a, r] * V[indices_b, r] * W[indices_c, r] ).sum()
        
        idx_a = np.array([x[0] for x in indices])
        idx_b = np.array([x[1] for x in indices])
        idx_c = np.array([x[2] for x in indices])
        
        # U_sparse: (80, R), V_sparse: (80, R), W_sparse: (80, R)
        U_sp = U[idx_a, :]
        V_sp = V[idx_b, :]
        W_sp = W[idx_c, :]
        
        inner = np.sum(U_sp * V_sp * W_sp) # Sum over all 80 and R
        
        loss = norm_sq - 2 * inner + len(indices) # len(indices) is ||T||_F^2
        
        # Gradients
        # dLoss/dU[ar] = 2 * (sum_s U[as] S_V[r,s] S_W[r,s] + U[ar] S_V[r,r] S_W[r,r]) - 2 * (sum_{(b,c) in U_grad_terms[a]} V[br] W[cr])
        
        # Term 1: sum_s U[as] (S_V * S_W)[r,s]
        # Let M_U[r,s] = S_V[r,s] * S_W[r,s]
        M_U = S_V * S_W
        # grad_norm_part_U = 2 * (U @ M_U.T + diag(M_U[:, diag]) * U)
        # Wait, sum_s U[as] M_U[r,s] is (U @ M_U.T)[a,r]?
        # (U @ M_U.T)[a,r] = sum_s U[a,s] M_U.T[s,r] = sum_s U[a,s] M_U[r,s]. Correct.
        # The diagonal term: U[ar] M_U[r,r].
        diag_M_U = np.diag(M_U)
        grad_U = 2 * (U @ M_U.T + U * diag_M_U[None, :])
        
        # Term 2: -2 * sum_{(b,c) in U_grad_terms[a]} V[br] W[cr]
        # We need to accumulate this for each a, r.
        # For each a, we have a list of (b, c).
        # grad_U[a, r] -= 2 * sum_{(b,c) in list} V[b, r] W[c, r]
        
        # Efficiently:
        # For each a, gather V[b, :] and W[c, :] for b,c in list.
        # This is small (list size <= p=5).
        for a in range(n*m):
            if U_grad_terms[a]:
                b_list = [x[0] for x in U_grad_terms[a]]
                c_list = [x[1] for x in U_grad_terms[a]]
                # V[b_list, :] is (len, R), W[c_list, :] is (len, R)
                # Product sum along axis 0
                contrib = np.sum(V[b_list, :] * W[c_list, :], axis=0)
                grad_U[a, :] -= 2 * contrib
                
        # Similarly for V
        M_V = S_U * S_W
        grad_V = 2 * (V @ M_V.T + V * np.diag(M_V)[None, :])
        for b in range(m*p):
            if V_grad_terms[b]:
                a_list = [x[0] for x in V_grad_terms[b]]
                c_list = [x[1] for x in V_grad_terms[b]]
                contrib = np.sum(U[a_list, :] * W[c_list, :], axis=0)
                grad_V[b, :] -= 2 * contrib
                
        # Similarly for W
        M_W = S_U * S_V
        grad_W = 2 * (W @ M_W.T + W * np.diag(M_W)[None, :])
        for c in range(p*n):
            if W_grad_terms[c]:
                a_list = [x[0] for x in W_grad_terms[c]]
                b_list = [x[1] for x in W_grad_terms[c]]
                contrib = np.sum(U[a_list, :] * V[b_list, :], axis=0)
                grad_W[c, :] -= 2 * contrib
                
        return loss, (grad_U, grad_V, grad_W)
        
    def verify(U, V, W, tol=1e-8):
        # Check exact decomposition
        # est[a,b,c] = sum_r U[ar] V[br] W[cr]
        # We only need to check the 80 non-zero entries are 1, and others are 0?
        # The problem says "exactly decompose".
        # We check the 80 entries. If they are 1, and the norm is close to 80, it's likely correct.
        # But to be safe, we can check a random subset of zeros or just rely on the loss.
        # Given the integer requirement, rounding and re-checking is better.
        idx_a = np.array([x[0] for x in indices])
        idx_b = np.array([x[1] for x in indices])
        idx_c = np.array([x[2] for x in indices])
        
        U_sp = U[idx_a, :]
        V_sp = V[idx_b, :]
        W_sp = W[idx_c, :]
        
        vals = np.sum(U_sp * V_sp * W_sp, axis=1)
        if not np.allclose(vals, 1.0, atol=tol):
            return False
            
        # Check that the total norm is close to 80 (implies zeros are close to 0)
        loss, _ = loss_and_grad(U, V, W)
        if loss > 1e-6:
            return False
        return True

    def try_rank(R, max_trials=50):
        if time.time() > deadline - 10:
            return None
            
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                return None
                
            np.random.seed(trial * 1000 + R)
            U = np.random.randn(n*m, R) * 0.5
            V = np.random.randn(m*p, R) * 0.5
            W = np.random.randn(p*n, R) * 0.5
            
            # Adam optimizer
            m_U = np.zeros_like(U)
            v_U = np.zeros_like(U)
            m_V = np.zeros_like(V)
            v_V = np.zeros_like(V)
            m_W = np.zeros_like(W)
            v_W = np.zeros_like(W)
            b1 = 0.9
            b2 = 0.999
            eps = 1e-8
            lr = 0.01
            
            for step in range(500):
                if time.time() > deadline - 3:
                    break
                    
                loss, (gU, gV, gW) = loss_and_grad(U, V, W)
                
                if loss < 1e-12:
                    break
                    
                t = step + 1
                m_U = b1 * m_U + (1 - b1) * gU
                v_U = b2 * v_U + (1 - b2) * (gU ** 2)
                m_U_hat = m_U / (1 - b1 ** t)
                v_U_hat = v_U / (1 - b2 ** t)
                
                m_V = b1 * m_V + (1 - b1) * gV
                v_V = b2 * v_V + (1 - b2) * (gV ** 2)
                m_V_hat = m_V / (1 - b1 ** t)
                v_V_hat = v_V / (1 - b2 ** t)
                
                m_W = b1 * m_W + (1 - b1) * gW
                v_W = b2 * v_W + (1 - b2) * (gW ** 2)
                m_W_hat = m_W / (1 - b1 ** t)
                v_W_hat = v_W / (1 - b2 ** t)
                
                U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
                V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
                W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
                
                # Periodic check
                if step % 50 == 0 and loss < 1e-8:
                    U_c = np.round(U * 2) / 2
                    V_c = np.round(V * 2) / 2
                    W_c = np.round(W * 2) / 2
                    if verify(U_c, V_c, W_c):
                        return (U_c, V_c, W_c)
                        
            # Final check
            U_c = np.round(U * 2) / 2
            V_c = np.round(V * 2) / 2
            W_c = np.round(W * 2) / 2
            if verify(U_c, V_c, W_c):
                return (U_c, V_c, W_c)
                
        return None

    # Try ranks from 60 down to 45
    for r in range(60, 44, -1):
        if time.time() > deadline - 15:
            break
        if r >= best_rank:
            continue
            
        result = try_rank(r, max_trials=30)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break
            
    return best_result
