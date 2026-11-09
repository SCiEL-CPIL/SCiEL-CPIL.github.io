import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_result = schoolbook()
    best_rank = n * m * p

    def residual(x, rank):
        # x is flattened U, V, W
        U = x[:n * m * rank].reshape(n * m, rank)
        V = x[n * m * rank:n * m * rank + m * p * rank].reshape(m * p, rank)
        W = x[n * m * rank + m * p * rank:].reshape(p * n, rank)
        
        # Compute tensor T[i', i, j, k', k] = sum_alpha U[i'm+j, alpha] V[jp+k', alpha] W[k'n+i', alpha]
        # Target is delta[i', i] * delta[k', k]
        
        # We can compute this efficiently using einsum or loops
        # T is shape (n, n, m, p, p) -> 3*3*5*6*6 = 1620
        # Let's compute it as a 5D array
        # U[i'm+j, :] shape (nm, r)
        # V[jp+k', :] shape (mp, r)
        # W[k'n+i', :] shape (pn, r)
        
        # Reshape U to (n, m, r)
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        
        # We want sum_r U[i', j, r] V[j, k', r] W[k', i', r]
        # Note: W index is k' * n + i'. So W_reshaped[k', i', r].
        
        # Let's use einsum
        # U[i', j, r]
        # V[j, k', r]
        # W[k', i', r]
        
        # Result R[i', j, k', i']? No, the output is for fixed i, j, k.
        # The condition is:
        # For all i, j, k:
        #   For all i', k':
        #     sum_r U[i'm+j, r] V[jp+k', r] W[k'n+i', r] = delta[i', i] delta[k', k]
        
        # So we compute the full tensor of products.
        # P[i', j, k', i] = sum_r U[i'm+j, r] V[jp+k', r] W[k'n+i', r]
        # Wait, U depends on i', j. V depends on j, k'. W depends on k', i'.
        # The sum is over r.
        # Let's compute S[i', j, k', i']? No, the last index of W is i'.
        # So S[i', j, k', i'] is not right.
        # S[i', j, k'] is a matrix of size (n, n)? No.
        
        # Let's just loop over i, j, k and check the n*p entries.
        # Or better, compute the full 5D array and compare.
        
        # U_reshaped: (n, m, r) -> indices i', j, r
        # V_reshaped: (m, p, r) -> indices j, k', r
        # W_reshaped: (p, n, r) -> indices k', i', r
        
        # We want to compute T[i', j, k', i']? No.
        # The term is U[i', j, r] * V[j, k', r] * W[k', i', r].
        # This sums to a value for each (i', j, k', i').
        # But the condition is for fixed (i, j, k), the result must be delta[i', i] delta[k', k].
        # So we need to check if the computed tensor matches the target tensor.
        
        # Let's compute the tensor C[i', j, k', i] where i is the "input i" from delta[i', i].
        # Wait, the input i is just an index.
        # The equation is:
        # For all i, j, k:
        #   For all i', k':
        #     Sum_r U[i'm+j, r] V[jp+k', r] W[k'n+i', r] == (1 if i'==i else 0) * (1 if k'==k else 0)
        
        # So we can compute the LHS for all i', j, k', i, k.
        # LHS[i', j, k', i, k] = Sum_r U[i'm+j, r] V[jp+k', r] W[k'n+i', r]
        # Note that LHS does not depend on i or k except through the RHS delta.
        # Wait, U depends on i' and j. V depends on j and k'. W depends on k' and i'.
        # So LHS is a function of i', j, k'.
        # Let L[i', j, k'] = Sum_r U[i'm+j, r] V[jp+k', r] W[k'n+i', r].
        # The condition is:
        # For all i, j, k:
        #   For all i', k':
        #     L[i', j, k'] == delta[i', i] * delta[k', k]
        
        # This implies that L[i', j, k'] must be 0 unless there exists i, k such that i'==i and k'==k.
        # But i and k are "existential" in the condition? No, "for all i, j, k".
        # This means for a fixed j, the matrix L[:, j, :] (of size n x p) must satisfy:
        # For all i, k:
        #   For all i', k':
        #     L[i', j, k'] == delta[i', i] delta[k', k]
        
        # This is only possible if L[:, j, :] is the zero matrix?
        # If i=0, k=0, then L[i', j, k'] must be 1 if i'=0, k'=0, else 0.
        # If i=1, k=0, then L[i', j, k'] must be 1 if i'=1, k'=0, else 0.
        # This is a contradiction unless n=1 and p=1.
        
        # I must have misinterpreted the tensor decomposition condition.
        
        # Let's re-read the tensor definition.
        # T(E_ij, F_jk) = G_ik.
        # This means the map is linear in A and B.
        # T(A, B) = sum_alpha (U_alpha . A) (V_alpha . B)^T ? No.
        
        # Standard form:
        # C_ik = sum_alpha (sum_j A_ij U_alpha_jk?) No.
        
        # Let's use the definition:
        # C = A B.
        # C_ik = sum_j A_ij B_jk.
        
        # Rank r decomposition:
        # C_ik = sum_{alpha=1}^r (sum_j A_ij u_alpha_ij?) No.
        
        # It is:
        # C_ik = sum_alpha (sum_j A_ij U_alpha_j) (sum_k B_jk V_alpha_k)? No.
        
        # Correct form:
        # C_ik = sum_alpha (A U_alpha)_i (V_alpha B)_k ?
        # (A U_alpha)_i = sum_j A_ij (U_alpha)_ji.
        # (V_alpha B)_k = sum_j (V_alpha)_jk B_jk.
        
        # So C_ik = sum_alpha (sum_j A_ij (U_alpha)_ji) (sum_j (V_alpha)_jk B_jk).
        
        # This is not a product of three independent vectors.
        # It is a sum of products of two vectors, where each vector is a linear combination of A or B.
        
        # The tensor T is in R^{nm} x R^{mp} x R^{np}.
        # The decomposition is T = sum_alpha u_alpha (in R^{nm}) x v_alpha (in R^{mp}) x w_alpha (in R^{np}).
        # This means:
        # T(A, B) = sum_alpha <u_alpha, A> <v_alpha, B> w_alpha.
        
        # Let's check this.
        # <u_alpha, A> = sum_ij (u_alpha)_ij A_ij.
        # <v_alpha, B> = sum_jk (v_alpha)_jk B_jk.
        # w_alpha is a matrix in R^{n x p}.
        
        # Then T(A, B) = sum_alpha (sum_ij u_ij A_ij) (sum_jk v_jk B_jk) w_alpha.
        
        # We want T(A, B) = A B.
        # So for all A, B:
        # AB = sum_alpha (sum_ij u_ij A_ij) (sum_jk v_jk B_jk) w_alpha.
        
        # Let's test with A = E_ab, B = E_cd.
        # AB = E_ac if b=c, else 0.
        
        # LHS: E_ac (if b=c).
        # RHS: sum_alpha (u_ab) (v_cd) w_alpha.
        
        # So we need:
        # For all a, b, c, d:
        # sum_alpha u_ab v_cd w_alpha = E_ac (if b=c) else 0.
        
        # This means:
        # If b != c, sum_alpha u_ab v_cd w_alpha = 0 (zero matrix).
        # If b = c, sum_alpha u_bc v_bd w_alpha = E_ad.
        
        # This is the condition!
        # Variables:
        # u_alpha is n x m.
        # v_alpha is m x p.
        # w_alpha is n x p.
        
        # Flattened:
        # U has shape (n*m, r). U[i*m+j, alpha] = u_alpha_ij.
        # V has shape (m*p, r). V[j*p+k, alpha] = v_alpha_jk.
        # W has shape (p*n, r). W[k*n+i, alpha] = w_alpha_ik. (Note: w_alpha is n x p, so index i, k. W index is k*n+i? No, usually row major. i*p+k? Problem says W[k*n+i]. So W is stored with k as major index? Or i as minor?
        # "W[k*n+i, r]". This implies index = k*n + i.
        # So W[k*n+i, alpha] = w_alpha_ik.
        
        # Condition:
        # For all a, b, c, d:
        # If b != c:
        #   For all i, k:
        #     sum_alpha U[a*m+b, alpha] V[c*p+d, alpha] W[d*n+i, alpha] = 0
        # If b == c:
        #   For all i, k:
        #     sum_alpha U[a*m+b, alpha] V[b*p+d, alpha] W[d*n+i, alpha] = delta_{i, a} delta_{k, d}
        
        # This is a set of equations.
        # Number of equations:
        # Case b != c: n * m * (m-1) * p * n * p = n^2 p^2 m (m-1).
        # Case b == c: n * m * p * n * p = n^2 m p^2.
        # Total: n^2 p^2 m^2.
        # For n=3, m=5, p=6:
        # 9 * 36 * 25 = 8100 equations.
        
        # Unknowns: r * (nm + mp + pn) = r * (15 + 30 + 18) = 63 r.
        
        # We can solve this system.
        
        def residual_vec(x, rank):
            U = x[:n * m * rank].reshape(n * m, rank)
            V = x[n * m * rank:n * m * rank + m * p * rank].reshape(m * p, rank)
            W = x[n * m * rank + m * p * rank:].reshape(p * n, rank)
            
            res = []
            # b != c
            for a in range(n):
                for b in range(m):
                    for c in range(m):
                        if b == c: continue
                        for d in range(p):
                            # Sum over alpha of U[am+b, alpha] V[cp+d, alpha] W[dn+i, alpha]
                            # This sum must be 0 for all i.
                            # We can compute the vector S_i = sum_alpha ... W[dn+i, alpha]
                            # S is a vector of length n.
                            # Actually, for fixed a, b, c, d, the term U[am+b, :] V[cp+d, :] is a vector of length r.
                            # Let Y = U[am+b, :] * V[cp+d, :] (element-wise product).
                            # Then the sum is W^T Y. W is (pn, r).
                            # We need W^T Y to be zero for all i?
                            # W[dn+i, alpha] is the i-th component of the d-th block of W.
                            # So we need the d-th block of W^T Y to be zero.
                            # W is stored as W[k*n+i, alpha].
                            # So W[d*n + :, :] is the block for d.
                            # Block = W[d*n:(d+1)*n, :].
                            # We need Block.T @ Y == 0.
                            
                            Y = U[a * m + b, :] * V[c * p + d, :]
                            block = W[d * n:(d + 1) * n, :]
                            res.extend((block.T @ Y).tolist())
                            
            # b == c
            for a in range(n):
                for b in range(m):
                    for d in range(p):
                        # Sum over alpha of U[am+b, alpha] V[bp+d, alpha] W[dn+i, alpha]
                        # Must equal delta_{i, a} delta_{k, d}?
                        # Wait, the result is E_ad.
                        # E_ad has 1 at (a, d) and 0 elsewhere.
                        # So for i=a, k=d, sum = 1.
                        # For i!=a or k!=d, sum = 0.
                        # But k is the column index of the result matrix.
                        # The result is a matrix.
                        # The sum gives the matrix M where M[i, k] = sum_alpha ...
                        # Wait, W[dn+i, alpha] is w_alpha_ik?
                        # Problem says W[k*n+i]. So W[d*n+i] is w_alpha_{d, i}?
                        # No, w_alpha is n x p. Indices i (row), k (col).
                        # If W[k*n+i], then index is k*n+i.
                        # So W[d*n+i] corresponds to k=d, row=i.
                        # So the matrix is formed by columns d.
                        # M[i, d] = sum_alpha U[am+b, alpha] V[bp+d, alpha] W[d*n+i, alpha].
                        # We want M = E_ad.
                        # So M[a, d] = 1.
                        # M[i, d] = 0 for i != a.
                        # And for k != d, M[i, k] = 0.
                        # But the sum only involves V[bp+d] and W[d*n+i].
                        # So the sum is only defined for column d.
                        # What about other columns k?
                        # The term for column k would involve V[bp+k] and W[k*n+i].
                        # So for fixed a, b, the matrix M has entries M[i, k] = sum_alpha U[am+b, alpha] V[bp+k, alpha] W[k*n+i, alpha].
                        # We want M = E_ad? No, for fixed d, we are looking at the d-th column.
                        # The condition is for ALL d.
                        # So for each d, the d-th column of M must be e_a (1 at a, 0 elsewhere).
                        # And for k != d, the k-th column must be 0.
                        
                        # So for fixed a, b:
                        # For all d:
                        #   For all i:
                        #     sum_alpha U[am+b, alpha] V[bp+d, alpha] W[d*n+i, alpha] = delta_{i, a}
                        # For all k != d:
                        #   For all i:
                        #     sum_alpha U[am+b, alpha] V[bp+k, alpha] W[k*n+i, alpha] = 0
                        
                        # This is equivalent to:
                        # For all k:
                        #   For all i:
                        #     sum_alpha U[am+b, alpha] V[bp+k, alpha] W[k*n+i, alpha] = delta_{i, a} delta_{k, d}
                        # But d is the specific column we are processing.
                        # Actually, the condition "b=c" implies that the input B has a non-zero at (b, k).
                        # So for a fixed input B = E_bk, the output column k must be e_a, and other columns 0.
                        
                        # So we just need to check:
                        # For all k:
                        #   Col_k = sum_alpha U[am+b, alpha] V[bp+k, alpha] W[k*n+:, alpha]
                        #   Col_k[i] = delta_{i, a} if k==d? No.
                        #   If B = E_bk, then C = A E_bk.
                        #   (A E_bk)[:, k] = A[:, b].
                        #   So C[:, k] = A[:, b].
                        #   If A = E_ab, then A[:, b] = e_a.
                        #   So C[:, k] = e_a.
                        #   And C[:, j] = 0 for j != k.
                        
                        # So for fixed a, b:
                        # For all k:
                        #   For all i:
                        #     sum_alpha U[am+b, alpha] V[bp+k, alpha] W[k*n+i, alpha] = delta_{i, a}
                        
                        # This is n * p equations for each a, b.
                        
                        Y = U[a * m + b, :]
                        for k in range(p):
                            Vk = V[b * p + k, :]
                            Wk = W[k * n:(k + 1) * n, :]
                            col = Wk.T @ (Y * Vk)
                            # col[i] should be 1 if i==a else 0
                            target = [1.0 if i == a else 0.0 for i in range(n)]
                            res.extend((col - np.array(target)).tolist())
                            
            return np.array(res)

    # Try to solve for lower ranks
    for rank in range(85, 29, -5):
        if time.time() > deadline - 10:
            break
            
        # Random initialization
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
                
            x0 = np.random.randn(n * m * rank + m * p * rank + p * n * rank) * 0.1
            try:
                res = least_squares(residual_vec, x0, args=(rank,), method='lm', max_nfev=1000)
                if res.cost < 1e-10:
                    # Found a solution
                    U = res.x[:n * m * rank].reshape(n * m, rank)
                    V = res.x[n * m * rank:n * m * rank + m * p * rank].reshape(m * p, rank)
                    W = res.x[n * m * rank + m * p * rank:].reshape(p * n, rank)
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = rank
                    break # Try next rank
            except:
                pass
                
    return best_result
