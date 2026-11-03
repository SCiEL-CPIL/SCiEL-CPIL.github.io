"""Tensor decomposition for 2x3x9 matrix multiplication.
Strategy:
1. Construct schoolbook (rank 54) instantly.
2. Attempt to find a lower-rank decomposition using a structured Kronecker product
   approach: decompose M_{2,3,9} as M_{2,1,3} \otimes M_{1,3,3} or similar.
   Specifically, we look for a decomposition of M_{2,3,3} with rank r, then
   M_{2,3,9} can have rank r * 3 if we can factor the 9 dimension as 3*3?
   Actually, M_{n,m,p} \otimes M_{n',m',p'} has rank R(n,m,p)R(n',m',p').
   M_{2,3,9} is NOT a Kronecker product of two smaller multiplication tensors in a simple way
   unless dimensions match.
   
   However, we can use the fact that M_{2,3,9} can be viewed as three copies of M_{2,3,3}
   if we block B into three 3x3 blocks? No, that's just schoolbook sum of ranks.
   
   Let's try a direct optimization approach but with a smarter initialization and
   a different optimization method: Alternating Least Squares (ALS) on the tensor
   completion problem, or simply a more robust gradient descent.
   
   Actually, the most promising "structural" change suggested by the stagnation is
   to try a different representation.
   
   Let's try to construct a rank 27 solution.
   Lower bound is 27.
   Is R(2,3,9) = 27?
   If we can find a rank 9 decomposition for M_{2,3,3}, then we could potentially
   use it? But M_{2,3,9} is not M_{2,3,3} \otimes M_{1,1,3}.
   M_{1,1,3} has rank 3.
   M_{2,3,3} \otimes M_{1,1,3} would be M_{2*1, 3*1, 3*3} = M_{2,3,9}.
   YES! The Kronecker product of M_{2,3,3} and M_{1,1,3} is M_{2,3,9}.
   
   Proof:
   (M_{n,m,p} \otimes M_{n',m',p'})_{i n' + i', j m' + j', k p' + k'} = M_{n,m,p}[i,j,k] M_{n',m',p'}[i',j',k']
   The multiplication tensor M_{n,m,p} has entries 1 for all valid indices?
   Wait, the standard definition is that the tensor M has M[i,j,k]=1 for all i,j,k.
   Let's check the Kronecker product of two "all ones" tensors (which represent the multiplication structure in the flattening sense).
   Actually, the matrix multiplication tensor is usually defined as the structural tensor.
   If T1 is the structural tensor for M_{2,3,3} and T2 is for M_{1,1,3},
   then T1 \otimes T2 is the structural tensor for M_{2,3,9}.
   
   Rank(T1 \otimes T2) <= Rank(T1) * Rank(T2).
   Rank(T2) = Rank(M_{1,1,3}) = 1*1*3 = 3. (Schoolbook is optimal for 1x1x3).
   So if we can find Rank(M_{2,3,3}) = 9, then Rank(M_{2,3,9}) <= 9 * 3 = 27.
   Since the lower bound for M_{2,3,9} is max(2*3, 3*9, 9*2) = max(6, 27, 18) = 27,
   if Rank(M_{2,3,3}) = 9, then Rank(M_{2,3,9}) = 27.
   
   Is Rank(M_{2,3,3}) = 9?
   Lower bound for M_{2,3,3} is max(6, 9, 6) = 9.
   Schoolbook is 18.
   If we can find a rank 9 decomposition for M_{2,3,3}, we win big.
   
   How to find a rank 9 decomposition for M_{2,3,3}?
   This is equivalent to finding a rank 9 decomposition for the 2x3 matrix multiplication.
   This is a known problem.
   Actually, R(2,3,3) is not known to be 9. It is suspected to be higher.
   However, we can try to optimize for it.
   
   Let's implement a search for a rank 9 decomposition of M_{2,3,3}.
   If found, we construct the Kronecker product with M_{1,1,3} (which is just 3 identity-like columns) to get M_{2,3,9} with rank 27.
   
   If rank 9 for M_{2,3,3} is not found, we fall back to schoolbook 54.
   
   To find rank 9 for M_{2,3,3}:
   We use scipy.optimize.minimize on the flattened factors.
   
   Let's refine the Kronecker construction:
   Let U233, V233, W233 be rank r factors for M_{2,3,3}.
   Let U113, V113, W113 be rank 3 factors for M_{1,1,3}.
   U113 is 1x3, V113 is 1x3, W113 is 3x3?
   M_{1,1,3}: A is 1x1, B is 1x3, C is 1x3.
   C = A B. C_k = A_0 B_k.
   Tensor T[i,j,k] where i in 0..0, j in 0..0, k in 0..2.
   T[0,0,k] = 1.
   Factors:
   U (1x3): U[0, c]
   V (1x3): V[0, c]
   W (3x3): W[k, c]
   sum_c U[0,c] V[0,c] W[k,c] = 1 for all k.
   Simple solution:
   c=0: U=1, V=1, W[0,0]=1
   c=1: U=1, V=1, W[1,1]=1
   c=2: U=1, V=1, W[2,2]=1
   So U113 = [[1,1,1]], V113 = [[1,1,1]], W113 = I_3.
   
   Now, Kronecker product of factors:
   U_239 = U233 \otimes U113
   V_239 = V233 \otimes V113
   W_239 = W233 \otimes W113
   
   Dimensions:
   U233: 6x9 (if rank 9)
   U113: 1x3
   U239: 6x27. Correct (n*m = 6).
   
   V233: 9x9
   V113: 1x3
   V239: 9x27. Correct (m*p = 9).
   
   W233: 6x9
   W113: 3x3
   W239: 18x27. Correct (p*n = 18).
   
   So if we find a rank 9 decomposition for M_{2,3,3}, we get a rank 27 decomposition for M_{2,3,9}.
"""

DIMENSIONS = [2, 3, 9]

import numpy as np
import time
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # 1. Schoolbook Fallback (Rank 54)
    rank_school = n * m * p
    U_s = np.zeros((n * m, rank_school))
    V_s = np.zeros((m * p, rank_school))
    W_s = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j, col] = 1
                V_s[j * p + k, col] = 1
                W_s[k * n + i, col] = 1
                col += 1
    
    def _verify_decomp(U, V, W, target_tensor):
        D = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(D, target_tensor, atol=1e-8)

    # 2. Attempt to find Rank 27 via Kronecker Product
    # Step 2a: Find Rank 9 decomposition for M_{2,3,3}
    n2, m2, p2 = 2, 3, 3
    T233 = np.zeros((n2 * m2, m2 * p2, p2 * n2))
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                T233[i * m2 + j, j * p2 + k, k * n2 + i] = 1.0

    rank_target = 9
    
    def _compute_error_233(params, rank):
        size_U = n2 * m2 * rank
        size_V = m2 * p2 * rank
        size_W = p2 * n2 * rank
        
        U = params[0:size_U].reshape(n2 * m2, rank)
        V = params[size_U:size_U + size_V].reshape(m2 * p2, rank)
        W = params[size_U + size_V:size_U + size_V + size_W].reshape(p2 * n2, rank)
        
        D = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        err = D - T233
        return np.sum(err * err)

    def _get_matrices_233(params, rank):
        size_U = n2 * m2 * rank
        size_V = m2 * p2 * rank
        size_W = p2 * n2 * rank
        U = params[0:size_U].reshape(n2 * m2, rank)
        V = params[size_U:size_U + size_V].reshape(m2 * p2, rank)
        W = params[size_U + size_V:size_U + size_V + size_W].reshape(p2 * n2, rank)
        return U, V, W

    found_233 = None
    best_err_233 = np.inf
    
    # Try to optimize for rank 9 M_{2,3,3}
    # Use a few random restarts
    for restart in range(10):
        if time.time() > deadline - 20:
            break
            
        np.random.seed(123 + restart * 10)
        U_init = np.random.randn(n2 * m2, rank_target) * 0.5
        V_init = np.random.randn(m2 * p2, rank_target) * 0.5
        W_init = np.random.randn(p2 * n2, rank_target) * 0.5
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            res = minimize(
                _compute_error_233, 
                params_init, 
                args=(rank_target,), 
                method='L-BFGS-B', 
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 10000}
            )
            
            if res.fun < best_err_233:
                best_err_233 = res.fun
                if best_err_233 < 1e-8:
                    U233, V233, W233 = _get_matrices_233(res.x, rank_target)
                    # Round to integers/half-integers if possible?
                    # For now, keep floats, but verify.
                    if _verify_decomp(U233, V233, W233, T233):
                        found_233 = (U233, V233, W233)
                        break
        except Exception:
            continue

    if found_233 is not None:
        U233, V233, W233 = found_233
        
        # Step 2b: Construct M_{1,1,3} factors (Rank 3)
        U113 = np.ones((1, 3))
        V113 = np.ones((1, 3))
        W113 = np.eye(3)
        
        # Step 2c: Kronecker Product
        U_final = np.kron(U233, U113)
        V_final = np.kron(V233, V113)
        W_final = np.kron(W233, W113)
        
        # Verify the final 2x3x9 decomposition
        T_final = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_final[i * m + j, j * p + k, k * n + i] = 1.0
        
        if _verify_decomp(U_final, V_final, W_final, T_final):
            # Round to half-integers for cleanliness if close
            U_rounded = np.round(U_final * 2) / 2
            V_rounded = np.round(V_final * 2) / 2
            W_rounded = np.round(W_final * 2) / 2
            
            if _verify_decomp(U_rounded, V_rounded, W_rounded, T_final):
                return {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
            else:
                return {
                    "U": U_final.tolist(),
                    "V": V_final.tolist(),
                    "W": W_final.tolist()
                }

    # 3. Fallback: Return Schoolbook
    return {
        "U": U_s.tolist(),
        "V": V_s.tolist(),
        "W": W_s.tolist()
    }

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
