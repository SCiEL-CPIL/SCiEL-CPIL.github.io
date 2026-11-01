import numpy as np
import time
import itertools
from collections import defaultdict

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Construct the known Rank-11 solution (Strassen for 2x2 + Schoolbook for 2x3)
    # This serves as the baseline best solution.
    # Strassen for 2x2 multiplication has rank 7.
    # The 2x3 multiplication can be seen as 2x2 * 2x1 (first col) + 2x2 * 2x1 (second col) + 2x2 * 2x1 (third col)?
    # Actually, simpler: 
    # A (2x2) * B (2x3)
    # We can use the rank-7 Strassen decomposition for the 2x2 part of B (cols 0,1) 
    # and add 3 terms for the 3rd column? No, that's not how it works directly.
    
    # Let's use the known Rank-11 construction:
    # The tensor T is defined by T[i, j, k] = 1 if A[i,j] and B[j,k] multiply to C[i,k].
    # Indices:
    # A: 2x2 -> 4 elements (0..3)
    # B: 2x3 -> 6 elements (0..5)
    # C: 3x2 -> 6 elements (0..5)
    # Note: W index is k*n + i. n=2. So W[k*2 + i].
    # C is 3x2. C[k, i].
    # Target Tensor T[a, b, c] where:
    # a = i*m + j (i in 0..1, j in 0..1) -> 0,1,2,3
    # b = j*p + k (j in 0..1, k in 0..2) -> 0,1,2,3,4,5
    # c = k*n + i (k in 0..2, i in 0..1) -> 0,1,2,3,4,5
    
    def get_target_tensor():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T

    T_target = get_target_tensor()
    
    def compute_tensor(U, V, W):
        return np.einsum('ir,jr,kr->ijk', U, V, W)

    def verify(U, V, W):
        T = compute_tensor(U, V, W)
        return np.allclose(T, T_target, atol=1e-8)

    # Build Rank 11 Solution
    # Strategy: Use Strassen (rank 7) for the 2x2 block (cols 0,1 of B) and Schoolbook (rank 4) for col 2?
    # Wait, Strassen multiplies 2x2 by 2x2.
    # Let B = [B1 | B2] where B1 is 2x2 (cols 0,1) and B2 is 2x1 (col 2).
    # C = A * B1 + A * B2.
    # If we use Strassen for A*B1 (rank 7) and Schoolbook for A*B2 (rank 4? No, 2x2 * 2x1 is rank 4 schoolbook, but maybe less?).
    # Actually, 2x2 * 2x1 multiplication tensor has rank 4?
    # Let's just use the standard Schoolbook for everything to get Rank 12, then try to improve.
    # But the prompt says "Combining Strassen (rank 7) for 2x2 block and schoolbook (rank 4) for remaining column achieves rank 11".
    # Let's construct Rank 11 explicitly.
    
    U11 = np.zeros((n*m, 11))
    V11 = np.zeros((m*p, 11))
    W11 = np.zeros((p*n, 11))
    
    # Strassen for 2x2 * 2x2 (indices for A: 0..3, B: 0..3 (cols 0,1), C: 0..3 (cols 0,1))
    # Strassen formulas:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Map to our indices:
    # A: i=0,j=0 (0), i=0,j=1 (1), i=1,j=0 (2), i=1,j=1 (3)
    # B (cols 0,1): j=0,k=0 (0), j=0,k=1 (1), j=1,k=0 (2), j=1,k=1 (3)
    # C (cols 0,1): k=0,i=0 (0), k=0,i=1 (1), k=1,i=0 (2), k=1,i=1 (3)
    
    strassen_defs = [
        # (U_indices, coeffs), (V_indices, coeffs), (W_indices, coeffs)
        # M1
        ([0, 3], [1, 1]), ([0, 3], [1, 1]), ([0, 3], [1, 1]),
        # M2
        ([2, 3], [1, 1]), ([0], [1]), ([1], [1]),
        # M3
        ([0], [1]), ([1, 3], [1, -1]), ([1], [1]),
        # M4
        ([3], [1]), ([2, 0], [1, -1]), ([2], [1]),
        # M5
        ([0, 1], [1, 1]), ([3], [1]), ([3], [1]),
        # M6
        ([2, 0], [1, -1]), ([0, 1], [1, 1]), ([3], [1]),
        # M7
        ([1, 3], [1, -1]), ([2, 3], [1, 1]), ([0], [1]),
    ]
    
    # Wait, the W coefficients in Strassen are not just 1/-1 in the factor matrices, they are in the reconstruction.
    # In tensor decomposition, we need T = sum r_u * v_u * w_u.
    # So for M1, U has 1s at 0,3. V has 1s at 0,3. W must produce C00 and C11 contributions.
    # C00 gets +M1, C11 gets +M1.
    # C indices: C00 is idx 0, C11 is idx 3.
    # So W for M1 has 1 at 0 and 1 at 3.
    
    # Let's re-verify Strassen W indices:
    # C00 = M1 + M4 - M5 + M7 -> W indices 0
    # C01 = M3 + M5 -> W indices 1
    # C10 = M2 + M4 -> W indices 2
    # C11 = M1 - M2 + M3 + M6 -> W indices 3
    
    # M1: U(0,3), V(0,3), W(0,3) with coeff 1
    # M2: U(2,3), V(0), W(2) with coeff 1, and W(3) with coeff -1?
    # No, the decomposition is a sum of rank-1 tensors.
    # T = sum_{r} U[:,r] V[:,r] W[:,r]
    # This means for each r, the outer product U[:,r] V[:,r] W[:,r] is added.
    # So if M2 contributes to C10 with +1 and C11 with -1, we can't do it in one rank-1 term unless W has both.
    # But W is a vector. If W[2]=1 and W[3]=-1, then the term adds U*V*W[2] to C10 and U*V*W[3] to C11.
    # Yes, that works.
    
    # Let's define the 7 Strassen terms for the 2x2 block (cols 0,1 of B)
    # A indices: 0,1,2,3
    # B indices (cols 0,1): 0,1,2,3
    # C indices (cols 0,1): 0,1,2,3
    
    s_terms = [
        # M1
        {0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1},
        # M2
        {2:1, 3:1}, {0:1}, {2:1, 3:-1},
        # M3
        {0:1}, {1:1, 3:-1}, {1:1},
        # M4
        {3:1}, {2:1, 0:-1}, {2:1},
        # M5
        {0:1, 1:1}, {3:1}, {3:1},
        # M6
        {2:1, 0:-1}, {0:1, 1:1}, {3:1},
        # M7
        {1:1, 3:-1}, {2:1, 3:1}, {0:1},
    ]
    
    # Fill U11, V11, W11 for first 7 columns
    for r in range(7):
        u_dict, v_dict, w_dict = s_terms[r]
        for idx, val in u_dict.items():
            U11[idx, r] = val
        for idx, val in v_dict.items():
            V11[idx, r] = val
        for idx, val in w_dict.items():
            W11[idx, r] = val
            
    # Now handle the 3rd column of B (index 4,5 in B, indices 4,5 in C)
    # A (2x2) * B_col2 (2x1) -> C_col2 (2x1)
    # This is a 2x2 * 2x1 multiplication.
    # Schoolbook rank for 2x2 * 2x1 is 4?
    # A00*B0 -> C0
    # A01*B0 -> C1
    # A10*B0 -> C0
    # A11*B0 -> C1
    # Wait, B is 2x3. Col 2 has elements B02 (idx 4) and B12 (idx 5).
    # C col 2 has elements C02 (idx 4) and C12 (idx 5).
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12
    
    # Terms:
    # 1. A00 * B02 -> C02
    # 2. A01 * B12 -> C02
    # 3. A10 * B02 -> C12
    # 4. A11 * B12 -> C12
    
    # Can we do it in less than 4?
    # Tensor for 2x2 * 2x1.
    # It is known that the rank of 2x2 * 2x1 is 4.
    # So we add 4 terms.
    
    extra_terms = [
        {0:1}, {4:1}, {4:1},
        {1:1}, {5:1}, {4:1},
        {2:1}, {4:1}, {5:1},
        {3:1}, {5:1}, {5:1},
    ]
    
    for r in range(4):
        u_dict, v_dict, w_dict = extra_terms[r]
        for idx, val in u_dict.items():
            U11[idx, r+7] = val
        for idx, val in v_dict.items():
            V11[idx, r+7] = val
        for idx, val in w_dict.items():
            W11[idx, r+7] = val
            
    if verify(U11, V11, W11):
        best_U, best_V, best_W = U11.tolist(), V11.tolist(), W11.tolist()
        best_rank = 11
    else:
        # Fallback to Schoolbook Rank 12
        best_rank = 12
        U12 = np.zeros((n*m, 12))
        V12 = np.zeros((m*p, 12))
        W12 = np.zeros((p*n, 12))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U12[i * m + j, col] = 1
                    V12[j * p + k, col] = 1
                    W12[k * n + i, col] = 1
                    col += 1
        best_U, best_V, best_W = U12.tolist(), V12.tolist(), W12.tolist()

    # 2. Search for Rank 10
    # We will try to find a rank-10 integer decomposition.
    # Strategy: 
    # The support of a rank-r decomposition corresponds to r triples (u, v, w).
    # For small ranks, we can search over the "support patterns".
    # However, the space is huge.
    # Alternative: Use the known rank-11 solution and try to merge terms or find a different structure.
    # Since we have time, let's try a randomized search for rank 10 with integer coefficients in {-1, 0, 1}.
    # This is a "randomized algebraic" approach.
    
    found_rank_10 = False
    
    # Heuristic: 
    # If rank 10 exists, it likely involves a mix of Strassen-like structures and cancellations.
    # Let's try random integer vectors with entries in {-1, 0, 1} and check.
    # To make it more effective, we can fix some structure.
    # But pure random is very unlikely to hit rank 10.
    
    # Better approach: 
    # Use the rank-11 solution. Try to express one of the terms as a linear combination of the others?
    # Or, use numerical optimization to find a rank-10 solution, then round.
    # The current program does this but with L-BFGS. 
    # Let's try a different optimization: 
    # Minimize the rank-10 approximation error using a gradient-free method or different initializations.
    
    # Actually, let's try a specific known structure for 2x2 * 2x3.
    # Is there a known rank-10 decomposition?
    # The lower bound is max(n*m, m*p, p*n) = max(4, 6, 6) = 6.
    # Upper bound is 12.
    # Strassen gives 11.
    # It is an open question if 10 is possible? Or maybe it is known?
    # If 10 is not possible, we return 11.
    
    # Let's try a quick numerical search for rank 10.
    # We will use a simplified version of the current program's logic but with a different optimizer: 
    # Coordinate Descent or Nelder-Mead on the flattened parameters, with random restarts.
    
    def try_find_rank_10():
        nm, mp, pn = n*m, m*p, p*n
        target_rank = 10
        total_params = (nm + mp + pn) * target_rank
        
        # Random initialization with integers {-1, 0, 1}
        # This ensures we stay in the integer domain if we just check.
        # But we need to search for a solution.
        
        # Let's use a "pattern search" over a small set of structured matrices.
        # Given the time limit, we will do a few random restarts of a local search.
        
        # We'll use the rank-11 solution as a starting point for rank 10 by zeroing out one column?
        # No, that will increase error.
        
        # Let's try to find a rank-10 solution using scipy.optimize.minimize with Nelder-Mead.
        # It is robust to non-smoothness if we use a large enough step.
        
        def objective(params):
            U = params[:nm*target_rank].reshape(nm, target_rank)
            V = params[nm*target_rank:nm*target_rank + mp*target_rank].reshape(mp, target_rank)
            W = params[nm*target_rank + mp*target_rank:].reshape(pn, target_rank)
            T_prime = compute_tensor(U, V, W)
            diff = T_prime - T_target
            return np.sum(diff**2)
        
        # Initial guess: Rank 11 solution, with one column set to 0 (and maybe scaled others?)
        # Or just random.
        
        attempts = 50
        for i in range(attempts):
            if time.time() > deadline:
                break
                
            # Random integer initialization
            params0 = np.random.randint(-1, 2, size=total_params)
            
            # Try Nelder-Mead
            try:
                res = minimize(objective, params0, method='Nelder-Mead', 
                               options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
                if res.fun < 1e-8:
                    U_num = res.x[:nm*target_rank].reshape(nm, target_rank)
                    V_num = res.x[nm*target_rank:nm*target_rank + mp*target_rank].reshape(mp, target_rank)
                    W_num = res.x[nm*target_rank + mp*target_rank:].reshape(pn, target_rank)
                    
                    # Check if it's close to integers
                    U_int = np.round(U_num)
                    V_int = np.round(V_num)
                    W_int = np.round(W_num)
                    
                    if verify(U_int, V_int, W_int):
                        return U_int, V_int, W_int, True
            except Exception:
                pass
        return None, None, None, False

    U10, V10, W10, ok = try_find_rank_10()
    if ok:
        best_U, best_V, best_W = U10.tolist(), V10.tolist(), W10.tolist()
        best_rank = 10

    return {"U": best_U, "V": best_V, "W": best_W}

# Import minimize at top level or inside?
# The current program imports it inside. I'll import it at the top to be safe, or inside construct.
# The prompt says "Keep the function names and signatures of the seed program."
# The seed program imports scipy.optimize.minimize inside construct. I will do the same.

# Re-define construct to include imports inside, matching seed style if needed, 
# but standard practice is top-level. The seed has `from scipy.optimize import minimize` at top.
# I will keep it at top.

from scipy.optimize import minimize
