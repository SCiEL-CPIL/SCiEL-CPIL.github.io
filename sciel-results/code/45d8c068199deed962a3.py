"""
Rewritten program for Matrix Multiplication Tensor Decomposition (3,3,6).
Aim: Find a rank-minimal exact tensor decomposition.
Target rank: < 54 (Schoolbook). Likely target: 46 (2 * Rank(3x3x3)) or lower.
Strategy: Pruning the Schoolbook decomposition (Rank 54) to Rank 46 by optimizing the remaining terms.
"""
import numpy as np
import time
from fractions import Fraction
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    n, m, p = 3, 3, 6
    
    # Construct the target tensor T of shape (n*m, m*p, p*n)
    dim_u = n * m  # 9
    dim_v = m * p  # 18
    dim_w = p * n  # 18
    
    T_target = np.zeros((dim_u, dim_v, dim_w))
    
    # Schoolbook indices
    schoolbook_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                schoolbook_indices.append((a, b, c))
                
    rank_sb = len(schoolbook_indices) # 54

    def verify(U, V, W, tol=1e-8):
        try:
            T_recon = np.einsum('at,bt,ct->abc', U, V, W)
            return np.allclose(T_recon, T_target, atol=tol)
        except Exception:
            return False

    def solve_pruned(target_rank, num_trials=50):
        """
        Attempts to find a decomposition of target_rank by starting from Schoolbook
        and zeroing out (54 - target_rank) terms, then optimizing the rest.
        """
        if target_rank >= rank_sb:
            # Just return schoolbook
            U = np.zeros((dim_u, rank_sb))
            V = np.zeros((dim_v, rank_sb))
            W = np.zeros((dim_w, rank_sb))
            for idx, (a, b, c) in enumerate(schoolbook_indices):
                U[a, idx] = 1.0
                V[b, idx] = 1.0
                W[c, idx] = 1.0
            # Truncate to target_rank if needed (though we assume target_rank <= 54)
            if target_rank < rank_sb:
                 # This logic is flawed if we just truncate, we must re-optimize.
                 pass
            
        from scipy.optimize import minimize
        
        # We want to keep 'target_rank' terms and zero out the rest.
        # We will randomly select which terms to keep.
        
        best_solution = None
        
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(42 + trial * 13)
            
            # Select indices to keep
            keep_indices = np.random.choice(rank_sb, target_rank, replace=False)
            
            # Initialize U, V, W with Schoolbook values for kept indices, 0 for others
            U = np.zeros((dim_u, target_rank))
            V = np.zeros((dim_v, target_rank))
            W = np.zeros((dim_w, target_rank))
            
            # Map kept indices to columns in the new matrices
            # We re-index the kept terms to 0..target_rank-1
            for new_idx, old_idx in enumerate(keep_indices):
                a, b, c = schoolbook_indices[old_idx]
                U[a, new_idx] = 1.0
                V[b, new_idx] = 1.0
                W[c, new_idx] = 1.0
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective_and_grad(x):
                U = x[:dim_u*target_rank].reshape(dim_u, target_rank)
                V = x[dim_u*target_rank : dim_u*target_rank + dim_v*target_rank].reshape(dim_v, target_rank)
                W = x[dim_u*target_rank + dim_v*target_rank : ].reshape(dim_w, target_rank)
                
                T_recon = np.einsum('at,bt,ct->abc', U, V, W)
                error = T_target - T_recon
                loss = np.sum(error**2)
                
                dU = 2 * np.einsum('abc,bt,ct->at', error, V, W)
                dV = 2 * np.einsum('abc,at,ct->bt', error, U, W)
                dW = 2 * np.einsum('abc,at,bt->ct', error, U, V)
                
                grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
                return loss, grad
            
            def objective(x):
                return objective_and_grad(x)[0]
            
            def grad(x):
                return objective_and_grad(x)[1]
            
            try:
                res = minimize(
                    objective, 
                    x0, 
                    jac=grad, 
                    method='L-BFGS-B', 
                    options={'maxiter': 100, 'gtol': 1e-12, 'maxfun': 1000}
                )
            except Exception:
                continue
            
            if time.time() > deadline:
                break
                
            U_final = res.x[:dim_u*target_rank].reshape(dim_u, target_rank)
            V_final = res.x[dim_u*target_rank : dim_u*target_rank + dim_v*target_rank].reshape(dim_v, target_rank)
            W_final = res.x[dim_u*target_rank + dim_v*target_rank : ].reshape(dim_w, target_rank)
            
            if verify(U_final, V_final, W_final, tol=1e-7):
                # Snap to integers if possible
                U_int = np.round(U_final * 2) / 2.0
                V_int = np.round(V_final * 2) / 2.0
                W_int = np.round(W_final * 2) / 2.0
                
                if verify(U_int, V_int, W_int, tol=1e-6):
                    return U_int, V_int, W_int, target_rank
                
                U_q = np.round(U_final * 4) / 4.0
                V_q = np.round(V_final * 4) / 4.0
                W_q = np.round(W_final * 4) / 4.0
                if verify(U_q, V_q, W_q, tol=1e-6):
                    return U_q, V_q, W_q, target_rank

                return U_final, V_final, W_final, target_rank

        return None

    # Fallback schoolbook
    def schoolbook():
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1.0
                    V[b, col] = 1.0
                    W[c, col] = 1.0
                    col += 1
        return U, V, W, rank

    U_sb, V_sb, W_sb, r_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, r_sb

    # Try to find rank 46
    # 46 is the known upper bound (2 * 23)
    res = solve_pruned(46, num_trials=20)
    if res is not None:
        best_U, best_V, best_W, best_rank = res

    # If 46 fails, try 48, 50?
    if best_rank == 54 and time.time() < deadline - 10:
        res = solve_pruned(48, num_trials=10)
        if res is not None:
            best_U, best_V, best_W, best_rank = res

    if best_rank == 54 and time.time() < deadline - 5:
        res = solve_pruned(50, num_trials=5)
        if res is not None:
            best_U, best_V, best_W, best_rank = res

    # Convert to lists
    def to_list(mat):
        res = []
        for row in mat:
            r = []
            for v in row:
                if not np.isfinite(v):
                    r.append(0.0)
                    continue
                try:
                    f = Fraction(v).limit_denominator(10)
                    if abs(f - v) < 1e-6:
                        if f.denominator == 1:
                            r.append(int(f.numerator))
                        else:
                            r.append(f.numerator / f.denominator)
                    else:
                        r.append(float(v))
                except (ValueError, OverflowError):
                    r.append(float(v))
            res.append(r)
        return res

    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }

# EVOLVE-BLOCK-END
