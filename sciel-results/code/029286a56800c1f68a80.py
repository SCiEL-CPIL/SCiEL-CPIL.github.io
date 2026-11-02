import time
import numpy as np
from fractions import Fraction
import sys

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    # Safety margin: The evaluator kills at 150s. We should aim to finish by 140s.
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    def fast_residual(params, R):
        """
        Compute the residuals for a given parameter vector and rank R.
        Returns the flattened residual vector.
        """
        U = params[:6 * R].reshape(6, R)
        V = params[6 * R:6 * R + 21 * R].reshape(21, R)
        W = params[6 * R + 21 * R:].reshape(14, R)
        
        # Pre-compute sums for efficiency? 
        # pred[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This is a 3-way tensor contraction.
        # We can use einsum for speed.
        # U shape: (6, R) -> indices (ij, r)
        # V shape: (21, R) -> indices (jk, r)
        # W shape: (14, R) -> indices (ki, r)
        
        # Reshape to 3D for easier einsum?
        # U_ijr, V_jkr, W_kir
        # Sum over r: U_ijr * V_jkr * W_kir
        
        # Let's just use loops for simplicity and control, or einsum.
        # Einsum:
        # U: (n*m, R)
        # V: (m*p, R)
        # W: (p*n, R)
        
        # Create full tensors? No, too memory heavy for large R, but R is small (max 42).
        # 6*42 = 252 floats. Small.
        
        # pred = np.einsum('ir,jr,kr->ijk', 
        #                  U.reshape(n, m, R), 
        #                  V.reshape(m, p, R), 
        #                  W.reshape(p, n, R))
        # Wait, indices:
        # U[i, j, r]
        # V[j, k, r]
        # W[k, i, r]
        # Result C[i, j, k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        
        pred = np.einsum('ijr,jkr,kr->ijk', U_3d, V_3d, W_3d)
        # Note: W is indexed by k, i. In W_3d, shape is (p, n, R). 
        # W_3d[k, i, r].
        # So einsum 'ijr,jkr,kr->ijk' is correct?
        # U: i, j, r
        # V: j, k, r
        # W: k, i, r (Wait, W_3d is (p, n, R) so index 0 is k, index 1 is i)
        # So W_3d[k, i, r].
        # Einsum string: 'ijr,jkr,kr->ijk'
        # Term 1: U_3d[i, j, r]
        # Term 2: V_3d[j, k, r]
        # Term 3: W_3d[k, i, r] -> The label 'k' is first, 'i' is second. 
        # In the string 'kr', 'k' corresponds to first axis, 'r' to third?
        # No, the string must match the axes.
        # W_3d axes: 0(k), 1(i), 2(r).
        # So the label for W_3d should be 'kir'.
        # So: np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        pred = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        return (pred - 1.0).flatten()

    def verify_exact(U, V, W, R):
        """
        Verify exact decomposition using Fractions.
        U, V, W are lists of lists of floats (which are actually half-integers).
        """
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = Fraction(0)
                    for r in range(R):
                        # Convert float to Fraction exactly if it's a half-integer
                        # We assume the input is already rounded to half-integers
                        u_val = Fraction(int(round(U[i * m + j][r] * 2)), 2)
                        v_val = Fraction(int(round(V[j * p + k][r] * 2)), 2)
                        w_val = Fraction(int(round(W[k * n + i][r] * 2)), 2)
                        val += u_val * v_val * w_val
                    if val != Fraction(1):
                        return False
        return True

    def try_rank(R, max_seeds=20):
        """
        Try to find a rank-R half-integer decomposition.
        """
        if time.time() > deadline - 5:
            return None

        n_params = 6 * R + 21 * R + 14 * R
        # 41 * R parameters

        try:
            from scipy.optimize import least_squares
        except ImportError:
            return None

        # Use a small number of seeds to spread out time
        for seed in range(max_seeds):
            if time.time() > deadline - 5:
                return None

            # Different scales for initialization
            # Sometimes small, sometimes larger
            scale = 0.1 * (1 + seed * 0.1)
            np.random.seed(seed * 1000 + 42)
            x0 = np.random.randn(n_params) * scale
            
            # Add some structured initialization?
            # e.g. random sparse?
            # For now, random dense is fine.

            try:
                # Use 'lm' (Levenberg-Marquardt) for small problems
                # If R is large, 'trf' (Trust Region Reflective) might be better
                # But LM is usually faster for zero-residual problems
                result = least_squares(fast_residual, x0, args=(R,), method='lm', max_nfev=1000)
            except Exception:
                continue

            # Check if solution is close to zero residual
            if result.cost < 1e-12:
                # Extract solution
                U_sol = result.x[:6 * R].reshape(6, R)
                V_sol = result.x[6 * R:6 * R + 21 * R].reshape(21, R)
                W_sol = result.x[6 * R + 21 * R:].reshape(14, R)

                # Round to nearest half-integer
                U_rounded = np.round(U_sol * 2) / 2.0
                V_rounded = np.round(V_sol * 2) / 2.0
                W_rounded = np.round(W_sol * 2) / 2.0

                # Convert to lists for verification
                U_list = [[float(U_rounded[i, r]) for r in range(R)] for i in range(6)]
                V_list = [[float(V_rounded[i, r]) for r in range(R)] for i in range(21)]
                W_list = [[float(W_rounded[i, r]) for r in range(R)] for i in range(14)]

                # Verify exactness
                if verify_exact(U_list, V_list, W_list, R):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Try ranks from lower bound up to schoolbook
    # Lower bound is 21.
    # We will try from 21 to 42.
    # If we find one, we return it.
    # If not, we return schoolbook.
    
    # Optimization: Start from the lowest possible rank.
    # However, lower ranks are harder to find.
    # Maybe start from a reasonable guess?
    # For 2x3x7, it's likely that 42 is the minimal rank for half-integers.
    # But let's try.
    
    # To save time, we can limit the number of seeds for lower ranks
    # and use more seeds for higher ranks if we suspect a solution exists there.
    # But since we don't know, we iterate.
    
    # Time budget allocation:
    # Total time ~ 140s.
    # Ranks: 22 values (21 to 42).
    # ~6s per rank on average.
    # That's enough for ~20-30 seeds per rank with LM.
    
    for R in range(21, 43):
        if time.time() > deadline - 10:
            break
        
        # Decrease seeds as R increases? No, higher R is easier.
        # Actually, lower R is harder.
        # Let's use 10 seeds for R < 30, 20 for R >= 30.
        seeds = 10 if R < 30 else 20
        
        res = try_rank(R, max_seeds=seeds)
        if res is not None:
            return res

    # Fallback to schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
