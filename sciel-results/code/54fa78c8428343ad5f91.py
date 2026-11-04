"""Combined program for matrix multiplication 2x3x7 rank minimization."""
import numpy as np
import random
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    deadline = time.time() + 140  # safety margin before 150s kill

    # Schoolbook as baseline
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
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

    def verify_fast(U, V, W, rank):
        """Vectorized verification using numpy."""
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        U3 = U_np.reshape(n, m, rank)
        V3 = V_np.reshape(m, p, rank)
        W3 = W_np.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.all(T == 1)

    # We try to find a decomposition of rank R.
    # We fix U and V randomly, and solve for W.
    # The system is M W = 1, where M is built from U and V.
    
    # Lower bound for rank is max(nm, mp, pn) = 21.
    # We iterate R from 21 to best_rank - 1.
    
    for target_rank in range(21, best_rank):
        if time.time() > deadline - 10:
            break
            
        # Number of attempts for this rank.
        # Lower ranks are harder, but the linear system approach is deterministic given U,V.
        # We just need to find U,V such that the solution W is integer/half-integer.
        max_attempts = 2000 if target_rank < 30 else 500
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
                
            # Generate random U and V with small integers
            # Using {-1, 0, 1} increases sparsity and chance of integer solutions?
            # Or maybe just 0, 1?
            U = np.array([[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(nm)], dtype=np.float64)
            V = np.array([[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(mp)], dtype=np.float64)
            
            # Build the linear system for W
            # Unknowns: W has shape (pn, target_rank) -> pn * target_rank variables
            # Equations: n * m * p equations
            num_vars = pn * target_rank
            num_eqs = n * m * p
            
            M = np.zeros((num_eqs, num_vars), dtype=np.float64)
            b = np.ones(num_eqs, dtype=np.float64)
            
            # Fill M
            # Equation index e corresponds to (i, j, k)
            # Variable index v corresponds to (k, i, c) in W -> (k*n + i) * target_rank + c
            # M[e, v] = U[i*m + j, c] * V[j*p + k, c]
            
            # We can vectorize the filling or just loop since dimensions are small
            # num_eqs = 42, num_vars <= 14 * 41 = 574. 
            # Looping 42 * 14 * 41 is ~24k operations. Very fast.
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        for c in range(target_rank):
                            # W variable index
                            w_row = k * n + i
                            var_idx = w_row * target_rank + c
                            u_val = U[i * m + j, c]
                            v_val = V[j * p + k, c]
                            M[eq_idx, var_idx] = u_val * v_val
            
            # Solve M W_flat = b
            # Use least squares
            try:
                W_flat, residuals, rank, sv = np.linalg.lstsq(M, b, rcond=None)
            except np.linalg.LinAlgError:
                continue
                
            W_sol = W_flat.reshape(pn, target_rank)
            
            # Check if W_sol is close to integer or half-integer
            # Multiply by 2 to check for integers
            W_check = W_sol * 2
            W_rounded = np.round(W_check)
            
            # Check closeness
            diff = np.abs(W_check - W_rounded)
            if np.all(diff < 1e-6):
                # It looks like a half-integer solution
                W_int = (W_rounded / 2)
                # Convert to exact fractions or just check if 2*W_int is integer
                # The problem allows integer/half-integer.
                # Let's convert to a list of floats that are exactly 0, 0.5, 1, etc.
                # Or just keep as float if they are exact halves.
                
                # Verify exactly
                U_list = U.astype(int).tolist()
                V_list = V.astype(int).tolist()
                W_list = W_int.tolist()
                
                # Ensure W_list contains only 0, 0.5, 1, -0.5, -1, etc.
                # The verification function expects integer/half-integer.
                # If we use floats in verify_fast, it might have precision issues.
                # Better to convert to a common denominator (2) and check integers.
                
                # Let's modify verify to handle this? 
                # No, verify_fast uses int64. 
                # If W has halves, we can't put them in int64 directly.
                # We can scale U, V, W by 2? 
                # If W has halves, let W = W' / 2.
                # Then sum U V (W'/2) = 1 => sum (2U) V W' = 2.
                # Or sum U (2V) W' = 2.
                # Or sum U V W' = 2.
                # We can scale one matrix by 2 and target 2.
                
                # Let's try to make U, V, W integers by scaling.
                # If W has halves, multiply W by 2 and multiply U by 2? No, that changes the product by 4.
                # Multiply W by 2, and we need the sum to be 2.
                # So we check if sum U V W' = 2.
                
                # Let's just check if W_sol is integer first.
                if np.all(np.abs(W_sol - np.round(W_sol)) < 1e-6):
                    W_int = np.round(W_sol).astype(int).tolist()
                    if verify_fast(U_list, V_list, W_int, target_rank):
                        return {"U": U_list, "V": V_list, "W": W_int}
                else:
                    # Try half-integers
                    # If W is half-integer, 2W is integer.
                    W_int_scaled = np.round(W_check).astype(int)
                    # We need sum U V W' = 2 where W' = 2W
                    # Let's verify this scaled version
                    # Create a temporary verify for target 2
                    U_np = np.array(U_list, dtype=np.int64)
                    V_np = np.array(V_list, dtype=np.int64)
                    W_np = W_int_scaled
                    U3 = U_np.reshape(n, m, target_rank)
                    V3 = V_np.reshape(m, p, target_rank)
                    W3 = W_np.reshape(pn, target_rank)
                    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                    if np.all(T == 2):
                        # We found a solution with W' = 2W.
                        # We need to return W = W' / 2.
                        # The interface expects entries. Can we return floats?
                        # "integer/half-integer entries".
                        # Yes, floats representing halves are fine.
                        W_final = (W_np / 2.0).tolist()
                        return {"U": U_list, "V": V_list, "W": W_final}
                        
    return best
# EVOLVE-BLOCK-END
