import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 9]

def construct():
    """
    Attempts to find a rank-minimal exact tensor decomposition for 2x3x9 matrix multiplication
    using continuous optimization (L-BFGS-B) starting from a lower rank than schoolbook.
    
    The tensor T[i,j,k] = 1 for all valid indices.
    We seek U (n*m, r), V (m*p, r), W (p*n, r) such that:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    Score is the rank r if the decomposition is exact (within tolerance), else infinity/invalid.
    """
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Target tensor T[i, j, k] = 1
    # We will flatten this for the loss function
    # The decomposition is a sum of rank-1 tensors.
    
    # Schoolbook rank is 54.
    # We try to beat this. Known lower bounds for 2x3x9 are not trivially lower than 54 in simple cases,
    # but optimization might find a sparse structure or a valid lower rank if one exists.
    # Actually, for small matrices, the rank is often close to nm*p.
    # Let's try ranks below 54. If it fails to converge to error ~0, we return the best valid one found.
    # If no lower rank works, we return schoolbook.
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Try ranks from a lower bound up to schoolbook
    # A trivial lower bound is max(nm, mp, pn) = max(6, 27, 18) = 27.
    # But tensor rank is usually higher.
    # Let's try a few candidates below 54.
    
    best_result = None
    best_rank = nm * p # 54
    
    # Helper to verify exactness
    def verify(U, V, W, tol=1e-5):
        # Reconstruct tensor
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We can compute this efficiently using einsum
        # But for verification, we can just check the max error.
        # To be fast, we can sample or use a fast check.
        # Let's use einsum for the full tensor if size is small enough.
        # nm=6, mp=27, pn=18. Tensor size 6*27*18 = 2916. Very small.
        
        r = U.shape[1]
        # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        
        # Compute T
        # T[i,j,k] = sum_r U_reshaped[i,j,r] * V_reshaped[j,k,r] * W_reshaped[k,i,r]
        # Using einsum: 'ijr,jkr,kir->ijk'
        T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        # Target is all ones
        error = np.max(np.abs(T - 1.0))
        return error < tol, error

    def loss(params):
        """
        params: flattened array of size r*(nm + mp + pn)
        """
        r = params.size // (nm + mp + pn)
        # Unpack
        offset = 0
        U_flat = params[offset:offset+nm*r]
        offset += nm*r
        V_flat = params[offset:offset+mp*r]
        offset += mp*r
        W_flat = params[offset:offset+pn*r]
        
        U = U_flat.reshape(nm, r)
        V = V_flat.reshape(mp, r)
        W = W_flat.reshape(pn, r)
        
        # Compute tensor
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        
        T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        # Loss: Frobenius norm of (T - 1)
        # We want T == 1
        diff = T - 1.0
        return np.sum(diff**2)

    # Try ranks
    # We'll try a few ranks below 54.
    # If we find a valid one, we return it.
    # If not, we return schoolbook.
    
    # To save time, we only try a few optimization steps per rank.
    # L-BFGS-B is efficient.
    
    ranks_to_try = [40, 45, 50, 54] # 54 is schoolbook, guaranteed to work.
    # Actually, we should try lower first.
    # If we don't find a lower one, we fall back to 54.
    
    for r in ranks_to_try:
        if time.time() > deadline - 10.0:
            break
            
        # Initialize with random values
        # Scale matters. If we start with 0, gradient is 0.
        # Start with small random values.
        np.random.seed(42 + r) # Reproducibility for debugging, but maybe random is better for diversity?
        # Let's use random seed based on time for exploration? No, deterministic is safer for evaluation consistency if it works.
        # But if it fails, we want to try different seeds?
        # Let's try a few seeds for each rank if time permits.
        
        found_valid = False
        best_params_for_r = None
        best_loss_for_r = float('inf')
        
        # Try a few random initializations
        for seed in range(5):
            if time.time() > deadline - 10.0:
                break
                
            np.random.seed(seed)
            size = r * (nm + mp + pn)
            # Initialize with small random numbers
            params = np.random.uniform(-0.5, 0.5, size)
            
            # Optimization
            try:
                res = minimize(loss, params, method='L-BFGS-B', options={'maxiter': 500, 'maxfun': 10000})
                if res.success or res.fun < 1e-4:
                    # Check validity
                    offset = 0
                    U_flat = res.x[offset:offset+nm*r]
                    offset += nm*r
                    V_flat = res.x[offset:offset+mp*r]
                    offset += mp*r
                    W_flat = res.x[offset:offset+pn*r]
                    
                    U = U_flat.reshape(nm, r)
                    V = V_flat.reshape(mp, r)
                    W = W_flat.reshape(pn, r)
                    
                    valid, err = verify(U, V, W)
                    if valid:
                        best_params_for_r = res.x
                        found_valid = True
                        break
                    elif res.fun < best_loss_for_r:
                        best_loss_for_r = res.fun
                        best_params_for_r = res.x
            except Exception:
                pass
        
        if found_valid:
            # Extract U, V, W
            offset = 0
            U_flat = best_params_for_r[offset:offset+nm*r]
            offset += nm*r
            V_flat = best_params_for_r[offset:offset+mp*r]
            offset += mp*r
            W_flat = best_params_for_r[offset:offset+pn*r]
            
            U = U_flat.reshape(nm, r).tolist()
            V = V_flat.reshape(mp, r).tolist()
            W = W_flat.reshape(pn, r).tolist()
            
            # Clean up values: snap to integers/half-integers if close?
            # The problem asks for integer/half-integer entries.
            # Optimization gives floats. We need to round them.
            # If we round, the exactness might break.
            # However, the evaluator checks "exactly decompose".
            # If the floats are very close to .0 or .5, we can round.
            # Let's check if the rounded version is still valid.
            
            U_rounded = [[round(val * 2) / 2 for val in row] for row in U]
            V_rounded = [[round(val * 2) / 2 for val in row] for row in V]
            W_rounded = [[round(val * 2) / 2 for val in row] for row in W]
            
            # Verify rounded version
            U_r = np.array(U_rounded)
            V_r = np.array(V_rounded)
            W_r = np.array(W_rounded)
            
            valid_rounded, err_rounded = verify(U_r, V_r, W_r, tol=1e-9)
            
            if valid_rounded:
                return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
            else:
                # If rounded version is not exact, the float version might be "exact" in floating point
                # but the contract says "integer/half-integer entries".
                # If we can't find a half-integer solution, we might have to return the float one?
                # "integer/half-integer entries that exactly decompose"
                # This implies the solution MUST be in that set.
                # If optimization finds a solution that isn't close to half-integers, it's invalid for the contract.
                # So we must continue searching or fall back to schoolbook (which is integer).
                pass
        
    # Fallback: Schoolbook
    rank = nm * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}

# Keep the function name and signature as required
# The prompt says "Keep the function names and signatures of the seed program."
# Seed program: def construct():
# So we are good.

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
