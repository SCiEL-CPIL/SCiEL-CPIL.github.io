"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    
    Strategy: 
    1. Construct the known rank-44 solution using Strassen's algorithm for 2x2x2 blocks 
       and schoolbook for the 2x1x8 part.
    2. Verify the solution against the correct matrix multiplication tensor (all 1s).
    3. Attempt to reduce the rank using numerical optimization and structured search.
    """
    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    size_U = n * m  # 6
    size_V = m * p  # 24
    size_W = p * n  # 16
    
    def get_target_tensor():
        """
        Returns the target tensor for matrix multiplication.
        T[i*m+j, j*p+k, k*n+i] = 1.0 for all valid i, j, k.
        """
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        return T

    def verify_solution(U, V, W, tol=1e-9):
        """Verify that the solution exactly decomposes the tensor."""
        T_target = get_target_tensor()
        # U, V, W are expected to be in (size, rank) format for this check
        # But the internal logic uses (rank, size). 
        # Let's assume U, V, W are (rank, size) here.
        T_decomp = np.einsum('ar,br,cr->abc', U.T, V.T, W.T)
        return np.allclose(T_decomp, T_target, atol=tol)

    def snap_to_half_integers(mat, tol=1e-6):
        """Snap values to nearest 0.5 and verify."""
        snapped = np.round(mat * 2) / 2
        if np.max(np.abs(snapped - mat)) > tol:
            return None
        return snapped

    def get_rank44_solution():
        """
        Return the known rank-44 solution as numpy arrays (R x size).
        Structure:
        - j=0: 16 schoolbook terms (2 i-values * 8 k-values)
        - j=1,2: 4 blocks of 2x2x2, each using Strassen (7 terms) = 28 terms
        Total: 16 + 28 = 44
        """
        rank = 44
        U = np.zeros((rank, size_U))
        V = np.zeros((rank, size_V))
        W = np.zeros((rank, size_W))
        col = 0
        
        # Part 1: j=0, schoolbook for 2x8
        for i in range(n):
            for k in range(p):
                U[col, i * m + 0] = 1
                V[col, 0 * p + k] = 1
                W[col, k * n + i] = 1
                col += 1
        
        # Part 2: j=1,2 in 4 blocks of 2x2x2 using Strassen
        for block in range(4):
            k0 = 2 * block
            k1 = 2 * block + 1
            
            # Strassen for 2x2x2:
            # m1 = (a00+a11)(b00+b11)
            U[col, 0 * m + 1] = 1; U[col, 1 * m + 2] = 1
            V[col, 1 * p + k0] = 1; V[col, 2 * p + k1] = 1
            W[col, k0 * n + 0] = 1; W[col, k1 * n + 1] = 1
            col += 1
            
            # m2 = (a10+a11)b00
            U[col, 1 * m + 1] = 1; U[col, 1 * m + 2] = 1
            V[col, 1 * p + k0] = 1
            W[col, k0 * n + 1] = 1; W[col, k1 * n + 1] = -1
            col += 1
            
            # m3 = a00(b01-b11)
            U[col, 0 * m + 1] = 1
            V[col, 1 * p + k1] = 1; V[col, 2 * p + k1] = -1
            W[col, k1 * n + 0] = 1; W[col, k1 * n + 1] = 1
            col += 1
            
            # m4 = a11(b10-b00)
            U[col, 1 * m + 2] = 1
            V[col, 2 * p + k0] = 1; V[col, 1 * p + k0] = -1
            W[col, k0 * n + 0] = 1; W[col, k0 * n + 1] = 1
            col += 1
            
            # m5 = (a00+a01)b11
            U[col, 0 * m + 1] = 1; U[col, 0 * m + 2] = 1
            V[col, 2 * p + k1] = 1
            W[col, k0 * n + 0] = -1; W[col, k1 * n + 0] = 1
            col += 1
            
            # m6 = (a10-a00)(b00+b01)
            U[col, 1 * m + 1] = 1; U[col, 0 * m + 1] = -1
            V[col, 1 * p + k0] = 1; V[col, 1 * p + k1] = 1
            W[col, k1 * n + 1] = 1
            col += 1
            
            # m7 = (a01-a11)(b10+b11)
            U[col, 0 * m + 2] = 1; U[col, 1 * m + 2] = -1
            V[col, 2 * p + k0] = 1; V[col, 2 * p + k1] = 1
            W[col, k0 * n + 0] = 1
            col += 1
        
        return U, V, W

    def to_output_format(U, V, W):
        """Convert numpy arrays (R x size) to list format (size x R)."""
        return {"U": U.T.tolist(), "V": V.T.tolist(), "W": W.T.tolist()}

    def error_func(params, R):
        """Compute residuals for given parameters."""
        U = params[0 : R*size_U].reshape(R, size_U)
        V = params[R*size_U : R*size_U + R*size_V].reshape(R, size_V)
        W = params[R*size_U + R*size_V : R*size_U + R*size_V + R*size_W].reshape(R, size_W)
        
        T_decomp = np.einsum('ar,br,cr->abc', U.T, V.T, W.T)
        T_target = get_target_tensor()
        
        # Flatten residuals
        residuals = (T_decomp - T_target).flatten()
        return residuals

    def try_rank_optimization(R, max_restarts=3, seed_base=1000):
        """Try to find rank-R solution using numerical optimization."""
        total_size = R * (size_U + size_V + size_W)
        
        for restart in range(max_restarts):
            if time.time() - start_time > timeout:
                return None
            
            np.random.seed(seed_base + restart)
            params = np.random.randn(total_size) * 0.5
            
            try:
                res = least_squares(error_func, params, args=(R,), method='lm', max_nfev=3000)
            except Exception:
                continue
            
            if res.cost < 1e-14:
                U = res.x[0 : R*size_U].reshape(R, size_U)
                V = res.x[R*size_U : R*size_U + R*size_V].reshape(R, size_V)
                W = res.x[R*size_U + R*size_V : R*size_U + R*size_V + R*size_W].reshape(R, size_W)
                
                # Try to snap to half-integers
                U_snapped = snap_to_half_integers(U)
                V_snapped = snap_to_half_integers(V)
                W_snapped = snap_to_half_integers(W)
                
                if U_snapped is not None and V_snapped is not None and W_snapped is not None:
                    if verify_solution(U_snapped, V_snapped, W_snapped):
                        return to_output_format(U_snapped, V_snapped, W_snapped)
        
        return None

    # 1. Generate Rank-44 solution
    U44, V44, W44 = get_rank44_solution()
    
    # Verify it
    if verify_solution(U44, V44, W44):
        fallback_result = to_output_format(U44, V44, W44)
    else:
        # If verification fails, something is wrong with the construction.
        # Return a dummy schoolbook solution (rank 48) as last resort
        rank = 48
        U = np.zeros((rank, size_U))
        V = np.zeros((rank, size_V))
        W = np.zeros((rank, size_W))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[col, i*m+j] = 1
                    V[col, j*p+k] = 1
                    W[col, k*n+i] = 1
                    col += 1
        fallback_result = to_output_format(U, V, W)
        return fallback_result

    # 2. Try to reduce rank
    # Try ranks 43, 42, 41
    for R in range(43, 40, -1):
        if time.time() - start_time > timeout:
            break
        result = try_rank_optimization(R, max_restarts=2)
        if result is not None:
            return result

    # Return the rank-44 fallback
    return fallback_result
# EVOLVE-BLOCK-END
