"""Matrix multiplication 2x4x6 rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 24
    pn = p * n  # 12
    
    # Schoolbook rank = n*m*p = 48
    # We'll try to find a lower rank decomposition using random initialization and iterative refinement.
    # The tensor T has entries T[i,j,k] = 1 (for the multiplication tensor, actually T is the structure tensor)
    # Actually, for matrix multiplication C = A * B:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor is T_{i,j,k} which is 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # But the standard decomposition uses rank-n terms where each term is a trilinear form.
    
    # The tensor for 2x4x6 matrix mult has 48 non-zero entries (one for each (i,j,k) triple)
    # We need to decompose this tensor into a sum of rank-1 tensors.
    
    # Let's use the schoolbook solution as a fallback and try to improve.
    # First, create the tensor as a 3D array.
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    
    # The tensor to decompose: we want sum_c U[:, i, j] * V[:, j, k] * W[:, k, i] = 1 for all i,j,k
    # Wait, the convention is:
    # U has shape (n*m, rank) -> U[i*m+j, c]
    # V has shape (m*p, rank) -> V[j*p+k, c]  
    # W has shape (p*n, rank) -> W[k*n+i, c]
    # The product: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1 for all i,j,k
    
    # Let's try to find a decomposition of rank r.
    # Start with a reasonable guess. Known results:
    # 2x2x2: rank 7 (Strassen)
    # 2x3x3: rank 16 or 17?
    # 2x4x4: rank 24 or so?
    # 2x4x6: let's try ranks from 32 to 48.
    
    def try_rank(r):
        """Try to find a rank-r decomposition."""
        # Random initialization
        np.random.seed(42)
        U = np.random.randn(nm, r)
        V = np.random.randn(mp, r)
        W = np.random.randn(pn, r)
        
        # Gradient-free approach: use alternating least squares or similar
        # Actually, for exact decomposition, we need to solve a system of equations.
        # The constraint is: for each (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        # This is a system of nonlinear equations. Let's use scipy.optimize if available.
        try:
            from scipy.optimize import least_squares
        except ImportError:
            return None
        
        # Flatten the variables
        def pack(U, V, W):
            return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def unpack(x):
            U_size = nm * r
            V_size = mp * r
            U = x[:U_size].reshape(nm, r)
            V = x[U_size:U_size+V_size].reshape(mp, r)
            W = x[U_size+V_size:].reshape(pn, r)
            return U, V, W
        
        def residuals(x):
            U, V, W = unpack(x)
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :]) * W[k*n+i, :]
                        # Wait, the sum is over c: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                        # This is not a simple dot product of two vectors.
                        # It's the element-wise product of three vectors, then summed.
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        res.append(val - 1.0)
            return np.array(res)
        
        # Initial guess
        x0 = pack(U, V, W)
        
        # Scale the initial guess to get closer to the solution
        # The product of three ~N(0,1) vectors summed over r terms: variance is r, so mean abs value is ~sqrt(r)
        # We want each term to contribute ~1/r on average
        scale = r ** 0.5
        x0 = x0 / scale
        
        result = least_squares(residuals, x0, method='lm', xtol=1e-12, ftol=1e-12, gtol=1e-12)
        
        if result.cost < 1e-10:
            U, V, W = unpack(result.x)
            # Check if we can round to integers or half-integers
            # Scale by 2 to check for half-integers
            U2 = np.round(2 * U) / 2
            V2 = np.round(2 * V) / 2
            W2 = np.round(2 * W) / 2
            
            # Verify the rounded solution
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U2[i*m+j, :] * V2[j*p+k, :] * W2[k*n+i, :])
                        max_err = max(max_err, abs(val - 1.0))
            
            if max_err < 1e-9:
                return U2, V2, W2
            
            # Try integer rounding
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_int[i*m+j, :] * V_int[j*p+k, :] * W_int[k*n+i, :])
                        max_err = max(max_err, abs(val - 1.0))
            
            if max_err < 1e-9:
                return U_int, V_int, W_int
            
            # Return the floating point solution (may not be exact but close)
            return result.x[:nm*r].reshape(nm, r), result.x[nm*r:nm*r+mp*r].reshape(mp, r), result.x[nm*r+mp*r:].reshape(pn, r)
        
        return None
    
    # Try ranks from low to high
    for r in [32, 34, 36, 38, 40, 42, 44, 46]:
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            # Convert to lists
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
