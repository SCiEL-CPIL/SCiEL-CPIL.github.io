"""Matrix multiplication 2x3x6 rank decomposition.

We find a low-rank exact tensor decomposition for the 2x3x6 matrix multiplication
tensor, then verify numerically and return the factor matrices.
"""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 3, 6]
n, m, p = DIMENSIONS
RANK = 24  # target rank for a 2x3x6 MM tensor (known lower than schoolbook 36)

# We will try to find a rank-24 decomposition by solving a linear system.
# The MM tensor T[i,j,k] = 1 if the corresponding output element C[i,k] = sum_l A[i,l]*B[l,k]
# We need U, V, W such that for all (i,k):
#   T_ik = sum_r U[i*m + ?, r] * V[?, k?, r] * W[k*n + i, r]
# Actually, the tensor is T[i,j,k] (A index i, B index j, output index k... wait)
#
# Let me re-read the task: "multiplying a 2 by 3 matrix by a 3 by 6 matrix"
# So A is 2x3, B is 3x6, C = A*B is 2x6.
# The tensor T has indices (i, j, k) where i=0..1 (row of A, row of C), j=0..2 (col of A = row of B), k=0..5 (col of B, col of C).
# T[i,j,k] = 1, and C[i,k] = sum_j A[i,j] * B[j,k] = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k]
#
# Rank-r decomposition: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k' ... ) 
# Actually, the standard form is:
# C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_l V[j*p+l, r] * B[j,l])
# But that's not right either. Let me think again.
#
# The matrix multiplication tensor T ∈ R^{n×m×p} where T[i,j,k]=1.
# A rank-1 tensor is u_i * v_j * w_k.
# A rank-r decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] ... no.
#
# Actually for matrix multiplication, the tensor is T[i,j,k] where the three modes are:
# mode 1: row index of A (i=0..n-1)
# mode 2: column index of A / row index of B (j=0..m-1)  
# mode 3: column index of B (k=0..p-1)
#
# But the output C has indices (i,k), not (i,j,k). So the tensor really encodes:
# C[i,k] = sum_j A[i,j] B[j,k]
#
# In tensor form: the multiplication map is a trilinear map.
# The tensor T has entries T[i,j,k] = δ_{j,j} (always 1) — wait, that's not right.
#
# Let me use the standard formulation. The MM tensor for n×m and m×p is in R^{(n×m) × (m×p) × (n×p)} but that's not a simple tensor either.
#
# Actually, the simplest way: we want to find U ∈ R^{(n*m)×r}, V ∈ R^{(m*p)×r}, W ∈ R^{(n*p)×r} such that:
# For all i in [0,n), k in [0,p):
#   1 = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] ... no, that doesn't work because j is summed out.
#
# Let me look at the seed program more carefully:
# U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
# And the decomposition is: for each (i,j,k) triple, one term.
#
# So the tensor entry T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
# And we need T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]B[j,k] and each (i,j,k) contributes).
#
# Wait, that's the standard MM tensor: T[i,j,k] = 1 for all i,j,k? No!
# T[i,j,k] should be such that sum_j T[i,j,k] A[i,j] B[j,k] = C[i,k].
# Since C[i,k] = sum_j A[i,j] B[j,k], we need T[i,j,k] = 1 for all i,j,k. Yes.
#
# So we need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
#
# This is a system of n*m*p = 36 equations. We want to solve for U, V, W with r=24 (or lower).
# This is a non-linear system (bilinear in pairs of variables), but we can fix two of them and solve linearly for the third.
#
# Strategy: fix U and V, then solve for W linearly. Each equation is linear in W entries.
# We have 36 equations and p*n*r = 12*24 = 288 unknowns in W. So it's underdetermined if we fix U,V.
# But we also need W to have specific structure.
#
# Alternative: use a known low-rank decomposition. For 2x3x6, the rank is known to be 24 (or maybe lower).
# Let me try to construct one using the fact that 6 = 2*3.
#
# Actually, let me try a different approach. I'll set up the problem as finding U, V, W with minimal r such that the tensor equation holds.
# I'll use a random initialization and then solve.
#
# For r=24:
# Fix U and V randomly (or with structure), solve for W.
# Check if W satisfies all 36 equations.

def try_decomposition(r, max_iters=10000, seed=42):
    """Try to find a rank-r decomposition by fixing U,V and solving for W."""
    np.random.seed(seed)
    nm = n * m  # 6
    mp = m * p  # 18
    pn = p * n  # 12
    
    # Initialize U and V randomly
    U = np.random.randn(nm, r)
    V = np.random.randn(mp, r)
    
    for it in range(max_iters):
        # Build the linear system for W
        # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # Group by (i,k): for fixed (i,k), we have m equations (one per j) that involve W[k*n+i, :]
        # Actually each (i,j,k) gives one equation, and the unknowns are W[k*n+i, 0..r-1]
        # For fixed (i,k), the equations over j=0..m-1 are:
        #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for j=0,1,2
        # This is a system of m=3 equations in r=24 unknowns (W[k*n+i, :])
        # So it's underdetermined. We can pick a solution (e.g., minimum norm) for each (i,k).
        
        # Build the full system
        # Unknowns: W flattened, size pn * r = 12 * 24 = 288
        # Equations: nm * p = 6 * 6 = 36
        # A * W_flat = b where A is 36 x 288
        
        A_mat = np.zeros((n * m * p, pn * r))
        b = np.ones(n * m * p)
        
        eq = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # U[i*m+j, :] * V[j*p+k, :] gives coefficients for W[k*n+i, :]
                    coeffs = U[i * m + j, :] * V[j * p + k, :]
                    row_start = (k * n + i) * r
                    A_mat[eq, row_start:row_start + r] = coeffs
                    eq += 1
        
        # Solve least squares
        W_flat, residuals, rank_, sv = np.linalg.lstsq(A_mat, b, rcond=None)
        W = W_flat.reshape(pn, r)
        
        # Verify
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    err = abs(val - 1)
                    if err > max_err:
                        max_err = err
        
        if max_err < 1e-10:
            return U, V, W, max_err
        
        # Perturb U and V slightly
        U += 0.1 * np.random.randn(nm, r)
        V += 0.1 * np.random.randn(mp, r)
    
    return None, None, None, max_err

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    # First, try rank 24
    best_result = None
    best_rank = None
    
    for r in [24, 27, 30, 36]:
        if time.time() > deadline - 10:
            break
        
        result = try_decomposition(r, max_iters=5000, seed=r * 100)
        if result[0] is not None:
            U, V, W, err = result
            best_result = (U, V, W, r)
            best_rank = r
            break
    
    if best_result is None:
        # Fallback: schoolbook rank 36
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        best_result = (U, V, W, r)
        best_rank = r
    
    U, V, W, r = best_result
    
    # Convert to list-of-lists format with integer/half-integer entries if possible
    # Check if entries are close to integers or half-integers
    def round_if_close(matrix):
        rounded = np.round(matrix * 2) / 2
        if np.allclose(matrix, rounded, atol=1e-9):
            return rounded
        return matrix
    
    U = round_if_close(U)
    V = round_if_close(V)
    W = round_if_close(W)
    
    # Final verification
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                err = abs(val - 1)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-8:
        # Fallback to schoolbook if verification fails
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        r = n * m * p
    
    # Convert to Python lists
    U_list = [[float(U[i, j]) for j in range(r)] for i in range(n * m)]
    V_list = [[float(V[i, j]) for j in range(r)] for i in range(m * p)]
    W_list = [[float(W[i, j]) for j in range(r)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
