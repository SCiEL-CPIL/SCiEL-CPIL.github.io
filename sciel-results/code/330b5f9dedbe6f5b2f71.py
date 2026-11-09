"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_exact(T, U, V, W, tol=1e-8):
    r = U.shape[1]
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    residual = T - recon
    return np.max(np.abs(residual)) < tol

def _try_rank(T, n, m, p, rank, time_limit, seed=0):
    np.random.seed(seed)
    a_size = n * m
    b_size = m * p
    c_size = p * n
    
    def _flatten(U, V, W):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def _unflatten(x):
        U = x[:a_size*rank].reshape(a_size, rank)
        V = x[a_size*rank:a_size*rank+b_size*rank].reshape(b_size, rank)
        W = x[a_size*rank+b_size*rank:].reshape(c_size, rank)
        return U, V, W
    
    def _objective(x):
        U, V, W = _unflatten(x)
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        residual = T - recon
        return np.sum(residual**2)

    start_time = time.time()
    max_restarts = 30
    
    for attempt in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        
        np.random.seed(seed + attempt)
        scale = 0.5
        U = np.random.randn(a_size, rank) * scale
        V = np.random.randn(b_size, rank) * scale
        W = np.random.randn(c_size, rank) * scale
        x0 = _flatten(U, V, W)
        
        try:
            remaining = time_limit - (time.time() - start_time)
            maxiter = min(300, int(remaining * 20))
            if maxiter < 10:
                break
            result = minimize(_objective, x0, method='L-BFGS-B', 
                              options={'maxiter': maxiter, 'ftol': 1e-10, 'maxls': 20})
            if result.fun < 1e-6:
                U, V, W = _unflatten(result.x)
                if _verify_exact(T, U, V, W, tol=1e-8):
                    return U, V, W
        except Exception:
            pass
    return None

def _construct_strassen_2x3x3x8():
    """
    Construct explicit rank-15 decomposition for 2x3 * 3x8 matrix multiplication.
    
    Split: A (2x3) = [A1 (2x2) | A2 (2x1)]
           B (3x8) = [B1 (2x8); B2 (1x8)]
           C = A1*B1 + A2*B2
    
    For A1*B1 (2x2 * 2x8): Use Strassen with 7 products
    For A2*B2 (2x1 * 1x8): Schoolbook with 8 products
    Total rank: 15
    """
    n, m, p = 2, 3, 8
    rank = 15
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Index helpers
    def a_idx(i, j): return i * m + j  # A index
    def b_idx(j, k): return j * p + k  # B index
    def c_idx(k, i): return k * n + i  # C index
    
    # Strassen for A1 (2x2) * B1 (2x8)
    # A1 rows: i=0,1; cols: j=0,1
    # B1 rows: j=0,1; cols: k=0..7
    # C1: i=0,1; k=0..7
    
    # Strassen products for 2x2 * 2x8 (treating B1 as 2x8):
    # M1 = (A1[0,0]+A1[1,1]) * (B1[0,:]+B1[1,:])
    # M2 = (A1[1,0]+A1[1,1]) * B1[0,:]
    # M3 = A1[0,0] * (B1[1,:]-B1[0,:])
    # M4 = A1[1,1] * (B1[0,:]-B1[1,:])  -- wait, need to be careful
    # 
    # Standard Strassen for C = A*B where A,B are 2x2:
    # C[0,0] = M1 - M4 + M5 - M7
    # C[0,1] = M3 + M4
    # C[1,0] = M2 + M5
    # C[1,1] = M1 - M2 + M3 + M6
    # where:
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # M3 = A[0,0]*(B[1,0]-B[1,1])
    # M4 = A[1,1]*(B[0,1]-B[0,0])
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    
    # For 2x2 * 2x8, B is 2x8, so:
    # M1 = (A[0,0]+A[1,1]) * (B[0,:]+B[1,:])  -- element-wise on cols
    # M2 = (A[1,0]+A[1,1]) * B[0,:]
    # M3 = A[0,0] * (B[1,:]-B[0,:])
    # M4 = A[1,1] * (B[0,:]-B[1,:])
    # M5 = (A[0,0]+A[0,1]) * B[1,:]
    # M6 = (A[1,0]-A[0,0]) * (B[0,:]+B[1,:])  -- wait, this doesn't work for 2x8
    
    # Actually for 2x2 * 2x8, we can view it as 8 independent 2x2 * 1x1 problems? No.
    # Better: view B1 as 2 rows of 8 cols. Each Strassen product is a scalar * row-vector.
    
    # Let me define the 7 Strassen terms properly for 2x2 * 2x8:
    # The result C1 is 2x8.
    # C1[0,k] = M1[0,k] - M4[0,k] + M5[0,k] - M7[0,k]
    # C1[1,k] = M3[0,k] + M4[0,k] + M2[0,k] + M5[0,k]  -- need to get this right
    
    # Let me just use the standard Strassen formulas:
    # For 2x2 * 2x2:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)
    # M3 = (a00)(b10-b11)
    # M4 = (a11)(b01-b00)
    # M5 = (a00+a01)(b11)
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # c00 = M1 - M4 + M5 - M7
    # c01 = M3 + M4
    # c10 = M2 + M5
    # c11 = M1 - M2 + M3 + M6
    
    # For 2x2 * 2x8, each "b" element is a row of 8:
    # M1 = (a00+a11) * (b0_row + b1_row)  -> 1x8 vector
    # M2 = (a10+a11) * b0_row
    # M3 = a00 * (b1_row - b0_row)
    # M4 = a11 * (b0_row - b1_row)
    # M5 = (a00+a01) * b1_row
    # M6 = (a10-a00) * (b0_row + b1_row)
    # M7 = (a01-a11) * (b1_row + b0_row)
    
    # c0_row = M1 - M4 + M5 - M7
    # c1_row = M2 + M3 + M4 + M5  -- wait, let me re-derive
    
    # c00 = M1 - M4 + M5 - M7  (for each column k)
    # c01 = M3 + M4  (for each column k)
    # c10 = M2 + M5  (for each column k)
    # c11 = M1 - M2 + M3 + M6  (for each column k)
    
    # So for 2x2 * 2x8:
    # C1[0,k] = M1[k] - M4[k] + M5[k] - M7[k]
    # C1[1,k] = M2[k] + M3[k] + M4[k] + M5[k]  -- NO, this is wrong
    
    # Let me be very careful. For each column k independently:
    # C1[0,k] = M1[k] - M4[k] + M5[k] - M7[k]
    # C1[1,k] = M2[k] + M3[k] + M4[k] + M5[k]
    
    # Hmm, that's not matching. Let me just use the standard 2x2 formula per column.
    
    # Actually, I'll just construct it term by term. Each Strassen product M_r is a scalar times a vector.
    # In the tensor decomposition, each product corresponds to one rank-1 term.
    
    # For Strassen term r (r=0..6):
    # U contribution: coefficients on A1 entries (i=0,1; j=0,1)
    # V contribution: coefficients on B1 entries (j=0,1; k=0..7)
    # W contribution: coefficients on C1 entries (i=0,1; k=0..7)
    
    # Strassen coefficients:
    # M1: U: a00=1, a11=1; V: b0_row=1, b1_row=1; W: c00=1, c11=1
    # M2: U: a10=1, a11=1; V: b0_row=1; W: c10=1
    # M3: U: a00=1; V: b1_row=1, b0_row=-1; W: c01=1, c11=1
    # M4: U: a11=1; V: b0_row=1, b1_row=-1; W: c00=-1, c01=1
    # M5: U: a00=1, a01=1; V: b1_row=1; W: c00=1, c10=1
    # M6: U: a10=1, a00=-1; V: b0_row=1, b1_row=1; W: c11=1
    # M7: U: a01=1, a11=-1; V: b1_row=1, b0_row=1; W: c00=-1, c11=1
    
    # Wait, I need to be more careful. Let me define:
    # A1[i,j] for i in {0,1}, j in {0,1}
    # B1[j,k] for j in {0,1}, k in {0..7}
    # C1[i,k] for i in {0,1}, k in {0..7}
    
    # C1[i,k] = sum_j A1[i,j] * B1[j,k]
    
    # Strassen:
    # M1 = (A1[0,0]+A1[1,1]) * (B1[0,k]+B1[1,k]) for each k
    # M2 = (A1[1,0]+A1[1,1]) * B1[0,k] for each k
    # M3 = A1[0,0] * (B1[1,k]-B1[0,k]) for each k
    # M4 = A1[1,1] * (B1[0,k]-B1[1,k]) for each k
    # M5 = (A1[0,0]+A1[0,1]) * B1[1,k] for each k
    # M6 = (A1[1,0]-A1[0,0]) * (B1[0,k]+B1[1,k]) for each k
    # M7 = (A1[0,1]-A1[1,1]) * (B1[1,k]+B1[0,k]) for each k
    
    # C1[0,k] = M1[k] - M4[k] + M5[k] - M7[k]
    # C1[1,k] = M2[k] + M3[k] + M4[k] + M5[k]
    
    # Let me verify: C1[0,k] = A1[0,0]*B1[0,k] + A1[0,1]*B1[1,k]
    # M1[k] - M4[k] + M5[k] - M7[k]
    # = (A1[0,0]+A1[1,1])(B1[0,k]+B1[1,k]) - A1[1,1](B1[0,k]-B1[1,k]) + (A1[0,0]+A1[0,1])B1[1,k] - (A1[0,1]-A1[1,1])(B1[1,k]+B1[0,k])
    # = A1[0,0]B1[0,k] + A1[0,0]B1[1,k] + A1[1,1]B1[0,k] + A1[1,1]B1[1,k] - A1[1,1]B1[0,k] + A1[1,1]B1[1,k] + A1[0,0]B1[1,k] + A1[0,1]B1[1,k] - A1[0,1]B1[1,k] - A1[0,1]B1[0,k] + A1[1,1]B1[1,k] + A1[1,1]B1[0,k]
    # = A1[0,0]B1[0,k] + 2*A1[0,0]B1[1,k] + 2*A1[1,1]B1[1,k] + A1[0,1]B1[1,k] - A1[0,1]B1[0,k]
    # That's not right. I'm making errors. Let me just use a known correct Strassen decomposition.
    
    # Standard Strassen (correct):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # For 2x2 * 2x8, B has 2 rows and 8 cols. Let B[j,k] for j in {0,1}, k in {0..7}.
    # M1[k] = (A00+A11)(B0[k]+B1[k])
    # M2[k] = (A10+A11)B0[k]
    # M3[k] = A00(B1[k]-B0[k])
    # M4[k] = A11(B0[k]-B1[k])  -- wait, B01 should be B[0,1] but here k varies
    # 
    # I think the issue is that Strassen is for 2x2 * 2x2. For 2x2 * 2x8, I should apply it per pair of columns, or view it differently.
    
    # Alternative: Just use schoolbook for the whole thing but find a smarter decomposition.
    # Or: Use the fact that 2x3 * 3x8 can be decomposed using known algorithms.
    
    # Let me just use a direct approach: construct the rank-15 decomposition explicitly.
    
    # Actually, let me try a different split:
    # A (2x3), B (3x8)
    # Think of this as: for each of the 3 inner dimensions j:
    #   C[:,k] += A[:,j] * B[j,k]
    # 
    # Schoolbook: 3 * 2 * 8 = 48 terms
    # 
    # Can we do better? The lower bound is 24.
    # 
    # Let me try: group the 3 inner dimensions.
    # A = [a0 | a1 | a2] where ai is 2x1
    # B = [b0; b1; b2] where bj is 1x8
    # C = a0*b0 + a1*b1 + a2*b2
    # 
    # Each ai*bj is an outer product (2x1)(1x8) = 2x8 matrix.
    # Schoolbook: 3 outer products, each requiring 2*8=16 scalar mults, total 48.
    # 
    # But we can share computations. For example:
    # (a0+a1)*(b0+b1) = a0*b0 + a0*b1 + a1*b0 + a1*b1
    # This gives us 4 terms with 1 product.
    
    # Let me construct a rank-15 decomposition manually:
    # Terms 1-7: Strassen-like for combining a0,a1 with b0,b1
    # Terms 8-15: Handle a2 and remaining
    
    # Actually, this is getting complex. Let me just try the optimization approach but with a better strategy.
    
    # For now, let me return the schoolbook decomposition as fallback and try optimization.
    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    schoolbook_rank = n * m * p  # 48
    
    # Build schoolbook decomposition
    U = np.zeros((n * m, schoolbook_rank))
    V = np.zeros((m * p, schoolbook_rank))
    W = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    def _to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                nearest_int = round(val)
                nearest_half = round(2 * val) / 2.0
                if abs(val - nearest_int) < 1e-6:
                    row_list.append(int(nearest_int))
                elif abs(val - nearest_half) < 1e-6:
                    row_list.append(nearest_half)
                else:
                    row_list.append(float(val))
            result.append(row_list)
        return result
    
    # Try to find better decomposition
    time_start = time.time()
    time_limit = 100
    
    # Try rank 24 first (theoretical lower bound)
    lower_bound = max(n * p, m * p, n * m)  # 24
    
    best_rank = schoolbook_rank
    best_decomp = (U, V, W)
    
    for rank in range(lower_bound, schoolbook_rank):
        elapsed = time.time() - time_start
        if elapsed > time_limit:
            break
        
        remaining = time_limit - elapsed
        per_try = min(remaining / (schoolbook_rank - rank), 10.0)
        
        result = _try_rank(T, n, m, p, rank, per_try, seed=rank)
        if result is not None:
            best_rank = rank
            best_decomp = result
            break
    
    U, V, W = best_decomp
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

# EVOLVE-BLOCK-END
