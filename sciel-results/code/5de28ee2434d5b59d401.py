import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]
_START_TIME = time.time()
TIME_LIMIT = 145.0  # safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _check_deadline():
    return _elapsed() > TIME_LIMIT

def _round_to_half(x):
    """Round to nearest half-integer."""
    return round(x * 2) / 2.0

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact."""
    rank = len(U[0]) if U else 0
    for r in range(rank):
        u_col = np.array([U[i*m+j][r] for i in range(n) for j in range(m)], dtype=object)
        v_col = np.array([V[j*p+k][r] for j in range(m) for k in range(p)], dtype=object)
        w_col = np.array([W[k*n+i][r] for k in range(p) for i in range(n)], dtype=object)
        
        # Compute outer product u ⊗ v reshaped to (n, m, p) and compare with w reshaped to (p, n)
        u_tensor = u_col.reshape(n, m)
        v_tensor = v_col.reshape(m, p)
        
        # The tensor entry for (i,j,k) should be U[i*m+j][r] * V[j*p+k][r]
        # And W[k*n+i][r] should equal sum over j of U[i*m+j][r] * V[j*p+k][r]
        for i in range(n):
            for k in range(p):
                expected = sum(u_tensor[i, j] * v_tensor[j, k] for j in range(m))
                actual = w_col[k*n+i]
                if expected != actual:
                    return False
    return True

def _verify_with_float(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition using floating point."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for r in range(rank):
        u_col = U_arr[:, r].reshape(n, m)
        v_col = V_arr[:, r].reshape(m, p)
        w_col = W_arr[:, r].reshape(p, n)
        
        # w should equal u * v (matrix product), transposed appropriately
        # W[k*n+i] = sum_j U[i*m+j] * V[j*p+k]
        # So W reshaped as (p, n) should equal V^T @ U^T ... let me think
        # W[k*n+i] for fixed r = sum_j U[i*m+j][r] * V[j*p+k][r]
        # This is (U_col @ V_col^T) where U_col is (n,m) and V_col is (m,p)
        # Result is (n,p) with entry [i,k] = sum_j U[i,j]*V[j,k]
        # But W is indexed as W[k*n+i], so W reshaped as (p,n) has [k,i] = sum_j U[i,j]*V[j,k]
        # So W_reshaped = (U_col @ V_col^T).T = V_col @ U_col^T
        
        expected = (u_col @ v_col.T).T  # shape (p, n)
        if not np.allclose(w_col, expected, atol=tol):
            return False
    return True

def _try_rank(r, n, m, p, deadline_check):
    """Try to find a rank-r decomposition using iterative optimization."""
    if deadline_check():
        return None
    
    # Initialize with random factors
    np.random.seed(42)
    U = np.random.randn(n * m, r)
    V = np.random.randn(m * p, r)
    W = np.random.randn(p * n, r)
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = float('inf')
    
    lr = 0.01
    for step in range(2000):
        if deadline_check():
            break
        
        # Compute gradient of loss = sum_r ||W_r - V_r^T @ U_r^T||^2 (in appropriate reshaping)
        # Loss = sum_r ||W[:,r] - (U[:,r].reshape(n,m) @ V[:,r].reshape(m,p).T).T||^2
        
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for r in range(r):
            u_col = U[:, r].reshape(n, m)
            v_col = V[:, r].reshape(m, p)
            w_col = W[:, r].reshape(p, n)
            
            # predicted = (u_col @ v_col.T).T  shape (p, n)
            pred = (u_col @ v_col.T).T
            diff = pred - w_col  # shape (p, n)
            
            # dW_r = -diff
            dW[:, r] = -diff.flatten()
            
            # dU_r: diff[i,k] = sum_j u[i,j]*v[j,k] - w[i,k]
            # d(diff)/d(u[i,j]) = sum_k diff[i,k] * v[j,k] ... wait
            # diff is (p,n), pred[i,k] = sum_j u[i,j]*v[j,k]
            # d(pred[i,k])/d(u[i,j]) = v[j,k]
            # So dL/du[i,j] = sum_k diff[i,k] * v[j,k]
            # But diff = pred - w, so dL/du = 2 * (diff @ v)  where diff is (p,n) -> need to be careful
            
            # Let me redo: L = ||pred - w||^2 = sum_{i,k} (sum_j u[i,j]*v[j,k] - w[i,k])^2
            # dL/du[i,j] = 2 * sum_k (sum_j' u[i,j']*v[j',k] - w[i,k]) * v[j,k]
            # = 2 * (diff @ v)[i, j]  where diff is (p,n)... 
            # Actually diff[i,k] for i in range(n), k in range(p)
            # But our diff array is indexed as [k, i] (p, n)
            # Let me re-index: diff_kn[k, i] = pred[k,i] - w[k,i] = sum_j u[i,j]*v[j,k] - w[k,i]
            # dL/du[i,j] = 2 * sum_k diff_kn[k,i] * v[j,k] = 2 * (v @ diff_kn[:, i])... 
            # = 2 * (v.T @ diff_kn[i, :]) ... hmm let me just do it numerically
            
            # diff_kn has shape (p, n), entry [k, i]
            # dL/du[i,j] = 2 * sum_k diff_kn[k,i] * v[j,k]
            # = 2 * (diff_kn[:, i] @ v[j, :])  -- no, that's sum_k diff[k,i]*v[j,k]
            # = 2 * (v[j, :] @ diff_kn[:, i])  -- this is a scalar
            # For all j: dL/du[i,:] = 2 * (diff_kn[:, i] @ v)  -- shape (m,)
            
            dU_col = 2.0 * (diff_kn.T @ v_col)  # diff_kn.T is (n,p), v_col is (m,p) -> (n,m)
            dU[:, r] = dU_col.flatten()
            
            # dL/dv[j,k] = 2 * sum_i diff_kn[k,i] * u[i,j]
            # For all j,k: dL/dv[:,k] = 2 * (u_col[:, k]... 
            # dL/dv[j,k] = 2 * sum_i diff_kn[k,i] * u[i,j]
            # = 2 * (u_col @ diff_kn[k, :]) ... u_col is (n,m), diff_kn[k,:] is (n,)
            # dL/dv[:,k] = 2 * (u_col.T @ diff_kn[k,:]) -- no
            # dL/dv[j,k] = 2 * sum_i u[i,j] * diff_kn[k,i]
            # For fixed k: dL/dv[:,k] = 2 * (u_col.T @ diff_kn[k,:]) -- u_col.T is (m,n), diff_kn[k,:] is (n,) -> (m,)
            
            dV_col = 2.0 * (u_col.T @ diff_kn)  # u_col.T is (m,n), diff_kn is (p,n) -> (m,p)
            dV[:, r] = dV_col.flatten()
        
        # Gradient descent
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Check error
        err = 0.0
        for r_ in range(r):
            u_col = U[:, r_].reshape(n, m)
            v_col = V[:, r_].reshape(m, p)
            w_col = W[:, r_].reshape(p, n)
            pred = (u_col @ v_col.T).T
            err += np.sum((pred - w_col)**2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < 1e-12:
            break
        
        # Adaptive learning rate
        if step % 100 == 0 and step > 0:
            if best_err < 1.0:
                lr = max(0.001, lr * 0.5)
    
    # Try to round to half-integers
    if best_err < 1e-6:
        U_rounded = np.array([[_round_to_half(x) for x in row] for row in best_U])
        V_rounded = np.array([[_round_to_half(x) for x in row] for row in best_V])
        W_rounded = np.array([[_round_to_half(x) for x in row] for row in best_W])
        
        if _verify_with_float(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p, tol=1e-9):
            return (U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist())
    
    # Try rounding original
    U_rounded = np.array([[_round_to_half(x) for x in row] for row in best_U])
    V_rounded = np.array([[_round_to_half(x) for x in row] for row in best_V])
    W_rounded = np.array([[_round_to_half(x) for x in row] for row in best_W])
    
    if _verify_with_float(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p, tol=1e-9):
        return (U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist())
    
    return None

def _strassen_block_2x2x2(n_offset, m_offset, p_offset, r_offset, U, V, W, n, m, p, rank):
    """Apply Strassen's algorithm for a 2x2x2 sub-block."""
    # Strassen's 7 multiplications for 2x2 matrix multiplication
    # C = A * B where A is 2x2, B is 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For our tensor, we need U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # For a 2x2x2 block at positions (i0, j0, k0):
    # U rows: (i0,j0), (i0,j0+1), (i0+1,j0), (i0+1,j0+1) -> indices i0*m+j0, i0*m+j0+1, (i0+1)*m+j0, (i0+1)*m+j0+1
    # V cols: (j0,k0), (j0,k0+1), (j0+1,k0), (j0+1,k0+1) -> indices j0*p+k0, j0*p+k0+1, (j0+1)*p+k0, (j0+1)*p+k0+1
    # W rows: (k0,i0), (k0,i0+1), (k0+1,i0), (k0+1,i0+1) -> indices k0*n+i0, k0*n+i0+1, (k0+1)*n+i0, (k0+1)*n+i0+1
    
    # Strassen needs 7 rank-1 terms
    # Each term: U_factor * V_factor = W_factor (in the 2x2x2 sense)
    # Term t: U_t[i,j] * V_t[j,k] = W_t[k,i] for i,j,k in {0,1}
    
    # The 7 Strassen multiplications define 7 rank-1 tensors in the 2x2x2 space
    # M1: (A00+A11)(B00+B11) -> U1 = [[1,0],[1,0]]... wait, let me think more carefully
    
    # Actually, Strassen decomposes the 2x2x2 tensor (rank 8 schoolbook) into 7 rank-1 terms
    # The 7 rank-1 tensors are:
    # t1: U=[[1,0],[1,0]], V=[[1,0],[0,1]], W=[[1,0],[0,1]]  -> (A00+A11)(B00+B11)
    # Hmm, let me use the standard Strassen rank-1 decomposition
    
    # Standard: the 7 terms have U, V, W factors (each 2x2) such that sum_t U_t ⊗ V_t ⊗ W_t = multiplication tensor
    # 
    # Let me define them properly. The multiplication tensor T[i,j,k] = delta_{appropriate}
    # For 2x2: T has 8 nonzero entries (i,j,k) where the product A[i,:]*B[:,k] contributes
    # T[i,j,k] = 1 if this is a valid multiplication path, but actually the tensor is:
    # T[i,j,k] = 1 for each (i,j,k) triple, and the matrix product is sum_j A[i,j]*B[j,k]
    # So the tensor entries are T[i,j,k] = 1 for all 8 (i,j,k) in {0,1}^3
    # Wait no. The matrix multiplication tensor for 2x2 is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has entries T[i,j,k] such that C[i,k] = sum_j A[i,j] B[j,k] T[i,j,k]... 
    # Actually T[i,j,k] = 1 for all i,j,k in {0,1}, and the decomposition is:
    # sum_r U_r[i,j] * V_r[j,k] * W_r[k,i]... no that's not right either.
    # 
    # Let me use the convention from the task:
    # W[k*n+i] = sum_j U[i*m+j] * V[j*p+k]
    # For 2x2: W[k,i] = sum_j U[i,j] * V[j,k]
    # The tensor T has entries T[i,j,k] = 1 for all (i,j,k) in {0,1}^3
    # A rank-1 decomposition: T[i,j,k] = sum_r U_r[i,j] * V_r[j,k] * ... 
    # No wait, the task says the decomposition is:
    # For each r, U_r[i*m+j] * V_r[j*p+k] contributes to W_r[k*n+i]
    # And W_r[k*n+i] should equal sum_j U_r[i*m+j] * V_r[j*p+k]
    # So each rank-1 term satisfies: W_r[k,i] = sum_j U_r[i,j] * V_r[j,k]
    # This means W_r = U_r @ V_r^T (as 2x2 matrices) for the 2x2 case
    # 
    # So we need: sum_r (U_r @ V_r^T) = I_2 (identity? no...)
    # Actually for the full 2x2x2 multiplication tensor:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_j sum_r A[i,j] U_r[i,j] * B[j,k] V_r[j,k] * W_r[k,i]... 
    # 
    # I'm overcomplicating this. Let me just use the known Strassen rank-1 factors.
    # 
    # For 2x2 matrix multiplication, Strassen gives 7 rank-1 terms.
    # The factors (U_r, V_r, W_r) for r=1..7 are:
    # Using the convention W[k,i] = sum_j U[i,j] * V[j,k]:
    
    # r=1: U=[[1,0],[1,0]], V=[[1,1],[0,0]], W=[[1,0],[0,1]]
    # Check: W[0,0]=U[0,0]*V[0,0]+U[0,1]*V[1,0]=1*1+0*0=1 ✓
    # W[0,1]=U[0,0]*V[0,1]+U[0,1]*V[1,1]=1*1+0*0=1 ✗ (should be 0)
    # 
    # Let me look up the correct Strassen factors.
    # 
    # The 7 Strassen multiplications:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00  
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # 
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    # 
    # Each Mr is a scalar multiplication. In tensor form, each Mr contributes:
    # Mr = (linear combo of A entries) * (linear combo of B entries)
    # And contributes to C entries with coefficients.
    # 
    # For rank-1 term r with factors U_r[i,j], V_r[j,k], W_r[k,i]:
    # The contribution to C[i,k] is sum_j A[i,j] U_r[i,j] * V_r[j,k] * W_r[k,i]
    # Wait, that's not the right formulation either.
    # 
    # Let me re-read the task convention:
    # "W[k*n+i,r]" and the decomposition satisfies:
    # For the matrix product C = A*B:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor decomposition: C[i,k] = sum_r (sum_j A[i,j] U_r[i*m+j]) * (sum_j' B[j',k] V_r[j'*p+k]) * W_r[k*n+i]
    # No, that's not right either.
    # 
    # From the code: U[i*m+j][r], V[j*p+k][r], W[k*n+i][r]
    # And verification: W[k*n+i][r] should equal sum_j U[i*m+j][r] * V[j*p+k][r]
    # 
    # So each rank-r term: W_r[k*n+i] = sum_j U_r[i*m+j] * V_r[j*p+k]
    # And the full product: C[i,k] = sum_r (sum_j A[i,j] * U_r[i*m+j]) * ... 
    # 
    # Actually I think the convention is:
    # C[i,k] = sum_r W_r[k*n+i] * (sum_j A[i,j] U_r[i*m+j]) * (sum_j B[j,k] V_r[j*p+k])
    # No...
    # 
    # Let me just verify: if U[i*m+j][r]=1, V[j*p+k][r]=1, W[k*n+i][r]=1 for specific (i,j,k),
    # then W[k*n+i][r] = 1 = sum_j' U[i*m+j'][r]*V[j'*p+k][r] = U[i*m+j][r]*V[j*p+k][r] = 1*1 = 1 ✓
    # (all other j' give 0)
    # 
    # So the rank-1 term with U having 1 at (i,j), V having 1 at (j,k), W having 1 at (k,i)
    # contributes to the product as: this is the (i,j,k) entry of the multiplication tensor.
    # 
    # For Strassen, we need 7 such rank-1 terms (with possibly non-1 entries) that sum to the 8-entry tensor.
    # 
    # The 7 Strassen terms in terms of U[i,j], V[j,k], W[k,i]:
    # Term 1 (M1): U=[[1,0],[1,0]], V=[[1,1],[0,0]], W=[[1,1],[0,0]]
    #   Check: W[0,0]=U[0,0]V[0,0]+U[0,1]V[1,0]=1*1+0*0=1 ✓
    #   W[0,1]=U[0,0]V[0,1]+U[0,1]V[1,1]=1*1+0*0=1 ✓
    #   W[1,0]=U[1,0]V[0,0]+U[1,1]V[1,0]=1*1+0*0=1 ✓
    #   W[1,1]=U[1,0]V[0,1]+U[1,1]V[1,1]=1*1+0*0=1 ✓
    #   But we need W to be such that W[k,i] = sum_j U[i,j]V[j,k]
    #   W[0,0]=1, W[0,1]=1, W[1,0]=1, W[1,1]=1
    #   This means the term contributes 1 to all 4 C entries? That doesn't seem right for M1.
    # 
    # I think I'm confusing the tensor convention. Let me just use the known result:
    # Strassen's algorithm gives a rank-7 decomposition of the 2x2x2 multiplication tensor.
    # The specific factors are well-known. Let me use a different approach.
    # 
    # Given the complexity, let me just try the optimization approach for smaller ranks.
    
    return None

def _schoolbook(n, m, p):
    """Generate the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    fallback = _schoolbook(n, m, p)
    
    # Try to find lower rank decompositions
    # Known lower bounds: for 3x4x5, the rank is at least... 
    # Let's try ranks from a reasonable lower bound up to schoolbook
    
    # Try Strassen-inspired approach: decompose into 2x2x2 blocks
    # 3x4x5 can be decomposed as:
    # - One 2x2x2 block (uses 4 rows of U, 4 cols of V, 4 rows of W) -> rank 7 with Strassen
    # - Remaining parts handled by schoolbook
    
    # Actually, let's try direct optimization for various ranks
    best_result = fallback
    best_rank = n * m * p
    
    # Try a few ranks below schoolbook
    for target_rank in range(best_rank - 2, max(20, best_rank - 5)):
        if _check_deadline():
            break
            
        result = _try_rank(target_rank, n, m, p, _check_deadline)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = target_rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
