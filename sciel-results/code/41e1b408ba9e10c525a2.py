"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from numpy.linalg import matrix_rank

DIMENSIONS = [4, 4, 8]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140  # 10s safety margin before 150s kill

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if we multiply A[i,j]*B[j,k]."""
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _schoolbook(n, m, p):
    """Return schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U,V,W exactly decompose the matmul tensor."""
    # For each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank_reduction(n, m, p, max_rank, time_budget=30):
    """Attempt to find a decomposition with rank <= max_rank using iterative refinement.
    
    Strategy: Start from schoolbook, randomly zero out some columns, then try to
    fix the remaining entries via least-squares on the free variables.
    """
    if time.time() > _DEADLINE:
        return None
    
    school_U, school_V, school_W = _schoolbook(n, m, p)
    full_rank = n * m * p
    
    # We'll try to find a subset of columns that still spans the tensor
    # Use a greedy approach: try removing columns one by one if the system stays solvable
    
    # Build the linear system: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # This is nonlinear in general, but we can try: fix U and V, solve for W, etc.
    
    # Alternative: use the fact that for small problems, we can do a direct search
    # over random rank-r subspaces.
    
    # Simpler approach: try to find a lower rank by exploiting structure.
    # For 4x4 * 4x8, the tensor has a special structure.
    # Let's try: can we do it with rank < 128?
    
    # Use a randomized approach: pick random rank-r U and V, then solve for W.
    # W[k*n+i, c] = sum over (i',j',k') such that... this gets complex.
    
    # Actually, let's use a different strategy: 
    # The tensor T[i,j,k] = 1 for all i,j,k is a very simple tensor.
    # Its rank as a tensor is min over decompositions.
    # For T[i,j,k] = 1, we can write it as:
    # T[i,j,k] = sum_c u_c[i] * v_c[j] * w_c[k]
    # where u_c[i] = 1 for all i, v_c[j] = 1 for all j, w_c[k] = 1 for all k
    # gives rank 1! But wait, that's for the tensor T[i,j,k] = 1.
    # 
    # BUT the matrix multiplication tensor is NOT T[i,j,k] = 1.
    # The matmul tensor is: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor is T[i,j,k] = 1 (the coefficient of A[i,j]*B[j,k] in the product).
    # Yes! The matmul tensor IS T[i,j,k] = 1 for all i,j,k.
    # 
    # Wait, but the decomposition constraint is:
    # (A*B)[i,k] = sum_c U[i*m+j... no wait.
    
    # Let me re-read the problem. The decomposition is:
    # (A*B)[i,k] = sum_c U[i, c] * V[c, k] ... no.
    # 
    # The standard tensor decomposition for matmul:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_c (sum_j A[i,j] * U[j, c]) * (sum_k B[j,k] * V[c, k])... 
    # 
    # Actually the standard form is:
    # (A*B)[i,k] = sum_c U[i, c] * V[c, k] where U depends on A and V on B.
    # But that's not right either.
    
    # The correct tensor decomposition form:
    # (A*B)[i,k] = sum_c (sum_j A[i,j] * U[j, c]) * (sum_k B[j,k] * V[c, k])
    # No, that's not standard.
    
    # The standard Strassen-style decomposition:
    # (A*B)[i,k] = sum_c M_c(A) * M_c(B) * M_c(C)
    # where M_c are linear maps.
    
    # In the given format:
    # U has shape (n*m, rank): U[i*m+j, c] is the coefficient for A[i,j] in the c-th product
    # V has shape (m*p, rank): V[j*p+k, c] is the coefficient for B[j,k] in the c-th product  
    # W has shape (p*n, rank): W[k*n+i, c] is the coefficient for C[i,k] in the c-th sum
    
    # So: C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
    # Wait, that's not quite right. Let me think again.
    
    # The constraint is:
    # For all A, B: (A*B)[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j' V[j'*p+k, c] * B[j',k])
    
    # Expanding: (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    
    # So we need: sum_j A[i,j]*B[j,k] = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c]*A[i,j]) * (sum_j' V[j'*p+k,c]*B[j',k])
    
    # This must hold for all A, B. So for each (i,k), the bilinear form on the left must equal the sum on the right.
    
    # The left side is a sum of rank-1 bilinear forms: for each j, A[i,j]*B[j,k].
    # The right side is a sum of rank-1 bilinear forms: for each c, (sum_j U[i*m+j,c]*A[i,j]) * (sum_j' V[j'*p+k,c]*B[j',k]) * W[k*n+i,c]
    
    # So we need to decompose the matmul tensor.
    
    # For the specific case where all entries are 1, the tensor is:
    # T[i,j,k] = 1 (coefficient of A[i,j]*B[j,k] in C[i,k])
    
    # This is indeed the all-ones tensor! And its tensor rank is 1:
    # T[i,j,k] = 1*1*1 for all i,j,k
    # So U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    
    # Let me verify: C[i,k] = W[k*n+i,0] * (sum_j U[i*m+j,0]*A[i,j]) * (sum_j' V[j'*p+k,0]*B[j',k])
    # = 1 * (sum_j 1*A[i,j]) * (sum_j' 1*B[j',k])
    # = (sum_j A[i,j]) * (sum_j' B[j',k])
    
    # But (A*B)[i,k] = sum_j A[i,j]*B[j,k], which is NOT equal to (sum_j A[i,j])*(sum_j' B[j',k]) in general!
    
    # So the all-ones tensor rank-1 decomposition does NOT work for matmul.
    # The issue is that the matmul tensor is not just "all ones" in the way I was thinking.
    
    # Let me reconsider. The matmul tensor T has T[i,j,k] = 1 meaning:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor decomposition says:
    # C[i,k] = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c]*A[i,j]) * (sum_j' V[j'*p+k,c]*B[j',k])
    
    # For this to equal sum_j A[i,j]*B[j,k] for all A,B, we need:
    # sum_j A[i,j]*B[j,k] = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c]*A[i,j]) * (sum_j' V[j'*p+k,c]*B[j',k])
    
    # The LHS is a bilinear form with coefficient 1 for each (i,j,k) term A[i,j]*B[j,k].
    # The RHS is a sum of products of linear forms.
    
    # For a fixed (i,k), the LHS is sum_j A[i,j]*B[j,k], which involves j as a free index.
    # The RHS for fixed (i,k) is sum_c W[k*n+i,c] * (sum_j U[i*m+j,c]*A[i,j]) * (sum_j' V[j'*p+k,c]*B[j',k])
    
    # For these to be equal as bilinear forms in A and B (with the constraint that only A[i,*] and B[*,k] appear), we need:
    # The coefficient of A[i,j]*B[j,k] on the RHS must be 1 for each j, and the coefficient of A[i,j]*B[j',k] must be 0 for j != j'.
    
    # On the RHS, the coefficient of A[i,j]*B[j',k] is:
    # sum_c W[k*n+i,c] * U[i*m+j,c] * V[j'*p+k,c]
    
    # We need: sum_c W[k*n+i,c] * U[i*m+j,c] * V[j'*p+k,c] = delta(j,j')
    
    # This is the actual constraint! It's not just "all ones tensor".
    
    # So the problem is to find U, V, W with minimal rank such that:
    # For all i, k, j, j': sum_c W[k*n+i,c] * U[i*m+j,c] * V[j'*p+k,c] = delta(j,j')
    
    # This is a much more constrained problem. The schoolbook solution has rank n*m*p = 128.
    
    # Can we do better? This is related to the rank of the matrix multiplication tensor.
    # For 4x4 * 4x8, the known upper bounds are... let me think.
    
    # The schoolbook is 128. Strassen's method for 4x4 is 49 (for square). 
    # For 4x4 * 4x8, we can use Strassen for 4x4 and extend.
    
    # Actually, let's just try a computational search for a lower rank.
    
    # Given the time constraints, let's try a simple approach:
    # 1. Start with schoolbook (rank 128)
    # 2. Try to reduce rank by checking if we can merge columns
    
    # For now, return schoolbook as a safe baseline.
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # First, try to find a better decomposition
    best = None
    if time.time() < _DEADLINE:
        # Try rank reduction search
        result = _try_rank_reduction(n, m, p, max_rank=n*m*p - 1, time_budget=20)
        if result is not None:
            best = result
    
    # Fall back to schoolbook
    if best is None:
        U, V, W = _schoolbook(n, m, p)
        # Convert to lists
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        U, V, W = best
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
