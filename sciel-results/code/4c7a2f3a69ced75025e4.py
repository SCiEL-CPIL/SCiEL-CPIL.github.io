"""Tensor decomposition for 3x3x7 matrix multiplication, rank-minimal search."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s safety margin within 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 63
    
    # Build the multiplication tensor T[i, j, k] where output = sum over j
    # T has indices: (i, k, j) -> we want U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # The tensor entry is 1 when the triple (i,j,k) is valid (always valid here)
    
    # Target: for each (i, k), sum_j A[i,j]*B[j,k] = C[i,k]
    # Decomposition: C[i,k] = sum_c U[i*m+j_c, c]*V[j_c*p+k, c]*W[k*n+i, c]
    # where j_c is implicit in U's row index
    
    # Let's use a direct approach: search for low-rank decompositions
    # using random perturbations and rank minimization via iterative methods
    
    best_rank = schoolbook_rank
    best_result = _schoolbook(n, m, p)
    
    # Try to find lower rank using random algebraic method
    # The tensor T_{i,k,j} = delta (always 1 for valid indices)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    
    # Strategy: parameterize and use random starts with gradient-free optimization
    # on the rank
    
    # First, try known lower bounds:
    # For 3x3x7, the tensor rank is at most min(63, ...)
    # Let's try ranks from a reasonable upper bound downward
    
    # Use a smarter approach: exploit that the 3x3x7 tensor can be seen as
    # a block structure. The "j" dimension (middle) has size 3.
    
    # Alternative: use the fact that rank(3x3xp) <= 9*p for schoolbook,
    # but we can do better. Let's try a randomized search.
    
    # Randomized Bini-Capovani style: 
    # Pick random U, V, solve for W, check if it works, then try to reduce rank.
    
    def _verify(U, V, W, n, m, p, tol=1e-10):
        """Verify that the decomposition exactly reproduces matrix multiplication."""
        # C[i,k] = sum_c sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # For each (i,k), the sum over j and c should equal 1 if we're decomposing
        # the identity-like tensor, but actually we need:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over j = delta-like
        # Actually the tensor is T[i,j,k] = 1 for all valid (i,j,k)
        # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = U_arr.shape[1]
        
        # Compute for each (i, j, k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def _try_rank(rank, n, m, p, max_iters=500, seed=None):
        """Try to find a decomposition of given rank using random initialization."""
        if seed is not None:
            np.random.seed(seed)
        
        # Random initialization
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Scale to make initial product closer to 1
        # For each (i,j,k), product should be ~1
        # Simple scaling: normalize so that typical product is ~1
        scale = (n*m*p)**(1.0/6.0)
        U *= scale
        V *= scale
        W *= scale
        
        # Iterative refinement: for each c, update one factor while fixing others
        # This is like ALS (alternating least squares) but for the tensor constraint
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update W: for each (i,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all j
            # This is a linear system in W for fixed U, V
            # For fixed (i,k): sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # Let M_{c,c'} = sum_j U[i*m+j,c]*V[j*p+k,c']
            # Then sum_c' M_{c,c'} * W[k*n+i, c'] ... no, W is indexed by (k*n+i, c)
            # Actually: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] summed over j = 1
            # = sum_c (sum_j U[i*m+j,c]*V[j*p+k,c]) * W[k*n+i,c]
            # Let A_{i,k,c} = sum_j U[i*m+j,c]*V[j*p+k,c]
            # Then sum_c A_{i,k,c} * W[k*n+i,c] = 1 for all (i,k)
            # This is a linear system: for each (i,k), a linear equation in W[k*n+i,:]
            
            # Build linear system for W
            # Variables: W[k*n+i, c] for all k, i, c -> p*n*rank variables
            # Equations: for each (i,k): sum_c A_{i,k,c} * W[k*n+i,c] = 1
            # But A_{i,k,c} = sum_j U[i*m+j,c]*V[j*p+k,c]
            
            # Actually this is underdetermined if rank > 1. Let's use a different approach.
            
            # Better: use the fact that we can solve for W given U, V via:
            # For each (i,k): sum_c [sum_j U[i*m+j,c]*V[j*p+k,c]] * W[k*n+i,c] = 1
            # This gives p*n linear equations in p*n*rank unknowns.
            # We can solve the least-squares problem.
            
            # Build the matrix for W
            # Row for each (i,k): coefficients are A_{i,k,c} for c=0..rank-1, placed at W[k*n+i, c]
            num_ik = p * n
            A_mat = np.zeros((num_ik, p*n*rank))
            rhs = np.ones(num_ik)
            
            idx = 0
            for i in range(n):
                for k in range(p):
                    for c in range(rank):
                        A_ik_c = 0.0
                        for j in range(m):
                            A_ik_c += U[i*m+j, c] * V[j*p+k, c]
                        A_mat[idx, (k*n+i)*rank + c] = A_ik_c
                    idx += 1
            
            # Solve least squares
            try:
                W_flat, _, _, _ = np.linalg.lstsq(A_mat, rhs, rcond=None)
                W = W_flat.reshape(p*n, rank)
            except:
                return None
            
            # Similarly update U
            # For each (i,j): sum_c U[i*m+j,c] * [sum_k V[j*p+k,c]*W[k*n+i,c]] = 1
            # Let B_{i,j,c} = sum_k V[j*p+k,c]*W[k*n+i,c]
            # sum_c U[i*m+j,c] * B_{i,j,c} = 1 for all (i,j)
            num_ij = n * m
            B_mat = np.zeros((num_ij, n*m*rank))
            rhs2 = np.ones(num_ij)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for c in range(rank):
                        B_ij_c = 0.0
                        for k in range(p):
                            B_ij_c += V[j*p+k, c] * W[k*n+i, c]
                        B_mat[idx, (i*m+j)*rank + c] = B_ij_c
                    idx += 1
            try:
                U_flat, _, _, _ = np.linalg.lstsq(B_mat, rhs2, rcond=None)
                U = U_flat.reshape(n*m, rank)
            except:
                return None
            
            # Similarly update V
            # For each (j,k): sum_c V[j*p+k,c] * [sum_i U[i*m+j,c]*W[k*n+i,c]] = 1
            # Let C_{j,k,c} = sum_i U[i*m+j,c]*W[k*n+i,c]
            num_jk = m * p
            C_mat = np.zeros((num_jk, m*p*rank))
            rhs3 = np.ones(num_jk)
            idx = 0
            for j in range(m):
                for k in range(p):
                    for c in range(rank):
                        C_jk_c = 0.0
                        for i in range(n):
                            C_jk_c += U[i*m+j, c] * W[k*n+i, c]
                        C_mat[idx, (j*p+k)*rank + c] = C_jk_c
                    idx += 1
            try:
                V_flat, _, _, _ = np.linalg.lstsq(C_mat, rhs3, rcond=None)
                V = V_flat.reshape(m*p, rank)
            except:
                return None
            
            # Check convergence
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        err += abs(s - 1.0)
            if err < 1e-10:
                # Found a valid decomposition!
                return U, V, W
        
        return None
    
    # Try ranks from a reasonable upper bound down
    # Known: rank(3x3x7) <= 63 (schoolbook)
    # Lower bound: at least max(9, 21, 21) = 21 (from dimension arguments)
    # Actually, the tensor rank is at least max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # But more carefully: the border rank gives better bounds.
    
    # Let's try ranks from 30 down to 21
    # For each rank, try a few random starts
    
    for rank in range(40, 20, -1):
        if time.time() > deadline:
            break
        
        if rank >= best_rank:
            continue
        
        found = False
        for trial in range(20):
            if time.time() > deadline:
                break
            
            result = _try_rank(rank, n, m, p, max_iters=200, seed=42+trial*1000+rank)
            if result is not None:
                U, V, W = result
                # Verify
                if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    best_rank = rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    found = True
                    break
        
        if found:
            # Try to go lower
            continue
    
    # If we found something better, return it; otherwise return schoolbook
    if best_rank < schoolbook_rank:
        return best_result
    else:
        return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
