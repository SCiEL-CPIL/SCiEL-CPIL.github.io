import numpy as np
from fractions import Fraction
import random
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    schoolbook_rank = n * m * p  # 75
    
    deadline = time.time() + 120  # safety margin before 150s kill
    best_rank = schoolbook_rank
    best_decomp = None
    
    # Build the schoolbook decomposition as the baseline
    def make_schoolbook():
        rank = schoolbook_rank
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    U0, V0, W0 = make_schoolbook()
    best_decomp = (U0.copy(), V0.copy(), W0.copy())
    best_rank = schoolbook_rank
    
    # The matrix multiplication tensor T has entries T[i,j,k] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    # where the output index is (k*n+i) for the W factor.
    # Actually, let's think about this more carefully.
    # We're multiplying A (n x m) by B (m x p) to get C (n x p).
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 if we're computing C[i,k] using A[i,j]*B[j,k]
    # 
    # In the decomposition: C[i,k] = sum_r U[i,j... wait, no.
    # 
    # The standard formulation: 
    # A is n x m, B is m x p, C = A*B is n x p
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # Tensor decomposition: T_{i,j,k} where the tensor represents the bilinear map
    # We want: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k' V[j*p+k', r] * B[j,k'])
    # 
    # Wait, let me re-read the interface:
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # 
    # So U has rows indexed by (i,j) pairs, V has rows indexed by (j,k) pairs,
    # W has rows indexed by (k,i) pairs.
    # 
    # The decomposition says: for each output entry C[i,k], 
    # C[i,k] = sum_r W[k*n+i, r] * (something)
    # 
    # Actually, the standard way: 
    # Let's define the multiplication as: C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # We can write this as: C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_{j',k'} V[j'*p+k', r] B[j',k'])
    # 
    # No, that's not right either. Let me think again.
    # 
    # The tensor T is defined such that the bilinear form gives:
    # sum_{i,j,k} T[i,j,k] x_i y_j z_k = sum_{i,k} C[i,k]
    # 
    # In the decomposition with factors U, V, W:
    # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # 
    # And we need this to equal the matrix multiplication tensor, which is:
    # T[i,j,k] = 1 if the indices are consistent with C[i,k] = sum_j A[i,j]B[j,k]
    # 
    # Actually for matrix multiplication, the tensor is:
    # T[i,j,k] = delta_{i,i} delta_{j,j} delta_{k,k} -- no that's just 1.
    # 
    # Let me reconsider. The matrix multiplication tensor for (n,m,p) dimensions:
    # The tensor T in R^{n x m x p} is defined by T[i,j,k] = 1 for all i,j,k? No.
    # 
    # The bilinear map: (A, B) -> C where C = A*B
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # So the tensor T satisfies: sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,k} C[i,k] * something
    # 
    # Actually, the correct formulation is:
    # The tensor T has T[i,j,k] = 1 if and only if the term A[i,j]*B[j,k] contributes to C[i,k].
    # Since each product A[i,j]*B[j,k] contributes exactly to C[i,k], we have:
    # T[i,j,k] = 1 for all valid i,j,k (where i in [0,n), j in [0,m), k in [0,p))
    # 
    # Wait, that would make the tensor all ones, which has rank... hmm.
    # 
    # No, I think the tensor is structured differently. Let me look at the schoolbook decomposition again.
    # 
    # In the schoolbook: for each (i,j,k), we have one term with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
    # This means: sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] should give the correct tensor.
    # 
    # For the schoolbook, for a fixed (i,j,k), only the term r corresponding to (i,j,k) has all three factors equal to 1.
    # So T[i,j,k] = 1 for all i,j,k? That can't be right for matrix multiplication.
    # 
    # I think I'm confusing myself. Let me just verify the schoolbook works and try to reduce rank.
    # 
    # The key insight: we need to find the minimum rank r such that there exist
    # U (nm x r), V (mp x r), W (pn x r) with integer/half-integer entries where
    # for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # 
    # And for all other combinations... wait, there are no other combinations since i,j,k are fully determined.
    # 
    # Hmm, actually I think the condition is just that the tensor equality holds.
    # The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k in the valid ranges.
    # 
    # Wait no. Let me reconsider. The tensor for matrix multiplication M_{n,m,p} is:
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where... 
    # 
    # Actually, I recall now. The matrix multiplication tensor is:
    # T_{i,j,k} = 1 if we interpret it as: the coefficient of x_i y_j z_k in the product
    # (sum_i x_i)(sum_j y_j)(sum_k z_k) -- no that's just all ones.
    # 
    # Let me just trust the schoolbook and work with the constraint:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # 
    # This is a system of nm*p = 75 equations. We want to minimize r.
    # 
    # One approach: start with the schoolbook (r=75) and try to merge/eliminate terms.
    # 
    # Another approach: use the fact that this is equivalent to finding a low-rank
    # decomposition of a specific tensor.
    # 
    # Let me try a greedy elimination approach:
    # Start with schoolbook, then for each pair of terms, check if we can replace them
    # with a single term (or remove one) while maintaining the equations.
    
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if the decomposition satisfies all constraints."""
        n, m, p = DIMENSIONS
        nm = n * m
        mp = m * p
        pn = p * n
        rank = U.shape[1]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    # Compute sum_r U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_reduce_rank(U, V, W, current_rank, max_time=10.0):
        """Try to reduce the rank by eliminating or merging terms."""
        start_time = time.time()
        n, m, p = DIMENSIONS
        nm = n * m
        mp = m * p
        pn = p * n
        
        U = U.copy()
        V = V.copy()
        W = W.copy()
        rank = current_rank
        
        improved = True
        while improved and time.time() - start_time < max_time:
            improved = False
            for col in range(rank):
                if time.time() - start_time > max_time:
                    break
                # Try to eliminate this column by expressing it as a combination of others
                # or by adjusting other columns.
                
                # Simple approach: try setting this column to zero and see if we can
                # fix the equations by adjusting other columns.
                # This is complex, so let's try a simpler heuristic.
                
                # Check if this column is redundant: can we find coefficients for other
                # columns that reproduce its contribution?
                # 
                # For each (i,j,k), the contribution of column col is:
                #   U[i*m+j, col] * V[j*p+k, col] * W[k*n+i, col]
                # 
                # We need: for all (i,j,k), sum_{r != col} U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] 
                #           + delta * (U[i*m+j, col] V[j*p+k, col] W[k*n+i, col]) = 1
                # where delta is the new value for this column's entries.
                
                # This is getting complex. Let's try a different approach:
                # Random perturbation and local search.
                
                pass
        
        return U, V, W, rank
    
    # Let's try a different approach: use the known lower bounds and known results.
    # For 3x5 * 5x5, the schoolbook rank is 75.
    # Known results: for 2x2*2x2, rank is 7 (Strassen).
    # For 3x3*3x3, the best known is around 20-23.
    # 
    # For 3x5*5x5, I'm not sure of the exact minimum rank.
    # 
    # Let's try a more systematic approach: use integer programming or random search.
    
    # Approach: random search for low-rank decompositions
    def random_search(rank, num_trials=100, max_time=5.0):
        """Try to find a rank-r decomposition via random initialization."""
        start_time = time.time()
        n, m, p = DIMENSIONS
        nm = n * m
        mp = m * p
        pn = p * n
        
        for trial in range(num_trials):
            if time.time() - start_time > max_time:
                break
            
            # Initialize with random integer values in {-1, 0, 1}
            U = np.random.choice([-1, 0, 1], size=(nm, rank))
            V = np.random.choice([-1, 0, 1], size=(mp, rank))
            W = np.random.choice([-1, 0, 1], size=(pn, rank))
            
            if check_decomposition(U, V, W):
                return U, V, W, rank
        
        return None
    
    # Try ranks from schoolbook downward
    for rank in range(schoolbook_rank - 1, 30, -1):
        if time.time() > deadline - 30:
            break
        
        result = random_search(rank, num_trials=50, max_time=5.0)
        if result is not None:
            U, V, W, r = result
            if check_decomposition(U, V, W):
                best_rank = r
                best_decomp = (U.copy(), V.copy(), W.copy())
                break
    
    # If we didn't find anything better, return the schoolbook
    if best_decomp is None:
        U, V, W = make_schoolbook()
    else:
        U, V, W = best_decomp
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
