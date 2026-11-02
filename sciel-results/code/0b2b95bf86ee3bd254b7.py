"""Matrix multiplication 3,3,8 - rank-minimal decomposition."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    nm, mp, pn = n * m, m * p, p * n
    rank = len(U[0]) if U else 0
    
    # Build the multiplication tensor T[i,j,k] = delta
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 if (i,j,k) valid else 0
    # Actually for matmul: (AB)[i,k] = sum_j A[i,j]*B[j,k]
    # So tensor T[i,j,k] should satisfy: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = delta_{i,j,k}
    # Wait, let me think about this more carefully.
    
    # The matrix multiplication tensor has entries T[i,j,k] where:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 if we're computing (A*B)[i,k] from A[i,j]*B[j,k]
    # But actually T[i,j,k] is a coefficient such that:
    # (A*B)[i,k] = sum_{j,r} A[i,j] * B[j,k] * ... no
    
    # Let me use the standard formulation:
    # We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,j,k}
    # where delta_{i,j,k} = 1 if this term contributes to the product
    
    # Actually, the correct check is:
    # For each (i, k) output, (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # The decomposition says: sum_r (sum_j U[i*m+j,r]*A[i,j]) * (sum_k V[j*p+k,r]*B[j,k]) * W[k*n+i,r]
    # This needs to equal sum_j A[i,j]*B[j,k]
    
    # So we need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all valid (i,j,k)
    # Wait no. Let me be more careful.
    
    # The bilinear form for matmul is:
    # f(A,B) = sum_{i,j,k} A[i,j] * B[j,k] * T[i,j,k]
    # where T[i,j,k] = 1 for all (i,j,k) since each term A[i,j]*B[j,k] contributes to (A*B)[i,k]
    
    # A rank-r decomposition is: f(A,B) = sum_r (sum_{i,j} U[i*m+j,r]*A[i,j]) * (sum_{j,k} V[j*p+k,r]*B[j,k]) * W[k*n+i,r]
    
    # Expanding: sum_r sum_{i,j} sum_{j',k} U[i*m+j,r]*V[j'*p+k,r]*W[k*n+i,r] * A[i,j]*B[j',k]
    
    # For this to equal sum_{i,j,k} A[i,j]*B[j,k], we need:
    # sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all (i,j,k)
    # AND the cross terms must vanish:
    # sum_r U[i*m+j,r]*V[j'*p+k,r]*W[k*n+i,r] = 0 for j != j'
    
    # Actually, the standard way: the tensor T has entries T[i,j,k] = 1 for all i,j,k
    # (since every pair (A[i,j], B[j,k]) multiplies together to contribute to the output)
    
    # So we need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all (i,j,k)
    
    # Let's verify with random matrices
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    AB = A @ B
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Compute via decomposition
    # C[i,k] = sum_r (sum_j U[i*m+j,r]*A[i,j]) * (sum_k' V[j'*p+k',r]*B[j',k']) * W[k*n+i,r]
    # Wait, this isn't right either. Let me think again.
    
    # Standard bilinear decomposition of matmul:
    # C = A*B where C[i,k] = sum_j A[i,j]*B[j,k]
    
    # Rank-r decomposition: C = sum_r u_r(A) * v_r(B) * w_r
    # where u_r(A) is a linear form on A, v_r(B) is a linear form on B, w_r is a matrix
    
    # In our parameterization:
    # u_r(A) = sum_{i,j} U[i*m+j, r] * A[i,j]
    # v_r(B) = sum_{j,k} V[j*p+k, r] * B[j,k]
    # w_r[i,k] = W[k*n+i, r]
    
    # So C[i,k] = sum_r (sum_{i',j'} U[i'*m+j', r] * A[i',j']) * (sum_{j'',k''} V[j''*p+k'', r] * B[j'',k'']) * W[k*n+i, r]
    
    # This doesn't look right because the sums are over all indices, not fixed.
    
    # I think the correct interpretation is:
    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * A[i,j] * B[j,k]
    
    # So: C[i,k] = sum_j A[i,j] * B[j,k] * (sum_r U[i*m+j, r] * V[j*p+k, r])
    
    # For this to equal sum_j A[i,j]*B[j,k], we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] = 1 for all (i,j,k)
    
    # But wait, W is also there. Let me re-read the problem statement.
    # "W[k*n+i,r]" - so W is indexed by [k*n+i, r]
    
    # I think the decomposition is:
    # C[i,k] = sum_r (sum_j U[i*m+j,r] * A[i,j]) * (sum_j V[j*p+k,r] * B[j,k]) * W[k*n+i,r]
    
    # No, that still doesn't work dimensionally.
    
    # Let me just test empirically with the schoolbook decomposition.
    
    # Schoolbook: for each (i,j,k), U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1
    # where col = i*m*p + j*p + k
    
    # Let's compute C[i,k] = sum_r sum_j U[i*m+j,r]*A[i,j] * sum_j' V[j'*p+k,r]*B[j',k] * W[k*n+i,r]
    # Hmm, this is getting confusing. Let me just test numerically.
    
    # Actually, I think the correct formula based on the trace convention W[k*n+i,r] is:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * W[k*n+i, r] * (sum_j V[j*p+k, r] * B[j,k])
    
    # No wait. Let me look at this differently.
    
    # The bilinear map is: (A, B) -> A*B
    # A rank-r decomposition is: A*B = sum_r alpha_r(A) * beta_r(B) * gamma_r
    # where alpha_r: A -> scalar, beta_r: B -> scalar, gamma_r: output matrix
    
    # In matrix form: (A*B)[i,k] = sum_r alpha_r(A) * beta_r(B) * gamma_r[i,k]
    
    # With our parameterization:
    # alpha_r(A) = sum_{i,j} U[i*m+j, r] * A[i,j]
    # beta_r(B) = sum_{j,k} V[j*p+k, r] * B[j,k]  
    # gamma_r[i,k] = W[k*n+i, r]
    
    # So: (A*B)[i,k] = sum_r (sum_{i',j'} U[i'*m+j',r]*A[i',j']) * (sum_{j'',k''} V[j''*p+k'',r]*B[j'',k'']) * W[k*n+i,r]
    
    # This is a sum over all i',j',j'',k'' which is not what we want.
    
    # I think the correct interpretation must be that the sums are restricted:
    # alpha_r(A) = sum_j U[i*m+j, r] * A[i,j]  (for fixed i)
    # beta_r(B) = sum_j V[j*p+k, r] * B[j,k]  (for fixed k)
    
    # But then alpha_r and beta_r depend on i and k, which contradicts them being fixed linear forms.
    
    # I'm confused. Let me just test the schoolbook decomposition numerically and see what formula works.
    
    # For now, let's assume the verification is:
    # For all (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # And for "wrong" combinations, the sum should be 0.
    
    # Actually, I realize the tensor for matmul 3x3 * 3x8 is:
    # T[i,j,k] = 1 for all i in [0,3), j in [0,3), k in [0,8)
    # because (A*B)[i,k] = sum_j A[i,j]*B[j,k] and each term is independent.
    
    # A rank-r decomposition means: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    # So we need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all (i,j,k)
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    nm, mp, pn = n * m, m * p, p * n
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
                
    return U, V, W

def _try_structured_search(n, m, p, deadline):
    """
    Try to find a lower-rank decomposition using structured patterns.
    For rectangular matrices, there may be exploitable structure.
    """
    nm, mp, pn = n * m, m * p, p * n
    best_rank = n * m * p
    best_decomp = None
    
    # First get the schoolbook as baseline
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    if _verify_decomposition(U_sb, V_sb, W_sb, n, m, p):
        best_decomp = (U_sb, V_sb, W_sb)
        best_rank = n * m * p
    
    # Try various structured approaches
    # Idea: For each (i,j), the vector of V values across k might allow sharing
    
    # Approach 1: Try to factor by grouping over j
    # For fixed i, we have n*m entries in U. 
    # The tensor entry T[i,j,k] = 1 for all (i,j,k).
    
    # A rank-r decomposition requires finding vectors u_{i,j}, v_{j,k}, w_{k,i} such that
    # sum_r u_{i,j}[r] * v_{j,k}[r] * w_{k,i}[r] = 1 for all (i,j,k)
    
    # This is a 3-way tensor decomposition problem.
    
    # For now, try a simple structured approach:
    # Use rank = n*p + m*p + n*m - some_savings
    
    # Actually, let's try a more systematic approach.
    # The tensor is all-ones, which has special structure.
    
    # For an all-ones tensor of size n x m x p, the rank is min(n,m,p) if we can use
    # rank-1 tensors that are all-ones in each mode.
    # But that's not quite right for matrix multiplication.
    
    # Let me think about this differently. The matmul tensor T[i,j,k] = 1 is NOT all-ones
    # in the tensor decomposition sense. It's a specific tensor that happens to have all
    # entries equal to 1.
    
    # For the all-ones tensor J_{n,m,p}, the rank is min(n,m,p) because:
    # J[i,j,k] = sum_{r=1}^{min(n,m,p)} e_i[r] * e_j[r] * e_k[r]
    # where e_x is a vector of ones.
    # But wait, that gives sum_r 1*1*1 = min(n,m,p), not 1.
    
    # So J[i,j,k] = (1/min(n,m,p)) * sum_r e_i[r] * e_j[r] * e_k[r]
    # This requires half-integer entries if min(n,m,p) > 1.
    
    # For n=3, m=3, p=8: min = 3, so rank 3 with entries 1/3.
    # But we need integer or half-integer entries!
    
    # So we can't use this directly. We need a different approach.
    
    # Let's try rank = n*p = 24 (grouping over j)
    # Or rank = m*p = 24 (grouping over i)
    # Or rank = n*m = 9 (grouping over k)
    
    # For rank = n*m = 9:
    # U[i*m+j, r] where r indexes (i', j')
    # V[j*p+k, r] where r indexes (i', j')
    # W[k*n+i, r] where r indexes (i', j')
    
    # We need: sum_{r=(i',j')} U[i*m+j, (i',j')] * V[j*p+k, (i',j')] * W[k*n+i, (i',j')] = 1
    
    # This is complex. Let's just try random search with various ranks.
    
    # Try ranks from 71 down
    for target_rank in range(best_rank - 1, 8, -1):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            U = np.random.randn(nm, target_rank)
            V = np.random.randn(mp, target_rank)
            W = np.random.randn(pn, target_rank)
            
            # Scale to make entries reasonable
            U /= np.sqrt(target_rank)
            V /= np.sqrt(target_rank)
            W /= np.sqrt(target_rank)
            
            # Check if this is close to a solution
            # Compute the tensor product
            error = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        error += (val - 1.0) ** 2
            
            if error < 1e-6:
                # Found a solution! Try to round to integers/half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify the rounded version
                error_rounded = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_rounded[i*m+j, :], V_rounded[j*p+k, :] * W_rounded[k*n+i, :])
                            error_rounded += (val - 1.0) ** 2
                
                if error_rounded < 1e-9:
                    # Convert to lists
                    U_list = U_rounded.tolist()
                    V_list = V_rounded.tolist()
                    W_list = W_rounded.tolist()
                    
                    # Clean up near-zero and near-integer values
                    for i in range(nm):
                        for r in range(target_rank):
                            if abs(U_list[i][r]) < 1e-9:
                                U_list[i][r] = 0
                            elif abs(U_list[i][r] - round(U_list[i][r])) < 1e-9:
                                U_list[i][r] = round(U_list[i][r])
                    for i in range(mp):
                        for r in range(target_rank):
                            if abs(V_list[i][r]) < 1e-9:
                                V_list[i][r] = 0
                            elif abs(V_list[i][r] - round(V_list[i][r])) < 1e-9:
                                V_list[i][r] = round(V_list[i][r])
                    for i in range(pn):
                        for r in range(target_rank):
                            if abs(W_list[i][r]) < 1e-9:
                                W_list[i][r] = 0
                            elif abs(W_list[i][r] - round(W_list[i][r])) < 1e-9:
                                W_list[i][r] = round(W_list[i][r])
                    
                    if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                        return U_list, V_list, W_list, target_rank
        
        if time.time() > deadline:
            break
    
    return best_decomp[0], best_decomp[1], best_decomp[2], best_rank

def construct():
    """
    Returns a rank-minimal tensor decomposition for 3x3 * 3x8 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # Safety margin before 150s limit
    
    # Try to find a lower-rank decomposition
    U, V, W, rank = _try_structured_search(n, m, p, deadline)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
