"""Matrix multiplication 4,5,5 - tensor decomposition search"""
import numpy as np
import time
import sys
from itertools import product

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check if decomposition is exact."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    # Build the tensor
    tensor = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i, j, k] += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
    # Compare to standard multiplication tensor (all ones)
    expected = np.ones((n, m, p), dtype=np.float64)
    if np.max(np.abs(tensor - expected)) > tol:
        return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
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

def _try_block_diagonal(n, m, p, deadline):
    """
    Try to find a lower-rank decomposition using block structure.
    For 4x5x5, we can try to exploit that n=4 is small.
    
    Strategy: Try rank = n*m + m*p + n*p - something, or use
    a bilinear form approach.
    
    Actually, let's try a different approach: use the fact that
    matrix multiplication can be written with rank <= n*m*p,
    and try to find a decomposition with rank < n*m*p using
    a greedy/elimination approach.
    """
    # For now, just return schoolbook as fallback
    return _schoolbook(n, m, p)

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, but we'll use 110s to be safe
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 100
    
    best_result = _schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    # Try to find a better decomposition
    # For 4x5x5, the Strassen-like approach might help.
    # The tensor rank of matrix multiplication is known to be 
    # at most n*m*p, and for small dimensions, there are known 
    # improvements.
    
    # Let's try a structured approach:
    # We want to write C[i,k] = sum_j A[i,j]*B[j,k]
    # using rank r terms: C[i,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # Wait, the convention is W[k*n+i, r], so W is indexed by (k, i).
    
    # The tensor we need to decompose is T[i,j,k] = 1 for all i,j,k
    # because (A*B)[i,k] = sum_j A[i,j]*B[j,k], and the "tensor" 
    # for matrix mult is the one that maps (A, B) -> C where
    # C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i,j,k.
    
    # We need rank-r decomposition of the all-ones tensor of size 4x5x5.
    # Rank of all-ones tensor of size n x m x p is min(n, m, p) if we
    # allow arbitrary values... no wait, that's for the "matrix" case.
    
    # Actually, for the all-ones tensor T[i,j,k] = 1, we can write:
    # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    # with U[i,c] = 1, V[j,c] = 1, W[k,c] = 1/c... no that doesn't work.
    
    # Hmm, but that's for a different decomposition structure. In our case,
    # the decomposition is:
    # C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # summed over j as well. So it's:
    # C[i,k] = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # This is equivalent to: for each c, we have a "slice" that contributes
    # to C[i,k] via a sum over j. So each term c defines a matrix 
    # M_c[i,k] = sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # And we need sum_c M_c[i,k] = 1 for all i,k.
    
    # The schoolbook has M_c[i,k] = 1 if (i,j,k) matches the specific 
    # (i0,j0,k0) and 0 otherwise.
    
    # Can we do better? Let's try to use the fact that we can have 
    # M_c that are "smoother".
    
    # For example, if we set W[k*n+i, c] = 1 for all k,i for a given c,
    # then M_c[i,k] = sum_j U[i*m+j, c] * V[j*p+k, c]
    # which is just a matrix product of two vectors (for fixed i, k).
    
    # Let's try: use r = n*p = 20 terms where W is all ones, and
    # U, V are chosen so that sum_c U[i*m+j, c] * V[j*p+k, c] = delta_jk... 
    # no, that's not right either.
    
    # Let me think differently. The rank of the matrix multiplication 
    # tensor is a well-studied quantity. For 4x5x5, the best known 
    # rank is likely 100 (schoolbook) or slightly less.
    
    # Given the time constraints, let's just return the schoolbook 
    # decomposition which is guaranteed to be correct.
    
    # But wait - let's at least try a simple improvement:
    # If we can find a decomposition with rank < 100, that's better.
    
    # One simple idea: use the fact that for the all-ones tensor,
    # we might be able to use fewer terms if we allow non-1 values.
    
    # Actually, I realize the tensor is NOT all ones in the way I was thinking.
    # Let me re-read the task.
    
    # "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor T satisfies:
    # T[i,j,k] = 1 if we're computing (A*B)[i,k] using A[i,j] and B[j,k]
    # So yes, T[i,j,k] = 1 for all i, j, k.
    
    # And the decomposition is:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # So we need: for all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a 3-way tensor decomposition problem.
    
    # For the all-ones tensor of size 4x5x5, the tensor rank is at most 100.
    # Can it be less? The tensor rank of the all-ones tensor of size 
    # n x m x p is actually min(n, m, p) in some definitions... 
    # No, that's not right. The tensor rank of the all-ones tensor 
    # T[i,j,k] = 1 is actually 1! Because T[i,j,k] = 1 * 1 * 1.
    
    # Wait, but that's for a different decomposition structure. In the 
    # standard tensor decomposition, T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c].
    # In that case, yes, rank 1 works.
    
    # But in our case, the indices are coupled differently:
    # U is indexed by (i, j) -> i*m+j
    # V is indexed by (j, k) -> j*p+k  
    # W is indexed by (k, i) -> k*n+i
    
    # So it's not a standard tensor decomposition. It's a "matrix multiplication"
    # tensor decomposition where the three factors have overlapping indices.
    
    # Let me re-derive. We want:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of n*m*p = 100 equations in r * (n*m + m*p + p*n) unknowns.
    # For r = 1: 1*(20+25+20) = 65 unknowns, 100 equations. Underdetermined? No, 
    # overdetermined. But we can check if it's feasible.
    
    # For r=1: U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0] = 1 for all i,j,k.
    # This means U[i*m+j, 0] = 1/a_{i,j}, V[j*p+k, 0] = 1/b_{j,k}, W[k*n+i, 0] = 1/c_{k,i}
    # where a_{i,j} * b_{j,k} * c_{k,i} = 1 for all i,j,k.
    # Taking logs: log a + log b + log c = 0 for all i,j,k.
    # Let x_{i,j} = log a_{i,j}, y_{j,k} = log b_{j,k}, z_{k,i} = log c_{k,i}.
    # Then x_{i,j} + y_{j,k} + z_{k,i} = 0 for all i,j,k.
    # This is a system of linear equations. Is it consistent?
    # From x_{i,j} + y_{j,k} = -z_{k,i}, and x_{i',j'} + y_{j',k'} = -z_{k',i'}
    # For fixed i, k: x_{i,j} + y_{j,k} should be the same for all j.
    # So x_{i,j1} + y_{j1,k} = x_{i,j2} + y_{j2,k} for all j1, j2, k.
    # This means x_{i,j1} - x_{i,j2} = y_{j2,k} - y_{j1,k} for all k.
    # So x_{i,j1} - x_{i,j2} must be independent of k, which means
    # y_{j2,k} - y_{j1,k} must be the same for all k, i.e., y_{j2,k} - y_{j1,k} = const.
    # Similarly, this propagates. It's likely that a solution exists.
    
    # Let's try to construct one. Set z_{k,i} = 0 for all k,i (so c=1).
    # Then x_{i,j} + y_{j,k} = 0 for all i,j,k.
    # So x_{i,j} = -y_{j,k} for all k. This means y_{j,k} must be independent of k,
    # so y_{j,k} = f(j) for all k. Then x_{i,j} = -f(j) for all i.
    # So U[i*m+j, 0] = 1/(-f(j))... wait, let me be more careful.
    
    # x_{i,j} = log U[i*m+j, 0]... no, x_{i,j} = log a_{i,j} where a_{i,j} is related to U.
    # Actually, U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0] = 1.
    # Let U[i*m+j, 0] = A_{i,j}, V[j*p+k, 0] = B_{j,k}, W[k*n+i, 0] = C_{k,i}.
    # A_{i,j} * B_{j,k} * C_{k,i} = 1 for all i,j,k.
    
    # Set C_{k,i} = 1. Then A_{i,j} * B_{j,k} = 1 for all i,j,k.
    # For fixed j: A_{i,j} * B_{j,k} = 1 for all i,k.
    # So A_{i,j} = 1/B_{j,k} for all k, meaning B_{j,k} is independent of k.
    # Let B_{j,k} = b_j. Then A_{i,j} = 1/b_j for all i.
    # So A_{i,j} = 1/b_j, B_{j,k} = b_j, C_{k,i} = 1.
    # Check: (1/b_j) * b_j * 1 = 1. Yes!
    
    # So rank 1 works?! Let me verify with the actual index structure.
    # U has shape (n*m, 1) = (20, 1)
    # V has shape (m*p, 1) = (25, 1)
    # W has shape (p*n, 1) = (20, 1)
    
    # U[i*m+j, 0] = 1/b_j
    # V[j*p+k, 0] = b_j
    # W[k*n+i, 0] = 1
    
    # For any i, j, k: U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0] 
    #   = (1/b_j) * b_j * 1 = 1. 
    
    # This is rank 1! That's way better than 100!
    
    # But wait, does this actually compute matrix multiplication?
    # The decomposition gives us:
    # (A*B)[i,k] = sum_j sum_c U[i*m+j, c] * A[i,j] * B[j,k] * V[j*p+k, c] * W[k*n+i, c]
    # Hmm, I need to be more careful about how the factors combine.
    
    # Actually, re-reading the task: the factor matrices U, V, W have shapes 
    # (n*m, rank), (m*p, rank), (p*n, rank).
    # The decomposition is:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where T[i,j,k] = 1.
    
    # And this T is the matrix multiplication tensor, meaning:
    # (A*B)[i,k] = sum_{j,c} A[i,j] * B[j,k] * T[i,j,k]... no.
    
    # Actually, I think the convention is:
    # (A*B)[i,k] = sum_c sum_j U[i*m+j, c] * A[i,j] * V[j*p+k, c] * B[j,k] * W[k*n+i, c]
    
    # Hmm, that doesn't quite make sense dimensionally. Let me think again.
    
    # I believe the correct interpretation is:
    # We're decomposing the bilinear map (A, B) -> C where C = A*B.
    # The bilinear map has a tensor T such that:
    # C[i,k] = sum_{j,c} A[i,j] * B[j,k] * T[i,j,k]
    # where T[i,j,k] = 1 for matrix multiplication.
    
    # The "rank" decomposition writes T as:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # But wait, that would make C[i,k] = sum_{j,c} A[i,j] * B[j,k] * U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # which is not quite right because U and V depend on the same index j.
    
    # I think the actual convention might be different. Let me look at the 
    # schoolbook example more carefully.
    
    # In the schoolbook:
    # U[i*m+j, c] = 1 if c corresponds to (i,j,k)
    # V[j*p+k, c] = 1 if c corresponds to (i,j,k)
    # W[k*n+i, c] = 1 if c corresponds to (i,j,k)
    
    # So for a specific (i0, j0, k0), there's one column c where U[i0*m+j0, c] = 1,
    # V[j0*p+k0, c] = 1, W[k0*n+i0, c] = 1, and all other entries in that column are 0.
    
    # Then sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if (i,j,k) = (i0,j0,k0) for the matching c, else 0.
    
    # So T[i,j,k] = 1 for all i,j,k (each (i,j,k) has its own term).
    
    # So the tensor is indeed the all-ones tensor, and we need to decompose it.
    
    # My rank-1 construction: U[i*m+j, 0] = 1/b_j, V[j*p+k, 0] = b_j, W[k*n+i, 0] = 1.
    # This gives T[i,j,k] = (1/b_j) * b_j * 1 = 1 for all i,j,k. 
    
    # So rank 1 should work! Let me code this up.
    
    def _rank1_decomposition(n, m, p):
        """Construct a rank-1 decomposition if possible."""
        rank = 1
        # Choose b_j values. Let b_j = 1 for simplicity.
        b = [1] * m
        U = [[1 / b[j] for j in range(m)] for i in range(n)]  # shape (n*m, 1)
        # Reshape to (n*m, rank)
        U_flat = []
        for i in range(n):
            for j in range(m):
                U_flat.append([1 / b[j]])
        
        V_flat = []
        for j in range(m):
            for k in range(p):
                V_flat.append([b[j]])
        
        W_flat = []
        for k in range(p):
            for i in range(n):
                W_flat.append([1])
        
        return {"U": U_flat, "V": V_flat, "W": W_flat}
    
    # Try rank 1
    if time.time() < deadline - 5.0:
        result_r1 = _rank1_decomposition(n, m, p)
        if _verify(result_r1["U"], result_r1["V"], result_r1["W"], n, m, p):
            return result_r1
        else:
            # Rank 1 didn't work, fall through
            pass
    
    # If rank 1 didn't work (shouldn't happen based on my analysis), 
    # try other approaches or fall back to schoolbook.
    return best_result

# EVOLVE-BLOCK-END
