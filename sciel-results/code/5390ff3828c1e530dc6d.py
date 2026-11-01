"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition
    rank0 = n * m * p
    U0 = np.zeros((n * m, rank0))
    V0 = np.zeros((m * p, rank0))
    W0 = np.zeros((p * n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1
    
    def verify(U, V, W, tol=1e-9):
        """Verify tensor decomposition: sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] matches matmul tensor"""
        r = U.shape[1]
        # Build the matmul tensor T[i,j,k] = 1 if (i,j,k) corresponds to a matmul term
        # Actually, the tensor is: T[i,j,k] = 1 for all valid (i,j,k) in matmul
        # The decomposition says: T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
        # For matrix multiplication C = A*B where A is n×m, B is m×p, C is n×p
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # And the decomposition is: sum_c U[i,c]*V[j,c]*W[k,c] = 1 for all (i,j,k)
        # Wait, that's not right. Let me re-read the task.
        
        # The task says: "exactly decompose the matrix multiplication tensor"
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # Output factor follows W[k*n+i, r]
        
        # So the indices are:
        # U[i*m + j, c] - row index is i*m+j where i in [0,n), j in [0,m)
        # V[j*p + k, c] - row index is j*p+k where j in [0,m), k in [0,p)
        # W[k*n + i, c] - row index is k*n+i where k in [0,p), i in [0,n)
        
        # The tensor decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # where T[i,j,k] = 1 for all valid (i,j,k)
        
        # So we need: for all i in [0,n), j in [0,m), k in [0,p):
        #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        # Let's verify this
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r))
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def verify_fast(U, V, W, tol=1e-9):
        """Faster verification using vectorization"""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to reduce rank by removing columns that can be absorbed
    # Strategy: try various rank-reduction approaches
    
    best_rank = rank0
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    
    # Approach 1: Try removing one column at a time and see if we can adjust others
    # For each column c, try to express it as a combination of others
    
    # Approach 2: Random perturbation + rank reduction via SVD-like methods
    # Since we need exact integer/half-integer entries, we'll use a different approach
    
    # Approach 3: Try specific structured decompositions
    
    # For 3x4x4, let's try to find a rank-47 or lower decomposition
    # by exploiting the structure
    
    # Try: can we merge some columns?
    # Column c corresponds to (i,j,k) triple
    # Two columns (i1,j1,k1) and (i2,j2,k2) can potentially be merged if
    # their contributions can be combined
    
    # Let's try a greedy approach: try to remove each column and see if the
    # remaining system is still solvable with the same other columns adjusted
    
    # Actually, let's try a different approach: use the fact that for matrix multiplication,
    # there might be algebraic identities that allow rank reduction
    
    # For 3x4x4, the schoolbook rank is 48.
    # Known results: rank(3x4x4) is at most 47 (from literature on matrix multiplication ranks)
    
    # Let me try to find a rank-47 decomposition by finding a linear dependency
    
    # The tensor T is in R^{12 x 16 x 12} (n*m=12, m*p=16, p*n=12)
    # Schoolbook gives 48 rank-1 tensors.
    # To get rank 47, we need to find one linear dependency among the 48 rank-1 tensors
    
    # Try: find vectors a, b, c such that sum over some subset of columns
    # a[i*m+j] * b[j*p+k] * c[k*n+i] = 0 for all (i,j,k)
    
    # This is equivalent to finding a rank-1 tensor in the span of the other 47
    
    # Let's try a simpler approach: try all pairs of columns and see if one can be
    # expressed in terms of the other with coefficient 1 or -1
    
    # Actually, for the standard matmul tensor, I don't think simple linear dependencies
    # exist. Let me try a numerical approach.
    
    # Try random linear combinations to see if we can find a dependency
    random.seed(42)
    
    # For each column, try to see if it's in the span of the others
    # This is a 3-tensor problem, so it's more complex
    
    # Let's try: pick a subset of columns, and see if the remaining tensor
    # can be decomposed with fewer terms
    
    # Approach: use the fact that T[i,j,k] = 1 for all (i,j,k)
    # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # This is equivalent to: the 12x16x12 all-ones tensor can be written as
    # sum of r rank-1 tensors
    
    # For the all-ones tensor of size a×b×c, the rank is max(a,b,c) in general
    # Wait, no. The all-ones tensor 1_{a×b×c} has tensor rank max(a,b,c)?
    # Actually, for the all-ones tensor, the rank is min(a,b,c) if we're talking about
    # the standard tensor rank? No, that's not right either.
    
    # Let me think again. The all-ones tensor T[i,j,k] = 1 for all i,j,k.
    # A rank-1 tensor is u ⊗ v ⊗ w where (u⊗v⊗w)[i,j,k] = u[i]*v[j]*w[k]
    # So we need: sum_c u_c[i] * v_c[j] * w_c[k] = 1 for all i,j,k
    
    # This means: the matrix sum_c u_c * v_c^T * w_c^T (in some sense) equals the all-ones matrix
    # for each fixed k, or similar.
    
    # Actually, for fixed k: sum_c u_c[i] * v_c[j] * w_c[k] = 1 for all i,j
    # This means: sum_c w_c[k] * (u_c v_c^T) = J (all-ones n×m matrix) for all k
    
    # Similarly, this is a constrained problem.
    
    # For the all-ones tensor, the tensor rank is actually min(n*m, m*p, p*n) in the worst case,
    # but can be lower.
    
    # Wait, I think I'm overcomplicating this. Let me just try to verify if the schoolbook
    # decomposition is correct and then try to find improvements.
    
    if not verify_fast(U0, V0, W0):
        # Should not happen
        pass
    
    # Try to reduce rank by finding linear dependencies
    # For each column c, check if it can be removed by adjusting others
    
    # This is a hard problem. Let's try a heuristic approach.
    
    # Heuristic: try random subsets of columns and see if the system is still consistent
    # with a lower rank
    
    # Actually, let's try a different approach: use the structure of the problem.
    # For 3x4x4, we have:
    # - 12 values of i*m+j (i in 0..2, j in 0..3)
    # - 16 values of j*p+k (j in 0..3, k in 0..3)  
    # - 12 values of k*n+i (k in 0..3, i in 0..2)
    
    # The tensor is the all-ones tensor in these indices.
    
    # Key insight: if we can find vectors u (length 12), v (length 16), w (length 12)
    # such that u[i*m+j] * v[j*p+k] * w[k*n+i] = 1 for all valid (i,j,k),
    # then we have a rank-1 decomposition. But this is impossible unless all entries are 1,
    # which would require u[i*m+j] = 1 for all, v[j*p+k] = 1 for all, w[k*n+i] = 1 for all.
    # That's a valid rank-1 decomposition!
    
    # Wait, let me check: if U[i*m+j, 0] = 1 for all, V[j*p+k, 0] = 1 for all, W[k*n+i, 0] = 1 for all,
    # then sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1*1*1 = 1 for all (i,j,k).
    # That's a rank-1 decomposition!
    
    # But wait, that can't be right. Let me re-examine the problem.
    
    # Oh, I see the issue. The matrix multiplication tensor is NOT the all-ones tensor.
    # Let me re-read the task.
    
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 4 matrix."
    
    # The matrix multiplication tensor T is defined by:
    # T[i, j, k] = 1 if the (i,k) entry of the product C = A*B involves A[i,j] * B[j,k]
    # Otherwise T[i,j,k] = 0
    
    # For standard matrix multiplication C = A*B where A is 3x4 and B is 4x4:
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    # So T[i,j,k] = 1 for all i in 0..2, j in 0..3, k in 0..3
    # Because every (i,j,k) triple contributes to some product entry.
    
    # Wait, that means T IS the all-ones tensor! And I just showed it has rank 1?
    
    # That can't be right. Let me reconsider.
    
    # Actually, I think I'm misunderstanding the tensor decomposition format.
    
    # The task says:
    # U has shape (n*m, rank)
    # V has shape (m*p, rank) 
    # W has shape (p*n, rank)
    # W[k*n+i, r]
    
    # And the decomposition is: for the matrix multiplication, we want
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # In tensor form: C[i,k] = sum_{j,c} A[i,j] * B[j,k] * (something)
    
    # I think the tensor decomposition is:
    # The matmul tensor T has T[i,j,k] = 1 for all valid (i,j,k)
    # And we decompose it as: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # But that would mean the all-ones tensor has rank 1, which is clearly wrong
    # for a non-trivial matrix multiplication.
    
    # Let me reconsider. Maybe the tensor is different.
    
    # Actually, I think the issue is that the "matrix multiplication tensor" in the
    # context of tensor rank for matrix multiplication is NOT the all-ones tensor.
    
    # For matrix multiplication of n×m by m×p matrices, the tensor T is in
    # (R^n ⊗ R^m) ⊗ (R^m ⊗ R^p) ⊗ (R^p ⊗ R^n) or similar.
    
    # The standard formulation: the matmul tensor M is defined by
    # M[i, j, k] = 1 if j is the "middle" index connecting the two matrices
    # In the decomposition, we have:
    # C[i,k] = sum_{j,c} A[i,j] * B[j,k] * M[i,j,k,c]
    
    # Hmm, I'm getting confused. Let me look at the schoolbook decomposition in the code.
    
    # In the schoolbook code:
    # For each (i,j,k): U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # So for a given (i,j,k), only the column corresponding to that (i,j,k) has 1s.
    
    # The verification should be:
    # For all (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # In the schoolbook, for a fixed (i,j,k), only the column c=(i,j,k) has all three
    # entries equal to 1, so the sum is 1. For other columns, at least one entry is 0.
    
    # So yes, the tensor is the all-ones tensor, and the schoolbook gives rank n*m*p.
    
    # But I argued that the all-ones tensor has rank 1. Let me check again.
    
    # If U[i*m+j, 0] = 1 for all (i,j), V[j*p+k, 0] = 1 for all (j,k), W[k*n+i, 0] = 1 for all (k,i),
    # then sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1*1*1 = 1 for all (i,j,k).
    
    # So the rank should be 1?! That seems too good to be true.
    
    # Wait, I think I'm missing something. Let me re-read the task more carefully.
    
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 4 matrix."
    
    # I think the key is that the tensor decomposition must be such that it can be used
    # to compute the matrix product. The way it works is:
    
    # Given A (n×m) and B (m×p), we compute C = A*B.
    # The decomposition gives us intermediate products:
    # For each c: P_c = (U^T A) ⊙ (V^T B) or something like that, and then
    # C = sum_c W * P_c or similar.
    
    # Actually, I think the correct interpretation is:
    # The matmul tensor T is in R^{(n*m) × (m*p) × (p*n)}
    # T[a, b, c] = 1 if the indices a, b, c are "compatible" in the matmul
    # Specifically, if a = i*m+j, b = j'*p+k, c = k'*n+i', then
    # T[a,b,c] = 1 if j = j' and i = i' and k = k'
    # Otherwise T[a,b,c] = 0
    
    # So T is NOT the all-ones tensor. It's a sparse tensor where T[a,b,c] = 1 only when
    # the indices are consistent.
    
    # Let me verify with the schoolbook:
    # For column corresponding to (i,j,k):
    # U[i*m+j, col] = 1, so a = i*m+j
    # V[j*p+k, col] = 1, so b = j*p+k
    # W[k*n+i, col] = 1, so c = k*n+i
    
    # And the tensor value T[a,b,c] for a=i*m+j, b=j*p+k, c=k*n+i should be 1.
    # For other combinations, it should be 0.
    
    # So the tensor T is defined by:
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if j=j' and i=i' and k=k'
    # T[...] = 0 otherwise
    
    # This makes more sense! The tensor is sparse.
    
    # Now, the rank-1 decomposition: T[a,b,c] = sum_c' U[a,c'] * V[b,c'] * W[c,c']
    # For the all-ones approach to work, we'd need:
    # U[a,0] * V[b,0] * W[c,0] = T[a,b,c] for all a,b,c
    
    # But T is not the all-ones tensor, so this doesn't work.
    
    # OK so now I understand the problem correctly. Let me redo the verification.
    
    def verify_correct(U, V, W, tol=1e-9):
        """Verify tensor decomposition for matmul tensor"""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # T[i*m+j, j*p+k, k*n+i] should be 1
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = sum(U[a, cc] * V[b, cc] * W[c, cc] for cc in range(r))
                    if abs(val - 1.0) > tol:
                        return False
                    # Also check that T is 0 for incompatible indices
                    # For example, T[i*m+j, j'*p+k, k'*n+i'] = 0 if j!=j' or i!=i' or k!=k'
                    # We need to check a few such cases
        # Check some zero entries
        # T[0, 1, 0] should be 0 (i=0,j=0; j'=0,k=1; k'=0,i'=0 -> j=j', i=i', but k!=k')
        # Wait, let me be more careful.
        # T[a,b,c] where a=i*m+j, b=j'*p+k', c=k''*n+i''
        # T[a,b,c] = 1 if j=j' and i=i'' and k=k''
        # So T[i*m+j, j'*p+k', k''*n+i''] = 1 iff j=j' and i=i'' and k=k''
        
        # Let's check a few zero entries
        # T[0*m+0, 0*p+1, 1*n+0] = T[0, 1, 3] (since n=3, 1*3+0=3)
        # Here i=0,j=0; j'=0,k'=1; k''=1,i''=0
        # j=j' (0=0), i=i'' (0=0), but k!=k'' (0!=1), so T should be 0
        a, b, c = 0, 1, 3
        val = sum(U[a, cc] * V[b, cc] * W[c, cc] for cc in range(r))
        if abs(val) > tol:
            return False
        
        # Check another: T[1, 0, 0]
        # i=0,j=1; j'=0,k'=0; k''=0,i''=0
        # j!=j' (1!=0), so T should be 0
        a, b, c = 1, 0, 0
        val = sum(U[a, cc] * V[b, cc] * W[c, cc] for cc in range(r))
        if abs(val) > tol:
            return False
        
        return True
    
    if not verify_correct(U0, V0, W0):
        print("Schoolbook verification failed!")
        # This should not happen, but let's continue
    
    # Now, to reduce rank, we need to find a lower-rank decomposition of this sparse tensor.
    
    # The tensor T is in R^{12 x 16 x 12} with T[i*m+j, j*p+k, k*n+i] = 1 for all (i,j,k)
    # and 0 elsewhere.
    
    # Number of non-zero entries: 3*4*4 = 48
    
    # The schoolbook decomposition uses 48 rank-1 tensors, one for each non-zero entry.
    
    # To reduce rank, we need to find linear dependencies among these rank-1 tensors.
    
    # Let's try a numerical approach: represent the tensor as a 3D array and try to find
    # a low-rank approximation.
    
    # But we need EXACT decomposition with integer/half-integer entries.
    
    # Let's try to find if there's a rank-47 decomposition by checking if one of the
    # rank-1 tensors is in the span of the others.
    
    # This is a hard problem in general. Let's try a heuristic.
    
    # For each column c, try to see if it can be removed by adjusting the other columns.
    
    # Actually, let's try a different approach: use the fact that the tensor has a
    # specific structure and try to find algebraic identities.
    
    # For 3x4x4, I'll try to find if there's a known low-rank decomposition.
    
    # Given the time constraints, let's try a simple heuristic:
    # Try to merge pairs of columns that have similar structure.
    
    # For now, let's just return the schoolbook decomposition, but with a time guard.
    
    if time.time() > deadline - 5:
        # Not enough time, return schoolbook
        return {
            "U": U0.tolist(),
            "V": V0.tolist(),
            "W": W0.tolist()
        }
    
    # Try to find rank reduction
    # For each column, try to express it as a linear combination of others
    # This requires solving a system of equations
    
    # Let's try: for column c0, can we find coefficients alpha_c for c != c0 such that
    # U[:,c0] = sum_{c!=c0} alpha_c * U[:,c]
    # V[:,c0] = sum_{c!=c0} alpha_c * V[:,c]  
    # W[:,c0] = sum_{c!=c0} alpha_c * W[:,c]
    
    # If such alpha_c exist, then we can remove column c0.
    
    # This is a system of 12 + 16 + 12 = 40 equations in 47 unknowns (alpha_c for c!=c0)
    
    # Let's try for a few columns
    
    for c0 in range(min(5, rank0)):  # Try first 5 columns
        if time.time() > deadline - 10:
            break
        
        # Build the system: [U[:,c1], U[:,c2], ..., V[:,c1], V[:,c2], ..., W[:,c1], W[:,c2], ...] * alpha = [U[:,c0], V[:,c0], W[:,c0]]
        # where c1, c2, ... are all columns except c0
        
        other_cols = [c for c in range(rank0) if c != c0]
        A_mat = np.zeros((40, len(other_cols)))
        b_vec = np.zeros(40)
        
        for idx, c in enumerate(other_cols):
            A_mat[:12, idx] = U0[:, c]
            A_mat[12:28, idx] = V0[:, c]
            A_mat[28:40, idx] = W0[:, c]
        
        b_vec[:12] = U0[:, c0]
        b_vec[12:28] = V0[:, c0]
        b_vec[28:40] = W0[:, c0]
        
        # Solve the least-squares problem
        try:
            alpha, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            if residuals[0] < 1e-9:
                # Column c0 can be removed!
                new_rank = rank0 - 1
                new_U = np.zeros((n*m, new_rank))
                new_V = np.zeros((m*p, new_rank))
                new_W = np.zeros((p*n, new_rank))
                
                col_idx = 0
                for c in other_cols:
                    new_U[:, col_idx] = U0[:, c]
                    new_V[:, col_idx] = V0[:, c]
                    new_W[:, col_idx] = W0[:, c]
                    col_idx += 1
                
                # Verify
                if verify_correct(new_U, new_V, new_W):
                    best_rank = new_rank
                    best_U, best_V, best_W = new_U, new_V, new_W
                    print(f"Found rank {new_rank} decomposition!")
                    break
        except Exception as e:
            pass
    
    # Return the best found
    if best_rank < rank0:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        return {
            "U": U0.tolist(),
            "V": V0.tolist(),
            "W": W0.tolist()
        }
# EVOLVE-BLOCK-END
