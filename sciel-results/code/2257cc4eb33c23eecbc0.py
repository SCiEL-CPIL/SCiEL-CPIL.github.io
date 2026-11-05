import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140
    
    n, m, p = DIMENSIONS
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    
    def verify(U, V, W):
        """Check if U,V,W exactly decompose T."""
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_recon, T, atol=1e-9)
    
    def build_strassen_rank50():
        """Build the known rank-50 decomposition."""
        rank = 50
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        
        col = 0
        
        def emit_strassen(i0, j0, k0):
            nonlocal col
            # M1 = (a00+a11)(b00+b11)
            U[i0*m + j0, col] = 1.0
            U[(i0+1)*m + (j0+1), col] = 1.0
            V[j0*p + k0, col] = 1.0
            V[(j0+1)*p + (k0+1), col] = 1.0
            W[k0*n + i0, col] = 1.0
            W[(k0+1)*n + (i0+1), col] = 1.0
            col += 1
            
            # M2 = (a10+a11)b00
            U[(i0+1)*m + j0, col] = 1.0
            U[(i0+1)*m + (j0+1), col] = 1.0
            V[j0*p + k0, col] = 1.0
            W[k0*n + (i0+1), col] = 1.0
            W[(k0+1)*n + (i0+1), col] = -1.0
            col += 1
            
            # M3 = a00(b01-b11)
            U[i0*m + j0, col] = 1.0
            V[j0*p + (k0+1), col] = 1.0
            V[(j0+1)*p + (k0+1), col] = -1.0
            W[(k0+1)*n + i0, col] = 1.0
            W[(k0+1)*n + (i0+1), col] = 1.0
            col += 1
            
            # M4 = a11(b10-b00)
            U[(i0+1)*m + (j0+1), col] = 1.0
            V[(j0+1)*p + k0, col] = 1.0
            V[j0*p + k0, col] = -1.0
            W[k0*n + (i0+1), col] = 1.0
            W[k0*n + i0, col] = 1.0
            col += 1
            
            # M5 = (a00+a01)b11
            U[i0*m + j0, col] = 1.0
            U[i0*m + (j0+1), col] = 1.0
            V[(j0+1)*p + (k0+1), col] = 1.0
            W[k0*n + i0, col] = -1.0
            W[(k0+1)*n + i0, col] = 1.0
            col += 1
            
            # M6 = (a10-a00)(b00+b01)
            U[(i0+1)*m + j0, col] = 1.0
            U[i0*m + j0, col] = -1.0
            V[j0*p + k0, col] = 1.0
            V[j0*p + (k0+1), col] = 1.0
            W[(k0+1)*n + (i0+1), col] = 1.0
            col += 1
            
            # M7 = (a01-a11)(b10+b11)
            U[i0*m + (j0+1), col] = 1.0
            U[(i0+1)*m + (j0+1), col] = -1.0
            V[(j0+1)*p + k0, col] = 1.0
            V[(j0+1)*p + (k0+1), col] = 1.0
            W[k0*n + i0, col] = 1.0
            col += 1

        def emit_schoolbook_2x2x1(i0, j0, k0):
            nonlocal col
            for di in range(2):
                for dj in range(2):
                    U[(i0+di)*m + (j0+dj), col] = 1.0
                    V[(j0+dj)*p + k0, col] = 1.0
                    W[k0*n + (i0+di), col] = 1.0
                    col += 1

        emit_strassen(0, 0, 0)
        emit_strassen(0, 0, 2)
        emit_strassen(0, 0, 4)
        emit_schoolbook_2x2x1(0, 0, 6)
        emit_strassen(0, 2, 0)
        emit_strassen(0, 2, 2)
        emit_strassen(0, 2, 4)
        emit_schoolbook_2x2x1(0, 2, 6)
        
        return U, V, W, rank
    
    def try_reduce_rank(U, V, W, rank, target_rank, max_iters=500):
        """
        Try to reduce rank by finding linear combinations that preserve the tensor.
        Use a structured approach: try to merge pairs of rank-1 terms.
        """
        if rank <= target_rank:
            return U, V, W, rank
        
        # For each pair of columns, try to see if they can be merged
        # into a single rank-1 term that still contributes correctly.
        # This is equivalent to checking if the sum of two rank-1 tensors
        # can be written as a single rank-1 tensor.
        
        # Alternative: use the fact that if two columns have proportional
        # U, V, W vectors, we can merge them.
        
        # Try a different approach: for each column r, try to express it
        # as a linear combination of other columns.
        
        # Build the "tensor contribution" matrix for each column
        # T_r = U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        
        # Check if any column's contribution is in the span of others
        # This is hard in general, but for small ranks we can try.
        
        # Simpler: try random linear combinations of pairs
        rng = np.random.RandomState(42)
        
        for _ in range(max_iters):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Pick two random columns to try to merge
            r1 = rng.randint(rank)
            r2 = rng.randint(rank - 1)
            if r2 >= r1:
                r2 += 1
            
            # Try to see if we can replace columns r1 and r2 with a single column
            # that preserves the tensor contribution.
            # This requires: U_r1⊗V_r1⊗W_r1 + U_r2⊗V_r2⊗W_r2 = U_new⊗V_new⊗W_new
            # which is generally impossible unless the two rank-1 tensors are proportional.
            
            # Check if they are proportional
            U1, U2 = U[:, r1], U[:, r2]
            V1, V2 = V[:, r1], V[:, r2]
            W1, W2 = W[:, r1], W[:, r2]
            
            # Check proportionality
            if np.linalg.norm(U1) < 1e-10 or np.linalg.norm(U2) < 1e-10:
                continue
            if np.linalg.norm(V1) < 1e-10 or np.linalg.norm(V2) < 1e-10:
                continue
            if np.linalg.norm(W1) < 1e-10 or np.linalg.norm(W2) < 1e-10:
                continue
            
            # Check if U1 = c*U2, V1 = c*V2, W1 = c*W2 for some c
            c_U = np.dot(U1, U2) / np.dot(U2, U2)
            c_V = np.dot(V1, V2) / np.dot(V2, V2)
            c_W = np.dot(W1, W2) / np.dot(W2, W2)
            
            if (np.allclose(U1, c_U * U2, atol=1e-9) and
                np.allclose(V1, c_V * V2, atol=1e-9) and
                np.allclose(W1, c_W * W2, atol=1e-9) and
                np.isclose(c_U, c_V, atol=1e-9) and
                np.isclose(c_V, c_W, atol=1e-9)):
                # They are proportional, merge them
                c = c_U + 1.0
                U[:, r2] = c * U2
                V[:, r2] = c * V2
                W[:, r2] = c * W2
                U[:, r1] = 0
                V[:, r1] = 0
                W[:, r1] = 0
                # Remove column r1
                U = np.delete(U, r1, axis=1)
                V = np.delete(V, r1, axis=1)
                W = np.delete(W, r1, axis=1)
                rank -= 1
                if rank <= target_rank:
                    return U, V, W, rank
        
        return U, V, W, rank
    
    def build_rank42_direct():
        """
        Try to build a rank-42 decomposition directly.
        
        Strategy: Use 3 Strassen blocks (21 terms) for a 2x2x6 subproblem
        covering output columns 0-5, and handle column 6 with 21 terms?
        No, that's too many.
        
        Alternative: Use a 2x2x3 decomposition with rank 11 (if it exists)
        for two blocks: 2*11 = 22 for columns 0-5, plus 20 for column 6?
        
        Actually, let me try: the 2x2x2 problem has rank 7 (Strassen).
        The 2x2x1 problem has rank 4.
        For 2x2x7: 3*7 + 4 = 25 per inner pair, total 50.
        
        Can we do 2x2x7 in fewer than 25?
        Lower bound: matricization ranks are max(4, 14, 7) = 14.
        So rank >= 14. Can we achieve 14?
        
        Let me try to construct a rank-14 decomposition for 2x2x7.
        """
        # For 2x2x7, try rank R
        for R in [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]:
            if time.time() - start_time > DEADLINE - 20:
                break
            
            # Try to build a rank-R decomposition for 2x2x7
            # using a structured approach.
            
            # The 2x2x7 tensor: T[i,j,k] = 1 for the multiplication pattern.
            # Actually, for the subproblem 2x2 @ 2x7:
            # C[i,k] = sum_j A[i,j] * B[j,k] for j in {0,1}
            # Tensor: T[i,j,k] where i in {0,1}, j in {0,1}, k in {0,6}
            # T[i,j,k] = 1 iff the term A[i,j]*B[j,k] contributes to C[i,k]
            # Which is always true, so T is all ones? No.
            
            # The tensor for 2x2x2 mat mult has T[i,j,k] = 1 for all i,j,k
            # since every pair (i,j) and (j,k) contributes to C[i,k].
            # Wait no, the tensor is defined by T[a,b,c] where a=i*m+j, b=j'*p+k, c=k'*n+i
            # and T[a,b,c] = 1 iff j=j' and k=k'.
            
            # For 2x2x2 (n=2,m=2,p=2):
            # T[i*2+j, j'*2+k, k'*2+i] = 1 iff j=j' and k=k'
            # This gives 8 non-zero entries (one for each (i,j,k) with j=j',k=k')
            # Actually 2*2*2 = 8 entries, all 1.
            
            # For 2x2x7 (n=2,m=2,p=7):
            # T[i*2+j, j'*7+k, k'*2+i] = 1 iff j=j' and k=k'
            # Non-zero: for each (i,j,k), one entry. Total 2*2*7 = 28 entries.
            
            # The rank of this tensor... let me compute matricization ranks.
            # Mode-1: (2*2) x (2*7 * 7*2) = 4 x 196
            # Mode-2: (2*7) x (2*2 * 7*2) = 14 x 56
            # Mode-3: (7*2) x (2*2 * 2*7) = 14 x 56
            
            # Let me just try to find a low-rank decomposition numerically.
            
            # Build the 2x2x7 tensor
            T_sub = np.zeros((4, 14, 14))
            for i in range(2):
                for j in range(2):
                    for k in range(7):
                    a = i*2 + j
                        b = j*7 + k
                        c = k*2 + i
                        T_sub[a, b, c] = 1.0
            
            # Try to find rank-R decomposition
            # Using a simple iterative approach: initialize randomly, then
            # use ALS (alternating least squares) to fit.
            
            U_sub = np.random.randn(4, R)
            V_sub = np.random.randn(14, R)
            W_sub = np.random.randn(14, R)
            
            # ALS iterations
            converged = False
            for it in range(200):
                if time.time() - start_time > DEADLINE - 20:
                    break
                
                # Update U
                # For each a: U[a,:] should satisfy sum_r U[a,r] * (V⊗W)[:,r] = T_sub[a,:,:].reshape(-1)
                VW = np.einsum('br,cr->bc', V_sub, W_sub).reshape(-1, R)
                U_sub = np.linalg.lstsq(VW, T_sub.reshape(4, -1).T, rcond=None)[0].T
                
                # Update V
                UW = np.einsum('ar,cr->ac', U_sub, W_sub).reshape(-1, R)
                V_sub = np.linalg.lstsq(UW, T_sub.transpose(1,0,2).reshape(14, -1).T, rcond=None)[0].T
                
                # Update W
                UV = np.einsum('ar,br->ab', U_sub, V_sub).reshape(-1, R)
                W_sub = np.linalg.lstsq(UV, T_sub.transpose(2,0,1).reshape(14, -1).T, rcond=None)[0].T
                
                # Check convergence
                T_recon = np.einsum('ar,br,cr->abc', U_sub, V_sub, W_sub)
                err = np.linalg.norm(T_recon - T_sub)
                if err < 1e-9:
                    converged = True
                    break
            
            if converged:
                # Found a rank-R decomposition for 2x2x7!
                # Now use it for both inner pairs
                U_full = np.zeros((8, 2*R))
                V_full = np.zeros((28, 2*R))
                W_full = np.zeros((14, 2*R))
                
                # First inner pair (j=0,1)
                U_full[0:4, 0:R] = U_sub
                V_full[0:14, 0:R] = V_sub
                W_full[:, 0:R] = W_sub
                
                # Second inner pair (j=2,3)
                U_full[4:8, R:2*R] = U_sub
                V_full[14:28, R:2*R] = V_sub
                W_full[:, R:2*R] = W_sub
                
                # Verify
                T_recon_full = np.einsum('ar,br,cr->abc', U_full, V_full, W_full)
                if np.allclose(T_recon_full, T, atol=1e-9):
                    return U_full, V_full, W_full, 2*R
        
        return None
    
    # First try to build rank-42 (or lower) directly
    result = build_rank42_direct()
    if result is not None:
        U, V, W, rank = result
        if verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to rank-50
    U, V, W, rank = build_strassen_rank50()
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Last resort: schoolbook rank 56
    rank = 56
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None
# EVOLVE-BLOCK-END
