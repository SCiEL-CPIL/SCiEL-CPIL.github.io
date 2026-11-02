"""Program for matrix multiplication 2,3,8 tensor rank minimization."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """Return a valid tensor decomposition for 2x3x8 matrix multiplication.
    
    Attempts to find a rank < 48 decomposition; falls back to schoolbook rank 48.
    """
    start_time = time.time()
    DEADLINE = 110.0  # 10s safety margin before 120s limit

    def time_left():
        return DEADLINE - (time.time() - start_time)

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 6, 24, 16

    def build_schoolbook():
        """Schoolbook decomposition of rank n*m*p = 48."""
        rank = n * m * p
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
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """Check that U[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c] for all (i,j,k).
        
        U has shape (n*m, rank), V (m*p, rank), W (p*n, rank).
        The multiplication tensor T[i*m+j, j*p+k, k*n+i] should equal
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
        """
        rank = U.shape[1]
        # Build the tensor via einsum
        T = np.einsum('ic,jc,kc->ijk', U, V, W)
        # T has shape (n*m, m*p, p*n)
        # Check: for each (i, j, k), T[i*m+j, j*p+k, k*n+i] should be 1, else 0
        # Actually the convention is: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
        # So the tensor T satisfies T[i, j, k] = 1 if the index corresponds to
        # (row of A, col of A / row of B, col of B) where A is n x m, B is m x p
        # T is indexed as (i*m+j, j*p+k, k*n+i)
        # We need T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k), 0 otherwise
        # But T from einsum gives us all entries; we just need to check the
        # specific pattern.
        # Actually, let's just check that the matrix multiplication works.
        # For random A (n x m) and B (m x p), C = A @ B
        # C[i,k] = sum_j A[i,j] B[j,k]
        # This equals sum_c (sum_j U[i*m+j, c] * W[k*n+i, c] * V[j*p+k, c])
        #   = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]  -- this is NOT quite right
        # The correct check: C[i,k] = sum_j A[i,j] B[j,k]
        # Using the decomposition: C[i,k] = sum_c (sum_j U[i*m+j,c] * A[i,j]) * ... 
        # No, the standard tensor decomposition is:
        # C[i,k] = sum_j A[i,j] B[j,k] = sum_c M_c[i] N_c[j] P_c[k]
        # where M_c is n-vector, N_c is m-vector, P_c is p-vector
        # And the rank-r decomposition means M_c[i] = U[i*m + j_c, c]... no.
        # 
        # The standard formulation: the multiplication tensor T_{i,j,k} (where i in [0,n),
        # j in [0,m), k in [0,p)) satisfies T_{i,j,k} = 1 if the indices form a valid
        # (A[i,j] * B[j,k]) term, i.e., always 1 for valid i,j,k.
        # The decomposition is: T_{i,j,k} = sum_{c=1}^{r} U_c[i] V_c[j] W_c[k]
        # where U_c is n-vector, V_c is m-vector, W_c is p-vector.
        #
        # In the matrix form given in the task:
        # U has shape (n*m, rank): U[i*m+j, c] corresponds to the c-th factor's
        #   U-component at position (i,j) -- but this is unusual.
        #
        # Let me re-read: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
        # "W[k*n+i,r]" -- so W is indexed as W[k*n + i, r]
        #
        # The tensor decomposition condition:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1  for all valid (i,j,k)
        # This is the standard 3-index tensor where index 1 is (i,j) with i<n, j<m,
        # index 2 is (j,k) with j<m, k<p, index 3 is (k,i) with k<p, i<n.
        # The tensor T[(i,j),(j,k),(k,i)] = 1 for all valid combinations.
        #
        # So the check is: for all i in [0,n), j in [0,m), k in [0,p):
        #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        # And for invalid combinations (which don't exist since all (i,j,k) are valid
        # in this structure), it should be 0.
        #
        # Wait, actually every (i,j,k) with i<n, j<m, k<p IS a valid term in the
        # matrix multiplication. So the tensor is all-ones of shape (n*m, m*p, p*n)
        # with the constraint that the middle index of the first mode matches the
        # first index of the second mode, etc. But since we're summing over c and
        # the indices are fully determined, every triple (i,j,k) with i<n, j<m, k<p
        # gives exactly one term A[i,j]*B[j,k].
        #
        # So the tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k),
        # and 0 elsewhere (but "elsewhere" doesn't occur since the indexing is
        # bijective between (i,j,k) triples and the three 2D index pairs).
        #
        # Actually no -- the three modes have sizes n*m=6, m*p=24, p*n=16.
        # A full tensor would have 6*24*16 = 2304 entries. But only n*m*p = 48
        # of them are non-zero (those where the indices are consistent).
        #
        # So the check: for each (i,j,k) with i<n, j<m, k<p:
        #   T[i*m+j, j*p+k, k*n+i] should equal 1
        # For all other (a, b, c) in [0,6) x [0,24) x [0,16):
        #   T[a, b, c] should equal 0
        #
        # Efficient check using einsum:
        # T_full = einsum('ac,bc,dc->abd', U, V, W)  -- shape (6, 24, 16)
        # Then verify the pattern.
        T_full = np.einsum('ac,bc,dc->abd', U, V, W)
        # Check non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if T_full[i*m+j, j*p+k, k*n+i] != 1:
                        return False
        # Check zero entries (spot check or full)
        # Full check is expensive (2304 entries), but for small matrices it's fine
        nonzero_mask = np.zeros((nm, mp, pn), dtype=bool)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    nonzero_mask[i*m+j, j*p+k, k*n+i] = True
        if np.any(T_full[~nonzero_mask] != 0):
            return False
        return True

    def to_half_integer_list(U, V, W):
        """Convert numpy arrays to lists with half-integer values."""
        def arr_to_list(A):
            L = []
            for row in A:
                L.append([x if abs(x - round(x*2)/2) < 1e-9 else round(x*2)/2 for x in row])
            return L
        return {"U": arr_to_list(U), "V": arr_to_list(V), "W": arr_to_list(W)}

    # Try to find rank 47 via numerical optimization with rounding
    # This is a heuristic; may not succeed but worth trying briefly.
    
    # First, let's try a structured approach: exploit the 2x3 structure
    # Split m=3 into m1=2 + m2=1
    # C = A1 @ B1 + A2 @ B2 where A1 is 2x2, B1 is 2x8, A2 is 2x1, B2 is 1x8
    # If we had a rank-r decomposition for 2x2x8, plus rank for 2x1x8 (= 2*1*8=16)
    # Total would be r + 16. If r < 32, we beat 48.
    # Schoolbook for 2x2x8 is 32, so no gain unless we find better for 2x2x8.
    # 
    # For 2x2x2, rank is 7 (Strassen). For 2x2xp, the rank is at most 7*p/2 + ... 
    # Actually R(2,2,p) is known for some p. R(2,2,2)=7, R(2,2,3)=11, etc.
    # For 2x2x8, I don't know the exact rank offhand.
    
    # Alternative: use the fact that one dimension is 8 = 2^3
    # Maybe we can use a Kronecker product structure with Strassen?
    # 2x3x8: if we could express 3 as 2+1 and 8 as 2*4, use Strassen on 2x2 part
    # But this is getting complex. Let me try a numerical search for rank 47.
    
    # Numerical approach: initialize random rank-r decomposition and use
    # alternating least squares to fit the tensor, then round.
    
    def try_rank(r, num_restarts=5, max_iter=200):
        """Try to find a rank-r decomposition via ALS with rounding."""
        if time_left() < 5.0:
            return None
        
        # Target tensor: T[i*m+j, j*p+k, k*n+i] = 1, else 0
        T_target = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i*m+j, j*p+k, k*n+i] = 1.0
        
        best_result = None
        best_score = float('inf')
        
        for restart in range(num_restarts):
            if time_left() < 3.0:
                break
            
            # Initialize with random values in {-1, 0, 1}
            np.random.seed(42 + restart)
            U = np.random.choice([-1, 0, 1], size=(nm, r))
            V = np.random.choice([-1, 0, 1], size=(mp, r))
            W = np.random.choice([-1, 0, 1], size=(pn, r))
            # Scale to make initial product close to target
            # T_approx = einsum('ac,bc,dc->abd', U, V, W)
            # We want this to approximate T_target
            
            # ALS: alternate fixing two factors and solving for the third
            for it in range(max_iter):
                if time_left() < 2.0:
                    break
                
                # Update U: for each row a, U[a,:] should satisfy
                # sum_c U[a,c] * (V * W)[a,c] = T_target[a, :, :] (flattened)
                # (V*W) has shape (mp*p n, r) where (V*W)[b*p n + d, c] = V[b,c]*W[d,c]
                VW = np.einsum('bc,dc->bd c', V, W)  # shape (mp, pn, r)
                VW_flat = VW.reshape(mp * pn, r)
                # For each a: sum_c U[a,c] * VW_flat[:, c] = T_target[a, :, :] flattened
                # T_target[a, :, :] has shape (mp, pn)
                # This is: U[a, :] @ VW_flat.T = T_target[a, :, :].reshape(mp*pn)
                # Solve: VW_flat @ U[a, :].T = T_target[a, :, :].reshape(mp*pn)
                # i.e., U[a, :] @ VW_flat.T = target_a
                # This is underdetermined if r < mp*pn, which it is.
                # Use pseudoinverse: U[a, :] = target_a @ pinv(VW_flat.T)
                # But this is expensive. Use least squares.
                
                # Actually, the update for U given V, W:
                # Minimize ||sum_c U[a,c] * V[:,c] outer W[:,c] - T_target[a,:,:]||_F^2
                # This is equivalent to: for each a, minimize ||U[a,:] @ (VW_flat).T - T_target[a,:,:].reshape(-1)||^2
                # Solution: U[a,:] = T_target[a,:,:].reshape(-1) @ pinv(VW_flat.T)
                # Or use lstsq: VW_flat @ x = T_target[a,:,:].reshape(-1), solve for x = U[a,:]
                
                for a in range(nm):
                    target_a = T_target[a].reshape(-1)  # length mp*pn
                    # Solve VW_flat @ x = target_a in least squares sense
                    # x has length r
                    try:
                        U[a, :] = np.linalg.lstsq(VW_flat, target_a, rcond=None)[0]
                    except:
                        break
                
                # Similarly update V
                UW = np.einsum('ac,dc->ad c', U, W)  # shape (nm, pn, r)
                UW_flat = UW.reshape(nm * pn, r)
                for b in range(mp):
                    target_b = T_target[:, b, :].reshape(-1)  # length nm*pn
                    # We need: sum_c V[b,c] * U[:,c] outer W[:,c] = T_target[:, b, :]
                    # (U*W)[a*d, c] = U[a,c]*W[d,c], so (U*W) has shape (nm*pn, r)
                    # V[b,:] @ (UW_flat).T should equal target_b
                    try:
                        V[b, :] = np.linalg.lstsq(UW_flat, target_b, rcond=None)[0]
                    except:
                        break
                
                # Update W
                UV = np.einsum('ac,bc->ab c', U, V)  # shape (nm, mp, r)
                UV_flat = UV.reshape(nm * mp, r)
                for d in range(pn):
                    target_d = T_target[:, :, d].reshape(-1)  # length nm*mp
                    try:
                        W[d, :] = np.linalg.lstsq(UV_flat, target_d, rcond=None)[0]
                    except:
                        break
                
                if it % 50 == 0:
                    # Check if we're close to integer/half-integer
                    pass
        
        # After ALS, round to half-integers and check
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        if verify(U_r, V_r, W_r):
            result = to_half_integer_list(U_r, V_r, W_r)
            return result
        
        return None
    
    # Try rank 47
    if time_left() > 15.0:
        result = try_rank(47, num_restarts=3, max_iter=100)
        if result is not None:
            return result
    
    # Try rank 46
    if time_left() > 10.0:
        result = try_rank(46, num_restarts=2, max_iter=50)
        if result is not None:
            return result
    
    # Fallback: schoolbook rank 48
    return build_schoolbook()
# EVOLVE-BLOCK-END
