import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m  # 8
    V_dim = m * p  # 32
    W_dim = p * n  # 16

    start_time = time.time()
    deadline = start_time + 130

    def make_schoolbook():
        rank = n * m * p  # 64
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    def verify(U, V, W, A, B, tol=1e-6):
        C = A @ B
        n, m, p = DIMENSIONS
        for i in range(n):
            for k in range(p):
                total = 0.0
                for j in range(m):
                    for t in range(U.shape[1]):
                        total += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                if abs(total - C[i,k]) > tol:
                    return False
        return True

    def verify_fast(U, V, W, A, B, tol=1e-6):
        """Vectorized verification."""
        C = A @ B
        n, m, p = DIMENSIONS
        # Build the full bilinear form
        # C[i,k] = sum_j sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
        # For each t, the contribution to C is: sum_j U[i*m+j,t] * A[i,j] * V[j*p+k,t] * W[k*n+i,t]
        # = sum_t W[k*n+i,t] * (sum_j U[i*m+j,t] * A[i,j] * V[j*p+k,t])
        # Let's compute via matrix ops
        C_decomp = np.zeros((n, p))
        for t in range(U.shape[1]):
            # For each t, we have vectors u_i (length m), v_j (length p), w_k (scalar per k)
            # Contribution to C[i,k] = w_k * sum_j u_i[j] * A[i,j] * v_j[k]
            # = w_k * (A @ (u_i * v_k^T)) ... not quite
            # Actually: sum_j U[i*m+j,t] * A[i,j] * V[j*p+k,t]
            # Let a_i = U[i*m:(i+1)*m, t] (length m)
            # Let b_k = V[:, t] reshaped... V[j*p+k, t] for fixed t is a vector of length m*p
            # Reshape V_t to (m, p): V_t[j*p+k] = V[j*p+k, t]
            V_t = V[:, t].reshape(m, p)
            for i in range(n):
                U_t_i = U[i*m:(i+1)*m, t]  # length m
                # sum_j U_t_i[j] * A[i,j] * V_t[j,k] for each k
                # = (U_t_i * A[i,:]) @ V_t
                contrib = np.dot(U_t_i * A[i, :], V_t)  # length p
                W_t = W[:, t]  # length p*n
                for k in range(p):
                    C_decomp[i, k] += W_t[k*n + i] * contrib[k]
        return np.allclose(C_decomp, C, atol=tol)

    # Strategy: Use 8 Strassen 2x2 blocks for rank 56, but try to find a lower rank
    # by using a (2,2,4) decomposition of rank 8 for each half.
    # 
    # (2,2,4) decomposition of rank 8:
    # A is 2x2, B is 2x4, C is 2x4
    # C = A @ B
    # We can write B = [B1, B2] where B1, B2 are 2x2
    # C = [A@B1, A@B2]
    # Use Strassen for A@B1 (rank 7) and A@B2 (rank 7) = 14 total
    # But we can share: the Strassen intermediates for A are the same!
    # M1 = (a+d)(e+h) for B1, M1' = (a+d)(g+i) for B2
    # These share (a+d) but not the B part.
    # Actually, let's just use 2 independent Strassen: rank 14 for (2,2,4)
    #
    # For (2,4,8): split into two (2,2,8) problems
    # Each (2,2,8) = 4 independent Strassen 2x2 = rank 28
    # Total: 56
    #
    # Alternative: use a (2,2,4) rank-8 construction
    # Let me construct one explicitly.
    #
    # Actually, let me try a different structural approach:
    # Use the fact that (2,4,8) can be done with rank 40 via:
    # - 5 independent (2,2,2) Strassen decompositions, each of rank 7, plus 15 extra
    # - Or: use a "ladder" decomposition
    #
    # Let me just go with the 8-Strassen-block rank-56 construction (verified) 
    # and add a greedy rank-reduction step.

    def make_strassen_block():
        """2x2*2x2 Strassen, rank 7."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1
        U[2,1]=1; U[3,1]=1; V[0,1]=1
        U[0,2]=1; V[1,2]=1; V[3,2]=-1
        U[3,3]=1; V[2,3]=1; V[0,3]=-1
        U[0,4]=1; U[2,4]=1; V[3,4]=1
        U[1,5]=1; U[3,5]=-1; V[0,5]=1; V[2,5]=1
        U[2,6]=1; U[0,6]=-1; V[1,6]=1; V[3,6]=1
        W[0,0]=1; W[0,4]=-1; W[0,3]=1; W[0,6]=1
        W[1,2]=1; W[1,4]=1
        W[2,1]=1; W[2,3]=1
        W[3,0]=1; W[3,2]=-1; W[3,5]=1; W[3,1]=-1
        return U, V, W

    # Build 8 Strassen blocks: rank 56
    U_block, V_block, W_block = make_strassen_block()
    rank_total = 8 * 7  # 56
    U_full = np.zeros((U_dim, rank_total))
    V_full = np.zeros((V_dim, rank_total))
    W_full = np.zeros((W_dim, rank_total))

    col_offset = 0
    for a_block in range(2):
        a_col_base = a_block * 2
        b_row_base = a_block * 2
        for c_block in range(4):
            c_col_base = c_block * 2
            for t in range(7):
                for i in range(2):
                    for j in range(2):
                        U_full[i*4 + a_col_base + j, col_offset] += U_block[i*2+j, t]
                for j in range(2):
                    for k in range(2):
                        V_full[(b_row_base+j)*8 + c_col_base + k, col_offset] += V_block[j*2+k, t]
                for i in range(2):
                    for k in range(2):
                        W_full[(c_col_base+k)*2 + i, col_offset] += W_block[k*2+i, t]
                col_offset += 1

    # Verify
    rng = np.random.RandomState(42)
    A_test = rng.randint(-3, 4, size=(n, m)).astype(float)
    B_test = rng.randint(-3, 4, size=(m, p)).astype(float)
    
    if not verify_fast(U_full, V_full, W_full, A_test, B_test):
        U0, V0, W0, r0 = make_schoolbook()
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

    # Try to reduce rank by finding linear dependencies among columns
    # A column t is redundant if its (U_t, V_t, W_t) can be expressed
    # as a linear combination that doesn't change the bilinear form.
    # Simple approach: check if any two columns have proportional U or V vectors
    
    best_rank = rank_total
    best_U, best_V, best_W = U_full.copy(), V_full.copy(), W_full.copy()
    
    # Try removing columns one at a time and checking validity
    # This is O(rank^2 * verify_cost) which is expensive, so be selective
    
    if time.time() < deadline - 60:
        # Try to merge columns that share the same W vector
        # Group columns by W vector
        from collections import defaultdict
        w_groups = defaultdict(list)
        for t in range(rank_total):
            w_key = tuple(np.round(W_full[:, t], 8))
            w_groups[w_key].append(t)
        
        # For groups with >1 columns, check if they can be merged
        # (they can't simply be merged unless U and V are also proportional)
        # Instead, try a different approach: use SVD-like analysis
        
        # Try: for each pair of columns, check if their "bilinear contribution" 
        # can be combined. This is complex, so try a simpler heuristic:
        # Remove a column if the remaining columns still span the same bilinear form
        
        # Actually, let's try a more direct approach: 
        # The 56-rank construction might have redundant columns.
        # Check rank of the U, V, W matrices
        u_rank = np.linalg.matrix_rank(U_full, tol=1e-8)
        v_rank = np.linalg.matrix_rank(V_full, tol=1e-8)
        w_rank = np.linalg.matrix_rank(W_full, tol=1e-8)
        
        # If any has rank < 56, there are linear dependencies
        # But linear dependencies in U alone don't mean the column is redundant
        # in the tensor sense.
        
        # Try a greedy removal: for each column, try removing it and see if
        # the decomposition still works (with a quick check)
        # This is too expensive for 56 columns, so try a subset
        
        # Alternative structural approach: use a (2,2,4) rank-8 decomposition
        # to get (2,4,8) at rank 2*8*2 = 32? Let me construct this.
        
        # (2,2,4) rank-8 construction:
        # A = [[a,b],[c,d]], B = [[e,f,g,h],[i,j,k,l]], C = A@B = [[ae+bi, af+bj, ag+bk, ah+bl], [ce+di, cf+dj, cg+dk, ch+dl]]
        # 
        # Use 8 bilinear terms:
        # M1 = (a+d)(e+i)
        # M2 = (c+d)(e)
        # M3 = a(f-j)
        # M4 = d(g-i)  -- wait, this doesn't work for 4 columns
        # 
        # Let me use a different approach. For (2,2,4), the minimal rank is 8.
        # One construction:
        # M1 = (a+d)(e+i)
        # M2 = (c+d)e
        # M3 = a(f-i)  -- no, this mixes columns
        # 
        # Actually, for (2,2,4), I can use:
        # Split B into two 2x2 blocks: B = [B1 | B2]
        # C = [A@B1 | A@B2]
        # Use Strassen for each: rank 7+7 = 14
        # But the (2,2,4) rank is 8, so there's a better way.
        #
        # The rank-8 construction for (2,2,4):
        # M1 = (a+d)(e+i)
        # M2 = (c+d)e
        # M3 = a(f+j)
        # M4 = d(g+k)
        # M5 = (a+c)(i+l)
        # M6 = (b-d)(e+g)
        # M7 = (c-a)(f+k)
        # M8 = (b)(h-l)  -- I'm not sure this is correct
        #
        # Let me just verify the rank-56 construction and try a different 
        # structural approach: use a (4,4,4) decomposition of rank 20 (Laderman-like)
        # embedded in (2,4,8).
        #
        # Actually, the cleanest improvement: use 4 independent Strassen 2x2 blocks
        # for the "diagonal" part and handle the cross terms separately.
        #
        # For now, return the verified rank-56 construction.
        pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
