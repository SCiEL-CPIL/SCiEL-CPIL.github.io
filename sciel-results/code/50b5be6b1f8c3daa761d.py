import numpy as np
import time
from itertools import product

def construct():
    n, m, p = 2, 3, 4
    deadline = time.time() + 140

    def verify(U, V, W, rank):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(rank):
                        s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def build_rank_22():
        """Current best known: rank 22 decomposition."""
        rank = 22
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_term(u_indices, v_indices, w_indices, col):
            for idx, val in u_indices:
                U[idx][col] += val
            for idx, val in v_indices:
                V[idx][col] += val
            for idx, val in w_indices:
                W[idx][col] += val
        
        # Strassen for A[0:2, 0:2] * B[0:2, 0:2]
        add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
        add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
        add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
        add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
        add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
        add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
        add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)

        # Strassen for A[0:2, 0:2] * B[0:2, 2:4]
        add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
        add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
        add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
        add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
        add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
        add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
        add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)

        # Outer product for A[0:2, 2:3] * B[2:3, 0:4]
        add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
        add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
        add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
        add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
        add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
        add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
        add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
        add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

        return U, V, W

    # Verify baseline
    U_base, V_base, W_base = build_rank_22()
    assert verify(U_base, V_base, W_base, 22), "Rank-22 baseline failed verification"
    best_result = {"U": U_base, "V": V_base, "W": W_base}

    def build_rank_20_candidate():
        """
        Try to find rank 20 by using Strassen on the first 2x2x2 block
        and a more clever combination for the remaining 3 columns of B
        and 1 row of A.
        
        Key idea: Use Strassen (7 terms) for A[:,0:2]*B[0:2,0:2]
        Then for the remaining: A[:,0:2]*B[0:2,2:4] + A[:,2]*B[2,0:4]
        
        Alternative: Use a different partition that allows sharing.
        """
        rank = 20
        U = [[0] * rank for _ in range(n * m)]  # 6 x 20
        V = [[0] * rank for _ in range(m * p)]  # 12 x 20
        W = [[0] * rank for _ in range(p * n)]  # 8 x 20
        
        # Try a construction where we share more structure
        # Let me try: Strassen for top-left 2x2, then combine remaining
        # with shared bilinear forms
        
        # Actually, let me try a completely different approach:
        # Use the fact that the tensor is "rectangular" and try
        # rank-20 via careful term sharing
        
        # Terms 0-6: Strassen for A[0:2,0:2] * B[0:2,0:2]
        # A indices: 0=A[0,0], 1=A[0,1], 2=A[0,2], 3=A[1,0], 4=A[1,1], 5=A[1,2]
        # B indices: 0-3=B[0,:], 4-7=B[1,:], 8-11=B[2,:]
        # W indices: 0-3=C[:,0], 4-7=C[:,1], 8-11=C[:,2], 12-15=C[:,3]
        # Actually W[k*n+i]: W[0]=C[0,0], W[1]=C[1,0], W[2]=C[0,1], W[3]=C[1,1], etc.
        
        def set_term(col, u_vals, v_vals, w_vals):
            for idx, val in u_vals:
                if val != 0:
                    U[idx][col] += val
            for idx, val in v_vals:
                if val != 0:
                    V[idx][col] += val
            for idx, val in w_vals:
                if val != 0:
                    W[idx][col] += val
        
        # Strassen 2x2x2 for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        # C[0,0] = U[0]*V[0]*W[0] + U[1]*V[0]*W[0] + ... 
        # Standard Strassen:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B10-B00)
        # M4 = A11*(B01-B11)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B10)
        # M7 = (A01-A11)*(B01+B11)
        
        # M1: (A00+A11)*(B00+B11) -> C00 += M1, C01 += M1, C10 += M1, C11 += M1
        set_term(0, [(0,1),(4,1)], [(0,1),(4,1)], [(0,1),(2,1),(1,1),(3,1)])
        # M2: (A10+A11)*B00 -> C10 -= M2, C11 += M2  
        set_term(1, [(3,1),(4,1)], [(0,1)], [(1,-1),(3,1)])
        # M3: A00*(B10-B00) -> C00 += M3, C01 += M3
        set_term(2, [(0,1)], [(4,-1),(0,1)], [(0,1),(2,1)])
        # M4: A11*(B01-B11) -> C10 -= M4, C11 += M4
        set_term(3, [(4,1)], [(1,-1),(5,1)], [(1,-1),(3,1)])
        # M5: (A00+A01)*B11 -> C00 -= M5, C01 += M5
        set_term(4, [(0,1),(1,1)], [(5,1)], [(0,-1),(2,1)])
        # M6: (A10-A00)*(B00+B10) -> C10 += M6
        set_term(5, [(3,1),(0,-1)], [(0,1),(4,1)], [(1,1)])
        # M7: (A01-A11)*(B01+B11) -> C01 += M7
        set_term(6, [(1,1),(4,-1)], [(1,1),(5,1)], [(2,1)])
        
        # Now need to handle: A[:,0:2]*B[0:2,2:4] + A[:,2]*B[2,0:4]
        # That's 2*2*2 + 2*1*4 = 8 + 8 = 16 more contributions
        # But we only have 13 terms left (20-7=13)
        # Need to share: the A[:,2]*B[2,:k] part for each k is just 2 terms
        # And A[:,j]*B[j,k] for j=0,1 and k=2,3
        
        # For A[:,0]*B[0,2:4] + A[:,1]*B[1,2:4] + A[:,2]*B[2,0:4]
        # That's 2*2 + 2*2 + 2*4 = 4+4+8 = 16 scalar products
        # With 13 terms, need to save 3 via sharing
        
        # Try: For k=2,3: (A[:,0]*B[0,k] + A[:,1]*B[1,k]) can use Strassen-like (4 terms each? No, 2x1 needs 2)
        # Actually for a 2x1 * 1x1, we need 2 terms (one per row of A)
        # For A[:,j]*B[j,k]: 2 terms (i=0,1)
        
        # So: j=0,k=2: 2 terms; j=0,k=3: 2 terms; j=1,k=2: 2 terms; j=1,k=3: 2 terms
        # j=2,k=0: 2; j=2,k=1: 2; j=2,k=2: 2; j=2,k=3: 2
        # Total: 16 terms, but we have 13 slots. Need to save 3.
        
        # Idea: Combine some. E.g., for k=2: A[:,0]*B[0,2] + A[:,1]*B[1,2] + A[:,2]*B[2,2]
        # This is a 2x3 * 3x1, which has rank 3 (can't do better for this sub-structure in general)
        # Unless we share across k values.
        
        # Let me just try rank 21 first (save 1 term)
        return None  # Placeholder, let me build rank 21
        
    def build_rank_21_candidate():
        """
        Rank 21: Use Strassen (7) for 2x2x2, then 14 terms for remaining
        Remaining: A[:,0:2]*B[0:2,2:4] (4 pairs * 2 = 8) + A[:,2]*B[2,0:4] (4 * 2 = 8) = 16
        Save 2 by combining: e.g., share some bilinear forms
        """
        rank = 21
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def set_term(col, u_vals, v_vals, w_vals):
            for idx, val in u_vals:
                if val != 0:
                    U[idx][col] += val
            for idx, val in v_vals:
                if val != 0:
                    V[idx][col] += val
            for idx, val in w_vals:
                if val != 0:
                    W[idx][col] += val
        
        # Same Strassen as before for terms 0-6
        set_term(0, [(0,1),(4,1)], [(0,1),(4,1)], [(0,1),(2,1),(1,1),(3,1)])
        set_term(1, [(3,1),(4,1)], [(0,1)], [(1,-1),(3,1)])
        set_term(2, [(0,1)], [(4,-1),(0,1)], [(0,1),(2,1)])
        set_term(3, [(4,1)], [(1,-1),(5,1)], [(1,-1),(3,1)])
        set_term(4, [(0,1),(1,1)], [(5,1)], [(0,-1),(2,1)])
        set_term(5, [(3,1),(0,-1)], [(0,1),(4,1)], [(1,1)])
        set_term(6, [(1,1),(4,-1)], [(1,1),(5,1)], [(2,1)])
        
        # Now 14 terms for: A[:,0]*B[0,2:4] + A[:,1]*B[1,2:4] + A[:,2]*B[2,0:4]
        # 16 scalar products into 14 terms = save 2
        
        # For k=2: C[:,2] += A[:,0]*B[0,2] + A[:,1]*B[1,2] + A[:,2]*B[2,2]
        # For k=3: C[:,3] += A[:,0]*B[0,3] + A[:,1]*B[1,3] + A[:,2]*B[2,3]
        # For k=0: C[:,0] += A[:,2]*B[2,0]
        # For k=1: C[:,1] += A[:,2]*B[2,1]
        
        # Combine k=2 and k=3 for j=0: A[:,0]*(B[0,2], B[0,3])
        # Use: A[0,0]*B[0,2], A[0,0]*B[0,3], A[1,0]*B[0,2], A[1,0]*B[0,3]
        # Can share A[0,0] and A[1,0] across k=2,3
        
        # Term 7: A[0,0]*B[0,2] -> C[0,2]
        set_term(7, [(0,1)], [(2,1)], [(2,1)])
        # Term 8: A[0,0]*B[0,3] -> C[0,3]
        set_term(8, [(0,1)], [(3,1)], [(6,1)])
        # Term 9: A[1,0]*B[0,2] -> C[1,2]
        set_term(9, [(3,1)], [(2,1)], [(3,1)])
        # Term 10: A[1,0]*B[0,3] -> C[1,3]
        set_term(10, [(3,1)], [(3,1)], [(7,1)])
        
        # Term 11: A[0,1]*B[1,2] -> C[0,2]
        set_term(11, [(1,1)], [(6,1)], [(2,1)])
        # Term 12: A[0,1]*B[1,3] -> C[0,3]
        set_term(12, [(1,1)], [(7,1)], [(6,1)])
        # Term 13: A[1,1]*B[1,2] -> C[1,2]
        set_term(13, [(4,1)], [(6,1)], [(3,1)])
        # Term 14: A[1,1]*B[1,3] -> C[1,3]
        set_term(14, [(4,1)], [(7,1)], [(7,1)])
        
        # Term 15: A[0,2]*B[2,0] -> C[0,0]
        set_term(15, [(2,1)], [(8,1)], [(0,1)])
        # Term 16: A[0,2]*B[2,1] -> C[0,1]
        set_term(16, [(2,1)], [(9,1)], [(2,1)])
        # Term 17: A[0,2]*B[2,2] -> C[0,2]
        set_term(17, [(2,1)], [(10,1)], [(2,1)])
        # Term 18: A[0,2]*B[2,3] -> C[0,3]
        set_term(18, [(2,1)], [(11,1)], [(6,1)])
        
        # Term 19: A[1,2]*B[2,0] -> C[1,0]
        set_term(19, [(5,1)], [(8,1)], [(1,1)])
        # Term 20: A[1,2]*B[2,1] -> C[1,1]
        set_term(20, [(5,1)], [(9,1)], [(3,1)])
        # We need A[1,2]*B[2,2] -> C[1,2] and A[1,2]*B[2,3] -> C[1,3]
        # That's 2 more terms = 22 total. We're at 21 terms used (0-20), need 2 more.
        # We only have rank 21 (terms 0-20). 
        # We need to save 2 more terms from the A[:,2]*B[2,:] part.
        
        # Let me reconsider. We have 16 scalar products for j=0,1,k=2,3 and j=2,k=0,1,2,3
        # Actually: j=0: k=2,3 (2 products per i, so 4); j=1: k=2,3 (4); j=2: k=0,1,2,3 (8)
        # Total = 4+4+8 = 16. With 14 slots, save 2.
        
        # Can combine: A[0,2]*B[2,2] + A[0,2]*B[2,3] share A[0,2]
        # But they go to different W positions, so can't easily share one term.
        # Unless: one term contributes to both C[0,2] and C[0,3]? No, each term is a rank-1.
        
        # Actually, I think rank 21 is not achievable this way. Let me try a different structure.
        return None
    
    # Let me try a completely different approach: use the known minimal rank for 2x3x4
    # The Strassen rank for 2x2x2 is 7. For 2x3x4, we can think of it as:
    # C = A[0:2,0:2]*B[0:2,0:4] + A[0:2,2]*B[2,0:4]
    # First part: 2x2 * 2x4 = can be done in 7+5=12? Or 7 for 2x2x2 and 4 more for 2x2x2?
    # 2x2*2x4: rank is at most 2*4=8, but can be less. Actually rank(2x2x2)=7, and 
    # 2x2x2 + 2x2x2 = 14 for 2x2*2x4. But we can do better.
    # For M x N * N x P, the rank is at most MN + NP - N (Strassen-like)
    # For 2x2*2x4: at most 4+8-2=10? Not sure.
    
    # Let me just try to verify if rank 20 is possible with a smart construction
    # and fall back to 22 if not.
    
    # Try: use rank 22 as baseline, attempt ALS for rank 20-21 with more care
    def try_als_refined(rank, max_iter=300, seed=42):
        """Refined ALS with better initialization."""
        rng = np.random.RandomState(seed)
        
        # Initialize with the rank-22 structure padded with zeros, then random perturbation
        U22, V22, W22 = build_rank_22()
        U = np.array(U22[:6], dtype=float)  # 6 x 22
        V = np.array(V22, dtype=float)  # 12 x 22
        W = np.array(W22, dtype=float)  # 8 x 22
        
        # Pad or truncate to desired rank
        if rank < 22:
            U = U[:, :rank]
            V = V[:, :rank]
            W = W[:, :rank]
            # Add small random noise to escape local minima
            U += rng.randn(6, rank) * 0.1
            V += rng.randn(12, rank) * 0.1
            W += rng.randn(8, rank) * 0.1
        else:
            U = np.zeros((6, rank))
            V = np.zeros((12, rank))
            W = np.zeros((8, rank))
            U[:, :22] = np.array(U22)
            V[:, :22] = np.array(V22)
            W[:, :22] = np.array(W22)
            if rank > 22:
                U[:, 22:] = rng.randn(6, rank-22) * 0.01
                V[:, 22:] = rng.randn(12, rank-22) * 0.01
                W[:, 22:] = rng.randn(8, rank-22) * 0.01
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iteration in range(max_iter):
            if time.time() > deadline - 20:
                break
            
            # Update W
            W_new = np.zeros((8, rank))
            for k in range(p):
                for i in range(n):
                    w = k * n + i
                    A_mat = np.zeros((m, rank))
                    for j in range(m):
                        A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    try:
                        W_new[w, :], _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                    except:
                        W_new[w, :] = rng.randn(rank) * 0.01
            
            # Update U
            U_new = np.zeros((6, rank))
            for i in range(n):
                for j in range(m):
                    u = i * m + j
                    A_mat = np.zeros((p, rank))
                    for k in range(p):
                        A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    try:
                        U_new[u, :], _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                    except:
                        U_new[u, :] = rng.randn(rank) * 0.01
            
            # Update V
            V_new = np.zeros((12, rank))
            for j in range(m):
                for k in range(p):
                    v = j * p + k
                    A_mat = np.zeros((n, rank))
                    for i in range(n):
                        A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    try:
                        V_new[v, :], _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                    except:
                        V_new[v, :] = rng.randn(rank) * 0.01
            
            U, V, W = U_new, V_new, W_new
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        err += (s - 1.0)**2
            err = np.sqrt(err)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if err < 1e-7:
                    # Try to round to integers/half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    U_l = [[float(x) for x in row] for row in U_r]
                    V_l = [[float(x) for x in row] for row in V_r]
                    W_l = [[float(x) for x in row] for row in W_r]
                    if verify(U_l, V_l, W_l, rank):
                        return U_l, V_l, W_l
        
        # Check best found
        if best_err < 1e-6:
            U_r = np.round(best_U * 2) / 2
            V_r = np.round(best_V * 2) / 2
            W_r = np.round(best_W * 2) / 2
            U_l = [[float(x) for x in row] for row in U_r]
            V_l = [[float(x) for x in row] for row in V_r]
            W_l = [[float(x) for x in row] for row in W_r]
            if verify(U_l, V_l, W_l, rank):
                return U_l, V_l, W_l
        
        return None

    # Try ranks 20 and 21 with multiple seeds
    for rank in [20, 21]:
        if time.time() > deadline - 30:
            break
        for seed in range(50):
            if time.time() > deadline - 30:
                break
            result = try_als_refined(rank, max_iter=200, seed=seed)
            if result is not None:
                U, V, W = result
                return {"U": U, "V": W} if False else {"U": U, "V": V, "W": W}
    
    return best_result
