import numpy as np
import time
from itertools import combinations

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    # Build the target tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Build the rank-28 Strassen-based decomposition
    def build_strassen_28():
        rank = 28
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0

        def add_strassen(i1, i2, j1, j2, k1, k2):
            nonlocal col
            # M1 = (A00 + A11)(B00 + B11)
            U[i1*m+j1, col] = 1; U[i2*m+j2, col] = 1
            V[j1*p+k1, col] = 1; V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = 1; W[k2*n+i2, col] = 1
            col += 1
            # M2 = (A10 + A11) B00
            U[i2*m+j1, col] = 1; U[i2*m+j2, col] = 1
            V[j1*p+k1, col] = 1
            W[k1*n+i2, col] = 1; W[k2*n+i2, col] = -1
            col += 1
            # M3 = A00 (B01 - B11)
            U[i1*m+j1, col] = 1
            V[j1*p+k2, col] = 1; V[j2*p+k2, col] = -1
            W[k2*n+i1, col] = 1; W[k2*n+i2, col] = 1
            col += 1
            # M4 = A11 (B10 - B00)
            U[i2*m+j2, col] = 1
            V[j2*p+k1, col] = 1; V[j1*p+k1, col] = -1
            W[k1*n+i1, col] = 1; W[k1*n+i2, col] = 1
            col += 1
            # M5 = (A00 + A01) B11
            U[i1*m+j1, col] = 1; U[i1*m+j2, col] = 1
            V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = -1; W[k2*n+i1, col] = 1
            col += 1
            # M6 = (A10 - A00)(B00 + B01)
            U[i2*m+j1, col] = 1; U[i1*m+j1, col] = -1
            V[j1*p+k1, col] = 1; V[j1*p+k2, col] = 1
            W[k2*n+i2, col] = 1
            col += 1
            # M7 = (A01 - A11)(B10 + B11)
            U[i1*m+j2, col] = 1; U[i2*m+j2, col] = -1
            V[j2*p+k1, col] = 1; V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = 1
            col += 1

        def add_outer(i, j, k):
            nonlocal col
            U[i*m+j, col] = 1
            V[j*p+k, col] = 1
            W[k*n+i, col] = 1
            col += 1

        # 2x2 * 2x2 Strassen blocks for j in {0,1}, k in {0,1} and k in {2,3}
        add_strassen(0, 1, 0, 1, 0, 1)
        add_strassen(0, 1, 0, 1, 2, 3)
        # Schoolbook for remaining
        # A[:,2] * B[2, 0:2]
        for i in range(n):
            for k in range(2):
                add_outer(i, 2, k)
        # A[:,0:2] * B[0:2, 4]
        for j in range(2):
            for i in range(n):
                add_outer(i, j, 4)
        # A[:,2] * B[2, 2:5]
        for i in range(n):
            for k in range(2, 5):
                add_outer(i, 2, k)

        assert col == rank, f"col={col} != {rank}"
        return U, V, W

    def check_decomposition(U, V, W, T, tol=1e-6):
        T_rec = np.einsum('ir,vr,wr->ivw', U, V, W, optimize=True)
        return np.max(np.abs(T_rec - T)) < tol

    # Build rank-28 and verify
    U28, V28, W28 = build_strassen_28()
    valid_28 = check_decomposition(U28, V28, W28, T)

    if not valid_28:
        # Fallback to schoolbook rank 30
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Now try to reduce rank below 28 using a smarter search
    # Strategy: Use the Lovasz Local Lemma inspired approach
    # 1. Identify independent components in the tensor structure
    # 2. Try to find rank-1 terms that can be "peeled" while maintaining decomposability

    best_rank = 28
    best_result = (U28, V28, W28)

    # Approach 1: Try to find rank-27 decomposition via structured search
    # The key insight: in the 2x3x5 case, we can try to merge some schoolbook terms
    # with Strassen terms in a way that reduces total rank

    # Let's try a different block decomposition
    # Instead of 2x2 Strassen + schoolbook, try 3x1 or other partitions

    def build_alternative_decomposition():
        """Try alternative block decompositions that might give rank < 28"""
        
        # Try: partition A into [2x2, 1x2] and B into [2x2, 2x2, 1x2]
        # Use Strassen for 2x2*2x2 blocks, schoolbook for cross terms
        
        # Actually, let's try a more systematic approach:
        # Enumerate all possible 2x2 block positions and use Strassen where beneficial
        
        # For 2x3x5:
        # A is 2x3, B is 3x5
        # Possible 2x2 sub-blocks of A: columns (0,1), (1,2)
        # Possible 2x2 sub-blocks of B: rows (0,1), (1,2), (2,3), (3,4)
        
        # We already use: A[:,0:2] * B[0:2,0:2] and A[:,0:2] * B[0:2,2:4]
        # Remaining: A[:,0:2]*B[0:2,4], A[:,2]*B[2,0:5]
        
        # Can we do better? Let's try A[:,1:3] * B[1:3,1:3] with Strassen
        # This would cover: (1,1,1), (1,1,2), (1,2,1), (1,2,2)
        # But we'd double-count (1,1,1) and (1,2,1) etc.
        
        # The issue is that Strassen requires disjoint blocks.
        
        # Let's try a different approach: use the fact that for 2x3x5,
        # the optimal rank might be 27 or lower.
        
        # Try to find a rank-27 decomposition by searching over
        # structured families of decompositions.
        
        return None  # Placeholder

    # Approach 2: Use alternating minimization with multiple restarts
    # to try to find a rank-27 decomposition
    
    if time_left() > 30:
        # Try rank 27
        R = 27
        found = False
        
        # Try multiple random initializations
        for attempt in range(10):
            if time_left() < 20:
                break
            
            # Initialize from rank-28 by dropping one column and adding noise
            U = U28[:, :R].copy()
            V = V28[:, :R].copy()
            W = W28[:, :R].copy()
            
            # Add small noise to escape local minima
            scale = 0.01 * (attempt + 1)
            U += scale * np.random.randn(nm, R)
            V += scale * np.random.randn(mp, R)
            W += scale * np.random.randn(pn, R)
            
            # Alternating minimization with more iterations
            for it in range(50):
                if time_left() < 15:
                    break
                
                # Solve for W
                B = np.zeros((nm * mp, R))
                for r in range(R):
                    B[:, r] = np.outer(U[:, r], V[:, r]).flatten()
                W_new = np.zeros((pn, R))
                for w in range(pn):
                    b_w = T[:, :, w].flatten()
                    x, _, _, _ = np.linalg.lstsq(B, b_w, rcond=None)
                    W_new[w, :] = x
                W = W_new
                
                # Solve for U
                C = np.zeros((mp * pn, R))
                for r in range(R):
                    C[:, r] = np.outer(V[:, r], W[:, r]).flatten()
                U_new = np.zeros((nm, R))
                for i in range(nm):
                    b_i = T[i, :, :].flatten()
                    x, _, _, _ = np.linalg.lstsq(C, b_i, rcond=None)
                    U_new[i, :] = x
                U = U_new
                
                # Solve for V
                D = np.zeros((nm * pn, R))
                for r in range(R):
                    D[:, r] = np.outer(U[:, r], W[:, r]).flatten()
                V_new = np.zeros((mp, R))
                for v in range(mp):
                    b_v = T[:, v, :].flatten()
                    x, _, _, _ = np.linalg.lstsq(D, b_v, rcond=None)
                    V_new[v, :] = x
                V = V_new
            
            # Check and snap to half-integers
            if check_decomposition(U, V, W, T, tol=1e-4):
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                if check_decomposition(U_snap, V_snap, W_snap, T, tol=1e-6):
                    best_rank = 27
                    best_result = (U_snap, V_snap, W_snap)
                    found = True
                    break
        
        # If we found rank 27, try to go lower
        if found and time_left() > 20:
            R = 26
            for attempt in range(5):
                if time_left() < 10:
                    break
                
                U = best_result[0][:, :R].copy()
                V = best_result[1][:, :R].copy()
                W = best_result[2][:, :R].copy()
                
                U += 0.01 * np.random.randn(nm, R)
                V += 0.01 * np.random.randn(mp, R)
                W += 0.01 * np.random.randn(pn, R)
                
                for it in range(30):
                    if time_left() < 5:
                        break
                    
                    B = np.zeros((nm * mp, R))
                    for r in range(R):
                        B[:, r] = np.outer(U[:, r], V[:, r]).flatten()
                    W_new = np.zeros((pn, R))
                    for w in range(pn):
                        b_w = T[:, :, w].flatten()
                        x, _, _, _ = np.linalg.lstsq(B, b_w, rcond=None)
                        W_new[w, :] = x
                    W = W_new
                    
                    C = np.zeros((mp * pn, R))
                    for r in range(R):
                        C[:, r] = np.outer(V[:, r], W[:, r]).flatten()
                    U_new = np.zeros((nm, R))
                    for i in range(nm):
                        b_i = T[i, :, :].flatten()
                        x, _, _, _ = np.linalg.lstsq(C, b_i, rcond=None)
                        U_new[i, :] = x
                    U = U_new
                    
                    D = np.zeros((nm * pn, R))
                    for r in range(R):
                        D[:, r] = np.outer(U[:, r], W[:, r]).flatten()
                    V_new = np.zeros((mp, R))
                    for v in range(mp):
                        b_v = T[:, v, :].flatten()
                        x, _, _, _ = np.linalg.lstsq(D, b_v, rcond=None)
                        V_new[v, :] = x
                    V = V_new
                
                if check_decomposition(U, V, W, T, tol=1e-4):
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    if check_decomposition(U_snap, V_snap, W_snap, T, tol=1e-6):
                        best_rank = 26
                        best_result = (U_snap, V_snap, W_snap)
                        break

    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
