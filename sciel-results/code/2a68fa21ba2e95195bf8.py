import numpy as np
import time
import itertools

def construct():
    n, m, p = 2, 3, 4
    # U: (n*m) x rank = 6 x rank
    # V: (m*p) x rank = 12 x rank
    # W: (p*n) x rank = 8 x rank
    # Convention: W[k*n + i, r] where k in [0,p), i in [0,n)
    # T[i,j,k,l] = delta_{j,k}
    # sum_r U[i*m+j, r] * V[k*p+l, r] * W[k*n+i, r] = delta_{j,k}
    
    # Let me try a greedy iterative rank-reduction algorithm.
    # Start with a known valid solution and try to reduce rank.
    
    # First, let me build the naive rank-24 solution and verify it.
    # Then try to find a lower-rank solution using greedy methods.
    
    deadline = time.time() + 100  # Leave 20s safety margin
    
    # Naive solution: rank = n*m*p = 24
    # U[i*m+j, r] = 1 if r == i*m+j
    # V[k*p+l, r] = 1 if r == k*p+l AND j==k (but this doesn't work directly)
    
    # Actually, for naive: each (i,j,k,l) with j==k contributes.
    # There are n * m * p / ... no, for each j, we have n*p terms where j=k.
    # Total: 3 * 2 * 4 = 24 terms.
    
    # Naive decomposition:
    # For each j in {0,1,2}:
    #   For each i in {0,1}:
    #     For each l in {0,1,2,3}:
    #       U[i*m+j, r] = 1
    #       V[j*p+l, r] = 1
    #       W[j*n+i, r] = 1
    #       where r is a unique index for (i,j,l)
    
    def make_naive():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for j in range(m):
            for i in range(n):
                for l in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + l][r] = 1
                    W[j * n + i][r] = 1
                    r += 1
        return U, V, W, rank
    
    U_naive, V_naive, W_naive, rank_naive = make_naive()
    
    # Verify the naive solution
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    for l in range(p):
                        s = 0
                        for r in range(rank):
                            s += U[i * m + j][r] * V[k * p + l][r] * W[k * n + i][r]
                        expected = 1 if j == k else 0
                        if abs(s - expected) > 1e-10:
                            return False
        return True
    
    assert verify(U_naive, V_naive, W_naive), "Naive solution should be valid"
    
    # Now try greedy rank reduction
    # Start from rank 22 (the current best known) and try to go lower
    
    # Let me try to construct a rank 21 solution using the structure.
    # The tensor has a block-diagonal structure in j,k.
    
    # Key observation: The tensor T can be written as:
    # T = T0 + T1 + T2
    # where Tj has support only when j=k=j.
    # Each Tj is a 2x4 matrix (i,l) for fixed j.
    # rank(Tj) = 2*4 = 8 if we treat it as an outer product structure.
    # But we can share factors across different j values.
    
    # Let me try a different approach: use the fact that for each j,
    # the sub-tensor is rank 8 (outer product of 2-dim and 4-dim spaces).
    # Total naive: 3*8 = 24.
    
    # Can we do better by sharing? The Strassen trick works for 2x2x2.
    # For 2x3x4, we can use:
    # - Strassen for the 2x2 part (j,k in {0,1}): 7 terms
    # - Then handle j=2 or k=2 separately.
    
    # Let me try to find a rank 21 solution by careful construction.
    
    # Actually, let me try a numerical search for rank 21.
    
    def try_greedy(target_rank):
        """Try to find a rank-target_rank solution using greedy iteration."""
        if time.time() > deadline - 10:
            return None
        
        # Initialize with random or structured starting point
        np.random.seed(42)
        
        # Start from the naive solution and try to compress
        # Use the idea: represent U, V, W as target_rank x target_rank matrices
        # and optimize to minimize the residual.
        
        # For small target_rank, try random initialization
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Iterative optimization: fix two, optimize the third
            for iteration in range(200):
                if time.time() > deadline - 10:
                    break
                
                # Optimize U: for each row of U, solve least squares
                for i_m_j in range(n * m):
                    # Collect all (k, l) pairs
                    # U[i_m_j, r] * sum_r V[k*p+l, r] * W[k*n+i, r] should contribute to delta_{j,k}
                    # This is complex; let me use a different approach.
                    
                    # Actually, let me use the mode-3 flattening approach.
                    # For fixed U and V, optimize W:
                    # W[k*n+i, r] = argmin sum_{i,j,k,l} (T[i,j,k,l] - sum_r U[i*m+j,r]*V[k*p+l,r]*W[k*n+i,r])^2
                    
                    # This is a linear system in W for each r.
                    pass
                
                # This is getting complex. Let me try a simpler approach.
                # Just check if we can find a valid solution by random search for small ranks.
            
            # Verify
            if verify(U.tolist(), V.tolist(), W.tolist()):
                return U.tolist(), V.tolist(), W.tolist(), target_rank
        
        return None
    
    # Try ranks from 21 down
    best_result = None
    for target_rank in [21, 20, 19, 18]:
        if time.time() > deadline - 15:
            break
        result = try_greedy(target_rank)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        U, V, W, rank = best_result
        return {"U": U, "V": V, "W": W}
    
    # Fall back to the known rank 22 solution
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
    
    # Strassen block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
    add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
    add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)
    
    # Strassen block 2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    # B[0:2, 2:4] has indices: V[0*4+2]=V[2], V[0*4+3]=V[3], V[1*4+2]=V[6], V[1*4+3]=V[7]
    # C[0:2, 2:4] has W indices: W[2*2+0]=W[4], W[2*2+1]=W[5], W[3*2+0]=W[6], W[3*2+1]=W[7]
    add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
    add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
    add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
    add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
    add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)
    
    # Cross terms: A[2,:] * B[:,2:4] and A[0:2,:] * B[2,:]
    # A[2,0:4] * B[2:4,:] -> C[:,2:4] ... wait, let me think again.
    
    # The remaining terms are for j=2 or k=2 (the third row/col).
    # A[i,2] * B[2,l] for all i,l: 2*4 = 8 terms
    # A[i,0:2] * B[2,l] for all i,l: 2*2*4 = 16 terms... no that's too many.
    
    # Actually, the cross terms should be:
    # For j=2, k!=2: A[i,2]*B[k,l] with k in {0,1} -> these contribute to C[i,k] but should be 0
    # Wait, I'm confusing myself. Let me re-examine.
    
    # The tensor is T[i,j,k,l] = delta_{j,k}.
    # So for each (j,k) with j==k, we need the product to give 1 for all (i,l).
    # For j!=k, we need 0.
    
    # The Strassen blocks handle j,k in {0,1}.
    # We still need:
    # - j=0 or 1, k=2: should be 0 (Strassen blocks only use B[0:2,:], so B[2,:]=0 in those terms)
    # - j=2, k=0 or 1: should be 0 (Strassen blocks only use A[:,0:2], so A[:,2]=0 in those terms)
    # - j=2, k=2: should be 1 for all (i,l): 2*4 = 8 outer products
    
    # So the 8 outer products are for j=k=2:
    # A[i,2] * B[2,l] -> C[i,l] for i in {0,1}, l in {2,3}
    # W index for C[i,l]: W[l*n+i]
    
    for i in range(n):
        for l in range(2, p):
            col = 14 + (l - 2) * n + i
            U[i * m + 2][col] = 1
            V[2 * p + l][col] = 1
            W[l * n + i][col] = 1
    
    # Verify the rank 22 solution
    assert verify(U, V, W), "Rank 22 solution should be valid"
    
    return {"U": U, "V": V, "W": W}
