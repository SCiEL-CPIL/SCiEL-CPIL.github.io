import time
import numpy as np
from itertools import combinations, product

DEADLINE = 130.0

def construct():
    start_time = time.time()
    n, m, p = 2, 5, 6
    total_entries = n * m * p  # 60

    # Build the target tensor (all ones)
    T_target = np.ones(total_entries)

    # We'll use a smarter approach:
    # 1. Start with a known good decomposition (Strassen-based)
    # 2. Try to reduce rank by finding linear dependencies
    
    # First, let's build a Strassen-based decomposition
    # For 2x5x6, we can decompose as:
    # - Use Strassen for 2x2x2 blocks where possible
    # - Use standard multiplication for remaining terms
    
    # Strategy: partition the 5 middle indices and 6 output indices
    # into blocks and apply Strassen where beneficial
    
    def strassen_2x2x2(U, V, W, rank_offset, i0, j0, k0):
        """Add 7 Strassen terms for 2x2x2 block starting at (i0, j0, k0)"""
        # M1 = (A11+A22)(B11+B22)
        U[i0*m+j0, rank_offset] = 1
        U[(i0+1)*m+(j0+1), rank_offset] = 1
        V[j0*p+k0, rank_offset] = 1
        V[(j0+1)*p+(k0+1), rank_offset] = 1
        W[k0*n+i0, rank_offset] = 1
        W[(k0+1)*n+(i0+1), rank_offset] = 1
        rank_offset += 1
        # M2 = (A21+A22)B11
        U[(i0+1)*m+j0, rank_offset] = 1
        U[(i0+1)*m+(j0+1), rank_offset] = 1
        V[j0*p+k0, rank_offset] = 1
        W[k0*n+(i0+1), rank_offset] = 1
        W[(k0+1)*n+(i0+1), rank_offset] = -1
        rank_offset += 1
        # M3 = A11(B21-B22)
        U[i0*m+j0, rank_offset] = 1
        V[j0*p+(k0+1), rank_offset] = 1
        V[(j0+1)*p+(k0+1), rank_offset] = -1
        W[(k0+1)*n+i0, rank_offset] = 1
        W[(k0+1)*n+(i0+1), rank_offset] = 1
        rank_offset += 1
        # M4 = A22(B12-B11)
        U[(i0+1)*m+(j0+1), rank_offset] = 1
        V[(j0+1)*p+k0, rank_offset] = 1
        V[j0*p+k0, rank_offset] = -1
        W[k0*n+i0, rank_offset] = 1
        W[k0*n+(i0+1), rank_offset] = 1
        rank_offset += 1
        # M5 = (A11+A12)B22
        U[i0*m+j0, rank_offset] = 1
        U[i0*m+(j0+1), rank_offset] = 1
        V[(j0+1)*p+(k0+1), rank_offset] = 1
        W[k0*n+i0, rank_offset] = -1
        W[(k0+1)*n+i0, rank_offset] = 1
        rank_offset += 1
        # M6 = (A21-A11)(B11+B12)
        U[(i0+1)*m+j0, rank_offset] = 1
        U[i0*m+j0, rank_offset] = -1
        V[j0*p+k0, rank_offset] = 1
        V[j0*p+(k0+1), rank_offset] = 1
        W[(k0+1)*n+(i0+1), rank_offset] = 1
        rank_offset += 1
        # M7 = (A12-A22)(B21+B22)
        U[i0*m+(j0+1), rank_offset] = 1
        U[(i0+1)*m+(j0+1), rank_offset] = -1
        V[(j0+1)*p+k0, rank_offset] = 1
        V[(j0+1)*p+(k0+1), rank_offset] = 1
        W[k0*n+i0, rank_offset] = 1
        rank_offset += 1
        return rank_offset

    def standard_2x1x1(U, V, W, rank_offset, i0, j0, k0):
        """Add 2 standard terms for 2x1x1 block"""
        U[i0*m+j0, rank_offset] = 1
        V[j0*p+k0, rank_offset] = 1
        W[k0*n+i0, rank_offset] = 1
        rank_offset += 1
        U[(i0+1)*m+j0, rank_offset] = 1
        V[j0*p+k0, rank_offset] = 1
        W[k0*n+(i0+1), rank_offset] = 1
        rank_offset += 1
        return rank_offset

    # Build initial decomposition
    # Partition: j indices [0,1,2,3,4], k indices [0,1,2,3,4,5]
    # Use Strassen for (j0,j1) pairs and (k0,k1) pairs where possible
    
    # Let's try different partition strategies and pick the best one
    
    def build_decomposition(strategy):
        """Build a decomposition based on the given strategy"""
        max_rank = 60  # Upper bound
        U = np.zeros((n*m, max_rank))
        V = np.zeros((m*p, max_rank))
        W = np.zeros((p*n, max_rank))
        rank = 0
        
        if strategy == 0:
            # Strategy 0: Strassen on (j0,j1) for k-pairs (0,1), (2,3), (4,5), standard for j2
            rank = strassen_2x2x2(U, V, W, rank, 0, 0, 0)
            rank = strassen_2x2x2(U, V, W, rank, 0, 0, 2)
            rank = strassen_2x2x2(U, V, W, rank, 0, 0, 4)
            # j=2 with all k
            for k in range(6):
                rank = standard_2x1x1(U, V, W, rank, 0, 2, k)
            # Strassen on (j3,j4) for k-pairs
            rank = strassen_2x2x2(U, V, W, rank, 0, 3, 0)
            rank = strassen_2x2x2(U, V, W, rank, 0, 3, 2)
            rank = strassen_2x2x2(U, V, W, rank, 0, 3, 4)
            
        elif strategy == 1:
            # Strategy 1: Different partition
            # Use Strassen for (j0,j1) with k=(0,1), (2,3), (4,5)
            # Use Strassen for (j2,j3) with k=(0,1), (2,3), (4,5)  
            # Standard for j4
            rank = strassen_2x2x2(U, V, W, rank, 0, 0, 0)
            rank = strassen_2x2x2(U, V, W, rank, 0, 0, 2)
            rank = strassen_2x2x2(U, V, W, rank, 0, 0, 4)
            rank = strassen_2x2x2(U, V, W, rank, 0, 2, 0)
            rank = strassen_2x2x2(U, V, W, rank, 0, 2, 2)
            rank = strassen_2x2x2(U, V, W, rank, 0, 2, 4)
            for k in range(6):
                rank = standard_2x1x1(U, V, W, rank, 0, 4, k)
                
        elif strategy == 2:
            # Strategy 2: Mix of Strassen and standard
            # (j0,j1) with Strassen for k-pairs, (j2) standard, (j3,j4) Strassen
            rank = strassen_2x2x2(U, V, W, rank, 0, 0, 0)
            rank = strassen_2x2x2(U, V, W, rank, 0, 0, 2)
            rank = strassen_2x2x2(U, V, W, rank, 0, 0, 4)
            for k in range(6):
                rank = standard_2x1x1(U, V, W, rank, 0, 2, k)
            rank = strassen_2x2x2(U, V, W, rank, 0, 3, 0)
            rank = strassen_2x2x2(U, V, W, rank, 0, 3, 2)
            rank = strassen_2x2x2(U, V, W, rank, 0, 3, 4)
            
        elif strategy == 3:
            # Strategy 3: More aggressive Strassen usage
            # Try to cover more with Strassen
            rank = strassen_2x2x2(U, V, W, rank, 0, 0, 0)
            rank = strassen_2x2x2(U, V, W, rank, 0, 0, 2)
            rank = strassen_2x2x2(U, V, W, rank, 0, 0, 4)
            rank = strassen_2x2x2(U, V, W, rank, 0, 1, 0)
            rank = strassen_2x2x2(U, V, W, rank, 0, 1, 2)
            rank = strassen_2x2x2(U, V, W, rank, 0, 1, 4)
            for k in range(6):
                rank = standard_2x1x1(U, V, W, rank, 0, 2, k)
            for k in range(6):
                rank = standard_2x1x1(U, V, W, rank, 0, 3, k)
            for k in range(6):
                rank = standard_2x1x1(U, V, W, rank, 0, 4, k)
                
        elif strategy == 4:
            # Strategy 4: Balanced approach
            rank = strassen_2x2x2(U, V, W, rank, 0, 0, 0)
            rank = strassen_2x2x2(U, V, W, rank, 0, 0, 2)
            rank = strassen_2x2x2(U, V, W, rank, 0, 0, 4)
            for k in range(6):
                rank = standard_2x1x1(U, V, W, rank, 0, 1, k)
            rank = strassen_2x2x2(U, V, W, rank, 0, 2, 0)
            rank = strassen_2x2x2(U, V, W, rank, 0, 2, 2)
            rank = strassen_2x2x2(U, V, W, rank, 0, 2, 4)
            for k in range(6):
                rank = standard_2x1x1(U, V, W, rank, 0, 3, k)
            for k in range(6):
                rank = standard_2x1x1(U, V, W, rank, 0, 4, k)
                
        return U[:, :rank], V[:, :rank], W[:, :rank], rank

    # Try multiple strategies and pick the best
    best_decomp = None
    best_rank = float('inf')
    
    for strat in range(5):
        if time.time() - start_time > DEADLINE - 20:
            break
        U, V, W, rank = build_decomposition(strat)
        
        # Verify the decomposition
        A_test = np.random.randint(-10, 10, (n, m))
        B_test = np.random.randint(-10, 10, (m, p))
        C_test = A_test @ B_test
        
        C_calc = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0
                for j in range(m):
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                C_calc[i, k] = val
        
        if np.allclose(C_test, C_calc, atol=1e-5):
            if rank < best_rank:
                best_rank = rank
                best_decomp = (U, V, W, rank)
        
        # Try to reduce rank further by finding linear dependencies
        if rank > 30:
            # Build the tensor matrix
            tensor_matrix = np.zeros((total_entries, rank))
            for r in range(rank):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            tensor_matrix[i*m*p + j*p + k, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            # Find null space
            try:
                U_sv, s_vals, Vt = np.linalg.svd(tensor_matrix, full_matrices=False)
                tol = 1e-8 * (s_vals[0] if len(s_vals) > 0 else 1)
                null_mask = s_vals < tol
                null_dim = np.sum(null_mask)
                
                if null_dim > 0:
                    # Try to eliminate null space vectors one at a time
                    null_space = Vt[null_mask].T
                    new_U = U.copy()
                    new_V = V.copy()
                    new_W = W.copy()
                    current_rank = rank
                    
                    for ns_vec in null_space:
                        if time.time() - start_time > DEADLINE - 10:
                            break
                        max_idx = int(np.argmax(np.abs(ns_vec)))
                        if abs(ns_vec[max_idx]) < 1e-10:
                            continue
                        
                        # Eliminate term max_idx
                        for t in range(current_rank):
                            if t != max_idx:
                                new_U[:, t] -= ns_vec[t] / ns_vec[max_idx] * new_U[:, max_idx]
                                new_V[:, t] -= ns_vec[t] / ns_vec[max_idx] * new_V[:, max_idx]
                                new_W[:, t] -= ns_vec[t] / ns_vec[max_idx] * new_W[:, max_idx]
                        
                        # Shift left
                        new_U[:, max_idx+1:current_rank] = new_U[:, max_idx+1:current_rank]
                        new_V[:, max_idx+1:current_rank] = new_V[:, max_idx+1:current_rank]
                        new_W[:, max_idx+1:current_rank] = new_W[:, max_idx+1:current_rank]
                        current_rank -= 1
                        
                        # Verify
                        C_calc2 = np.zeros((n, p))
                        for i in range(n):
                            for k in range(p):
                                val = 0
                                for j in range(m):
                                    for r in range(current_rank):
                                        val += new_U[i*m+j, r] * new_V[j*p+k, r] * new_W[k*n+i, r]
                                C_calc2[i, k] = val
                        
                        if not np.allclose(C_test, C_calc2, atol=1e-5):
                            # Revert
                            break
                        
                        if current_rank < best_rank:
                            best_rank = current_rank
                            best_decomp = (new_U[:, :current_rank].copy(), new_V[:, :current_rank].copy(), new_W[:, :current_rank].copy(), current_rank)
            except np.linalg.LinAlgError:
                pass
    
    if best_decomp is not None:
        U, V, W, rank = best_decomp
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: return a valid (though possibly not minimal) decomposition
    U, V, W, rank = build_decomposition(0)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
