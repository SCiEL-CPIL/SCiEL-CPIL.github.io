"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin for 120s limit
    
    # Pre-compute the target tensor T
    # T is a 3D array of shape (n*m, m*p, p*n)
    # T[u, v, w] = 1 if indices are consistent, else 0
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    T = np.zeros((n*m, m*p, p*n), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1

    # Fallback: Rank 42 Strassen decomposition
    def get_strassen_rank_42():
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen_block(i_start, j_start, k_start, offset):
            # Local indices a,b in {0,1}
            # Global A index: (i_start+a)*m + (j_start+b)
            # Global B index: (j_start+a)*p + (k_start+b)
            # Global C index: (k_start+b)*n + (i_start+a)
            
            # M1: (A00+A11)(B00+B11) -> C00+1, C11+1
            col = offset
            U[(i_start+0)*m + (j_start+0)][col] += 1
            U[(i_start+1)*m + (j_start+1)][col] += 1
            V[(j_start+0)*p + (k_start+0)][col] += 1
            V[(j_start+1)*p + (k_start+1)][col] += 1
            W[(k_start+0)*n + (i_start+0)][col] += 1
            W[(k_start+1)*n + (i_start+1)][col] += 1
            
            # M2: (A10+A11)B00 -> C10+1, C11-1
            col = offset + 1
            U[(i_start+1)*m + (j_start+0)][col] += 1
            U[(i_start+1)*m + (j_start+1)][col] += 1
            V[(j_start+0)*p + (k_start+0)][col] += 1
            W[(k_start+0)*n + (i_start+1)][col] += 1
            W[(k_start+1)*n + (i_start+1)][col] -= 1
            
            # M3: A00(B01-B11) -> C01+1, C11+1
            col = offset + 2
            U[(i_start+0)*m + (j_start+0)][col] += 1
            V[(j_start+0)*p + (k_start+1)][col] += 1
            V[(j_start+1)*p + (k_start+1)][col] -= 1
            W[(k_start+1)*n + (i_start+0)][col] += 1
            W[(k_start+1)*n + (i_start+1)][col] += 1
            
            # M4: A11(B10-B00) -> C00+1, C10+1
            col = offset + 3
            U[(i_start+1)*m + (j_start+1)][col] += 1
            V[(j_start+1)*p + (k_start+0)][col] += 1
            V[(j_start+0)*p + (k_start+0)][col] -= 1
            W[(k_start+0)*n + (i_start+0)][col] += 1
            W[(k_start+0)*n + (i_start+1)][col] += 1
            
            # M5: (A00+A01)B11 -> C00-1, C01+1
            col = offset + 4
            U[(i_start+0)*m + (j_start+0)][col] += 1
            U[(i_start+0)*m + (j_start+1)][col] += 1
            V[(j_start+1)*p + (k_start+1)][col] += 1
            W[(k_start+0)*n + (i_start+0)][col] -= 1
            W[(k_start+1)*n + (i_start+0)][col] += 1
            
            # M6: (A10-A00)(B00+B01) -> C11+1
            col = offset + 5
            U[(i_start+1)*m + (j_start+0)][col] += 1
            U[(i_start+0)*m + (j_start+0)][col] -= 1
            V[(j_start+0)*p + (k_start+0)][col] += 1
            V[(j_start+0)*p + (k_start+1)][col] += 1
            W[(k_start+1)*n + (i_start+1)][col] += 1
            
            # M7: (A01-A11)(B10+B11) -> C00+1
            col = offset + 6
            U[(i_start+0)*m + (j_start+1)][col] += 1
            U[(i_start+1)*m + (j_start+1)][col] -= 1
            V[(j_start+1)*p + (k_start+0)][col] += 1
            V[(j_start+1)*p + (k_start+1)][col] += 1
            W[(k_start+0)*n + (i_start+0)][col] += 1

        add_strassen_block(0, 0, 0, 0)
        add_strassen_block(0, 0, 2, 7)
        add_strassen_block(0, 0, 4, 14)
        
        add_strassen_block(0, 2, 0, 21)
        add_strassen_block(0, 2, 2, 28)
        add_strassen_block(0, 2, 4, 35)
        
        return {"U": U, "V": V, "W": W}

    # Try to find a lower rank solution using local search
    # We try ranks 41, 40, 39
    # Values must be half-integers: k/2 for integer k
    # We restrict values to {-1, -0.5, 0, 0.5, 1} to keep search space manageable
    
    def verify(U, V, W):
        # U, V, W are numpy arrays of shape (dim1, rank), etc.
        # Compute sum_r U[:, r] * V[:, r] * W[:, r] ? No.
        # The tensor product is outer product.
        # We need to check if sum_r U_{u,r} V_{v,r} W_{w,r} == T_{u,v,w}
        # This is equivalent to checking the tensor reconstruction.
        # For speed, we can check a subset of (u,v,w) or use matrix multiplication tricks.
        # Since dimensions are small (8, 24, 12), full check is 8*24*12*rank ~ 2304*40 ~ 92k ops. Very fast.
        rank = U.shape[1]
        # Reshape to (dim1, rank, dim2, rank, dim3, rank) is too complex.
        # Loop over r
        # Rec = np.zeros((n*m, m*p, p*n))
        # for r in range(rank):
        #     Rec += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
        # return np.allclose(Rec, T)
        
        # Faster:
        # Rec = sum_r U_r \otimes V_r \otimes W_r
        # We can flatten T
        # But let's just do the einsum loop, it's fast enough for rank < 42.
        Rec = np.zeros((n*m, m*p, p*n), dtype=np.int16) # Values are small integers/halves
        for r in range(rank):
            # Outer product
            # u_term = U[:, r]
            # v_term = V[:, r]
            # w_term = W[:, r]
            # Rec += np.einsum('i,j,k->ijk', u_term, v_term, w_term)
            # To avoid float issues, we work with integers (multiply by 2)
            pass
        
        # Let's work in integer domain (multiply by 2) to avoid floating point errors
        # U_int = U * 2, etc.
        # T_int = T * 2 (T is 0 or 1, so 0 or 2)
        # Check if sum_r (2*U_{u,r}) (2*V_{v,r}) (2*W_{w,r}) / 4 == T_{u,v,w} ?
        # No, the entries are half-integers.
        # Let U_{u,r} = u_r / 2.
        # Sum_r (u_r/2) (v_r/2) (w_r/2) = T_{u,v,w}
        # Sum_r u_r v_r w_r = 8 * T_{u,v,w}
        
        # We will store U, V, W as integers representing 2*value.
        # So U_int entries are in {-2, -1, 0, 1, 2}
        
        Rec = np.zeros((n*m, m*p, p*n), dtype=np.int32)
        for r in range(rank):
            u_col = U[:, r]
            v_col = V[:, r]
            w_col = W[:, r]
            # Rec += np.einsum('i,j,k->ijk', u_col, v_col, w_col)
            # This creates a large intermediate array. 
            # Alternative: 
            # for u in range(n*m):
            #    if u_col[u] != 0:
            #       for v in range(m*p):
            #          if v_col[v] != 0:
            #             for w in range(p*n):
            #                if w_col[w] != 0:
            #                    Rec[u, v, w] += u_col[u] * v_col[v] * w_col[w]
            # This is O(rank * nnz(U) * nnz(V) * nnz(W)).
            # Random initialization might have many non-zeros.
            # Let's stick to einsum for simplicity and speed in numpy.
            Rec += np.einsum('i,j,k->ijk', u_col, v_col, w_col, optimize=True)
            
        return np.array_equal(Rec, T * 8)

    def search_rank(rank):
        if time.time() > deadline:
            return None
            
        # Initialize with random half-integers (represented as integers /2)
        # Values: -2, -1, 0, 1, 2
        vals = np.array([-2, -1, 0, 1, 2], dtype=np.int32)
        
        best_U = None
        best_V = None
        best_W = None
        best_err = np.inf
        
        # Try a few random restarts
        max_restarts = 10
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            U = np.random.choice(vals, size=(n*m, rank))
            V = np.random.choice(vals, size=(m*p, rank))
            W = np.random.choice(vals, size=(p*n, rank))
            
            # Local search: perturb one element at a time
            # We calculate error as sum (Rec - 8*T)^2
            # To speed up, we only check if error is 0.
            
            # Initial check
            if verify(U, V, W):
                return U, V, W
                
            # Simple local search
            # We don't have a fast error function, so we rely on random restarts
            # and maybe a few greedy steps if we had a gradient.
            # Without gradient, random restarts are the only option.
            # Given the discrete space, this might not find a solution if one exists.
            # But it's a structural change from analytic construction.
            
            # Let's try a few more random variations
            for _ in range(5):
                if time.time() > deadline:
                    break
                # Mutate
                U[np.random.randint(n*m), np.random.randint(rank)] = np.random.choice(vals)
                V[np.random.randint(m*p), np.random.randint(rank)] = np.random.choice(vals)
                W[np.random.randint(p*n), np.random.randint(rank)] = np.random.choice(vals)
                
                if verify(U, V, W):
                    return U, V, W
        
        return None

    # Try ranks 41, 40
    for rank in [41, 40]:
        res = search_rank(rank)
        if res:
            U, V, W = res
            # Convert back to half-integers (floats)
            U_out = [[float(x)/2.0 for x in row] for row in U]
            V_out = [[float(x)/2.0 for x in row] for row in V]
            W_out = [[float(x)/2.0 for x in row] for row in W]
            return {"U": U_out, "V": V_out, "W": W_out}

    # Fallback to rank 42
    return get_strassen_rank_42()
# EVOLVE-BLOCK-END
