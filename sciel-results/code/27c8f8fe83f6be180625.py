"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product as iproduct

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    n, m, p = DIMENSIONS

    # Target tensor T[a, b, c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i
    # Dimensions: (4, 24, 24)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W):
        """
        Checks if U, V, W decompose T.
        T_rec[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        """
        if U is None:
            return False
        # Check shapes
        if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
            return False
        if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
            return False
        
        # Compute reconstruction
        # For small ranks, einsum is fine.
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T_rec - T)
        return err < 1e-9

    def get_strassen_42():
        """
        Generates the rank-42 decomposition using Strassen's algorithm
        applied to 6 disjoint 2x2x2 blocks (grouping the 12 columns into 6 pairs).
        """
        rank = 42
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Strassen 2x2x2 factors (u, v, w) for local indices (i,j), (j,k), (k,i)
        # Note: local k is 0,1. Global k = 2t + local_k.
        
        prods = [
            ([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]), # M1
            ([(2,1), (3,1)], [(0,1)], [(1,1)]), # M2
            ([(0,1)], [(1,1), (3,-1)], [(2,1)]), # M3
            ([(3,1)], [(2,1), (0,-1)], [(0,1), (1,1)]), # M4
            ([(0,1), (1,1)], [(3,1)], [(0,-1), (2,1)]), # M5
            ([(2,1), (0,-1)], [(0,1), (1,1)], [(1,1)]), # M6
            ([(1,1), (3,-1)], [(2,1), (3,1)], [(0,1)]) # M7
        ]
        
        for t in range(6):
            for s in range(7):
                col = 7 * t + s
                u_entries, v_entries, w_entries = prods[s]
                
                for idx, coeff in u_entries:
                    U[idx, col] = coeff
                    
                for idx, coeff in v_entries:
                    j = idx // 2
                    k_loc = idx % 2
                    k = 2 * t + k_loc
                    V[j * p + k, col] = coeff
                    
                for idx, coeff in w_entries:
                    k_loc = idx // 2
                    i = idx % 2
                    k = 2 * t + k_loc
                    W[k * n + i, col] = coeff
                    
        return U, V, W

    U_42, V_42, W_42 = get_strassen_42()
    
    # Verify 42
    if not verify(U_42, V_42, W_42):
        # Fallback to schoolbook if Strassen logic is broken
        rank_fb = 48
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U_fb[a, col] = 1
                    V_fb[b, col] = 1
                    W_fb[c, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    best_rank = 42
    best_U, best_V, best_W = U_42, V_42, W_42

    # NEW STRATEGY: Block-Circulant Tensor Decomposition
    # Partition k (0..11) into 4 groups of 3: g=0,1,2,3.
    # Let k = 3g + r, where r in {0, 1, 2}.
    # The multiplication is C = A B.
    # C_{i, 3g+r} = sum_j A_{i,j} B_{j, 3g+r}.
    # This is 4 independent 2x2x3 multiplications?
    # No, because A and B are full 2x2 and 2x12.
    # But if we assume the decomposition respects the block structure,
    # we can look for a solution where the factors are block-circulant.
    
    # Let's try a "Tensor Train" or "MPS" style decomposition.
    # But first, let's try a simpler algebraic construction.
    
    # Consider the ring R = Q[x]/(x^3-1).
    # x^3-1 = (x-1)(x^2+x+1).
    # R is isomorphic to Q x Q(omega), where omega = e^(2pi i / 3).
    # A 2x2 matrix over R can be multiplied by a 2x2 matrix over R.
    # The rank of 2x2x2 over R is 7 (Strassen).
    # The rank of 2x2x2 over Q is 7? No, it's 7 over any field of char != 2.
    # Wait, the rank of 2x2x2 is 7.
    # If we map the 12 columns to 4 blocks of 3, and treat each block as an element of R?
    # This would mean B is a 2x2 matrix over R? No, B is 2x12.
    # If we view B as a 2x2 matrix over R, where each entry is a polynomial of degree < 3?
    # Then B_{j, g} = sum_{r=0}^2 B_{j, 3g+r} x^r.
    # Then C_{i, g} = sum_j A_{i,j} B_{j, g} in R.
    # This decomposes the 2x2x12 problem into a 2x2x2 problem over R.
    # The rank of 2x2x2 over R is 7 * rank(R)? No.
    # The tensor rank of M_{2,2,2} over R is 7 if R is a field.
    # But R is not a field, it's Q x Q(omega).
    # The rank over R is the max of the ranks over the components?
    # Rank over Q x Q(omega) is Rank(Q) + Rank(Q(omega))?
    # Rank of 2x2x2 over Q is 7.
    # Rank of 2x2x2 over Q(omega) is 7.
    # So total rank would be 14?
    # But we have 4 blocks. So 4 * 14 = 56? That's worse than 48.
    
    # Let's rethink.
    # The tensor T is a sum of 48 rank-1 tensors.
    # We want to find a lower rank decomposition.
    
    # Let's try a "Randomized SVD" style approach on the flattening.
    # Flatten T into a matrix T_mat of shape (4, 576).
    # The rank of T_mat is an upper bound on the tensor rank.
    # Let's compute the rank of T_mat.
    
    T_flat = T.reshape(n*m, -1)
    U_svd, s, Vh = np.linalg.svd(T_flat, full_matrices=False)
    # The number of non-zero singular values is the rank of the flattening.
    rank_flat = np.sum(s > 1e-9)
    # For 2x2x12, the rank of the flattening is likely 48 or less.
    # If it's 48, SVD doesn't help directly for tensor rank.
    
    # Let's try a "Greedy" approach to find a rank-40 solution.
    # Start with the rank-42 Strassen solution.
    # Try to merge columns.
    
    # Actually, let's try a "Local Search" that allows "column swaps" and "scaling".
    # But the current program already tries continuous relaxation.
    
    # Let's try a different structural change: "Symmetry Breaking" with "Grouped Search".
    # Assume the 12 columns are grouped into 3 sets of 4.
    # And the decomposition is a Kronecker product of a 2x2x4 solution and a 3x3x1 "cyclic" structure?
    
    # Let's try to find a rank-40 solution by "perturbing" the Strassen solution.
    # We will use a "Simulated Annealing" style search on the set of active columns.
    
    def local_search(U, V, W, max_iter, seed):
        np.random.seed(seed)
        U = U.copy()
        V = V.copy()
        W = W.copy()
        rank = U.shape[1]
        
        best_err = np.linalg.norm(np.einsum('ar,br,cr->abc', U, V, W) - T)
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
                
            # Pick a random column to perturb
            r = np.random.randint(rank)
            
            # Perturb U[:, r], V[:, r], W[:, r]
            U[:, r] += np.random.randn(n*m) * 0.01
            V[:, r] += np.random.randn(m*p) * 0.01
            W[:, r] += np.random.randn(p*n) * 0.01
            
            # Try to "prune" the column if it becomes small
            if np.linalg.norm(U[:, r]) < 0.1 and np.linalg.norm(V[:, r]) < 0.1 and np.linalg.norm(W[:, r]) < 0.1:
                U[:, r] = 0
                V[:, r] = 0
                W[:, r] = 0
            
            err = np.linalg.norm(np.einsum('ar,br,cr->abc', U, V, W) - T)
            
            if err < best_err:
                best_err = err
                # Round to nearest integer/half-integer
                U_r = np.round(U)
                V_r = np.round(V)
                W_r = np.round(W)
                if verify(U_r, V_r, W_r):
                    # Check if rank can be reduced
                    active = [i for i in range(rank) if not (np.allclose(U_r[:, i], 0) and np.allclose(V_r[:, i], 0) and np.allclose(W_r[:, i], 0))]
                    if len(active) < rank:
                        return U_r[:, active], V_r[:, active], W_r[:, active], len(active)
                    return U_r, V_r, W_r, rank
        return None, None, None, best_err

    # Try local search from Strassen 42
    for seed in range(10):
        if time.time() > deadline - 20:
            break
        U_ls, V_ls, W_ls, rank_ls = local_search(U_42, V_42, W_42, 100, seed)
        if U_ls is not None and rank_ls < best_rank:
            best_rank = rank_ls
            best_U, best_V, best_W = U_ls, V_ls, W_ls

    # Final verification
    if verify(best_U, best_V, best_W):
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        # Fallback to Strassen 42
        return {
            "U": U_42.tolist(),
            "V": V_42.tolist(),
            "W": W_42.tolist()
        }
# EVOLVE-BLOCK-END
