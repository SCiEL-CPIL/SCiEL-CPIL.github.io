"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product as iproduct

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit (using 140 to be safe)

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

    def time_left():
        return deadline - time.time()

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
        
        # M1: (a00+a11)(b00+b11) -> c00+=1, c11+=1
        # M2: (a10+a11)b00 -> c10+=1
        # M3: a00(b01-b11) -> c01+=1
        # M4: a11(b10-b00) -> c00+=1, c10+=1
        # M5: (a00+a01)b11 -> c00-=1, c01+=1
        # M6: (a10-a00)(b00+b01) -> c11+=1
        # M7: (a01-a11)(b10+b11) -> c00-=1, c11+=1
        
        # Indices for U (i,j): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Indices for V (j,k): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Indices for W (k,i): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        # Note on W indexing in the tensor definition: c = k*n + i.
        # For local k=0, i=0 -> c=0. Local k=0, i=1 -> c=1.
        # Local k=1, i=0 -> c=2. Local k=1, i=1 -> c=3.
        # So W indices map directly to (k,i) -> k*2+i.
        
        prods = [
            # M1
            ([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]),
            # M2
            ([(2,1), (3,1)], [(0,1)], [(2,1)]), # c10 is k=0, i=1 -> idx 1? 
            # Wait. c = k*n + i.
            # If k=0, i=1 -> c = 0*2 + 1 = 1.
            # Let's re-verify Strassen output mapping.
            # C = [[c00, c01], [c10, c11]]
            # c00: i=0, k=0 -> idx 0
            # c01: i=0, k=1 -> idx 2
            # c10: i=1, k=0 -> idx 1
            # c11: i=1, k=1 -> idx 3
            
            # Let's re-map the prods carefully.
            # M1: (a00+a11)(b00+b11) = c00 + c11
            #   U: a00(0), a11(3) -> [(0,1), (3,1)]
            #   V: b00(0), b11(3) -> [(0,1), (3,1)]
            #   W: c00(0), c11(3) -> [(0,1), (3,1)]
            
            # M2: (a10+a11)b00 = c10
            #   U: a10(2), a11(3) -> [(2,1), (3,1)]
            #   V: b00(0) -> [(0,1)]
            #   W: c10(1) -> [(1,1)]
            
            # M3: a00(b01-b11) = c01
            #   U: a00(0) -> [(0,1)]
            #   V: b01(1), b11(3) -> [(1,1), (3,-1)]
            #   W: c01(2) -> [(2,1)]
            
            # M4: a11(b10-b00) = c00 + c10
            #   U: a11(3) -> [(3,1)]
            #   V: b10(2), b00(0) -> [(2,1), (0,-1)]
            #   W: c00(0), c10(1) -> [(0,1), (1,1)]
            
            # M5: (a00+a01)b11 = c00 - c01? No.
            # Standard Strassen:
            # M1 = (A11+A22)(B11+B22)
            # M2 = (A21+A22)B11
            # M3 = A11(B12-B22)
            # M4 = A22(B21-B11)
            # M5 = (A11+A12)B22
            # M6 = (A21-A11)(B11+B12)
            # M7 = (A12-A22)(B21+B22)
            
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 - M4 + M6
            # C22 = M1 - M2 + M3 - M6
            
            # Let's check M5: (A11+A12)B22 = C11 - C12?
            # C11 - C12 = (M1+M4-M5+M7) - (M3+M5) = M1+M4+M7-M3-2M5. Not quite.
            # Actually, C11 = M1 + M4 - M5 + M7.
            # C12 = M3 + M5.
            # So M5 contributes +1 to C12 and -1 to C11.
            
            # M5: (a00+a01)b11
            #   U: a00(0), a01(1) -> [(0,1), (1,1)]
            #   V: b11(3) -> [(3,1)]
            #   W: c01(2) is +1, c00(0) is -1 -> [(2,1), (0,-1)]
            
            # M6: (a10-a00)(b00+b01) = C21 - (C21 - M4 + M6)? No.
            # C21 = M2 - M4 + M6.
            # So M6 contributes +1 to C21.
            #   U: a10(2), a00(0) -> [(2,1), (0,-1)]
            #   V: b00(0), b01(1) -> [(0,1), (1,1)]
            #   W: c10(1) -> [(1,1)]
            
            # M7: (a01-a11)(b10+b11)
            # C11 = ... + M7.
            # So M7 contributes +1 to C11.
            #   U: a01(1), a11(3) -> [(1,1), (3,-1)]
            #   V: b10(2), b11(3) -> [(2,1), (3,1)]
            #   W: c00(0) -> [(0,1)]
            
            # Let's re-list with correct W indices:
            # c00: 0, c10: 1, c01: 2, c11: 3
            
            ([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]), # M1: C00+1, C11+1
            ([(2,1), (3,1)], [(0,1)], [(1,1)]), # M2: C10+1
            ([(0,1)], [(1,1), (3,-1)], [(2,1)]), # M3: C01+1
            ([(3,1)], [(2,1), (0,-1)], [(0,1), (1,1)]), # M4: C00+1, C10+1
            ([(0,1), (1,1)], [(3,1)], [(0,-1), (2,1)]), # M5: C00-1, C01+1
            ([(2,1), (0,-1)], [(0,1), (1,1)], [(1,1)]), # M6: C10+1
            ([(1,1), (3,-1)], [(2,1), (3,1)], [(0,1)]) # M7: C00+1
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

    # Strategy: "Ring Homomorphism" / "Block Cyclic" search
    # Hypothesis: The 2x2x12 problem might be decomposable into 3 independent 2x2x4 blocks
    # if the 12 columns are viewed as a cyclic group of order 12?
    # Or, more likely, we can try to find a decomposition that respects a cyclic symmetry
    # in the k-dimension (columns of B, rows of C).
    
    # Let's try a specialized search that constrains the factors to have a "block-cyclic" structure.
    # Specifically, assume the 12 columns of B (and corresponding rows of C) can be grouped into 3 sets of 4,
    # and the decomposition is a Kronecker product of a 2x2x4 solution and a 3x3x1 "cyclic" structure?
    # This is a stretch.
    
    # Alternative Structural Change: "Tensor Network" / "MPS" style optimization.
    # Represent U, V, W as sums of Kronecker products?
    # U = sum_alpha U_alpha^A \otimes U_alpha^B ?
    # This might be too restrictive.
    
    # Let's try a different structural change: "Integer Linear Programming" style local search with "Column Swapping" and "Scaling".
    # Current search is random coordinate descent.
    # New search: "Simulated Annealing" on the set of active columns and their values, allowing "jumps" to escape local minima.
    
    # But first, let's try a "Symmetry Breaking" approach.
    # The tensor T is invariant under cyclic shifts of (i, k) if we also shift j? No.
    # T[a,b,c] depends on i,j,k.
    # If we shift k -> k+1, c -> c+n.
    # If we shift i -> i+1, a -> a+m, c -> c+1.
    
    # Let's try a "Grouped Search".
    # Instead of treating all 42 columns independently, treat them as 6 groups of 7 (Strassen blocks).
    # Allow the "offset" of each block to vary? No, the blocks are fixed by the column pairing.
    
    # What if we try to find a rank-40 solution by "merging" two columns from different blocks?
    # This is what the previous program attempted with `merge_search`, but it only checked for exact proportionality.
    # We can generalize this: try to replace two columns c1, c2 with a single column c_new such that
    # u1 v1 w1 + u2 v2 w2 = u_new v_new w_new.
    # This is only possible if the two rank-1 tensors are linearly dependent (proportional).
    # So we can't merge arbitrary columns.
    
    # However, we CAN try to find a rank-41 solution by "absorbing" a column.
    # If we have a rank-42 solution, and we want rank 41, we need to find a linear dependency?
    # No, tensor rank is not linear.
    
    # Let's try a "Perturbation" of the Strassen solution.
    # Take the rank-42 solution.
    # Randomly pick two columns, c1, c2.
    # Try to replace them with a single column c_new = u1 v1 w1 + u2 v2 w2? No, that's not rank 1.
    
    # Let's try a different strategy: "Random Restart with Intensity".
    # Generate random rank-40, 39, 38 solutions and try to fit them using a "Least Squares" approach on the flattened tensor?
    # T is 4x24x24 = 2304 entries.
    # Rank 40 has 40 * (4+24+24) = 40 * 52 = 2080 variables.
    # We have 2304 constraints (mostly 0, some 1).
    # This is an overdetermined system of polynomial equations.
    
    # We can use `scipy.optimize.least_squares` on the flattened tensor difference.
    # Objective: min || flatten(U V W^T) - flatten(T) ||^2
    # But U, V, W are coupled.
    
    # Let's try a "Gradient Descent" with "Integer Projection".
    # 1. Initialize U, V, W with random integers.
    # 2. Compute gradient of the squared error.
    # 3. Update U, V, W by -lr * grad.
    # 4. Project back to nearest integers? Or keep as floats and round at the end?
    # If we keep as floats, we might find a real-valued decomposition.
    # But the task requires "integer/half-integer entries".
    # If we find a real-valued solution with small error, we can try to round it.
    # If the solution is unique, rounding might break it.
    # But if the solution space is large, rounding might work.
    
    # Let's try this "Continuous Relaxation + Rounding" approach.
    
    def continuous_search(rank, max_iter, seed):
        np.random.seed(seed)
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        # Scale up to match T's magnitude
        # T has 48 ones.
        # If U,V,W are small, T_rec is very small.
        # We need T_rec ~ 1.
        # So U,V,W should be ~ 1.
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        lr = 0.01
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time_left() < 5:
                break
                
            # Compute T_rec
            T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T_rec - T
            err = np.sum(diff**2)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if err < 1e-6:
                break
                
            # Compute gradients
            Gu = np.einsum('abc,br,cr->ar', diff, V, W)
            Gv = np.einsum('abc,ar,cr->br', diff, U, W)
            Gw = np.einsum('abc,ar,br->cr', diff, U, V)
            
            # Update
            U -= lr * Gu
            V -= lr * Gv
            W -= lr * Gw
            
            # Decay learning rate
            if it % 100 == 0:
                lr *= 0.9
                
        return best_U, best_V, best_W, best_err

    # Try continuous search for rank 40, 39, 38
    for rank in [40, 39, 38]:
        if time_left() < 10:
            break
        
        for seed in range(5):
            if time_left() < 5:
                break
            
            U_c, V_c, W_c, err_c = continuous_search(rank, 200, seed)
            
            # Try to round to integers
            # Nearest integer
            U_r = np.round(U_c)
            V_r = np.round(V_c)
            W_r = np.round(W_c)
            
            # Check if rounded solution is valid
            if verify(U_r, V_r, W_r):
                best_rank = rank
                best_U, best_V, best_W = U_r, V_r, W_r
                break # Found a valid one for this rank, move to next lower rank
            
            # Also try half-integers?
            # Multiply by 2, round, divide by 2
            U_h = np.round(U_c * 2) / 2.0
            V_h = np.round(V_c * 2) / 2.0
            W_h = np.round(W_c * 2) / 2.0
            
            if verify(U_h, V_h, W_h):
                best_rank = rank
                best_U, best_V, best_W = U_h, V_h, W_h
                break

    # If we still have time, try a "Local Search" from the best found solution
    # to see if we can reduce the rank further by "pruning" zero columns.
    # (The continuous search might produce columns that are effectively zero.)
    
    if best_U is not None:
        # Prune zero columns
        active_cols = []
        for r in range(best_U.shape[1]):
            if not (np.allclose(best_U[:, r], 0) and np.allclose(best_V[:, r], 0) and np.allclose(best_W[:, r], 0)):
                active_cols.append(r)
                
        if len(active_cols) < best_rank:
            best_U = best_U[:, active_cols]
            best_V = best_V[:, active_cols]
            best_W = best_W[:, active_cols]
            best_rank = len(active_cols)

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
