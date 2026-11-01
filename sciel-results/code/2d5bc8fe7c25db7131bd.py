"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor as a 24-dim vector
    # T[(i,j),(j,k),(k,i)] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Flatten index for each mode:
    #   U index: i*m + j  (range [0, nm))
    #   V index: j*p + k  (range [0, mp))
    #   W index: k*n + i  (range [0, pn))
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                tensor_entries.append((u_idx, v_idx, w_idx))
    
    num_entries = len(tensor_entries)  # should be 24
    
    # Generate candidate half-integer vectors
    # Values: -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2
    vals = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    
    # We'll search for rank-1 tensors u ⊗ v ⊗ w
    # u has length nm=6, v has length mp=12, w has length pn=8
    # Total candidates: 9^6 * 9^12 * 9^8 = 9^26 which is way too many
    # 
    # Instead, use a smarter approach: enumerate sparse candidates
    # or use a greedy approach with random sampling + local optimization
    
    # Strategy: Start with schoolbook rank 24, try to reduce
    # by finding linear dependencies among the 24 basis tensors
    
    # The 24 schoolbook rank-1 tensors are:
    # For each (i,j,k): e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    # where e_{i,j} is unit vector in U space, etc.
    
    # Let's check if these 24 vectors in the tensor space are linearly independent
    # Tensor space dimension: nm * mp * pn = 6 * 12 * 8 = 576
    # But our tensor only has 24 nonzero entries, so we work in a 24-dim subspace
    
    # Actually, the key question: can we represent T as sum of r rank-1 tensors?
    # T = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r
    # This means for all (i,j,k): sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1
    
    # Let's try rank 15, 16, 17, ... and use optimization
    
    # First, let's try a structured approach:
    # Parameterize with small integer/half-integer entries
    
    # For small rank, try exhaustive search over limited value sets
    # with pruning
    
    best_result = None
    best_rank = n * m * p  # start with schoolbook
    
    # Generate schoolbook as fallback
    U_sb = [[0] * best_rank for _ in range(nm)]
    V_sb = [[0] * best_rank for _ in range(mp)]
    W_sb = [[0] * best_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try to find lower rank using randomized search with half-integer values
    # For each candidate rank r, try to find u_1,...,u_r, v_1,...,v_r, w_1,...,w_r
    # such that sum_r u_r[i*m+j]*v_r[j*p+k]*w_r[k*n+i] = 1 for all 24 entries
    
    # Use a greedy/iterative approach:
    # 1. Randomly initialize with half-integer values
    # 2. Use coordinate descent or Newton's method to minimize error
    # 3. Snap to nearest half-integers and verify
    
    np.random.seed(42)
    
    # Try ranks from 15 upward
    for target_rank in range(15, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Multiple random restarts
        num_restarts = 20
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random half-integer values
            scale = 1.0
            U_init = np.random.choice([-1.5, -1, -0.5, 0, 0.5, 1, 1.5], size=(nm, target_rank))
            V_init = np.random.choice([-1.5, -1, -0.5, 0, 0.5, 1, 1.5], size=(mp, target_rank))
            W_init = np.random.choice([-1.5, -1, -0.5, 0, 0.5, 1, 1.5], size=(pn, target_rank))
            
            # Build the linear system: for each of the 24 entries, we need
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is nonlinear. Use iterative refinement.
            
            # Alternative: fix two factors, solve for the third linearly
            # Given U and V, the equations are linear in W:
            # For entry (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # This is: sum_r (U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r] = 1
            # Let A_{(i,j,k), (k',i',r)} = U[i*m+j,r] * V[j*p+k,r] if k'=k, i'=i else 0
            # Then A * W_flat = 1
            
            # This is a 24 x (pn * target_rank) system. Underdetermined if pn*target_rank > 24.
            # pn * target_rank = 8 * 15 = 120 > 24, so underdetermined.
            
            # Use least squares with regularization, then snap
            
            # Actually, let's use a different approach:
            # Enumerate all possible "patterns" for U, V, W with limited support
            
            # Simpler: use the fact that we need exact half-integer solutions
            # Try all combinations of a small set of basis vectors
            
            # For rank 15, we have 15 columns. Each column is (u_col, v_col, w_col).
            # The constraint is: for all 24 entries, sum_col u_col[u_idx] * v_col[v_idx] * w_col[w_idx] = 1
            
            # This is equivalent to: the 15 rank-1 tensors sum to T.
            
            # Let's try a structured construction:
            # Use symmetry and small integer patterns
            
            # Try: each column has entries from {-1, 0, 1}
            # and is sparse (at most 2-3 nonzeros per vector)
            
            # This is getting complex. Let's use a direct linear algebra approach:
            # If we fix U and V, solve for W using least squares, then snap W to half-integers,
            # then fix W and V, solve for U, snap, etc.
            
            U_curr = U_init.copy()
            V_curr = V_init.copy()
            W_curr = W_init.copy()
            
            # Iterative refinement
            for iteration in range(50):
                if time.time() > deadline - 3:
                    break
                
                # Solve for W given U, V
                # System: for each entry e=(i,j,k) with u_idx=i*m+j, v_idx=j*p+k, w_idx=k*n+i:
                # sum_r U_curr[u_idx, r] * V_curr[v_idx, r] * W_curr[w_idx, r] = 1
                # Linear in W_curr: for fixed w_idx, the coefficient of W_curr[w_idx, r] is U_curr[u_idx,r]*V_curr[v_idx,r]
                # But different entries may share the same w_idx.
                # Group by w_idx:
                # For w_idx = k*n+i, entries with this w_idx are those with specific (k,i) and varying j
                # Entry (i,j,k) has w_idx = k*n+i, u_idx = i*m+j, v_idx = j*p+k
                
                # Build linear system for W:
                # Unknowns: W_curr[pn, target_rank] -> pn*target_rank unknowns
                # Equations: 24 (one per tensor entry)
                # A * W_flat = b where b = ones(24)
                
                A_mat = np.zeros((num_entries, pn * target_rank))
                b_vec = np.ones(num_entries)
                
                for e_idx, (u_idx, v_idx, w_idx) in enumerate(tensor_entries):
                    for r in range(target_rank):
                        col_idx = w_idx * target_rank + r
                        A_mat[e_idx, col_idx] = U_curr[u_idx, r] * V_curr[v_idx, r]
                
                # Solve least squares
                W_flat, residuals, rank_A, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W_curr = W_flat.reshape(pn, target_rank)
                
                # Snap to nearest half-integer
                W_curr = np.round(W_curr * 2) / 2
                
                # Solve for U given V, W
                A_mat_U = np.zeros((num_entries, nm * target_rank))
                for e_idx, (u_idx, v_idx, w_idx) in enumerate(tensor_entries):
                    for r in range(target_rank):
                        col_idx = u_idx * target_rank + r
                        A_mat_U[e_idx, col_idx] = V_curr[v_idx, r] * W_curr[w_idx, r]
                
                U_flat, _, _, _ = np.linalg.lstsq(A_mat_U, b_vec, rcond=None)
                U_curr = U_flat.reshape(nm, target_rank)
                U_curr = np.round(U_curr * 2) / 2
                
                # Solve for V given U, W
                A_mat_V = np.zeros((num_entries, mp * target_rank))
                for e_idx, (u_idx, v_idx, w_idx) in enumerate(tensor_entries):
                    for r in range(target_rank):
                        col_idx = v_idx * target_rank + r
                        A_mat_V[e_idx, col_idx] = U_curr[u_idx, r] * W_curr[w_idx, r]
                
                V_flat, _, _, _ = np.linalg.lstsq(A_mat_V, b_vec, rcond=None)
                V_curr = V_flat.reshape(mp, target_rank)
                V_curr = np.round(V_curr * 2) / 2
                
                # Check if we have an exact solution
                error = 0.0
                for e_idx, (u_idx, v_idx, w_idx) in enumerate(tensor_entries):
                    val = 0.0
                    for r in range(target_rank):
                        val += U_curr[u_idx, r] * V_curr[v_idx, r] * W_curr[w_idx, r]
                    error += abs(val - 1.0)
                
                if error < 1e-9:
                    # Found exact solution!
                    best_rank = target_rank
                    U_list = [[float(x) for x in U_curr[i]] for i in range(nm)]
                    V_list = [[float(x) for x in V_curr[i]] for i in range(mp)]
                    W_list = [[float(x) for x in W_curr[i]] for i in range(pn)]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    found = True
                    break
            
            if found:
                break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
