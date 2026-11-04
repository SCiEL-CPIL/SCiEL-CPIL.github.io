"""
Strategy: Direct stochastic search for low-rank tensor decomposition.
Instead of recursive block decomposition (Strassen), we search for a set of 
rank-1 terms (u, v, w) that sum to the matrix multiplication tensor T_{i,j,k} = delta_{i,j,k} (in the appropriate index mapping).
We use a binary search space (entries in {-1, 0, 1}) to keep the search tractable and aligned with known fast algorithms.
We use a deadline guard to ensure we finish within 120s.
"""

import numpy as np
import time
import random

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.
    Strategy:
    1. Define the target tensor T for 4x4 * 4x6.
    2. Initialize with a known valid decomposition (e.g., schoolbook rank 96 or a heuristic Strassen-like rank 77).
    3. Use stochastic local search to reduce rank by:
       a. Trying to remove a term and re-fitting the remaining terms (if residual is small).
       b. Trying to replace a term with a different pattern.
    4. Verify exactness at the end.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin for 120s limit
    
    # Target Tensor T: T[i, j, k] = 1 if A_{i,j} and B_{j,k} contribute to C_{i,k}.
    # In our interface:
    # U index: i*m + j (size n*m = 16)
    # V index: j*p + k (size m*p = 24)
    # W index: k*n + i (size p*n = 24)
    # The tensor entry T[idx_u, idx_v, idx_w] is 1 if the indices correspond to a valid matmul term.
    # Valid term: U_{i*m+j}, V_{j*p+k}, W_{k*n+i} where i in [0,n), j in [0,m), k in [0,p).
    
    def build_target_tensor():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    T[idx_u, idx_v, idx_w] = 1.0
        return T

    T = build_target_tensor()

    # Helper to compute residual norm for a given set of terms (U, V, W columns)
    # We represent terms as a list of (u_vec, v_vec, w_vec).
    # To speed up, we precompute the outer products? No, rank is small, we can just sum.
    # Residual R = T - sum_r u_r \otimes v_r \otimes w_r
    
    def compute_residual_norm(U_cols, V_cols, W_cols):
        # U_cols: list of vectors of length n*m
        # V_cols: list of vectors of length m*p
        # W_cols: list of vectors of length p*n
        # Rank = len(U_cols)
        if not U_cols:
            return np.linalg.norm(T)
        
        rank = len(U_cols)
        # We can accumulate the tensor
        # For efficiency with small rank, we can use einsum or loops.
        # Let's try to vectorize: 
        # Residual = T - sum_r U[:,r] V[:,r] W[:,r]
        # This is hard to do in one shot without large memory if rank is high, but rank < 100 is fine.
        
        U_mat = np.array(U_cols).T # (n*m, rank)
        V_mat = np.array(V_cols).T # (m*p, rank)
        W_mat = np.array(W_cols).T # (p*n, rank)
        
        # Reconstruct tensor
        # T_rec[i,j,k] = sum_r U_mat[i,r] V_mat[j,r] W_mat[k,r]
        # We can do this by iterating over r or using einsum.
        # einsum: 'ir,jr,kr->ijk'
        T_rec = np.einsum('ir,jr,kr->ijk', U_mat, V_mat, W_mat)
        R = T - T_rec
        return np.linalg.norm(R)

    # Helper to verify exactness
    def verify_exact(U_cols, V_cols, W_cols):
        if not U_cols:
            return False
        U_mat = np.array(U_cols).T
        V_mat = np.array(V_cols).T
        W_mat = np.array(W_cols).T
        T_rec = np.einsum('ir,jr,kr->ijk', U_mat, V_mat, W_mat)
        return np.allclose(T, T_rec, atol=1e-9)

    # Initialize with a valid decomposition: Schoolbook (Rank 96)
    # We can try to start with a better initial guess if we have one, but Schoolbook is safe.
    # Let's try to generate a "Strassen-like" initial guess if possible, but for now, 
    # let's use a random subset of the schoolbook terms as a starting point for search?
    # No, we need a valid start to prune. 
    # Actually, the search can start from scratch if we allow "adding" terms.
    # But minimizing from a high rank is easier if we start high and remove.
    
    # Let's start with the full schoolbook decomposition.
    U_init = []
    V_init = []
    W_init = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n*m)
                v = np.zeros(m*p)
                w = np.zeros(p*n)
                u[i*m + j] = 1.0
                v[j*p + k] = 1.0
                w[k*n + i] = 1.0
                U_init.append(u)
                V_init.append(v)
                W_init.append(w)
    
    current_U = U_init
    current_V = V_init
    current_W = W_init
    best_rank = len(current_U)
    best_U = [u.copy() for u in current_U]
    best_V = [v.copy() for v in current_V]
    best_W = [w.copy() for w in current_W]
    
    # Stochastic Local Search
    # Move 1: Remove a random term. If residual is 0, we found a lower rank.
    # Move 2: Change a random entry in a random term to -1, 0, or 1. If residual decreases, accept.
    # Move 3: Add a random term (random pattern). If residual decreases, accept.
    
    # To make search effective, we should focus on "structured" patterns.
    # Let's restrict the search space to vectors with entries in {-1, 0, 1}.
    
    def random_pattern(size):
        return np.array([random.choice([-1, 0, 1]) for _ in range(size)], dtype=float)

    temp = 1.0
    min_temp = 0.01
    
    while time.time() < deadline:
        # Simulated Annealing step
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to remove a term
            if len(current_U) > 1:
                idx = random.randrange(len(current_U))
                # Temporarily remove
                new_U = current_U[:idx] + current_U[idx+1:]
                new_V = current_V[:idx] + current_V[idx+1:]
                new_W = current_W[:idx] + current_W[idx+1:]
                
                # Check if exact
                if verify_exact(new_U, new_V, new_W):
                    current_U = new_U
                    current_V = new_V
                    current_W = new_W
                    if len(current_U) < best_rank:
                        best_rank = len(current_U)
                        best_U = [u.copy() for u in current_U]
                        best_V = [v.copy() for v in current_V]
                        best_W = [w.copy() for w in current_W]
                        temp = max(min_temp, temp * 0.9) # Cool down on success
                else:
                    # Calculate residual change to decide acceptance (optional, but exactness is key)
                    # For speed, we only accept removals that are exact.
                    pass
                    
        elif move_type < 0.8:
            # Modify a random term
            if current_U:
                idx = random.randrange(len(current_U))
                # Pick a random vector to modify
                vec_idx = random.choice([0, 1, 2])
                if vec_idx == 0:
                    vec = current_U[idx]
                elif vec_idx == 1:
                    vec = current_V[idx]
                else:
                    vec = current_W[idx]
                    
                # Pick a random entry to flip
                entry_idx = random.randrange(len(vec))
                old_val = vec[entry_idx]
                new_val = random.choice([-1, 0, 1])
                if new_val == old_val:
                    continue
                vec[entry_idx] = new_val
                
                # Check new residual
                # We need to compare with old residual.
                # For simplicity, let's just check if it's exact. If not, we might want to accept if it's better.
                # But calculating full residual every step is expensive.
                # Let's only accept if it leads to an exact solution or if we do a periodic full check.
                # Actually, for tensor decomposition, "exact" is a very sparse target.
                # A better heuristic: Accept if the residual norm decreases.
                
                # Let's compute the new residual norm.
                # This is O(R * N*M*P) which is ~ 100 * 16 * 24 * 24 ~ 90k ops. Fast enough.
                new_U = [u.copy() for u in current_U]
                new_V = [v.copy() for v in current_V]
                new_W = [w.copy() for w in current_W]
                
                if vec_idx == 0:
                    new_U[idx] = vec
                elif vec_idx == 1:
                    new_V[idx] = vec
                else:
                    new_W[idx] = vec
                    
                # We don't have old residual stored. Let's store it.
                # This is getting complex. Let's simplify: 
                # Only accept moves that result in an EXACT solution.
                # This is a "greedy exact search". It might get stuck.
                # To escape, we can use random restarts or accept non-exact moves with penalty.
                
                if verify_exact(new_U, new_V, new_W):
                    current_U = new_U
                    current_V = new_V
                    current_W = new_W
                    if len(current_U) < best_rank:
                        best_rank = len(current_U)
                        best_U = [u.copy() for u in current_U]
                        best_V = [v.copy() for v in current_V]
                        best_W = [w.copy() for w in current_W]
                else:
                    # Revert
                    vec[entry_idx] = old_val

        else:
            # Add a random term
            u_new = random_pattern(n*m)
            v_new = random_pattern(m*p)
            w_new = random_pattern(p*n)
            
            new_U = current_U + [u_new]
            new_V = current_V + [v_new]
            new_W = current_W + [w_new]
            
            if verify_exact(new_U, new_V, new_W):
                current_U = new_U
                current_V = new_V
                current_W = new_W
                if len(current_U) < best_rank:
                    best_rank = len(current_U)
                    best_U = [u.copy() for u in current_U]
                    best_V = [v.copy() for v in current_V]
                    best_W = [w.copy() for w in current_W]

        # Periodic cooling
        temp = max(min_temp, temp * 0.999)

    # Final verification of best found
    if verify_exact(best_U, best_V, best_W):
        U_mat = np.array(best_U).T
        V_mat = np.array(best_V).T
        W_mat = np.array(best_W).T
        return {
            "U": U_mat.tolist(),
            "V": V_mat.tolist(),
            "W": W_mat.tolist()
        }
    else:
        # Fallback to schoolbook
        U, V, W = [], [], []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = np.zeros(n*m)
                    v = np.zeros(m*p)
                    w = np.zeros(p*n)
                    u[i*m + j] = 1.0
                    v[j*p + k] = 1.0
                    w[k*n + i] = 1.0
                    U.append(u)
                    V.append(v)
                    W.append(w)
        return {
            "U": np.array(U).T.tolist(),
            "V": np.array(V).T.tolist(),
            "W": np.array(W).T.tolist()
        }

# EVOLVE-BLOCK-END
