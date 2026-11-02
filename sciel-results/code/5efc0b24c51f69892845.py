"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 3x4x5 matrix multiplication.
    Uses divide-and-conquer: partition the inner dimension j (size 4) into two
    subspaces of size 2, decomposing 3x2*2x5 and 3x2*2x5 sub-problems.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_schoolbook = n * m * p  # 60
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 135
    
    # Build the target tensor T[a, b, c] where:
    # a = i*m + j, b = j*p + k, c = k*n + i
    # T[a, b, c] = 1 if i==i', j==j', k==k' else 0
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Schoolbook decomposition as guaranteed fallback
    def make_schoolbook(n_sub, m_sub, p_sub):
        r = n_sub * m_sub * p_sub
        nm_sub = n_sub * m_sub
        mp_sub = m_sub * p_sub
        pn_sub = p_sub * n_sub
        U = np.zeros((nm_sub, r))
        V = np.zeros((mp_sub, r))
        W = np.zeros((pn_sub, r))
        col = 0
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    U[i * m_sub + j, col] = 1
                    V[j * p_sub + k, col] = 1
                    W[k * n_sub + i, col] = 1
                    col += 1
        return U, V, W
    
    U_school, V_school, W_school = make_schoolbook(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_schoolbook
    
    # Verification function - checks all non-zero entries and samples zero entries
    def verify(U, V, W, T, n, m, p, tol=1e-9):
        r = U.shape[1]
        nm, mp, pn = n * m, m * p, p * n
        
        # Check all non-zero entries (the 60 entries that should be 1)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val - 1.0) > tol:
                        return False
        
        # Check zero entries: all entries that should be 0
        # We can check all of them since the tensor is small (12*20*15 = 3600 entries)
        # But let's be efficient: only check entries where indices don't match
        for a in range(nm):
            i_a = a // m
            j_a = a % m
            for b in range(mp):
                j_b = b // p
                k_b = b % p
                for c in range(pn):
                    i_c = c // n
                    k_c = c % n
                    if i_a == i_c and j_a == j_b and k_b == k_c:
                        continue  # This is a non-zero entry, already checked
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > tol:
                        return False
        return True
    
    # Faster verification for optimization loop (only check non-zero entries)
    def verify_fast(U, V, W, n, m, p, tol=1e-8):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Find a rank-r decomposition for n_sub x m_sub * m_sub x p_sub
    def find_decomposition(n_sub, m_sub, p_sub, target_rank, max_iter=300, num_restarts=5):
        """Find a rank-r decomposition using ALS with multiple restarts."""
        nm_sub = n_sub * m_sub
        mp_sub = m_sub * p_sub
        pn_sub = p_sub * n_sub
        
        # Build sub-tensor
        T_sub = np.zeros((nm_sub, mp_sub, pn_sub), dtype=float)
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    a = i * m_sub + j
                    b = j * p_sub + k
                    c = k * n_sub + i
                    T_sub[a, b, c] = 1
        
        U_school_sub, V_school_sub, W_school_sub = make_schoolbook(n_sub, m_sub, p_sub)
        best_U_sub, best_V_sub, best_W_sub = U_school_sub, V_school_sub, W_school_sub
        best_rank_sub = n_sub * m_sub * p_sub
        
        np.random.seed(42)
        
        # Precompute for ALS efficiency
        # For updating U[a]: solve (V[b,:] * W[c,:])^T * U[a,:] = T_sub[a,b,c]
        # This is a linear system: A^T * x = b where A has rows V[b,:]*W[c,:]
        
        def als_step(U, V, W, T_sub, n_sub, m_sub, p_sub, max_iter, tol=1e-10):
            nm_sub = n_sub * m_sub
            mp_sub = m_sub * p_sub
            pn_sub = p_sub * n_sub
            r = U.shape[1]
            
            prev_err = float('inf')
            
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break
                
                # Update U
                for a in range(nm_sub):
                    # Build system: for each (b,c), U[a,:] * (V[b,:] * W[c,:]) = T_sub[a,b,c]
                    # This is: sum_r U[a,r] * V[b,r] * W[c,r] = T_sub[a,b,c]
                    # Linear in U[a,:]: A * U[a,:] = b_vec
                    # where A[i, r] = V[b_i, r] * W[c_i, r] for each (b,c) pair
                    A = np.zeros((mp_sub * pn_sub, r))
                    b_vec = np.zeros(mp_sub * pn_sub)
                    idx = 0
                    for b in range(mp_sub):
                        for c in range(pn_sub):
                            A[idx, :] = V[b, :] * W[c, :]
                            b_vec[idx] = T_sub[a, b, c]
                            idx += 1
                    U[a, :], _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
                
                # Update V
                for b in range(mp_sub):
                    A = np.zeros((nm_sub * pn_sub, r))
                    b_vec = np.zeros(nm_sub * pn_sub)
                    idx = 0
                    for a in range(nm_sub):
                        for c in range(pn_sub):
                            A[idx, :] = U[a, :] * W[c, :]
                            b_vec[idx] = T_sub[a, b, c]
                            idx += 1
                    V[b, :], _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
                
                # Update W
                for c in range(pn_sub):
                    A = np.zeros((nm_sub * mp_sub, r))
                    b_vec = np.zeros(nm_sub * mp_sub)
                    idx = 0
                    for a in range(nm_sub):
                        for b in range(mp_sub):
                            A[idx, :] = U[a, :] * V[b, :]
                            b_vec[idx] = T_sub[a, b, c]
                            idx += 1
                    W[c, :], _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
                
                # Compute error
                err = 0.0
                for i in range(n_sub):
                    for j in range(m_sub):
                        for k in range(p_sub):
                            a = i * m_sub + j
                            b = j * p_sub + k
                            c = k * n_sub + i
                            val = np.sum(U[a, :] * V[b, :] * W[c, :])
                            err += (val - 1.0) ** 2
                
                if err < tol or (it > 10 and abs(prev_err - err) < 1e-12):
                    break
                prev_err = err
            
            return U, V, W, err
        
        # Try different ranks
        for r in range(target_rank, max(10, target_rank - 15), -1):
            if time.time() > deadline - 10:
                break
            
            found = False
            for attempt in range(num_restarts):
                if time.time() > deadline - 8:
                    break
                
                # Initialize
                if attempt == 0:
                    # Structured init from schoolbook (truncated)
                    U = np.zeros((nm_sub, r))
                    V = np.zeros((mp_sub, r))
                    W = np.zeros((pn_sub, r))
                    col = 0
                    for i in range(n_sub):
                        for j in range(m_sub):
                            for k in range(p_sub):
                                if col >= r:
                                    break
                                U[i * m_sub + j, col] = 1
                                V[j * p_sub + k, col] = 1
                                W[k * n_sub + i, col] = 1
                                col += 1
                            if col >= r:
                                break
                        if col >= r:
                            break
                elif attempt == 1:
                    # Perturbed schoolbook
                    U = U_school_sub[:, :r].copy()
                    V = V_school_sub[:, :r].copy()
                    W = W_school_sub[:, :r].copy()
                    U += np.random.randn(nm_sub, r) * 0.01
                    V += np.random.randn(mp_sub, r) * 0.01
                    W += np.random.randn(pn_sub, r) * 0.01
                else:
                    # Random init
                    U = np.random.randn(nm_sub, r) * 0.3
                    V = np.random.randn(mp_sub, r) * 0.3
                    W = np.random.randn(pn_sub, r) * 0.3
                
                # Run ALS
                U, V, W, err = als_step(U, V, W, T_sub, n_sub, m_sub, p_sub, max_iter)
                
                if err < 1e-8:
                    # Full verification
                    if verify(U, V, W, T_sub, n_sub, m_sub, p_sub):
                        best_U_sub, best_V_sub, best_W_sub = U, V, W
                        best_rank_sub = r
                        found = True
                        break
            
            if found:
                break
        
        return best_U_sub, best_V_sub, best_W_sub, best_rank_sub
    
    # Try to find decomposition for the full 3x4x5 problem
    # First, try direct optimization on the full problem
    print(f"Trying direct optimization for rank {rank_schoolbook - 1}...", file=sys.stderr)
    
    # Try ranks from 59 down to 40 for the full problem
    for r in range(rank_schoolbook - 1, 39, -1):
        if time.time() > deadline - 30:
            break
        
        found = False
        for attempt in range(3):
            if time.time() > deadline - 25:
                break
            
            # Initialize
            if attempt == 0:
                # Truncated schoolbook
                U = np.zeros((nm, r))
                V = np.zeros((mp, r))
                W = np.zeros((pn, r))
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col >= r:
                                break
                            U[i * m + j, col] = 1
                            V[j * p + k, col] = 1
                            W[k * n + i, col] = 1
                            col += 1
                        if col >= r:
                            break
                    if col >= r:
                        break
            else:
                # Random
                U = np.random.randn(nm, r) * 0.2
                V = np.random.randn(mp, r) * 0.2
                W = np.random.randn(pn, r) * 0.2
            
            # Run ALS on full problem
            prev_err = float('inf')
            for it in range(200):
                if time.time() > deadline - 20:
                    break
                
                # Update U
                for a in range(nm):
                    A = np.zeros((mp * pn, r))
                    b_vec = np.zeros(mp * pn)
                    idx = 0
                    for b in range(mp):
                        for c in range(pn):
                            A[idx, :] = V[b, :] * W[c, :]
                            b_vec[idx] = T[a, b, c]
                            idx += 1
                    U[a, :], _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
                
                # Update V
                for b in range(mp):
                    A = np.zeros((nm * pn, r))
                    b_vec = np.zeros(nm * pn)
                    idx = 0
                    for a in range(nm):
                        for c in range(pn):
                            A[idx, :] = U[a, :] * W[c, :]
                            b_vec[idx] = T[a, b, c]
                            idx += 1
                    V[b, :], _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
                
                # Update W
                for c in range(pn):
                    A = np.zeros((nm * mp, r))
                    b_vec = np.zeros(nm * mp)
                    idx = 0
                    for a in range(nm):
                        for b in range(mp):
                            A[idx, :] = U[a, :] * V[b, :]
                            b_vec[idx] = T[a, b, c]
                            idx += 1
                    W[c, :], _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
                
                # Error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            val = np.sum(U[a, :] * V[b, :] * W[c, :])
                            err += (val - 1.0) ** 2
                
                if err < 1e-10:
                    break
                
                if it > 20 and abs(prev_err - err) < 1e-12:
                    break
                prev_err = err
            
            if err < 1e-8:
                if verify(U, V, W, T, n, m, p):
                    best_U, best_V, best_W = U, V, W
                    best_rank = r
                    found = True
                    break
        
        if found:
            break
    
    # If direct optimization didn't find anything better, try divide-and-conquer
    if best_rank == rank_schoolbook and time.time() < deadline - 40:
        print("Trying divide-and-conquer approach...", file=sys.stderr)
        
        # Partition j into j1={0,1} and j2={2,3}
        # Sub-problem 1: 3x2 * 2x5
        U1, V1, W1, rank1 = find_decomposition(3, 2, 5, target_rank=30, max_iter=150, num_restarts=3)
        
        if time.time() > deadline - 25:
            return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
        
        # Sub-problem 2: 3x2 * 2x5
        U2, V2, W2, rank2 = find_decomposition(3, 2, 5, target_rank=30, max_iter=150, num_restarts=3)
        
        if time.time() > deadline - 20:
            return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
        
        total_rank = rank1 + rank2
        
        if total_rank < best_rank:
            # Build combined decomposition
            U_comb = np.zeros((nm, total_rank))
            V_comb = np.zeros((mp, total_rank))
            W_comb = np.zeros((pn, total_rank))
            
            # Fill U
            for i in range(n):
                for j in range(m):
                    if j < 2:
                        U_comb[i * m + j, :rank1] = U1[i * 2 + j, :]
                    else:
                        U_comb[i * m + j, rank1:] = U2[i * 2 + (j - 2), :]
            
            # Fill V
            for j in range(m):
                for k in range(p):
                    if j < 2:
                        V_comb[j * p + k, :rank1] = V1[j * 5 + k, :]
                    else:
                        V_comb[j * p + k, rank1:] = V2[(j - 2) * 5 + k, :]
            
            # Fill W
            for k in range(p):
                for i in range(n):
                    W_comb[k * n + i, :rank1] = W1[k * 3 + i, :]
                    W_comb[k * n + i, rank1:] = W2[k * 3 + i, :]
            
            if verify(U_comb, V_comb, W_comb, T, n, m, p):
                best_U, best_V, best_W = U_comb, V_comb, W_comb
                best_rank = total_rank
    
    # Return the best decomposition found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
