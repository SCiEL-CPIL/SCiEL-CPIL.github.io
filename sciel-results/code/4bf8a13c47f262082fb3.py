import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition is valid."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check all (i,j,k) entries equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _greedy_decomposition(n, m, p, max_rank, seed, deadline):
    """
    Greedy integer decomposition:
    At each step, find the best rank-1 tensor (u,v,w) with entries in {-1,0,1}
    that best approximates the remaining residual tensor.
    
    We restrict to sparse vectors to make the search feasible.
    """
    rng = np.random.RandomState(seed)
    
    # Residual tensor: initially all ones
    R = np.ones((n, m, p), dtype=float)
    
    # Accumulate factor matrices
    U_list = []
    V_list = []
    W_list = []
    
    N_U = n * m
    N_V = m * p
    N_W = p * n
    
    # Precompute a pool of candidate vectors for each dimension
    # For efficiency, we use sparse vectors with support size 1 or 2
    # and values in {-1, 1}
    
    # Generate candidate vectors
    def gen_candidates(N, max_support, vals=(-1, 1), num_candidates=500):
        """Generate random sparse candidates."""
        cands = []
        # Include unit vectors
        for idx in range(N):
            v = np.zeros(N)
            v[idx] = 1
            cands.append(v)
        
        # Add random sparse vectors
        while len(cands) < num_candidates:
            support = rng.randint(1, max_support + 1)
            indices = rng.choice(N, support, replace=False)
            v = np.zeros(N)
            for idx in indices:
                v[idx] = rng.choice(vals)
            cands.append(v)
        
        return np.array(cands)
    
    # Use moderate candidate pool sizes
    U_cands = gen_candidates(N_U, max_support=3, num_candidates=300)
    V_cands = gen_candidates(N_V, max_support=3, num_candidates=300)
    W_cands = gen_candidates(N_W, max_support=3, num_candidates=300)
    
    for rank_idx in range(max_rank):
        if time.time() > deadline - 10:
            break
            
        # Current residual
        # We want to find u, v, w that minimize ||R - u⊗v⊗w||^2
        # Expanding: ||R||^2 - 2 * sum(R * u⊗v⊗w) + ||u||^2 ||v||^2 ||w||^2
        # Since ||R||^2 is constant, minimize: -2 * <R, u⊗v⊗w> + ||u||^2||v||^2||w||^2
        # For binary/sparse vectors, the norm term is small, so we mainly maximize <R, u⊗v⊗w>
        
        # Greedy: pick best u, then best v, then best w (or joint search)
        # To keep it fast, do coordinate-wise greedy
        
        # Step 1: For each candidate u, compute the "contraction" with R
        # C1[u_idx, j, k] = sum_i R[i,j,k] * U_cands[u_idx, i*m+j]
        # Then for each v, C2[v_idx, k, i] = sum_j C1[u_idx,j,k] * V_cands[v_idx, j*p+k]
        # Then for each w, score = sum_{k,i} C2[v_idx,k,i] * W_cands[w_idx, k*n+i]
        
        # This is expensive if done naively. Let's use a simpler heuristic:
        # Randomly sample a subset of candidates and evaluate.
        
        best_score = -np.inf
        best_u_idx = 0
        best_v_idx = 0
        best_w_idx = 0
        
        # Sample candidates for speed
        sample_size = min(50, len(U_cands))
        u_sample = rng.choice(len(U_cands), sample_size, replace=False)
        v_sample = rng.choice(len(V_cands), sample_size, replace=False)
        w_sample = rng.choice(len(W_cands), sample_size, replace=False)
        
        # For efficiency, we'll do a coarser search
        # Pick random combinations
        num_trials = 200
        for _ in range(num_trials):
            u_idx = u_sample[rng.randint(len(u_sample))]
            v_idx = v_sample[rng.randint(len(v_sample))]
            w_idx = w_sample[rng.randint(len(w_sample))]
            
            u = U_cands[u_idx].reshape(n, m)
            v = V_cands[v_idx].reshape(m, p)
            w = W_cands[w_idx].reshape(p, n)
            
            # Compute u⊗v⊗w
            term = np.einsum('ij,jk,ki->ijk', u, v, w)
            
            # Score: inner product with residual
            score = np.sum(R * term)
            
            # Penalize if the term has large norm (to keep values bounded)
            norm_penalty = 0.01 * np.sum(u**2) * np.sum(v**2) * np.sum(w**2)
            score -= norm_penalty
            
            if score > best_score:
                best_score = score
                best_u_idx = u_idx
                best_v_idx = v_idx
                best_w_idx = w_idx
        
        # Apply the best found term
        u = U_cands[best_u_idx].reshape(n, m)
        v = V_cands[best_v_idx].reshape(m, p)
        w = W_cands[best_w_idx].reshape(p, n)
        
        term = np.einsum('ij,jk,ki->ijk', u, v, w)
        
        # Check if this term makes the residual worse
        old_norm = np.sum(R**2)
        new_R = R - term
        new_norm = np.sum(new_R**2)
        
        if new_norm < old_norm - 1e-9:
            R = new_R
            U_list.append(U_cands[best_u_idx].tolist())
            V_list.append(V_cands[best_v_idx].tolist())
            W_list.append(W_cands[best_w_idx].tolist())
            
            # Check convergence
            if np.sum(R**2) < 1e-6:
                break
        else:
            # Try a smaller step or different direction
            # For now, just stop if no improvement
            break
    
    if len(U_list) == 0:
        return None
    
    # Build the factor matrices
    rank = len(U_list)
    U = [[0] * rank for _ in range(N_U)]
    V = [[0] * rank for _ in range(N_V)]
    W = [[0] * rank for _ in range(N_W)]
    
    for c in range(rank):
        for i in range(N_U):
            U[i][c] = U_list[c][i]
        for i in range(N_V):
            V[i][c] = V_list[c][i]
        for i in range(N_W):
            W[i][c] = W_list[c][i]
    
    # Round to nearest half-integer
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U_rounded = np.round(U_arr * 2) / 2
    V_rounded = np.round(V_arr * 2) / 2
    W_rounded = np.round(W_arr * 2) / 2
    
    if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
        return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
    
    # Try without rounding
    if _verify(U, V, W, n, m, p):
        return U, V, W
    
    return None

def _structured_search(n, m, p, rank, deadline):
    """
    Try structured decompositions with specific patterns.
    For 3x4x6, known bounds suggest rank around 48-60 might be achievable.
    """
    # Try a pattern where we group indices
    # For now, this is a placeholder for more sophisticated structured search
    
    # One idea: use Kronecker product structures
    # If 3=1*3, 4=2*2, 6=2*3, maybe there's a Kronecker decomposition
    
    # For matrix multiplication: (A⊗B) * (C⊗D) = (AC) ⊗ (BD)
    # This could reduce rank if dimensions factorize nicely
    
    # 3x4x6: 
    # 3 = 3, 4 = 4, 6 = 6
    # No obvious Kronecker factorization
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U, "V": V, "W": W}
    
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Schoolbook verification failed")
    
    # Theoretical lower bound
    lower_bound = max(n * p, m * p, n * m)
    
    # Try greedy decomposition with various seeds
    for seed in range(50):
        if time.time() > deadline - 20:
            break
        
        # Try greedy with max_rank starting from lower_bound + 1
        for max_rank in range(max(lower_bound + 1, 40), min(best_rank, 72)):
            if time.time() > deadline - 10:
                break
            
            result = _greedy_decomposition(n, m, p, max_rank, seed, deadline)
            if result is not None:
                U_new, V_new, W_new = result
                
                # Verify the result
                if _verify(U_new, V_new, W_new, n, m, p):
                    new_rank = len(U_new[0])
                    if new_rank < best_rank:
                        best_result = {"U": U_new, "V": V_new, "W": W_new}
                        best_rank = new_rank
                        # Found a better decomposition
                        return best_result
    
    # If no better found, return schoolbook
    return best_result
