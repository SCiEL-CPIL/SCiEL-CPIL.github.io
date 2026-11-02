import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Time limit
    start_time = time.time()
    time_limit = 140 # 10s margin
    
    # Target tensor M
    # Dimensions: (n*m, m*p, p*n)
    # Flatten to vector of length n*m * m*p * p*n
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    total_dim = dim_u * dim_v * dim_w
    
    M_flat = np.zeros(total_dim)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                # Linear index in the flattened tensor
                # Order: u, v, w
                idx = (idx_u * dim_v + idx_v) * dim_w + idx_w
                M_flat[idx] = 1.0
                
    # Generate candidates
    # Entries in {-1, 0, 1}
    # To keep it fast, we don't generate all 19683 if we can avoid it, but 19683 is small.
    # Let's generate them.
    candidates = []
    # u vectors
    us = []
    for u_vals in np.ndindex(3, 3):
        vec = np.array([x - 1 for x in u_vals]) # -1, 0, 1
        if np.any(vec != 0):
            us.append(vec)
    vs = []
    for v_vals in np.ndindex(3, 3, 3):
        vec = np.array([x - 1 for x in v_vals])
        if np.any(vec != 0):
            vs.append(vec)
    ws = []
    for w_vals in np.ndindex(3, 3, 3, 3):
        vec = np.array([x - 1 for x in w_vals])
        if np.any(vec != 0):
            ws.append(vec)
            
    # This loop is 9 * 27 * 81 = 19683 iterations.
    # Each iteration creates a tensor of size 24.
    # This is fast.
    for u in us:
        for v in vs:
            for w in ws:
                tensor = np.outer(np.outer(u, v), w).flatten()
                candidates.append(tensor)
                
    candidates = np.array(candidates)
    
    # Greedy Set Cover
    # We want to select a subset S such that sum(candidates[S]) == M_flat
    # Greedy:
    # Residual R = M_flat
    # Selected = []
    # While R != 0:
    #   Find candidate C that maximizes "coverage"
    #   Coverage metric: 
    #     We want to reduce the L2 norm of R.
    #     But we need exact match.
    #     A simple heuristic: Pick C that has the highest inner product with R, 
    #     but also check if it introduces "errors" (non-zero where R is 0).
    #     Actually, since M is 0/1, and candidates are -1/0/1, 
    #     we can try to pick candidates that match the 1s in M and cancel out elsewhere.
    
    # Let's try a few random restarts of a greedy algorithm.
    # In each restart, we shuffle candidates and pick greedily.
    
    def try_greedy(candidates, M_flat, max_iters=50):
        R = M_flat.copy()
        selected = []
        for _ in range(max_iters):
            if np.allclose(R, 0):
                break
            # Compute inner products
            # scores = candidates @ R
            # We want to pick a candidate that reduces the error.
            # Error = ||R - C||^2 = ||R||^2 - 2 R.C + ||C||^2
            # Minimizing error is maximizing 2 R.C - ||C||^2
            scores = candidates @ R
            norms_sq = np.sum(candidates**2, axis=1)
            scores = 2 * scores - norms_sq
            # Add some randomness
            scores += np.random.normal(0, 0.1, len(scores))
            best_idx = np.argmax(scores)
            C = candidates[best_idx]
            # Check if this step is "safe" or if we should backtrack?
            # For now, just take it.
            R = R - C
            selected.append(best_idx)
            
            # If R becomes too "weird" (e.g., large values), abort?
            if np.max(np.abs(R)) > 5:
                return None
                
        if np.allclose(R, 0, atol=1e-9):
            # Reconstruct U, V, W
            # selected contains indices into candidates
            # We need to extract u, v, w for each selected candidate
            # This requires storing the components, not just the flattened tensor.
            # So we should store (u, v, w) tuples.
            return selected
        return None

    # To allow reconstruction, we need to store the components.
    # Let's restructure candidates to be a list of (tensor, u, v, w)
    # Or just re-derive u, v, w from the tensor? No, that's hard.
    # So we store the components.
    
    candidate_components = []
    candidate_tensors = []
    for u in us:
        for v in vs:
            for w in ws:
                tensor = np.outer(np.outer(u, v), w).flatten()
                candidate_components.append((u, v, w))
                candidate_tensors.append(tensor)
                
    candidate_tensors = np.array(candidate_tensors)
    
    for restart in range(100): # 100 restarts
        if time.time() - start_time > time_limit:
            break
            
        # Shuffle indices
        indices = np.arange(len(candidate_components))
        np.random.shuffle(indices)
        shuffled_tensors = candidate_tensors[indices]
        
        R = M_flat.copy()
        selected_indices = []
        
        success = False
        for _ in range(30): # Max 30 terms
            if np.allclose(R, 0, atol=1e-9):
                success = True
                break
                
            scores = shuffled_tensors @ R
            norms_sq = np.sum(shuffled_tensors**2, axis=1)
            scores = 2 * scores - norms_sq
            # Randomness
            scores += np.random.normal(0, 0.5, len(scores))
            best_idx = np.argmax(scores)
            C = shuffled_tensors[best_idx]
            R = R - C
            selected_indices.append(indices[best_idx])
            
            if np.max(np.abs(R)) > 10:
                break
                
        if success:
            # Reconstruct
            rank = len(selected_indices)
            if rank < best_rank:
                U = np.zeros((dim_u, rank))
                V = np.zeros((dim_v, rank))
                W = np.zeros((dim_w, rank))
                for k, idx in enumerate(selected_indices):
                    u, v, w = candidate_components[idx]
                    U[:, k] = u
                    V[:, k] = v
                    W[:, k] = w
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
                # If we found a very good one, we can stop early?
                # Rank 18 is good.
                if best_rank <= 18:
                    break

    # Convert to lists for output
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
