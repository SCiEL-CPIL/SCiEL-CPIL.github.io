"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank

    best_U, best_V, best_W, best_rank = schoolbook()

    def verify_decomposition(U, V, W):
        """Verify that the decomposition is correct using numpy for speed."""
        rank = len(U[0])
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U_np[i * m + j, c] * V_np[j * p + k, c] * W_np[k * n + i, c]
                    if s != 1:
                        return False
        return True

    # Generate candidate rank-1 terms: sparse vectors with entries in {-1, 0, 1}
    # For efficiency, limit to vectors with at most 2 non-zero entries
    
    def generate_sparse_vectors(dim, max_nonzero=2, max_val=1):
        """Generate sparse vectors of given dimension with limited non-zero entries."""
        vecs = set()
        vecs.add(tuple([0] * dim))
        
        if max_nonzero >= 1:
            for pos in range(dim):
                for val in range(-max_val, max_val + 1):
                    if val != 0:
                        v = [0] * dim
                        v[pos] = val
                        vecs.add(tuple(v))
        
        if max_nonzero >= 2:
            for pos1 in range(dim):
                for pos2 in range(pos1 + 1, dim):
                    for val1 in range(-max_val, max_val + 1):
                        for val2 in range(-max_val, max_val + 1):
                            if val1 != 0 and val2 != 0:
                                v = [0] * dim
                                v[pos1] = val1
                                v[pos2] = val2
                                vecs.add(tuple(v))
        
        return list(vecs)

    # For 2x3*3x4: n*m=6, m*p=12, p*n=8
    u_vecs = generate_sparse_vectors(n * m, max_nonzero=2, max_val=1)
    v_vecs = generate_sparse_vectors(m * p, max_nonzero=2, max_val=1)
    w_vecs = generate_sparse_vectors(p * n, max_nonzero=2, max_val=1)

    # Filter to useful candidates (contribute to at least one tensor entry)
    useful_candidates = []
    for u in u_vecs:
        for v in v_vecs:
            for w in w_vecs:
                contributes = False
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            prod = u[i * m + j] * v[j * p + k] * w[k * n + i]
                            if prod != 0:
                                contributes = True
                                break
                        if contributes:
                            break
                    if contributes:
                        break
                if contributes:
                    useful_candidates.append((u, v, w))

    # Try to find a decomposition with rank < 24
    # Use a greedy + local search approach
    
    def try_find_decomposition(target_rank):
        """Try to find a decomposition with exactly target_rank terms."""
        if time.time() > deadline:
            return None
        
        if len(useful_candidates) < target_rank:
            return None
        
        # Precompute contribution vectors for each candidate
        # contribution[candidate_idx] = array of shape (n*m*p,) with the tensor contribution
        n_entries = n * m * p
        contribs = np.zeros((len(useful_candidates), n_entries), dtype=np.int64)
        for ci, (u, v, w) in enumerate(useful_candidates):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * m * p + j * p + k
                        contribs[ci, idx] = u[i * m + j] * v[j * p + k] * w[k * n + i]
        
        target = np.ones(n_entries, dtype=np.int64)
        
        # Greedy: iteratively pick the candidate that best reduces residual
        # Start with empty selection
        selected = []
        residual = target.copy()
        
        for step in range(target_rank):
            if time.time() > deadline:
                return None
            
            # Find the candidate that maximizes dot product with residual
            # (i.e., covers the most uncovered entries with correct sign)
            dots = contribs @ residual
            # Pick the one with maximum positive dot product
            best_idx = np.argmax(dots)
            
            if dots[best_idx] <= 0:
                # No progress possible
                return None
            
            selected.append(best_idx)
            residual -= contribs[best_idx]
            
            # Check if we've reached zero residual
            if np.all(residual == 0):
                # We found a solution with fewer than target_rank terms
                # Build the decomposition
                U = [[0] * len(selected) for _ in range(n * m)]
                V = [[0] * len(selected) for _ in range(m * p)]
                W = [[0] * len(selected) for _ in range(p * n)]
                
                for c, ci in enumerate(selected):
                    u, v, w = useful_candidates[ci]
                    for idx in range(n * m):
                        U[idx][c] = u[idx]
                    for idx in range(m * p):
                        V[idx][c] = v[idx]
                    for idx in range(p * n):
                        W[idx][c] = w[idx]
                
                return U, V, W
        
        # If greedy didn't work, try randomized local search
        max_iterations = 500
        for iteration in range(max_iterations):
            if time.time() > deadline:
                return None
            
            # Random restart: pick target_rank random candidates
            selected = random.sample(range(len(useful_candidates)), target_rank)
            
            # Check if this selection works
            total = np.zeros(n_entries, dtype=np.int64)
            for ci in selected:
                total += contribs[ci]
            
            if np.all(total == target):
                # Found a solution!
                U = [[0] * target_rank for _ in range(n * m)]
                V = [[0] * target_rank for _ in range(m * p)]
                W = [[0] * target_rank for _ in range(p * n)]
                
                for c, ci in enumerate(selected):
                    u, v, w = useful_candidates[ci]
                    for idx in range(n * m):
                        U[idx][c] = u[idx]
                    for idx in range(m * p):
                        V[idx][c] = v[idx]
                    for idx in range(p * n):
                        W[idx][c] = w[idx]
                
                return U, V, W
        
        return None

    # Try ranks from 23 down to 18 (reasonable lower bound)
    for rank in range(23, 17, -1):
        if time.time() > deadline:
            break
        
        result = try_find_decomposition(rank)
        if result is not None:
            U, V, W = result
            if verify_decomposition(U, V, W):
                return {"U": U, "V": V, "W": W}

    # If no better decomposition found, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
