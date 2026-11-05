import time
import numpy as np
import random

def construct(q, n, radius):
    t_start = time.time()
    deadline = 115.0  # Safety margin before 120s limit

    space_size = q ** n

    # Precompute neighbors (ball of radius 1) for each word
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i - 1] * q

    neighbors = np.empty(space_size, dtype=object)
    for idx in range(space_size):
        nbs = [idx]
        for pos in range(n):
            digit = (idx // q_powers[n - 1 - pos]) % q
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                delta = (new_digit - digit) * q_powers[n - 1 - pos]
                nbs.append(idx + delta)
        neighbors[idx] = np.array(nbs, dtype=np.int32)

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Spectral approach: build adjacency matrix and compute low-frequency eigenvectors
    # For the Hamming graph, the adjacency matrix A has A[i,j] = 1 if dist(i,j) == 1
    # We'll use a sparse representation via the neighbors list
    
    # Build adjacency matrix as sparse (using scipy if available, else dense for small cases)
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
        use_scipy = True
    except ImportError:
        use_scipy = False

    if use_scipy:
        # Build sparse adjacency matrix
        row_idx = []
        col_idx = []
        for i in range(space_size):
            for j in neighbors[i]:
                if j != i:  # exclude self-loops
                    row_idx.append(i)
                    col_idx.append(j)
        
        data = np.ones(len(row_idx), dtype=np.float64)
        A = csr_matrix((data, (row_idx, col_idx)), shape=(space_size, space_size))
        
        # Compute a few low-frequency eigenvectors (smallest eigenvalues)
        # For a connected graph, the smallest eigenvalue is -d (where d is degree)
        # We want eigenvectors corresponding to small |eigenvalue| or specific modes
        # Actually, for covering, we want to identify "central" vertices
        # Let's use the largest eigenvalue eigenvector (Perron-Frobenius) as primary signal
        # and a few others for diversity
        
        k_eig = min(10, space_size - 1)
        try:
            # Get largest magnitude eigenvalues
            eigvals, eigvecs = eigsh(A, k=k_eig, which='LM')
            # Sort by eigenvalue
            order = np.argsort(eigvals)
            eigvecs_sorted = eigvecs[:, order]
            
            # The Perron vector (largest eigenvalue) gives a centrality measure
            # Use absolute values of the top eigenvectors to rank vertices
            # Combine multiple eigenvectors for robustness
            spectral_scores = np.zeros(space_size, dtype=np.float64)
            for i in range(min(5, eigvecs_sorted.shape[1])):
                vec = eigvecs_sorted[:, i]
                # Normalize and add absolute values
                vec_abs = np.abs(vec)
                vec_abs = vec_abs / (np.max(vec_abs) + 1e-10)
                spectral_scores += vec_abs
            
            # Also add degree-based centrality (all vertices have same degree in Hamming graph,
            # but we can use the number of uncovered neighbors as a dynamic score)
            
        except Exception:
            # Fallback: use random ordering
            spectral_scores = np.random.rand(space_size)
    else:
        # Dense eigendecomposition fallback
        try:
            row_idx = []
            col_idx = []
            for i in range(space_size):
                for j in neighbors[i]:
                    if j != i:
                        row_idx.append(i)
                        col_idx.append(j)
            
            A_dense = np.zeros((space_size, space_size), dtype=np.float64)
            A_dense[np.array(row_idx), np.array(col_idx)] = 1.0
            
            eigvals, eigvecs = np.linalg.eigh(A_dense)
            # Take last few (largest eigenvalues)
            k_eig = min(10, space_size)
            eigvecs_sorted = eigvecs[:, -k_eig:]
            
            spectral_scores = np.zeros(space_size, dtype=np.float64)
            for i in range(min(5, eigvecs_sorted.shape[1])):
                vec = eigvecs_sorted[:, i]
                vec_abs = np.abs(vec)
                vec_abs = vec_abs / (np.max(vec_abs) + 1e-10)
                spectral_scores += vec_abs
        except Exception:
            spectral_scores = np.random.rand(space_size)

    # Normalize spectral scores
    spectral_scores = spectral_scores / (np.max(spectral_scores) + 1e-10)

    # Spectral greedy: repeatedly pick the vertex with highest spectral score
    # among those that cover the most uncovered words
    uncovered_mask = np.ones(space_size, dtype=bool)
    
    code = []
    in_set = np.zeros(space_size, dtype=bool)
    
    # Precompute cover_map for efficient scoring
    cover_map = np.empty(space_size, dtype=object)
    for w in range(space_size):
        cover_map[w] = []
    for c in range(space_size):
        for w in neighbors[c]:
            cover_map[w].append(c)
    for w in range(space_size):
        cover_map[w] = np.array(cover_map[w], dtype=np.int32)

    while time.time() - t_start < deadline - 15:
        if not uncovered_mask.any():
            break
        
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            break
        
        # Compute score for each candidate: combination of spectral score and coverage
        # score[c] = spectral_scores[c] * alpha + (number of uncovered words covered by c) * beta
        # We want to balance spectral centrality with actual coverage gain
        
        # For efficiency, only consider candidates that cover at least one uncovered word
        candidate_set = set()
        for w in uncovered_indices:
            for c in cover_map[w]:
                candidate_set.add(c)
        
        if not candidate_set:
            break
        
        candidates = np.array(list(candidate_set), dtype=np.int32)
        
        # Compute coverage gain for each candidate
        coverage_gain = np.zeros(len(candidates), dtype=np.int32)
        for i, c in enumerate(candidates):
            ball = neighbors[c]
            coverage_gain[i] = np.sum(uncovered_mask[ball])
        
        # Combine with spectral score
        spectral_cands = spectral_scores[candidates]
        
        # Weight: prioritize coverage, but use spectral as tie-breaker
        # Use a weighted sum
        weights = coverage_gain.astype(np.float64) * 1.0 + spectral_cands * 5.0
        
        best_idx = np.argmax(weights)
        best_c = candidates[best_idx]
        
        if coverage_gain[best_idx] == 0:
            # No candidate covers any uncovered word (shouldn't happen)
            break
        
        code.append(best_c)
        in_set[best_c] = True
        uncovered_mask[neighbors[best_c]] = False

    # Force-cover any remaining
    if uncovered_mask.any():
        remaining = np.where(uncovered_mask)[0]
        for w in remaining:
            code.append(w)
            uncovered_mask[neighbors[w]] = False

    # Local search: try to remove redundant codewords
    def local_search(code):
        code_set = set(code)
        cov = np.zeros(space_size, dtype=np.int32)
        for c in code:
            cov[neighbors[c]] += 1
        
        improved = True
        while improved:
            if time.time() - t_start > deadline - 10:
                break
            improved = False
            random.shuffle(code)
            for i in range(len(code)):
                c = code[i]
                ball = neighbors[c]
                if np.any(cov[ball] == 1):
                    continue  # critical, cannot remove
                cov[ball] -= 1
                code.pop(i)
                code_set.discard(c)
                improved = True
                break
        
        return code

    current_code = local_search(code)
    best_code = current_code[:]
    best_size = len(best_code)

    # ILS with spectral-informed kicks
    rng = random.Random(42)
    
    while time.time() - t_start < deadline - 5:
        # Kick: remove codewords with lowest spectral scores first (more likely to be less central)
        k = rng.randint(2, min(6, max(1, len(current_code) - 5)))
        
        # Sort current code by spectral score (ascending) and remove k lowest
        sorted_by_spectral = sorted(current_code, key=lambda c: spectral_scores[c])
        to_remove = sorted_by_spectral[:k]
        
        kicked = [c for c in current_code if c not in to_remove]
        
        # Re-cover uncovered words
        cov = np.zeros(space_size, dtype=np.int32)
        for c in kicked:
            cov[neighbors[c]] += 1
        
        uncovered_mask = (cov == 0)
        if uncovered_mask.any():
            # Greedy re-cover using spectral-informed selection
            while time.time() - t_start < deadline - 10:
                if not uncovered_mask.any():
                    break
                uncovered_indices = np.where(uncovered_mask)[0]
                if len(uncovered_indices) == 0:
                    break
                
                candidate_set = set()
                for w in uncovered_indices:
                    for c in cover_map[w]:
                        if c not in kicked:
                            candidate_set.add(c)
                
                if not candidate_set:
                    # Add any word from uncovered
                    w = uncovered_indices[0]
                    kicked.append(w)
                    uncovered_mask[neighbors[w]] = False
                    continue
                
                candidates = np.array(list(candidate_set), dtype=np.int32)
                coverage_gain = np.zeros(len(candidates), dtype=np.int32)
                for i, c in enumerate(candidates):
                    ball = neighbors[c]
                    coverage_gain[i] = np.sum(uncovered_mask[ball])
                
                spectral_cands = spectral_scores[candidates]
                weights = coverage_gain.astype(np.float64) * 1.0 + spectral_cands * 5.0
                
                best_idx = np.argmax(weights)
                best_c = candidates[best_idx]
                
                if coverage_gain[best_idx] == 0:
                    break
                
                kicked.append(best_c)
                uncovered_mask[neighbors[best_c]] = False
            
            # Force-cover remaining
            if uncovered_mask.any():
                remaining = np.where(uncovered_mask)[0]
                for w in remaining:
                    kicked.append(w)
                    uncovered_mask[neighbors[w]] = False
        
        searched = local_search(kicked)
        
        if len(searched) < best_size:
            best_code = searched[:]
            best_size = len(best_code)
            current_code = searched[:]
        else:
            if len(searched) <= len(current_code):
                current_code = searched[:]
            else:
                current_code = best_code[:]

    # Verify best_code is valid
    final_cov = np.zeros(space_size, dtype=np.int32)
    for c in best_code:
        final_cov[neighbors[c]] += 1

    if np.any(final_cov == 0):
        # Fallback: force cover
        uncovered_mask = (final_cov == 0)
        remaining = np.where(uncovered_mask)[0]
        for w in remaining:
            best_code.append(w)
            final_cov[neighbors[w]] += 1

    return [idx_to_word(c) for c in best_code]
