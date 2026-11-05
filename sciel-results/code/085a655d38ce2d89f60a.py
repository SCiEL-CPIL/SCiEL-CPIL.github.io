"""
Optimized Greedy MIS via Local Search on Vertex Scores.

This program generates priority scores for a greedy decoder.
It uses a two-phase approach:
1. Generate diverse initial score vectors based on graph structural features (Degree, Clustering, Spectral).
2. Perform local search (swaps and perturbations) on the score vectors to maximize the resulting independent set size, 
   using the greedy decoder as the objective function.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    DEADLINE = 125.0  # Safety margin below 150s limit, allowing ~10s for cleanup

    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]

    if N == 0:
        return np.array([])

    # Ensure symmetric and zero diagonal
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    A_bool = A > 0.5
    
    # Precompute neighbor lists for faster greedy evaluation
    # Using lists of indices is faster than boolean slicing in a loop for N=200
    neighbors = [np.where(A_bool[i])[0] for i in range(N)]
    # Also store neighbor sets for O(1) membership checks if needed
    neighbor_sets = [set(n.tolist()) for n in neighbors]

    # --- Fast Greedy Evaluator ---
    def evaluate(scores):
        """
        Simulates the greedy decoder:
        1. Sort vertices by score descending.
        2. Iterate through sorted list.
        3. If vertex is not blocked (not neighbor of any picked vertex), pick it.
        4. Mark it and its neighbors as blocked.
        Returns the count of picked vertices.
        """
        order = np.argsort(-scores, kind='mergesort')
        
        blocked = np.zeros(N, dtype=bool)
        count = 0
        
        for v in order:
            if not blocked[v]:
                count += 1
                blocked[v] = True
                if neighbors[v].size > 0:
                    blocked[neighbors[v]] = True
        
        return count
    
    # --- 2-Opt Neighborhood Refinement ---
    def two_opt_refine(scores, max_iterations=500):
        """
        Try swapping adjacent vertices in the sorted order to improve the independent set size.
        Returns (improved_scores, improved_size).
        """
        current_scores = scores.copy()
        current_size = evaluate(current_scores)
        order = np.argsort(-current_scores, kind='mergesort')
        
        improved = True
        iters = 0
        while improved and iters < max_iterations:
            improved = False
            iters += 1
            for i in range(N - 1):
                v1 = order[i]
                v2 = order[i + 1]
                # Swap scores of adjacent vertices in the order
                current_scores[v1], current_scores[v2] = current_scores[v2], current_scores[v1]
                new_size = evaluate(current_scores)
                if new_size > current_size:
                    current_size = new_size
                    improved = True
                    # Re-sort after improvement
                    order = np.argsort(-current_scores, kind='mergesort')
                else:
                    # Revert swap
                    current_scores[v1], current_scores[v2] = current_scores[v2], current_scores[v1]
        
        return current_scores, current_size

    # --- Feature Extraction for Initial Candidates ---
    
    # 1. Degree
    D = A.sum(axis=1)
    
    # 2. Clustering Coefficient (approximated or exact)
    # For N=200, exact triangle count is feasible
    A_sq = A @ A
    Tri = np.diag(A_sq) / 2.0
    D_safe = np.where(D > 0, D, 1.0)
    # Standard clustering coefficient: 2*Triangles / (deg*(deg-1))
    # Avoid division by zero
    Clus = np.zeros(N)
    mask = D >= 2
    Clus[mask] = 2.0 * Tri[mask] / (D[mask] * (D[mask] - 1.0))

    # 3. Mean Neighbor Degree
    SND = A @ D
    MeanND = SND / D_safe

    # 4. Spectral Centrality (Principal Eigenvector)
    # Vertices with high spectral centrality are central. We want low centrality for independent set.
    x = np.ones(N) / N
    for _ in range(30):
        x_new = A @ x
        norm = np.linalg.norm(x_new)
        if norm < 1e-12:
            break
        x_new /= norm
        if np.linalg.norm(x_new - x) < 1e-9:
            break
        x = x_new
    Spectral = x

    # 5. Reciprocal Neighbor Degree Sum (RND)
    # score_i = sum_{j in N(i)} 1/(d_j + 1)
    # Quantifies the "independence value" gained by selecting i:
    # it removes low-degree (low-cost) neighbors from contention.
    # Vertices whose neighbors are mostly low-degree are "cheap to pick"
    # because they block few high-impact vertices.
    inv_deg = 1.0 / (D + 1.0)
    RND = A @ inv_deg  # RND[i] = sum_{j in N(i)} 1/(d_j + 1)

    # 6. Fiedler Vector (second-smallest eigenvector of Laplacian)
    # Identifies vertices minimally entangled in graph connectivity
    # High absolute Fiedler values indicate vertices on periphery/sparse regions
    L = np.diag(D) - A
    try:
        # Use scipy for efficient eigenvalue computation
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import csr_matrix
        L_sparse = csr_matrix(L)
        # Get 3 smallest eigenvalues to ensure we get the Fiedler vector
        # (eigsh requires k < N for 'SM' target)
        k_eig = min(3, N - 1)
        eigvals, eigvecs = eigsh(L_sparse, k=k_eig, which='SM')
        # Sort by eigenvalue
        idx_sort = np.argsort(eigvals)
        eigvals = eigvals[idx_sort]
        eigvecs = eigvecs[:, idx_sort]
        # Fiedler vector is the second smallest (index 1)
        if len(eigvals) >= 2:
            Fiedler = np.abs(eigvecs[:, 1])
        else:
            # Fallback to principal eigenvector if only one mode found
            Fiedler = Spectral
    except Exception:
        # Fallback: use spectral as proxy
        Fiedler = Spectral

    # Normalize features to [0, 1] for consistent weighting
    def norm(x):
        min_val = np.min(x)
        max_val = np.max(x)
        if max_val == min_val:
            return np.zeros_like(x)
        return (x - min_val) / (max_val - min_val)

    D_n = norm(D)
    Clus_n = norm(Clus)
    MeanND_n = norm(MeanND)
    Spectral_n = norm(Spectral)
    Fiedler_n = norm(Fiedler)
    RND_n = norm(RND)
    
    # "Openness": Vertices with many neighbors that are NOT connected to each other.
    # This suggests the vertex is in a sparse region.
    # OpenPairs = C(D, 2) - Tri
    OpenPairs = np.maximum(0, 0.5 * D * (D - 1) - Tri)
    Open_n = norm(OpenPairs)

    # --- Phase 1: Initialize with Strong Heuristics ---
    
    candidates = []
    
    # H1: Min Degree (Classic)
    candidates.append(-D_n)
    
    # H2: Min Degree, High Openness
    candidates.append(-D_n + 0.5 * Open_n)
    
    # H3: Min Degree, Low Clustering
    candidates.append(-D_n - 0.5 * Clus_n)
    
    # H4: Min Degree, Low Mean Neighbor Degree (prefer peripheral)
    candidates.append(-D_n - 0.5 * MeanND_n)
    
    # H5: Low Spectral Centrality
    candidates.append(-Spectral_n)
    
    # H6: Mix of Degree and Spectral
    candidates.append(-0.7 * D_n - 0.3 * Spectral_n)
    
    # H7: High Fiedler (peripheral/sparse regions)
    candidates.append(Fiedler_n)
    
    # H8: Min Degree, High Fiedler
    candidates.append(-0.5 * D_n + 0.5 * Fiedler_n)
    
    # H9: Min Degree, High Fiedler, Low Clustering
    candidates.append(-0.4 * D_n + 0.4 * Fiedler_n - 0.2 * Clus_n)
    
    # H10: Random initialization
    rng = np.random.default_rng(42)
    candidates.append(rng.normal(0, 1, N))
    
    # H11: Strong Fiedler Focus (Directive: Prioritize vertices with largest absolute Fiedler values)
    # This directly tests the hypothesis that Fiedler vector alone is a strong guide
    candidates.append(2.0 * Fiedler_n)
    
    # H12: Fiedler + Low Degree (Hybrid)
    candidates.append(1.5 * Fiedler_n - 0.5 * D_n)
    
    # H13: Fiedler + Openness
    candidates.append(1.5 * Fiedler_n + 0.5 * Open_n)
    
    # H14: Reciprocal Neighbor Degree Sum (high RND = neighbors are low-degree, cheap to block)
    candidates.append(RND_n)
    
    # H15: Low Degree + High RND
    candidates.append(-0.5 * D_n + 0.5 * RND_n)
    
    # H16: Fiedler + RND
    candidates.append(1.0 * Fiedler_n + 0.5 * RND_n)
    
    # H17: Low Degree + RND + Fiedler
    candidates.append(-0.4 * D_n + 0.4 * RND_n + 0.3 * Fiedler_n)
    
    # H18: RND - Mean Neighbor Degree (prefer neighbors that are low-degree)
    candidates.append(RND_n - 0.5 * MeanND_n)
    
    # H19: Strong RND focus
    candidates.append(2.0 * RND_n)
    
    # H20: RND + Openness
    candidates.append(1.5 * RND_n + 0.5 * Open_n)

    best_size = -1
    best_scores = None

    for c in candidates:
        size = evaluate(c)
        if size > best_size:
            best_size = size
            best_scores = c.copy()
            
    # --- Phase 2: Local Search ---
    # Strategy:
    # 1. Try swapping scores of two random vertices.
    # 2. Try adjacent swaps in the ranking order.
    # 3. Try adding small Gaussian noise.
    # 4. Occasionally revert to a base heuristic + noise to escape local optima.
    # 5. Periodically run 2-opt refinement.

    # We can do many iterations in 120s. N=200.
    # Each eval is ~200 argsorts + 200 checks. Very fast.
    
    trial_count = 0
    max_trials = 100000 # Increased budget with faster evaluation
    
    while trial_count < max_trials:
        if time.time() > DEADLINE:
            break
            
        # Decide move type
        r = rng.random()
        new_scores = best_scores.copy()
        
        if r < 0.4:
            # Swap move: Pick two random indices and swap their scores.
            i = rng.integers(0, N)
            j = rng.integers(0, N)
            if i != j:
                new_scores[i], new_scores[j] = new_scores[j], new_scores[i]
            else:
                continue # No change
        elif r < 0.6:
            # Adjacent swap: Pick a vertex and swap with a nearby-ranked vertex
            # This makes smaller, more controlled changes
            order = np.argsort(-best_scores, kind='mergesort')
            rank_i = np.where(order == rng.integers(0, N))[0][0]
            # Pick a vertex within +/- 3 ranks
            delta_rank = rng.integers(-3, 4)
            rank_j = rank_i + delta_rank
            if 0 <= rank_j < N:
                j = order[rank_j]
                new_scores[order[rank_i]], new_scores[j] = new_scores[j], new_scores[order[rank_i]]
            else:
                continue
        elif r < 0.75:
            # Perturbation move: Add small noise to a few random vertices
            k = rng.integers(1, 6) # Perturb 1 to 5 vertices
            idxs = rng.choice(N, k, replace=False)
            noise = rng.normal(0, 0.03, k)
            new_scores[idxs] += noise
        elif r < 0.85:
            # 2-opt refinement on a copy
            new_scores, size = two_opt_refine(best_scores, max_iterations=100)
            if size > best_size:
                best_size = size
                best_scores = new_scores
                trial_count = 0
                continue
        elif r < 0.90:
            # Fiedler-Driven Move: Nudge scores towards Fiedler alignment
            # Vertices with high |Fiedler| should have higher scores (if they are in the independent set)
            # We create a target score vector based on Fiedler and blend it with current best.
            # This helps escape local optima that ignore the global structure.
            fiedler_target = Fiedler_n.copy()
            # Add some noise to avoid getting stuck in a single Fiedler configuration
            noise = rng.normal(0, 0.1, N)
            fiedler_target += noise
            
            # Blend with current best
            alpha = rng.uniform(0.6, 0.9)
            new_scores = alpha * best_scores + (1 - alpha) * fiedler_target
        elif r < 0.95:
            # RND-Driven Move: Nudge scores towards RND alignment
            # Vertices with high RND (neighbors are low-degree) are cheap to select
            rnd_target = RND_n.copy()
            noise = rng.normal(0, 0.1, N)
            rnd_target += noise
            
            # Blend with current best
            alpha = rng.uniform(0.6, 0.9)
            new_scores = alpha * best_scores + (1 - alpha) * rnd_target
        else:
            # Restart/Escape: Blend with a random base heuristic including Fiedler and RND
            w_d = rng.uniform(-1, 0)
            w_c = rng.uniform(-0.5, 0.5)
            w_s = rng.uniform(-0.5, 0.5)
            w_f = rng.uniform(0.5, 1.5) # Weight on Fiedler
            w_r = rng.uniform(0.3, 1.5)  # Weight on RND
            base = w_d * D_n + w_c * Clus_n + w_s * Spectral_n + w_f * Fiedler_n + w_r * RND_n
            # Blend with current best to stay in the neighborhood
            alpha = rng.uniform(0.5, 1.0)
            new_scores = alpha * best_scores + (1 - alpha) * base
            
        size = evaluate(new_scores)
        
        if size > best_size:
            best_size = size
            best_scores = new_scores
            # Reset trial counter for success-based patience
            trial_count = 0
        else:
            trial_count += 1
            
        # Early exit if stuck for too many trials
        if trial_count > 3000:
            # Diversify: jump to a new random neighborhood
            base_heuristics = [
                -D_n,
                -0.5 * D_n + 0.5 * Fiedler_n,
                Fiedler_n,
                -0.7 * D_n - 0.3 * Spectral_n,
                -0.6 * D_n + 0.4 * Open_n
            ]
            base = base_heuristics[rng.integers(0, len(base_heuristics))]
            alpha = rng.uniform(0.3, 0.7)
            best_scores = alpha * best_scores + (1 - alpha) * base
            best_size = evaluate(best_scores)
            trial_count = 0

    # --- Phase 3: Final Refinement (if time permits) ---
    if time.time() < DEADLINE - 5.0 and best_scores is not None:
        # Run 2-opt refinement for a longer period
        improved = True
        while improved and time.time() < DEADLINE - 3.0:
            best_scores, best_size = two_opt_refine(best_scores, max_iterations=200)
            # Check if further improvement is possible
            test_scores, test_size = two_opt_refine(best_scores, max_iterations=50)
            if test_size <= best_size:
                improved = False
        
        # Additional random swap refinement
        for _ in range(5000):
            if time.time() > DEADLINE - 2.0:
                break
            i = rng.integers(0, N)
            j = rng.integers(0, N)
            if i == j: continue
            
            # Store old values
            si = best_scores[i]
            sj = best_scores[j]
            
            # Swap
            best_scores[i] = sj
            best_scores[j] = si
            
            size = evaluate(best_scores)
            
            if size > best_size:
                best_size = size
                # Keep swap
            else:
                # Revert swap
                best_scores[i] = si
                best_scores[j] = sj

        # Fiedler Polish: Try to align the final scores more closely with the Fiedler vector
        # This is a final attempt to exploit the global structure
        if time.time() < DEADLINE - 1.0:
            for alpha in [0.9, 0.8, 0.7, 0.6, 0.5]:
                polished_scores = alpha * best_scores + (1 - alpha) * Fiedler_n
                polished_size = evaluate(polished_scores)
                if polished_size > best_size:
                    best_size = polished_size
                    best_scores = polished_scores
        
        # RND Polish: Try to align the final scores with the RND vector
        if time.time() < DEADLINE - 0.5:
            for alpha in [0.9, 0.8, 0.7, 0.6, 0.5]:
                polished_scores = alpha * best_scores + (1 - alpha) * RND_n
                polished_size = evaluate(polished_scores)
                if polished_size > best_size:
                    best_size = polished_size
                    best_scores = polished_scores
        
        # Combined Fiedler + RND Polish
        if time.time() < DEADLINE - 0.5:
            combined_target = 0.5 * Fiedler_n + 0.5 * RND_n
            for alpha in [0.9, 0.8, 0.7, 0.6, 0.5]:
                polished_scores = alpha * best_scores + (1 - alpha) * combined_target
                polished_size = evaluate(polished_scores)
                if polished_size > best_size:
                    best_size = polished_size
                    best_scores = polished_scores
                
    if best_scores is None:
        return -D.astype(float)

    return best_scores
# EVOLVE-BLOCK-END
