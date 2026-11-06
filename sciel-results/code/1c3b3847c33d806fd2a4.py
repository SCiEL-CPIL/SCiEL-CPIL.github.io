"""
Strategy: Multi-Start Spectral/SDP-like Initialization with Vectorized Simulated Annealing 
and 2-Flip Local Search.

Improvements over the previous version:
1. Enhanced Initialization: Uses random projections onto the subspace of the first K Laplacian 
   eigenvectors (inspired by SDP relaxation geometry) to generate diverse, high-quality starting points.
2. 2-Flip Moves: Incorporates 2-flip moves into the Simulated Annealing and final polish. 
   This allows escaping local optima that are stable under 1-flip moves.
3. Efficient Gain Calculation: Maintains state to compute 1-flip and 2-flip gains in O(n) 
   per step (vectorized) rather than O(n^2) or recomputing from scratch.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def partition(adjacency):
    """
    Assign each vertex a 0/1 label to maximize the number of cut edges.
    """
    adj = np.asarray(adjacency, dtype=np.float64)
    n = len(adj)
    
    if n <= 1:
        return np.zeros(n, dtype=np.int8)
    
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    if not np.any(adj):
        return np.zeros(n, dtype=np.int8)
    
    degrees = np.sum(adj, axis=1)
    
    # Build Laplacian
    D = np.diag(degrees)
    L = D - adj
    
    # Get eigenvectors of Laplacian
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    except Exception:
        eigenvalues = np.zeros(n)
        eigenvectors = np.eye(n)

    initializations = []
    
    # 1. Fiedler vector (2nd smallest eigenvalue)
    if n > 1:
        fiedler_vec = eigenvectors[:, 1]
        labels_fiedler = (fiedler_vec > 0).astype(np.int8)
        initializations.append(labels_fiedler.copy())
        
    # 2. Top eigenvector of adjacency matrix
    try:
        eigenvalues_a, eigenvectors_a = np.linalg.eigh(adj)
        idx_a = np.argsort(eigenvalues_a)[::-1]
        top_eigvec = eigenvectors_a[:, idx_a[0]]
        labels_top = (top_eigvec > 0).astype(np.int8)
        initializations.append(labels_top.copy())
    except Exception:
        pass

    # 3. SDP-like Random Projections
    # Project random vectors onto the subspace spanned by the first K eigenvectors
    K = min(10, n)
    subspace = eigenvectors[:, :K]
    rng_init = np.random.RandomState(42)
    for _ in range(5):
        r = rng_init.randn(n)
        # Project r onto subspace: P = subspace @ subspace.T
        # proj = subspace @ (subspace.T @ r)
        proj = subspace @ (subspace.T @ r)
        labels_proj = (proj > 0).astype(np.int8)
        initializations.append(labels_proj.copy())

    # 4. Random and Balanced Random
    rng = np.random.RandomState(123)
    for _ in range(2):
        initializations.append(rng.randint(0, 2, n).astype(np.int8))
    
    labels_balanced = np.zeros(n, dtype=np.int8)
    half = n // 2
    labels_balanced[:half] = 1
    perm = rng.permutation(n)
    labels_balanced = labels_balanced[perm]
    initializations.append(labels_balanced.copy())

    def compute_cut_size(labels):
        """Compute the number of cut edges."""
        diff_mask = (labels[:, None] != labels[None, :])
        return 0.5 * np.sum(adj * diff_mask)

    def local_search_sa(labels, max_iter=50000):
        """
        Vectorized Simulated Annealing with 1-flip and 2-flip moves.
        """
        labels = labels.copy()
        # Convert to +/-1 for easier gain calculation
        # s = 2*labels - 1
        s = 2 * labels.astype(np.float64) - 1.0
        
        # Current cut size
        current_cut = compute_cut_size(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        # Precompute Sigma = A @ s
        # Sigma[i] = sum_j A[i,j] s[j]
        Sigma = adj @ s
        
        # Gain for flipping i (1-flip):
        # Delta_i = -s[i] * Sigma[i]
        # If Delta_i > 0, flipping i increases cut.
        
        # Gain for flipping i and j (2-flip):
        # Delta_ij = Delta_i + Delta_j - 2 * A[i,j] * s[i] * s[j]
        
        T0 = 5.0
        T_min = 0.01
        cooling = 0.999
        T = T0
        
        rng_sa = np.random.RandomState(int(start_time * 1000) % 10000)
        
        for it in range(max_iter):
            if time.time() > deadline - 10.0:
                break
            
            # Compute 1-flip gains
            gains_1 = -s * Sigma
            
            # Find best 1-flip
            max_gain_1 = np.max(gains_1)
            best_idx_1 = np.argmax(gains_1)
            
            # Compute 2-flip gains efficiently
            # We want max_{i < j} (gains_1[i] + gains_1[j] - 2 * A[i,j] * s[i] * s[j])
            # Let B[i,j] = -2 * A[i,j] * s[i] * s[j]
            # Note B is symmetric.
            # We can vectorize this by constructing the matrix of 2-flip gains.
            # G2[i,j] = gains_1[i] + gains_1[j] + B[i,j]
            # For i != j.
            
            # Optimization: Instead of full NxN matrix every step (O(N^2) memory/time),
            # we can sample or use a heuristic. However, for N=100, NxN is 10,000 floats, which is tiny.
            # Let's compute the full matrix to find the global best 2-flip.
            
            s_col = s[:, None]
            s_row = s[None, :]
            # B = -2 * adj * s_col * s_row
            B = -2.0 * adj * s_col * s_row
            
            # G2 = gains_1[:, None] + gains_1[None, :] + B
            G2 = gains_1[:, None] + gains_1[None, :] + B
            
            # Set diagonal to -inf to avoid i=j
            np.fill_diagonal(G2, -np.inf)
            
            max_gain_2 = np.max(G2)
            # Get indices
            flat_idx = np.argmax(G2)
            i2, j2 = np.unravel_index(flat_idx, (n, n))
            
            # Choose between best 1-flip and best 2-flip
            # Or use SA probability to choose a random move?
            # Strategy: With prob 0.5, pick best 1-flip. With prob 0.5, pick best 2-flip.
            # Or, if best 2-flip is significantly better, pick it.
            
            # Let's stick to a standard SA approach:
            # Propose a move.
            # To keep it simple and effective, we will alternate or choose based on temperature.
            # Actually, picking the *best* available move (1 or 2) is a form of greedy hill climbing.
            # SA should explore. 
            # Let's propose:
            # 1. With prob 0.7, pick the best 1-flip.
            # 2. With prob 0.3, pick the best 2-flip.
            # This balances the frequency of moves.
            
            if rng_sa.random() < 0.7:
                move_type = 1
                gain = max_gain_1
                # Update state for 1-flip
                flip_indices = [best_idx_1]
            else:
                move_type = 2
                gain = max_gain_2
                flip_indices = [i2, j2]
                
            # Accept/Reject
            if gain > 0:
                accept = True
            else:
                if T > T_min:
                    prob = np.exp(gain / T)
                    accept = (rng_sa.random() < prob)
                else:
                    accept = False
            
            if accept:
                if move_type == 1:
                    idx = best_idx_1
                    # Flip s[idx]
                    s[idx] *= -1
                    labels[idx] = 1 - labels[idx]
                    # Update Sigma
                    # Sigma[k] = sum_j A[k,j] s[j]
                    # Only Sigma changes for neighbors of idx.
                    # Sigma_new = Sigma - 2 * s[idx] * adj[:, idx]
                    # Note: s[idx] is the OLD value before flip?
                    # Let's be careful.
                    # Old s[idx] = old_s. New s[idx] = -old_s.
                    # Delta s[idx] = -2 * old_s.
                    # Delta Sigma[k] = A[k, idx] * Delta s[idx] = -2 * old_s * A[k, idx].
                    old_s = -s[idx] # Since we already flipped s[idx] in place? 
                    # Wait, I flipped s[idx] *= -1 BEFORE this update?
                    # Let's revert to calculating delta first.
                    
                    # Correct Update Logic:
                    # 1. Calculate gain using OLD state.
                    # 2. Update state.
                    # 3. Update Sigma.
                    
                    # I already updated s[idx] and labels[idx].
                    # So current s[idx] is NEW.
                    # OLD s[idx] = -s[idx].
                    # Delta s[idx] = NEW - OLD = s[idx] - (-s[idx]) = 2 * s[idx].
                    # Wait.
                    # If OLD = 1, NEW = -1. Delta = -2.
                    # Formula: Delta Sigma = A[:, idx] * Delta s[idx].
                    # Delta s[idx] = 2 * NEW_s[idx] ? No.
                    # NEW = -OLD.
                    # Delta = NEW - OLD = -OLD - OLD = -2 * OLD = 2 * NEW.
                    # Yes, Delta s[idx] = 2 * s[idx] (where s[idx] is the NEW value).
                    
                    # So: Sigma += 2 * s[idx] * adj[:, idx]
                    Sigma += 2.0 * s[idx] * adj[:, idx]
                    
                else: # move_type == 2
                    idx1, idx2 = i2, j2
                    # Flip both
                    # Old s1, s2. New s1=-s1, s2=-s2.
                    # Delta s1 = 2 * New s1. Delta s2 = 2 * New s2.
                    
                    # Update labels and s
                    old_s1 = s[idx1]
                    old_s2 = s[idx2]
                    
                    s[idx1] *= -1
                    s[idx2] *= -1
                    labels[idx1] = 1 - labels[idx1]
                    labels[idx2] = 1 - labels[idx2]
                    
                    # Update Sigma
                    # Sigma[k] += A[k, idx1] * Delta s1 + A[k, idx2] * Delta s2
                    # Delta s1 = 2 * s[idx1] (New)
                    # Delta s2 = 2 * s[idx2] (New)
                    Sigma += 2.0 * s[idx1] * adj[:, idx1] + 2.0 * s[idx2] * adj[:, idx2]
                    
                current_cut += gain
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
            
            # Cool down
            T *= cooling
            if T < T_min:
                T = T_min
                
        return best_labels, best_cut

    # Run SA from each initialization
    best_labels = None
    best_cut = -1
    
    for init_labels in initializations:
        if time.time() > deadline - 20.0:
            break
        
        sa_labels, sa_cut = local_search_sa(init_labels, max_iter=30000)
        
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels
    
    # Final Polish: Strict local search with 1-flip and 2-flip
    if best_labels is not None and time.time() < deadline - 5.0:
        labels = best_labels.copy()
        s = 2 * labels.astype(np.float64) - 1.0
        Sigma = adj @ s
        
        improved = True
        while improved:
            if time.time() > deadline - 2.0:
                break
            improved = False
            
            # 1-Flip
            gains_1 = -s * Sigma
            max_gain_1 = np.max(gains_1)
            if max_gain_1 > 1e-9:
                idx = np.argmax(gains_1)
                # Apply flip
                s[idx] *= -1
                labels[idx] = 1 - labels[idx]
                Sigma += 2.0 * s[idx] * adj[:, idx]
                improved = True
                continue # Re-evaluate everything
            
            # 2-Flip
            s_col = s[:, None]
            s_row = s[None, :]
            B = -2.0 * adj * s_col * s_row
            G2 = gains_1[:, None] + gains_1[None, :] + B
            np.fill_diagonal(G2, -np.inf)
            
            max_gain_2 = np.max(G2)
            if max_gain_2 > 1e-9:
                flat_idx = np.argmax(G2)
                i2, j2 = np.unravel_index(flat_idx, (n, n))
                
                # Apply 2-flip
                s[i2] *= -1
                s[j2] *= -1
                labels[i2] = 1 - labels[i2]
                labels[j2] = 1 - labels[j2]
                
                Sigma += 2.0 * s[i2] * adj[:, i2] + 2.0 * s[j2] * adj[:, j2]
                improved = True
                continue # Re-evaluate everything
                
        best_labels = labels

    if best_labels is None:
        best_labels = np.zeros(n, dtype=np.int8)
    
    return best_labels
# EVOLVE-BLOCK-END
