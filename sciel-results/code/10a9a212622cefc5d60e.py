"""Multi-Start Spectral Initialization with Vectorized SA, 2-Flip Polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def partition(adjacency):
    """
    Assign each vertex a 0/1 label to maximize the number of cut edges.
    Uses spectral initialization + vectorized Simulated Annealing with
    2-flip neighborhood exploration and a final high-precision polish pass.
    """
    adj = np.asarray(adjacency, dtype=np.int8)
    n = len(adj)
    
    if n <= 1:
        return np.zeros(n, dtype=np.int8)
    
    total_edges = adj.sum() / 2.0
    deadline = time.time() + 115.0  # Safety margin before 150s kill
    
    best_labels = None
    best_cut = -1.0
    
    def compute_cut(labels):
        s = 2.0 * labels - 1.0
        Q = s @ adj @ s
        return (total_edges - Q) / 2.0

    def spectral_init(rng):
        """Initialize using the leading eigenvector of the adjacency matrix."""
        # Use power iteration for speed (n=100 is small, but still faster than full eigh)
        v = rng.uniform(-1, 1, n)
        v /= np.linalg.norm(v)
        for _ in range(50):
            v_new = adj.astype(np.float64) @ v
            norm = np.linalg.norm(v_new)
            if norm < 1e-12:
                break
            v_new /= norm
            if np.allclose(v, v_new, atol=1e-10):
                break
            v = v_new
        # Threshold: positive components -> 1, negative -> 0
        labels = (v > 0).astype(np.int8)
        return labels

    def local_search_sa(labels, time_limit, use_2flip=True):
        """Simulated Annealing with optional 2-flip moves."""
        current_labels = labels.copy()
        current_cut = compute_cut(current_labels)
        
        T = 1.0
        cooling_rate = 0.9995
        min_T = 0.005
        
        while time.time() < time_limit and T > min_T:
            s = 2.0 * current_labels - 1.0
            neighbor_sum = adj @ s
            
            if use_2flip and np.random.random() < 0.3:
                # Try a 2-flip move: pick two distinct vertices
                v1 = np.random.randint(n)
                v2 = np.random.randint(n)
                if v1 == v2:
                    continue
                    
                # Gain of flipping both v1 and v2 simultaneously
                # Delta cut for flipping v1 alone: g1 = s[v1] * neighbor_sum[v1]
                # Delta cut for flipping v2 alone: g2 = s[v2] * neighbor_sum[v2]
                # Cross term: if both flipped, the edge (v1,v2) interaction changes
                # Let's compute directly:
                # New labels: l1' = 1-l1, l2' = 1-l2
                # Gain = change in cut
                # For each neighbor u of v1 (u != v2): edge (v1,u) changes status
                # For each neighbor u of v2 (u != v1): edge (v2,u) changes status
                # Edge (v1,v2): if A[v1,v2]=1, its cut status changes twice (net 0)
                # Actually let's compute gain directly:
                # g1 = s[v1] * neighbor_sum[v1]  (flip v1 alone)
                # g2 = s[v2] * neighbor_sum[v2]  (flip v2 alone)
                # When flipping both, the cross term is:
                # The interaction between v1 and v2: 
                # If A[v1,v2]=1, flipping both means the edge status doesn't change
                # (both endpoints flip, so same/diff relationship is preserved)
                # So gain_2flip = g1 + g2 - 2*A[v1,v2]*s[v1]*s[v2]
                # Wait, let me think more carefully.
                # g1 = sum_u A[v1,u] * s[v1]*s[u]
                # g2 = sum_u A[v2,u] * s[v2]*s[u]
                # When we flip both, the actual gain is:
                # For edges from v1 to u (u != v2): gain contribution = s[v1]*s[u]*A[v1,u]
                # For edges from v2 to u (u != v1): gain contribution = s[v2]*s[u]*A[v2,u]
                # For edge (v1,v2): both endpoints flip, so s[v1]->-s[v1], s[v2]->-s[v2]
                #   The product s[v1]*s[v2] stays the same, so no change.
                # So gain_2flip = (g1 - A[v1,v2]*s[v1]*s[v2]) + (g2 - A[v1,v2]*s[v2]*s[v1])
                # = g1 + g2 - 2*A[v1,v2]*s[v1]*s[v2]
                
                g1 = s[v1] * neighbor_sum[v1]
                g2 = s[v2] * neighbor_sum[v2]
                cross = 2.0 * adj[v1, v2] * s[v1] * s[v2]
                gain = g1 + g2 - cross
                
                if gain >= 0:
                    current_labels[v1] = 1 - current_labels[v1]
                    current_labels[v2] = 1 - current_labels[v2]
                    current_cut += gain
                else:
                    if np.random.random() < np.exp(gain / T):
                        current_labels[v1] = 1 - current_labels[v1]
                        current_labels[v2] = 1 - current_labels[v2]
                        current_cut += gain
            else:
                # 1-flip move
                v = np.random.randint(n)
                gain = s[v] * neighbor_sum[v]
                
                if gain >= 0:
                    current_labels[v] = 1 - current_labels[v]
                    current_cut += gain
                else:
                    if np.random.random() < np.exp(gain / T):
                        current_labels[v] = 1 - current_labels[v]
                        current_cut += gain
            
            T *= cooling_rate
            
            if time.time() > time_limit:
                break
        
        return current_labels, current_cut

    def polish(labels, time_limit):
        """Exhaustive 1-flip and 2-flip local search until no improvement."""
        current_labels = labels.copy()
        current_cut = compute_cut(current_labels)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            s = 2.0 * current_labels - 1.0
            neighbor_sum = adj @ s
            
            # Try all 1-flip moves
            for v in range(n):
                gain = s[v] * neighbor_sum[v]
                if gain > 0:
                    current_labels[v] = 1 - current_labels[v]
                    current_cut += gain
                    improved = True
                    # Recompute s and neighbor_sum since labels changed
                    s = 2.0 * current_labels - 1.0
                    neighbor_sum = adj @ s
            
            if not improved and time.time() < time_limit:
                # Try 2-flip moves
                s = 2.0 * current_labels - 1.0
                neighbor_sum = adj @ s
                for v1 in range(n):
                    for v2 in range(v1 + 1, n):
                        g1 = s[v1] * neighbor_sum[v1]
                        g2 = s[v2] * neighbor_sum[v2]
                        cross = 2.0 * adj[v1, v2] * s[v1] * s[v2]
                        gain = g1 + g2 - cross
                        if gain > 0:
                            current_labels[v1] = 1 - current_labels[v1]
                            current_labels[v2] = 1 - current_labels[v2]
                            current_cut += gain
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    # Recompute
                    s = 2.0 * current_labels - 1.0
                    neighbor_sum = adj @ s
        
        return current_labels, current_cut

    rng = np.random.default_rng()
    
    # Allocate time: 70% for SA restarts, 30% for final polish
    sa_time = 80.0
    polish_time = 30.0
    
    n_restarts = 60
    time_per_restart = sa_time / n_restarts
    
    for i in range(n_restarts):
        if time.time() > deadline - polish_time:
            break
        
        # Spectral initialization with random perturbation for diversity
        labels = spectral_init(rng)
        # Add some randomness to break degeneracy
        flip_mask = rng.random(n) < 0.1
        labels[flip_mask] = 1 - labels[flip_mask]
        
        # Run SA with 2-flip moves
        labels, cut = local_search_sa(labels, time.time() + time_per_restart, use_2flip=True)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
        best_cut = compute_cut(best_labels)
    
    # Final high-precision polish pass
    remaining = deadline - time.time()
    if remaining > 1.0:
        polished, polished_cut = polish(best_labels, time.time() + min(remaining - 0.5, polish_time))
        if polished_cut > best_cut:
            best_cut = polished_cut
            best_labels = polished
    
    return best_labels
# EVOLVE-BLOCK-END
