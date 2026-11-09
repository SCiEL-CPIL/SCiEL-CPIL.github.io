# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    """
    Solve Maximum Cut using Simulated Annealing with Restarts and Optimized Delta Updates.
    
    Args:
        adjacency: bool matrix (n x n) representing the graph adjacency
        
    Returns:
        0/1 label per vertex (numpy array of length n)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int8)
    
    # Convert to efficient representation
    adj = np.asarray(adjacency, dtype=np.int8)
    degrees = adj.sum(axis=1)
    
    # Get neighbors for fast iteration
    neighbors = [np.where(adj[i] == 1)[0] for i in range(n)]
    
    def compute_initial_labels():
        """Generate initial labels using Fiedler vector or random."""
        # Try Fiedler vector first for a good start
        try:
            from scipy.linalg import eigh
            L = np.diag(degrees.astype(float)) - adj.astype(float)
            eigenvalues, eigenvectors = eigh(L)
            # Handle potential degeneracy or disconnected components
            # Fiedler vector is usually index 1 if connected
            # If disconnected, there are multiple zero eigenvalues.
            # We just pick the second smallest.
            fiedler = eigenvectors[:, 1]
            return (fiedler > 0).astype(np.int8)
        except Exception:
            rng = np.random.default_rng(42)
            return rng.integers(0, 2, size=n)
    
    def initialize_state(labels):
        """Initialize same_count array and current cut."""
        same_count = np.zeros(n, dtype=np.int32)
        for i in range(n):
            same_count[i] = np.sum(adj[i] * (labels == labels[i]))
        
        # Cut size = 0.5 * sum(degrees) - 0.5 * sum(same_count)
        # Or just sum(degrees - same_count) / 2
        current_cut = int(np.sum(degrees - same_count) // 2)
        return same_count, current_cut
    
    def run_sa(labels, time_limit, rng):
        """Run Simulated Annealing for a specific time limit."""
        same_count, current_cut = initialize_state(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        # Cooling schedule
        # We want T to drop from T_start to T_end during the time_limit
        # Let's estimate iterations. 
        # Each iteration is O(degree) ~ O(n) for dense, but n=100 is small.
        # Let's assume we can do ~100,000 iterations per second? Maybe less due to Python overhead.
        # Let's target ~10,000 iterations per restart to be safe and allow many restarts.
        # Or just run until time runs out.
        
        T_start = 10.0
        T_end = 0.1
        # Geometric cooling: T_k = T_start * alpha^k
        # We want T_K = T_end where K is total iterations.
        # But we don't know K in advance if we loop by time.
        # Better: Fix number of iterations per restart, or estimate.
        # Let's try a fixed number of iterations per restart, say 2000.
        # With n=100, 2000 iterations is very fast.
        # We can do many restarts.
        
        num_iterations = 5000
        alpha = (T_end / T_start) ** (1.0 / num_iterations)
        T = T_start
        
        for _ in range(num_iterations):
            if time.time() > deadline:
                break
            
            v = rng.integers(0, n)
            
            # Delta calculation
            # delta = same_count[v] - (degrees[v] - same_count[v])
            same_v = same_count[v]
            opp_v = degrees[v] - same_v
            delta = same_v - opp_v
            
            # Acceptance
            if delta >= 0 or rng.random() < np.exp(delta / T):
                # Flip v
                old_label = labels[v]
                labels[v] = 1 - old_label
                
                # Update same_count for v and its neighbors
                # v's new same_count is opp_v
                same_count[v] = opp_v
                
                for u in neighbors[v]:
                    if labels[u] == old_label:
                        # u was same as v, now u is opposite to v
                        # u's same_count decreases by 1
                        same_count[u] -= 1
                    else:
                        # u was opposite to v, now u is same as v
                        # u's same_count increases by 1
                        same_count[u] += 1
                
                current_cut += delta
                
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
            
            T *= alpha
        
        return best_labels, best_cut

    def polish(labels):
        """Greedy hill climbing to ensure local optimality."""
        same_count, current_cut = initialize_state(labels)
        improved = True
        while improved:
            improved = False
            for v in range(n):
                same_v = same_count[v]
                opp_v = degrees[v] - same_v
                delta = same_v - opp_v
                if delta > 0:
                    # Flip v
                    old_label = labels[v]
                    labels[v] = 1 - old_label
                    same_count[v] = opp_v
                    for u in neighbors[v]:
                        if labels[u] == old_label:
                            same_count[u] -= 1
                        else:
                            same_count[u] += 1
                    current_cut += delta
                    improved = True
        return labels, current_cut

    best_global_labels = None
    best_global_cut = 0
    
    rng_master = np.random.default_rng(12345)
    
    # Run multiple restarts
    # Estimate time per restart.
    # Let's allow up to 20 restarts or until deadline.
    max_restarts = 20
    
    for i in range(max_restarts):
        if time.time() > deadline - 5.0: # Leave 5s for final polish
            break
        
        # Initialize
        if i == 0:
            labels = compute_initial_labels()
        else:
            # Random initialization for diversity
            labels = rng_master.integers(0, 2, size=n)
        
        rng_run = np.random.default_rng(12345 + i)
        
        # Run SA
        sa_labels, sa_cut = run_sa(labels, 0.5, rng_run)
        
        # Polish
        final_labels, final_cut = polish(sa_labels)
        
        if final_cut > best_global_cut:
            best_global_cut = final_cut
            best_global_labels = final_labels
        
        # Check time
        if time.time() > deadline - 2.0:
            break

    if best_global_labels is None:
        # Fallback if something went wrong
        best_global_labels = compute_initial_labels()
        best_global_labels, _ = polish(best_global_labels)
        
    return best_global_labels
# EVOLVE-BLOCK-END
