"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = labels[None, :] != labels[:, None]
    return int((adjacency & diff).sum() // 2)

def _spectral_init(adjacency):
    """Initialize using the sign of the Fiedler vector (2nd smallest eigenvector of Laplacian)."""
    n = adjacency.shape[0]
    degrees = adjacency.sum(axis=1)
    L = np.diag(degrees) - adjacency.astype(float)
    
    try:
        vals, vecs = np.linalg.eigh(L)
        # The Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        fiedler = vecs[:, 1]
        labels = (fiedler >= 0).astype(np.int8)
        return labels
    except Exception:
        return np.random.randint(0, 2, size=n, dtype=np.int8)

def _batch_simulated_annealing(adjacency, labels, start_temp, end_temp, max_iters, deadline):
    """
    Vectorized Simulated Annealing for Max Cut.
    Instead of flipping one node at a time, we flip a random subset of nodes in each iteration.
    This allows us to leverage NumPy's vectorization for the delta calculation.
    """
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    best_cut = current_cut
    best_labels = labels.copy()
    
    # Precompute adjacency as float for matrix ops
    adj_float = adjacency.astype(float)
    
    start_time = time.time()
    
    if max_iters <= 0:
        return best_labels, best_cut
        
    alpha = (end_temp / start_temp) ** (1.0 / max_iters)
    
    # We will flip a random fraction of nodes in each step.
    # Let's try flipping ~5-10% of nodes per step to balance exploration and exploitation.
    # Actually, for Max Cut, flipping a random set is a valid move.
    # Delta for flipping a set S:
    # New cut = Old cut + sum_{v in S} (gain_v) - 2 * sum_{(u,v) in E, u,v in S} 1
    # Wait, let's derive properly.
    # Let x be the label vector.
    # Cut = 0.5 * sum_{i,j} A_{ij} (1 - x_i x_j)  (if x in {-1, 1})
    # Or using 0/1: Cut = sum_{i<j} A_{ij} (x_i XOR x_j)
    
    # Let's stick to the 0/1 formulation for simplicity in code, but use vectorized ops.
    # If we flip a set S of vertices:
    # 1. Edges within S: If (u,v) in S, their relative label doesn't change. So internal edges don't affect cut change.
    # 2. Edges between S and V\S: Their relative label flips. So if they were cut, they become non-cut, and vice versa.
    # 3. Edges within V\S: No change.
    
    # So Delta = (# edges between S and V\S that are currently non-cut) - (# edges between S and V\S that are currently cut)
    # Let M be the adjacency matrix.
    # Let x be the label vector (0/1).
    # Edge (u,v) is cut if x_u != x_v.
    # Let S be the set of flipped nodes.
    # For u in S, v not in S:
    #   Before: cut if x_u != x_v
    #   After: cut if (1-x_u) != x_v  => x_u == x_v
    # So the cut status flips.
    # Delta = sum_{u in S, v not in S} A_{uv} * (1 if x_u == x_v else -1)
    
    # This can be computed as:
    # Let s be a binary vector where s_i = 1 if i in S, else 0.
    # Let diff_matrix = (x[None, :] != x[:, None])  # True if cut
    # We want sum over u in S, v not in S of A_{uv} * (1 if NOT diff else -1)
    # Note: sum over u in S, v not in S is not symmetric. The total change is symmetric?
    # Actually, the total change in cut size is:
    # New_Cut - Old_Cut = sum_{u in S, v not in S} A_{uv} * ( (1-x_u) != x_v ? 1 : 0 ) - ( x_u != x_v ? 1 : 0 )
    # = sum_{u in S, v not in S} A_{uv} * ( (x_u == x_v) - (x_u != x_v) )
    # = sum_{u in S, v not in S} A_{uv} * ( 1 - 2*(x_u != x_v) )
    
    # Let's compute this vectorized.
    # mask_S = s
    # mask_not_S = 1 - s
    # term = (1 - 2 * diff_matrix)  # 1 if same, -1 if diff
    # delta = sum(A * term * mask_S[None, :] * mask_not_S[:, None])
    # Note: This counts each edge (u,v) with u in S, v not in S once.
    # Since the graph is undirected, the edge (v,u) is the same.
    # Does the formula account for both directions?
    # The sum is over u in S, v not in S.
    # If we swap S and not S, we get the same sum?
    # sum_{v not in S, u in S} A_{vu} (1 - 2 diff_{vu}) = same sum.
    # So the total change is exactly this sum.
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
            
        T = start_temp * (alpha ** i)
        
        # Pick a random subset of nodes to flip.
        # Size of subset: maybe 10% of nodes?
        # Let's vary the size slightly or keep it fixed. Fixed is simpler.
        # For N=100, flipping 5-10 nodes is reasonable.
        # Let's use a fixed fraction, e.g., 0.1
        k = max(1, int(n * 0.1))
        indices = np.random.choice(n, k, replace=False)
        s = np.zeros(n, dtype=np.int8)
        s[indices] = 1
        
        # Calculate delta
        # diff_matrix: True if labels differ
        diff_matrix = (labels[None, :] != labels[:, None])
        # term: 1 if same, -1 if diff
        term = 1 - 2 * diff_matrix.astype(np.int8)
        
        # Mask: u in S, v not in S
        # We need A_{uv} where u in S, v not in S
        # mask = s[None, :] * (1 - s)[:, None]  # u in S, v not in S
        # delta = sum(A * term * mask)
        
        # Optimization: We can precompute A * term? No, term changes with labels.
        # But A is constant.
        # Let's just compute the sum.
        # mask_u_in_S = s[None, :]  # broadcast over rows? No, s is (n,). s[None, :] is (1, n).
        # We want rows u in S, cols v not in S.
        # mask = s[None, :] * (1 - s)[:, None]
        # This creates an (n, n) matrix.
        
        mask = s[None, :] * (1 - s)[:, None]
        delta = np.sum(adj_float * term * mask)
        
        # Acceptance probability
        if delta > 0:
            accept = True
        else:
            # P(accept) = exp(-delta / T)
            # delta is negative here
            if T > 0:
                prob = np.exp(-delta / T)
                accept = np.random.rand() < prob
            else:
                accept = False
                
        if accept:
            # Flip labels in S
            labels[indices] = 1 - labels[indices]
            current_cut += delta
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
                
    return best_labels, best_cut

def partition(adjacency):
    """
    Main partition function.
    1. Spectral Init + Batch SA
    2. Weighted Random Restarts + Batch SA
    """
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_labels = None
    best_cut = -1
    
    # Phase 1: Spectral Initialization + Batch SA
    labels = _spectral_init(adjacency)
    
    # Estimate max possible delta for temp scaling
    # Max delta for flipping 10% of nodes (10 nodes)
    # Each node has max degree ~ n.
    # Max change per edge is 1.
    # Max edges involved ~ 10 * n = 1000.
    # So max delta ~ 1000.
    start_temp = 10.0 * n  # Heuristic
    end_temp = 0.1
    
    # How many iterations?
    # Batch SA is much faster per iteration than single-flip SA.
    # One iteration does O(N^2) work in C (NumPy).
    # N=100 -> 10,000 ops.
    # Should be able to do 100k-500k iterations in 10s?
    # Let's measure or estimate.
    # Let's just run until deadline - 10s.
    
    labels, cut = _batch_simulated_annealing(adjacency, labels, start_temp, end_temp, 100000, deadline - 10.0)
    if cut > best_cut:
        best_cut = cut
        best_labels = labels.copy()
    
    # Phase 2: Weighted Random Restarts
    # Bias initialization towards high-degree nodes being in one group?
    # Or just random restarts with shorter SA.
    remaining = deadline - time.time() - 5.0
    if remaining > 2.0:
        # Do several random restarts
        # We can't do too many if each takes time.
        # Let's estimate time per restart.
        # If we have 5s left, and we want 5 restarts, each gets 1s.
        # In 1s, how many batch SA iterations?
        # Let's just run a fixed number of iterations or time-based.
        
        num_restarts = 10
        time_per_restart = remaining / num_restarts
        
        # Precompute degrees for weighted random init
        degrees = adjacency.sum(axis=1)
        
        for r in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
            
            # Weighted Random Init:
            # Assign label 0 with probability proportional to degree?
            # Or just random?
            # Let's try random first, it's robust.
            labels = np.random.randint(0, 2, size=n, dtype=np.int8)
            
            # Run Batch SA for a shorter time
            # We need to pass a deadline for this specific run
            sub_deadline = time.time() + time_per_restart - 0.1
            labels, cut = _batch_simulated_annealing(adjacency, labels, start_temp, end_temp, 50000, sub_deadline)
            
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
    
    if best_labels is None:
        best_labels = _spectral_init(adjacency)
        
    return best_labels
# EVOLVE-BLOCK-END
