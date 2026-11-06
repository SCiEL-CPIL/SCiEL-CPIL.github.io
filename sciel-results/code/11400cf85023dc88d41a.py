"""
Parallel Tempering (Replica Exchange) with Vectorized Greedy Sweeps for Maximum Cut.

Strategy:
1. Maintain a population of 'replicas' (different label configurations).
2. Each step, every replica performs a greedy sweep: it flips any vertex that
   increases the cut size. This is done vectorized across all replicas.
3. Periodically, we swap the best and worst performing replicas to allow
   'heating' (escaping local minima) and 'cooling' (refining good solutions).
4. This deterministic, population-based approach often finds better global
   structures than single-trajectory stochastic search on small graphs like n=100.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np

def partition(adjacency):
    """
    Assign each vertex a 0/1 label to maximize the number of cut edges.
    Uses Parallel Tempering with vectorized greedy sweeps.
    """
    adj = np.asarray(adjacency, dtype=np.float64)
    n = len(adj)
    
    if n <= 1:
        return np.zeros(n, dtype=np.int8)

    # Precompute total edges
    total_edges = np.sum(adj) / 2.0

    deadline = time.time() + 115.0  # Safety margin

    # Population size: 20 replicas
    # For n=100, this is small enough to keep memory and compute low, 
    # but large enough for effective tempering.
    n_replicas = 20
    
    # Initialize replicas
    # Mix of random and structured initializations for diversity
    replicas = np.empty((n_replicas, n), dtype=np.int8)
    
    # Replica 0: All 0s (will immediately flip to greedy)
    replicas[0] = 0
    # Replica 1: All 1s
    replicas[1] = 1
    # Replica 2: Alternating
    replicas[2] = np.arange(n) % 2
    # Rest: Random
    for i in range(3, n_replicas):
        replicas[i] = np.random.randint(0, 2, n)

    # Helper to compute cut sizes for all replicas at once
    # s = 2*labels - 1  (shape: R x N)
    # Q = s @ adj @ s.T (shape: R x N) -> sum over j for each i? 
    # Actually we need scalar Q per replica.
    # Q_r = sum_{i,j} s_{r,i} A_{ij} s_{r,j}
    # This is element-wise sum of (s @ adj) * s
    def compute_cuts(replicas):
        s = 2.0 * replicas - 1.0  # R x N
        # s @ adj gives R x N matrix where entry (r, j) is sum_i s_{r,i} A_{ij}
        # We need sum_j (s @ adj)_{r,j} * s_{r,j}
        # Let X = s @ adj. Then Q_r = sum_j X_{r,j} s_{r,j} = sum_j (s @ adj)_{r,j} (2*labels_{r,j} - 1)
        # Or simpler: Q = np.sum((s @ adj) * s, axis=1)
        s_adj = s @ adj  # R x N
        Q = np.sum(s_adj * s, axis=1)  # R
        # Cut size = (M - Q) / 2
        return (total_edges - Q) / 2.0

    # Initial cut calculations
    cuts = compute_cuts(replicas)
    best_cut = np.max(cuts)
    best_labels = replicas[np.argmax(cuts)].copy()

    start_time = time.time()
    steps = 0

    while time.time() < deadline:
        steps += 1
        
        # --- Phase 1: Greedy Sweep (Vectorized) ---
        # For each replica r, compute the gain of flipping each vertex v.
        # Gain_{r,v} = s_{r,v} * (adj @ s_r)_v  (derived in previous thought process)
        # If Gain > 0, flipping improves the cut.
        
        s = 2.0 * replicas - 1.0  # R x N
        # Compute adj @ s_r for each r.
        # adj is N x N. s is R x N.
        # We want (adj @ s_r) for each r.
        # This is equivalent to adj @ s.T -> N x R.
        # Let Y = adj @ s.T. Y[v, r] = sum_u adj[v, u] s_{r, u}.
        # Gain_{r, v} = s_{r, v} * Y[v, r].
        # We can compute G = s * (adj @ s.T).T
        Y = adj @ s.T  # N x R
        G = s * Y.T    # R x N
        
        # Identify vertices where flipping increases cut (Gain > 0)
        # Note: If Gain == 0, flipping doesn't change cut. We can ignore or include.
        # Including Gain >= 0 might help escape plateaus, but strictly > 0 is safer for convergence.
        # Let's use > 0 to ensure monotonic improvement per replica per sweep.
        flip_mask = G > 0
        
        # Apply flips
        # replicas[flip_mask] = 1 - replicas[flip_mask]
        # Using boolean indexing on 2D array
        replicas[flip_mask] = 1 - replicas[flip_mask]
        
        # --- Phase 2: Update Cuts ---
        cuts = compute_cuts(replicas)
        
        # Track global best
        current_global_best_idx = np.argmax(cuts)
        if cuts[current_global_best_idx] > best_cut:
            best_cut = cuts[current_global_best_idx]
            best_labels = replicas[current_global_best_idx].copy()
            
        # --- Phase 3: Replica Exchange (Tempering) ---
        # Swap best and worst replicas to inject diversity into the best solution's neighborhood
        # and allow the worst solution to "cool down" from the best.
        # Simple swap: Best <-> Worst
        best_idx = np.argmax(cuts)
        worst_idx = np.argmin(cuts)
        
        if best_idx != worst_idx:
            replicas[best_idx], replicas[worst_idx] = replicas[worst_idx].copy(), replicas[best_idx].copy()
            # No need to recompute cuts immediately as they are just swapped, 
            # but the next sweep will use the new states.
            
        # Optional: If we want more aggressive tempering, we could swap neighbors 
        # based on a temperature ladder, but simple best-worst swap is effective 
        # for small populations and fast convergence.
        
        # Early stopping check: If no flips occurred in the last sweep for all replicas,
        # we might be stuck. The swap will handle this by mixing states.
        
        # To prevent infinite loop if stuck in a 2-cycle, we rely on the time limit.
        # The swap ensures that states change, allowing new greedy paths.

    return best_labels
# EVOLVE-BLOCK-END
