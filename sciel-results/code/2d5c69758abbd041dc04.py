"""
Optimized Greedy MIS via Weighted Heuristic Search and Local Polish.

This program improves upon the seed by:
1. Using a lower-dimensional search space (heuristic weights) to find a robust scoring function.
2. Applying a final local search on vertex scores to fine-tune the greedy order.
3. Incorporating stronger structural features (Core-Number) into the heuristic.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def priority(adjacency):
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin below 150s limit

    A = np.asarray(adjacency, dtype=float)
    N = A.shape[0]

    if N == 0:
        return np.array([])

    # Ensure symmetric and zero diagonal
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    A_bool = A > 0.5
    
    # Precompute neighbor lists
    neighbors = [np.where(A_bool[i])[0] for i in range(N)]
    degrees = np.array([len(neighbors[i]) for i in range(N)], dtype=float)

    # --- Feature Extraction ---
    
    # 1. Degree
    D = degrees
    
    # 2. Clustering Coefficient
    A_sq = A @ A
    Tri = np.diag(A_sq) / 2.0
    D_safe = np.where(D > 0, D, 1.0)
    Clus = np.zeros(N)
    mask = D >= 2
    Clus[mask] = 2.0 * Tri[mask] / (D[mask] * (D[mask] - 1.0))

    # 3. Core Number (k-core decomposition)
    # Vertices in low k-cores are peripheral and often in MIS
    core_numbers = np.zeros(N, dtype=int)
    remaining = np.ones(N, dtype=bool)
    deg_rem = D.copy()
    # Simple O(N^2) k-core decomposition for N=200 is fast enough
    for k in range(1, int(D.max()) + 1):
        changed = True
        while changed:
            changed = False
            for i in range(N):
                if remaining[i] and deg_rem[i] < k:
                    remaining[i] = False
                    changed = True
                    for nb in neighbors[i]:
                        if remaining[nb]:
                            deg_rem[nb] -= 1
        # Assign core number k to vertices removed in this step? 
        # Actually, standard definition: core number is max k such that vertex is in k-core.
        # Let's just use the final remaining set size or a simpler proxy: 
        # Number of neighbors with high degree? 
        # Let's stick to a simpler feature: "Periphery" = 1 - (Degree / MaxDegree)
        # Or use the actual k-core number calculation properly.
        
    # Proper K-Core Number Calculation
    core = np.zeros(N, dtype=int)
    deg = D.copy().astype(int)
    active = np.ones(N, dtype=bool)
    for k in range(1, int(D.max()) + 1):
        changed = True
        while changed:
            changed = False
            for i in range(N):
                if active[i] and deg[i] < k:
                    active[i] = False
                    core[i] = k - 1 # Removed at step k, so max core was k-1
                    changed = True
                    for nb in neighbors[i]:
                        if active[nb]:
                            deg[nb] -= 1
    # Vertices still active have core >= last k
    last_k = int(D.max())
    for i in range(N):
        if active[i]:
            core[i] = last_k
            
    # Normalize features to [0, 1]
    def norm(x):
        x = np.asarray(x, dtype=float)
        min_val = np.min(x)
        max_val = np.max(x)
        if max_val == min_val:
            return np.zeros_like(x)
        return (x - min_val) / (max_val - min_val)

    D_n = norm(D)
    Clus_n = norm(Clus)
    Core_n = norm(core)
    
    # 4. Spectral (Fiedler)
    # Use power iteration on Laplacian for smallest non-zero eigenvalue?
    # Or just use the principal eigenvector of A (centrality) and invert it.
    # Low centrality is good for MIS.
    x = np.ones(N) / N
    for _ in range(20):
        x_new = A @ x
        norm_x = np.linalg.norm(x_new)
        if norm_x < 1e-12: break
        x_new /= norm_x
        if np.linalg.norm(x_new - x) < 1e-9: break
        x = x_new
    Spec_n = norm(x)

    # Features: [D_n, Clus_n, Core_n, Spec_n]
    # We want to minimize D, Clus, Spec and maximize Core? 
    # Actually, high Core means central. Low Core means peripheral.
    # So we want to MINIMIZE Core_n as well? 
    # Let's check: In a star graph, center has high core, leaves have low core. 
    # MIS is leaves. So we want LOW core.
    # So all features should be minimized? 
    # Let's define features such that LOWER is BETTER.
    F_D = D_n
    F_Clus = Clus_n
    F_Core = Core_n
    F_Spec = Spec_n

    # --- Fast Greedy Evaluator ---
    def evaluate(scores):
        order = np.argsort(-scores, kind='stable')
        blocked = np.zeros(N, dtype=np.uint8)
        count = 0
        for v in order:
            if blocked[v] == 0:
                count += 1
                blocked[v] = 1
                if neighbors[v].size > 0:
                    blocked[neighbors[v]] = 1
        return count

    # --- Phase 1: Optimize Heuristic Weights ---
    # Score = -w1*D - w2*Clus - w3*Core - w4*Spec
    # We search for w in R^4
    
    rng = np.random.default_rng(42)
    best_size = -1
    best_weights = np.array([1.0, 0.5, 0.5, 0.5])
    
    # Initial random restarts
    for _ in range(50):
        w = rng.uniform(0.5, 2.0, 4)
        scores = -w[0]*F_D - w[1]*F_Clus - w[2]*F_Core - w[3]*F_Spec
        size = evaluate(scores)
        if size > best_size:
            best_size = size
            best_weights = w.copy()
            
    # Local search on weights
    for _ in range(2000):
        if time.time() > DEADLINE - 20:
            break
        w_new = best_weights + rng.normal(0, 0.1, 4)
        w_new = np.clip(w_new, 0.1, 5.0)
        scores = -w_new[0]*F_D - w_new[1]*F_Clus - w_new[2]*F_Core - w_new[3]*F_Spec
        size = evaluate(scores)
        if size > best_size:
            best_size = size
            best_weights = w_new
        elif rng.random() < 0.05: # Occasional restart
             w_new = rng.uniform(0.5, 2.0, 4)
             scores = -w_new[0]*F_D - w_new[1]*F_Clus - w_new[2]*F_Core - w_new[3]*F_Spec
             size = evaluate(scores)
             if size > best_size:
                 best_size = size
                 best_weights = w_new

    # --- Phase 2: Local Search on Vertex Scores ---
    # Start with the best weighted scores
    best_scores = -best_weights[0]*F_D - best_weights[1]*F_Clus - best_weights[2]*F_Core - best_weights[3]*F_Spec
    current_size = best_size
    
    # Polish: Swap vertices to improve greedy outcome
    # Strategy: 
    # 1. Identify "bad" picks: vertices that block many high-potential neighbors.
    # 2. Swap them with their blocked neighbors or random low-ranked vertices.
    
    for _ in range(20000):
        if time.time() > DEADLINE - 2:
            break
            
        # Random swap of two vertices
        i = rng.integers(0, N)
        j = rng.integers(0, N)
        if i == j:
            continue
            
        # Try swapping
        old_si = best_scores[i]
        old_sj = best_scores[j]
        best_scores[i] = old_sj
        best_scores[j] = old_si
        
        new_size = evaluate(best_scores)
        
        if new_size > current_size:
            current_size = new_size
            # Keep swap
            # Reset patience? 
        else:
            # Revert
            best_scores[i] = old_si
            best_scores[j] = old_sj
            
        # Occasionally try a targeted swap: 
        # Swap a vertex with one of its neighbors if it's currently blocking it
        if rng.random() < 0.2:
             # Get current greedy order and blocked status
             order = np.argsort(-best_scores, kind='stable')
             blocked = np.zeros(N, dtype=np.uint8)
             picked = []
             for v in order:
                 if blocked[v] == 0:
                     picked.append(v)
                     blocked[v] = 1
                     if neighbors[v].size > 0:
                         blocked[neighbors[v]] = 1
             
             # Pick a random picked vertex
             if picked:
                 v = picked[rng.integers(0, len(picked))]
                 # Pick a random neighbor of v
                 if neighbors[v].size > 0:
                     u = neighbors[v][rng.integers(0, neighbors[v].size)]
                     # Swap v and u
                     old_sv = best_scores[v]
                     old_su = best_scores[u]
                     best_scores[v] = old_su
                     best_scores[u] = old_sv
                     
                     new_size = evaluate(best_scores)
                     if new_size > current_size:
                         current_size = new_size
                     else:
                         best_scores[v] = old_sv
                         best_scores[u] = old_su

    return best_scores
# EVOLVE-BLOCK-END
