"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def partition(adjacency):
    """
    Solves Max-Cut using Spectral Initialization, Multi-Start Greedy Refinement,
    and Simulated Annealing for escape from local optima.
    """
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin for 150s limit

    # Convert to numpy array
    adj = np.array(adjacency, dtype=np.int8)
    n = len(adj)
    
    # Total number of edges (for normalization if needed, but we maximize raw cut)
    # Pre-compute degrees for potential use
    degrees = np.sum(adj, axis=1)

    # Helper to compute cut size efficiently
    def cut_size(labels):
        # labels: 0/1 array
        # s = 2*labels - 1
        s = 2.0 * labels - 1.0
        # cut = 0.5 * (sum(A) - s^T A s)
        # sum(A) is 2 * E
        # s^T A s = sum_{i,j} s_i A_ij s_j
        # Since A is symmetric and 0-diagonal, this works.
        total_sum = np.sum(adj)
        s_As = s @ (adj @ s)
        return (total_sum - s_As) / 2.0

    # Greedy Refinement using Gain Updates
    # Returns labels and cut size
    def greedy_refine(adj, init_labels, max_steps=5000):
        labels = init_labels.copy()
        n = len(labels)
        s = 2.0 * labels - 1.0
        
        # G[v] = sum_u A_uv s_v s_u = s_v * (adj @ s)[v]
        As = adj @ s
        G = s * As
        
        current_cut = cut_size(labels)
        
        for _ in range(max_steps):
            if time.time() - start_time > DEADLINE:
                break
                
            v = np.argmax(G)
            gain = G[v]
            
            if gain <= 1e-9:
                break
                
            # Flip v
            s[v] = -s[v]
            
            # Update G for neighbors of v
            neighbors = np.nonzero(adj[v])[0]
            
            # G[u] <- G[u] + 2 * A_uv * s_u * s_v_new
            # Note: s[v] is already flipped to s_v_new
            G[neighbors] += 2.0 * adj[v, neighbors] * s[neighbors] * s[v]
            
            # G[v] flips sign
            G[v] = -G[v]
            
            # Update current cut
            # The gain in cut size is gain/2? 
            # Let's check:
            # Cut = 0.5 * (Sum(A) - s^T A s)
            # Delta(s^T A s) when flipping v:
            # Old term involving v: 2 * s_v * (adj @ s)_v (since s_v^2=1 and off-diag)
            # Actually, s^T A s = sum_i s_i (As)_i.
            # When s_v flips, As doesn't change (depends on other s_u).
            # New s^T A s = Old s^T A s - s_v_old (As)_v + s_v_new (As)_v
            # = Old - s_v_old (As)_v - s_v_old (As)_v = Old - 2 s_v_old (As)_v
            # Gain in objective (Cut) = -0.5 * Delta(s^T A s) = s_v_old (As)_v
            # G[v] = s_v_old (As)_v.
            # So current_cut += gain
            current_cut += gain
            
        final_labels = ((s + 1.0) // 2).astype(np.int8)
        return final_labels, current_cut

    # Simulated Annealing
    def simulated_annealing(adj, init_labels, steps=2000, T_start=1.0, T_end=0.1):
        labels = init_labels.copy()
        n = len(labels)
        s = 2.0 * labels - 1.0
        current_cut = cut_size(labels)
        best_cut = current_cut
        best_labels = labels.copy()
        
        # Precompute neighbors for each node to speed up move generation?
        # Or just pick random node.
        
        # For N=100, picking a random node is fine.
        # We can optimize by maintaining G like in greedy, but for SA we need to accept bad moves.
        # Let's use the same G update mechanism but allow negative gains.
        
        As = adj @ s
        G = s * As
        
        for i in range(steps):
            if time.time() - start_time > DEADLINE:
                break
                
            T = T_start * (T_end / T_start) ** (i / max(1, steps - 1))
            
            v = random.randint(0, n - 1)
            gain = G[v]
            
            # Acceptance probability
            if gain > 0:
                accept = True
            else:
                # P(accept) = exp(gain / T)
                # If T is very small and gain is negative, exp is ~0
                if T < 1e-9:
                    accept = False
                else:
                    accept = random.random() < np.exp(gain / T)
            
            if accept:
                # Flip v
                s[v] = -s[v]
                
                # Update G for neighbors
                neighbors = np.nonzero(adj[v])[0]
                G[neighbors] += 2.0 * adj[v, neighbors] * s[neighbors] * s[v]
                G[v] = -G[v]
                
                current_cut += gain
                
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = ((s + 1.0) // 2).astype(np.int8)
                    
        return best_labels, best_cut

    # 1. Spectral Initialization
    try:
        # Laplacian L = D - A
        L = np.diag(degrees) - adj
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # Pick Fiedler vector (smallest non-zero eigenvalue)
        non_zero_indices = np.where(eigenvalues > 1e-10)[0]
        if len(non_zero_indices) > 0:
            idx = non_zero_indices[0]
        else:
            idx = 1 if n > 1 else 0
        fiedler_vec = eigenvectors[:, idx]
        median_val = np.median(fiedler_vec)
        spectral_labels = (fiedler_vec >= median_val).astype(np.int8)
    except Exception:
        spectral_labels = np.random.randint(0, 2, n).astype(np.int8)

    best_labels = None
    best_cut = -1

    # 2. Multi-Start Strategy
    # Start 1: Spectral
    refined, cut_val = greedy_refine(adj, spectral_labels)
    if cut_val > best_cut:
        best_cut = cut_val
        best_labels = refined

    # Start 2-20: Random
    num_restarts = 20
    
    for i in range(num_restarts):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        random_labels = np.random.randint(0, 2, n).astype(np.int8)
        refined, cut_val = greedy_refine(adj, random_labels)
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = refined

    # 3. Simulated Annealing from Best Found
    # Use remaining time for SA
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5:
        # Run SA from the best labels found so far
        # Also try SA from a few random starts if time permits
        sa_steps = 5000 # Adjust based on remaining time? 
        # Let's just run a fixed number of steps, but break if deadline hits.
        
        sa_labels, sa_cut = simulated_annealing(adj, best_labels, steps=sa_steps)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels
        
        # If time permits, run SA from another random start
        if time.time() - start_time < DEADLINE - 5:
            random_labels2 = np.random.randint(0, 2, n).astype(np.int8)
            refined2, cut_val2 = greedy_refine(adj, random_labels2)
            if cut_val2 > best_cut:
                best_cut = cut_val2
                best_labels = refined2
            
            sa_labels2, sa_cut2 = simulated_annealing(adj, best_labels, steps=sa_steps)
            if sa_cut2 > best_cut:
                best_cut = sa_cut2
                best_labels = sa_labels2

    if best_labels is None:
        best_labels = np.zeros(n, dtype=np.int8)
        
    return best_labels
# EVOLVE-BLOCK-END
