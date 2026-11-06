"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def partition(adjacency):
    """
    Maximum Cut via Spectral Initialization + Vectorized Best-Improvement Local Search.
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array for fast operations
    adj = np.asarray(adjacency, dtype=np.float64)
    
    # Compute initial cut size
    def get_cut_size(labels):
        s = 2 * labels - 1
        E = 0.5 * np.sum(adj)
        sAs = s @ (adj @ s)
        return E - 0.5 * sAs

    # 1. Spectral Initialization
    # Construct Laplacian L = D - A
    A_sp = sp.csr_matrix(adj)
    D = np.sum(adj, axis=1)
    L = sp.diags(D) - A_sp
    
    try:
        # Get the smallest eigenvalues. The first is 0 (constant vector).
        # We want the second smallest (Fiedler vector).
        vals, vecs = spla.eigsh(L, k=2, which='SM')
        idx = np.argsort(vals)
        fiedler_vec = vecs[:, idx[1]]
        
        # Initialize labels based on sign of Fiedler vector
        labels = (fiedler_vec >= 0).astype(int)
        
        # Ensure it's not a trivial solution (all 0 or all 1)
        if np.all(labels == 0) or np.all(labels == 1):
            labels = np.arange(n) % 2
            
    except Exception:
        # Fallback to alternating if spectral fails
        labels = np.arange(n) % 2

    curr_labels = labels.copy()
    best_labels = curr_labels.copy()
    best_cut = get_cut_size(curr_labels)
    
    # 2. Vectorized Best-Improvement Local Search (BILS)
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit (actually 150s kill, but safe)
    
    # We will run the BILS loop. Since it's deterministic and fast, 
    # we can potentially afford multiple passes or restarts if time permits,
    # but BILS usually converges quickly.
    
    # To make it a "different strategy" and robust, we can do a few 
    # randomized perturbations + BILS (Iterated Local Search style) 
    # but the core is the vectorized BILS.
    
    # Let's stick to pure BILS first, as the inspiration says it's better.
    # If time remains, we can do a second phase with random restarts.
    
    # Optimization: Pre-compute degrees? No, we need dynamic s.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        s = 2 * curr_labels - 1
        As = adj @ s
        # delta[v] is the change in cut size if we flip vertex v
        # Delta = s[v] * As[v]
        deltas = s * As
        
        # Find the vertex with the maximum positive delta
        max_delta = np.max(deltas)
        
        if max_delta > 1e-9:
            # Flip the best vertex
            best_v = np.argmax(deltas)
            curr_labels[best_v] = 1 - curr_labels[best_v]
            curr_cut = best_cut + max_delta  # Wait, this accumulation is risky if we don't recompute?
            # Actually, delta is exact for a single flip.
            # But after flipping, the deltas for other vertices change.
            # So we MUST recompute s and As in the next iteration.
            # This is the cost of BILS: O(n^2) per step.
            # For n=100, n^2 = 10,000 ops. Very fast.
            
            if curr_cut > best_cut:
                best_cut = curr_cut
                best_labels = curr_labels.copy()
            
            improved = True
        else:
            improved = False

    # If we have time left, try a few random restarts with BILS
    # This combines the robustness of multi-start with the quality of BILS
    restarts_done = 0
    max_restarts = 20  # Cap to avoid infinite loop if BILS is slow (unlikely)
    
    while time.time() < deadline - 5.0 and restarts_done < max_restarts:
        # Random restart
        curr_labels = np.random.randint(0, 2, n)
        curr_cut = get_cut_size(curr_labels)
        
        # Run BILS
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            s = 2 * curr_labels - 1
            As = adj @ s
            deltas = s * As
            max_delta = np.max(deltas)
            
            if max_delta > 1e-9:
                best_v = np.argmax(deltas)
                curr_labels[best_v] = 1 - curr_labels[best_v]
                curr_cut += max_delta
                
                if curr_cut > best_cut:
                    best_cut = curr_cut
                    best_labels = curr_labels.copy()
                
                improved = True
            else:
                improved = False
        
        restarts_done += 1

    return best_labels
# EVOLVE-BLOCK-END
