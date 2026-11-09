"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def partition(adjacency):
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.float64)
    
    # Deadline guard: 110 seconds safety margin
    start_time = time.time()
    deadline = start_time + 110.0
    
    def cut_size(labels):
        # Vectorized cut size calculation
        # Cut = sum(adj[i,j] * (labels[i] != labels[j]))
        # Using matrix operations:
        # Let L be the label vector (0 or 1).
        # We want to count edges where endpoints differ.
        # This can be done via:
        # cut = 0.5 * (sum(degrees) - sum(edges within same set))
        # Or more directly using the adjacency matrix and label matrix.
        # Let's use a fast trick:
        # If we define x = 2*labels - 1 (values -1, 1), then
        # cut = 0.25 * ( (1-x)^T A (1-x) ... no that's not right)
        
        # Standard fast calculation:
        # cut = sum_{i<j} A[i,j] * (labels[i] != labels[j])
        # Let's use the property:
        # cut = 0.5 * ( (sum_i A[i,:])^T * 1 - sum_{i,j} A[i,j] * (labels[i] == labels[j]) )
        # Actually, simpler:
        # cut = sum_{i} sum_{j} A[i,j] * (labels[i] != labels[j]) / 2 ? No, A is symmetric usually, but let's assume general.
        # The task says "unweighted 100-node graphs", likely undirected.
        # Let's use:
        # cut = sum_{i} (sum_{j: labels[j]!=labels[i]} A[i,j]) / 2
        # Vectorized:
        # diff = np.abs(labels[:, None] - labels[None, :])
        # return np.sum(adj * diff) / 2.0
        
        # Even faster for small N (100):
        # Precompute degrees
        # Let's just do the matrix multiplication approach if possible.
        # x = 2*labels - 1
        # cut = 0.25 * (n * sum(deg) - x^T L x) ? No.
        
        # Let's stick to a robust vectorized difference:
        # This creates an NxN matrix, which is 100x100 = 10k ops. Very fast.
        diff = np.abs(labels[:, None] - labels[None, :])
        return np.sum(adj * diff) / 2.0

    def local_search(labels):
        # Greedy local improvement
        improved = True
        while improved:
            improved = False
            # Calculate gain for flipping each vertex
            # Gain = (edges to opposite) - (edges to same)
            # If Gain > 0, flip.
            
            # Vectorized gain calculation:
            # For vertex v, gain = sum_{u: label[u]!=label[v]} A[v,u] - sum_{u: label[u]==label[v]} A[v,u]
            # Note: A[v,v] is 0 usually.
            
            # Let L = labels (0/1)
            # Let A = adjacency
            # deg_same[v] = sum_{u: L[u]==L[v]} A[v,u]
            # deg_opp[v] = sum_{u: L[u]!=L[v]} A[v,u]
            # gain[v] = deg_opp[v] - deg_same[v]
            
            # We can compute this using matrix multiplication:
            # Let M be a matrix where M[v,u] = A[v,u] if L[u] != L[v] else -A[v,u]
            # This is hard to vectorize directly without N^2 memory which is fine for N=100.
            
            # Alternative:
            # sum_opp = adj @ (1 - labels)  # Sum of edges to nodes with label 0
            # sum_same = adj @ labels       # Sum of edges to nodes with label 1
            # If labels[v] == 1:
            #   deg_same = sum_same[v] - adj[v,v] (if self loop, but assume none)
            #   deg_opp = sum_opp[v]
            #   gain = deg_opp - deg_same
            # If labels[v] == 0:
            #   deg_same = sum_opp[v]
            #   deg_opp = sum_same[v]
            #   gain = deg_opp - deg_same
            
            sum_to_1 = adj @ labels
            sum_to_0 = adj @ (1 - labels)
            
            # For v with label 1:
            # deg_same = sum_to_1[v]
            # deg_opp = sum_to_0[v]
            # gain = sum_to_0[v] - sum_to_1[v]
            
            # For v with label 0:
            # deg_same = sum_to_0[v]
            # deg_opp = sum_to_1[v]
            # gain = sum_to_1[v] - sum_to_0[v]
            
            # General formula:
            # gain = 2 * (labels[v] * sum_to_0[v] + (1 - labels[v]) * sum_to_1[v]) - (sum_to_0[v] + sum_to_1[v])
            # Let's verify:
            # If labels[v]=1: 2*sum_to_0 - (sum_to_0+sum_to_1) = sum_to_0 - sum_to_1. Correct.
            # If labels[v]=0: 2*sum_to_1 - (sum_to_0+sum_to_1) = sum_to_1 - sum_to_0. Correct.
            
            gains = 2 * (labels * sum_to_0 + (1 - labels) * sum_to_1) - (sum_to_0 + sum_to_1)
            
            # Find vertices with positive gain
            candidates = np.where(gains > 0)[0]
            if len(candidates) == 0:
                break
            
            # Flip all candidates? Or one by one?
            # Flipping all at once can be unstable if they are neighbors.
            # However, for Max Cut, if gain > 0, flipping improves the cut.
            # If we flip a set of vertices, the gains for others change.
            # Standard greedy flips one at a time in order of gain.
            # For speed, we can flip all with gain > threshold, but let's do top-k or just all positive?
            # Let's flip all positive gains. This is a "parallel" greedy step.
            # It might oscillate, but usually converges.
            # To be safer, we can just pick the best one?
            # Given N=100, picking the best is fast.
            # But flipping all positive is often faster to converge.
            # Let's try flipping all positive.
            
            labels[candidates] = 1 - labels[candidates]
            improved = True
            
            # Check time
            if time.time() > deadline:
                break
                
        return labels

    def simulated_annealing(labels, start_temp, end_temp, steps):
        current_labels = labels.copy()
        current_cut = cut_size(current_labels)
        best_labels = current_labels.copy()
        best_cut = current_cut
        
        for i in range(steps):
            if time.time() > deadline:
                break
            
            # Pick random vertex
            v = np.random.randint(n)
            
            # Calculate gain of flipping v
            # Reuse logic from local search for single vertex
            sum_to_1 = adj @ current_labels
            sum_to_0 = adj @ (1 - current_labels)
            
            if current_labels[v] == 1:
                gain = sum_to_0[v] - sum_to_1[v]
            else:
                gain = sum_to_1[v] - sum_to_0[v]
                
            new_cut = current_cut + gain
            
            if new_cut > current_cut:
                current_labels[v] = 1 - current_labels[v]
                current_cut = new_cut
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = current_labels.copy()
            else:
                # Accept with probability exp(-delta / T)
                temp = start_temp * (end_temp / start_temp) ** (i / steps)
                if np.random.rand() < np.exp(-gain / temp):
                    current_labels[v] = 1 - current_labels[v]
                    current_cut = new_cut
        
        return best_labels, best_cut

    best_labels_global = None
    best_cut_global = 0
    
    # Strategy:
    # 1. Spectral Initialization (Fiedler Vector)
    # 2. Local Search
    # 3. Simulated Annealing from that result
    # 4. Random Restarts with Local Search + SA if time permits
    
    # Spectral Init
    try:
        # Laplacian L = D - A
        degrees = adj.sum(axis=1)
        L = np.diag(degrees) - adj
        
        # Find eigenvalues and eigenvectors
        # Use scipy if available, else numpy
        try:
            from scipy.sparse.linalg import eigsh
            # eigsh is for sparse, but we have dense. Use np.linalg.eigh
            evals, evecs = np.linalg.eigh(L)
            # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
            # Sort eigenvalues
            idx = np.argsort(evals)
            fiedler_vec = evecs[:, idx[1]]
        except Exception:
            evals, evecs = np.linalg.eigh(L)
            idx = np.argsort(evals)
            fiedler_vec = evecs[:, idx[1]]
            
        # Initialize labels based on sign of Fiedler vector
        init_labels = (fiedler_vec > 0).astype(int)
        
    except Exception:
        # Fallback to random
        init_labels = np.random.randint(0, 2, n)

    # Run Local Search on spectral init
    ls_labels = local_search(init_labels)
    ls_cut = cut_size(ls_labels)
    
    if ls_cut > best_cut_global:
        best_cut_global = ls_cut
        best_labels_global = ls_labels.copy()
        
    # Run Simulated Annealing from LS result
    # Tune SA params
    # N=100, let's do 5000 steps?
    # Each step is O(N) due to matrix-vector mult? No, we computed sum_to_1/0 in LS.
    # In SA loop above, I recomputed adj @ labels every step. That is O(N^2) per step.
    # 5000 steps * 100^2 = 50M ops. Should be < 1s.
    
    sa_labels, sa_cut = simulated_annealing(ls_labels, start_temp=10.0, end_temp=0.1, steps=5000)
    
    if sa_cut > best_cut_global:
        best_cut_global = sa_cut
        best_labels_global = sa_labels.copy()
        
    # Random Restarts
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline - 5: # Leave 5s buffer
            break
            
        random_labels = np.random.randint(0, 2, n)
        
        # Local Search
        ls_r_labels = local_search(random_labels)
        ls_r_cut = cut_size(ls_r_labels)
        
        if ls_r_cut > best_cut_global:
            best_cut_global = ls_r_cut
            best_labels_global = ls_r_labels.copy()
            
        # SA from this LS result
        sa_r_labels, sa_r_cut = simulated_annealing(ls_r_labels, start_temp=10.0, end_temp=0.1, steps=3000)
        
        if sa_r_cut > best_cut_global:
            best_cut_global = sa_r_cut
            best_labels_global = sa_r_labels.copy()

    # Final check
    if best_labels_global is None:
        best_labels_global = init_labels
        
    return best_labels_global
# EVOLVE-BLOCK-END
