"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def partition(adjacency):
    """
    Solves Maximum Cut using Simulated Annealing with multiple random restarts.
    
    Strategy:
    1. Convert adjacency matrix to numpy array for speed.
    2. Define a fast function to calculate the change in cut size for a vertex flip.
    3. Run multiple restarts of Simulated Annealing:
       - Initialize labels randomly.
       - Iteratively flip vertices.
       - Accept improving moves always.
       - Accept worsening moves with probability exp(-delta / T), where T is temperature.
       - Cool down temperature over time.
    4. Track the best solution found across all restarts and iterations.
    5. Enforce a strict time deadline (115s) to avoid timeout.
    """
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin for 120s limit
    
    n = len(adjacency)
    # Ensure adjacency is a numpy array of floats/ints for fast operations
    # The input is a bool matrix, but we might need integer arithmetic for sums
    adj = np.array(adjacency, dtype=np.int8)
    
    # Pre-compute degrees? Not strictly necessary if we calculate delta dynamically,
    # but calculating delta for a flip v:
    # Let S be the set of neighbors of v with same label as v.
    # Let T be the set of neighbors of v with different label as v.
    # Current contribution of v to cut: |T|
    # New contribution of v to cut: |S|
    # Delta = |S| - |T|
    # Note: This only accounts for edges incident to v.
    # When we flip v, the cut status of edges (u, v) changes.
    # Edges (u, v) where u in S were NOT in cut, now ARE in cut. (+|S|)
    # Edges (u, v) where u in T WERE in cut, now ARE NOT in cut. (-|T|)
    # So Delta = |S| - |T|.
    
    def calc_cut(labels):
        """Calculate total cut size."""
        # Cut size = sum over edges (i,j) where labels[i] != labels[j]
        # Efficiently: 0.5 * sum_{i} sum_{j} (1 - 2*labels[i]*labels[j] + 1) * A[i][j] ? 
        # Simpler: 
        # Cut = sum_{i<j, A[i][j]=1, labels[i]!=labels[j]} 1
        # Using vectorization:
        # L = labels (0/1)
        # Diff = 1 - 2*L (values -1, 1)
        # This is getting complex to vectorize simply without memory blowup for 100x100.
        # For n=100, a simple loop or np.sum is fast enough.
        # Let's use: 
        # cut = 0
        # for i in range(n):
        #    cut += np.sum(adj[i] * (labels != labels[i]))
        # This is O(n^2). 100*100 = 10,000 ops. Very fast.
        
        # Vectorized approach:
        # labels is shape (n,)
        # We want sum of A[i,j] where L[i] != L[j]
        # Let's construct a matrix of matches: M = (L[:, None] == L[None, :])
        # Cut = total_edges - sum(A * M)
        # total_edges is constant.
        # This creates a 100x100 bool matrix. 10,000 elements. Negligible.
        matches = (labels[:, None] == labels[None, :])
        # Sum of edges where labels are same
        same_count = np.sum(adj * matches)
        # Total edges (undirected, so divide by 2 if adj is symmetric, but let's just count pairs)
        # The task says "number of edges crossing". 
        # If adjacency is symmetric, total edges = sum(adj)//2.
        # Cut = TotalEdges - SameCount//2 ? 
        # Wait, matches[i][j] is true if L[i]==L[j].
        # sum(adj * matches) counts directed edges (i,j) where L[i]==L[j].
        # Since adj is symmetric, this is 2 * (number of undirected edges with same label).
        # So SameUndirected = sum(adj * matches) // 2.
        # TotalUndirected = sum(adj) // 2.
        # Cut = TotalUndirected - SameUndirected.
        
        total_edges = np.sum(adj) // 2
        same_undirected = np.sum(adj * matches) // 2
        return total_edges - same_undirected

    def delta_flip(labels, v):
        """Calculate change in cut size if we flip vertex v."""
        # Current label of v
        lv = labels[v]
        # Neighbors with same label as v
        same = np.sum(adj[v] * (labels == lv))
        # Neighbors with different label as v
        diff = np.sum(adj[v] * (labels != lv))
        # If we flip v:
        # Edges to 'same' become crossing (+same)
        # Edges to 'diff' become non-crossing (-diff)
        return same - diff

    best_labels = np.zeros(n, dtype=np.int8)
    best_cut = -1
    
    # Multi-start Simulated Annealing
    # We will run restarts until time is up.
    # Each restart:
    # 1. Random init
    # 2. Annealing loop
    
    num_restarts = 0
    max_restarts = 1000 # Safety cap, though time will likely stop it first
    
    while time.time() - start_time < DEADLINE and num_restarts < max_restarts:
        # Initialize randomly
        labels = np.random.randint(0, 2, size=n, dtype=np.int8)
        current_cut = calc_cut(labels)
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
            
        # Simulated Annealing parameters
        # T0: Initial temperature. Should be high enough to accept bad moves.
        # Max delta is roughly max degree. For n=100, max degree <= 99.
        # Let's set T0 = 10.0
        T = 10.0
        cooling_rate = 0.99
        min_T = 0.01
        max_iter = 5000 # Per restart
        
        for i in range(max_iter):
            if time.time() - start_time > DEADLINE:
                break
                
            # Pick a random vertex to flip
            v = np.random.randint(0, n)
            
            d = delta_flip(labels, v)
            
            if d > 0:
                # Always accept improvement
                labels[v] = 1 - labels[v]
                current_cut += d
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
            else:
                # Accept worsening move with probability exp(d/T)
                if T > min_T:
                    prob = np.exp(d / T)
                    if np.random.random() < prob:
                        labels[v] = 1 - labels[v]
                        current_cut += d
                        if current_cut > best_cut:
                            best_cut = current_cut
                            best_labels = labels.copy()
            
            T *= cooling_rate
            
        num_restarts += 1
        
    return best_labels
# EVOLVE-BLOCK-END
