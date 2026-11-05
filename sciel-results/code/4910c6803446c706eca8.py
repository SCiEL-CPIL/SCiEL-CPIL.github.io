"""Rewritten program for Maximum Cut task."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    """
    Solve Maximum Cut using:
    1. Multiple random restarts with fast greedy local search (1-pass improvement).
    2. A final simulated-annealing polish on the best solution found.
    
    The key insight is that for 100-node graphs, we can do many fast 1-pass
    improvements per second. Each pass is O(n^2) via vectorized ops.
    We use a time-bounded approach: run as many 1-pass improvements as possible,
    then polish the best result with simulated annealing.
    """
    start_time = time.time()
    DEADLINE = 115.0  # seconds, leave 5s safety margin
    
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert adjacency to numpy array of floats for faster computation
    adj = np.array(adjacency, dtype=np.float32)
    
    # Precompute degree for each vertex
    degrees = adj.sum(axis=1)
    
    def one_pass_improve(labels):
        """
        Do one full pass over vertices, flipping any vertex that improves cut size.
        Returns improved labels and whether any flip was made.
        This is O(n^2) per pass due to the matrix-vector products.
        """
        labels = labels.copy()
        any_flip = False
        
        # We process vertices in a fixed order for determinism within a pass
        # For each vertex, compute gain from flipping
        # gain[v] = same_count[v] - diff_count[v]
        # where same_count[v] = number of neighbors with same label as v
        # and diff_count[v] = number of neighbors with different label as v
        
        for v in range(n):
            # Compute gain for flipping vertex v
            # same_count[v] = adj[v] @ (labels == labels[v])
            if labels[v] == 0:
                same_count = adj[v] @ (labels == 0)
            else:
                same_count = adj[v] @ (labels == 1)
            
            diff_count = degrees[v] - same_count
            gain = same_count - diff_count
            
            if gain > 1e-6:
                labels[v] = 1 - labels[v]
                any_flip = True
        
        return labels, any_flip
    
    def fast_improve(labels, max_passes=10):
        """
        Do multiple passes of 1-pass improvement until no more improvements or max_passes.
        """
        for _ in range(max_passes):
            labels, any_flip = one_pass_improve(labels)
            if not any_flip:
                break
        return labels
    
    def cut_size(labels):
        """Compute the cut size."""
        l = labels.astype(np.float32)
        return float(l @ adj - l @ (adj @ l))
    
    def simulated_annealing(labels, start_temp=10.0, end_temp=0.1, max_iter=5000):
        """
        Simulated annealing to escape local optima.
        """
        labels = labels.copy()
        current_cut = cut_size(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        # Precompute for faster gain calculation
        labels_float = labels.astype(np.float32)
        
        for it in range(max_iter):
            if time.time() > DEADLINE - 3:
                break
            
            # Temperature schedule: exponential decay
            t = start_temp * (end_temp / start_temp) ** (it / max_iter)
            
            # Pick a random vertex to flip
            v = np.random.randint(n)
            
            # Compute gain for flipping v
            if labels[v] == 0:
                same_count = adj[v] @ (labels == 0)
            else:
                same_count = adj[v] @ (labels == 1)
            
            diff_count = degrees[v] - same_count
            gain = same_count - diff_count
            
            # Accept if improvement, or with probability exp(gain/t) if worsening
            if gain >= 0 or np.random.rand() < np.exp(gain / max(t, 1e-10)):
                labels[v] = 1 - labels[v]
                current_cut += gain
                
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
        
        return best_labels, best_cut
    
    best_labels = None
    best_cut = -1
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Many fast 1-pass improvements from diverse starts
    # For 100 nodes, each pass is ~100 * (100 ops) = 10k ops, very fast
    # We can do thousands of passes per second
    
    # Initializations to try first
    initializations = []
    initializations.append(np.arange(n) % 2)  # Alternating
    initializations.append(1 - (np.arange(n) % 2))  # Reverse alternating
    initializations.append(np.zeros(n, dtype=int))  # All zeros
    initializations.append(np.ones(n, dtype=int))  # All ones
    # Add some random starts
    for _ in range(20):
        initializations.append(rng.integers(0, 2, size=n))
    
    for init_labels in initializations:
        if time.time() > DEADLINE - 20:
            break
        
        # Do a few passes of improvement
        improved = fast_improve(init_labels, max_passes=5)
        cut = cut_size(improved)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = improved.copy()
    
    # Phase 2: Continue with more random restarts if time permits
    while time.time() < DEADLINE - 25:
        labels = rng.integers(0, 2, size=n)
        improved = fast_improve(labels, max_passes=3)
        cut = cut_size(improved)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = improved.copy()
    
    # Phase 3: Polish the best solution with simulated annealing
    if best_labels is not None and time.time() < DEADLINE - 5:
        polished, polished_cut = simulated_annealing(best_labels, start_temp=5.0, end_temp=0.05, max_iter=20000)
        if polished_cut > best_cut:
            best_cut = polished_cut
            best_labels = polished
    
    if best_labels is None:
        # Fallback: alternating
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
