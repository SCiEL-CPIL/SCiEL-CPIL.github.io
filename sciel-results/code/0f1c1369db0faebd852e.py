"""Rewritten program for Maximum Cut task."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    """
    Solve Maximum Cut using:
    1. Goemans-Williamson style randomized rounding from SDP-like heuristics (simplified)
       or simply multiple random restarts with local search.
    2. Hill-climbing local search (flip vertices that improve cut size).
    3. Simulated annealing or multi-start to escape local optima.
    
    We use a time-bounded approach: run multiple random restarts with hill climbing,
    keeping the best solution found within the time budget.
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
    
    def cut_size(labels):
        """Compute the cut size: number of edges crossing the partition."""
        # For each vertex, count neighbors with different label
        # cut = sum over v of (number of neighbors u where label[u] != label[v]) / 2
        # More efficiently: for each v, count neighbors with label != label[v]
        # Using broadcasting: adj @ (labels != labels[v]) per vertex is slow
        # Instead: cut = sum_v sum_u adj[v,u] * (labels[v] != labels[u]) / 2
        # = 0.5 * sum_v sum_u adj[v,u] * (labels[v] XOR labels[u])
        # For binary labels: XOR = labels[v] + labels[u] - 2*labels[v]*labels[u]
        # So cut = 0.5 * (sum_v sum_u adj[v,u]*(labels[v]+labels[u]) - 2*sum_v sum_u adj[v,u]*labels[v]*labels[u])
        # sum_v sum_u adj[v,u]*labels[v] = labels @ adj
        # sum_v sum_u adj[v,u]*labels[u] = adj @ labels (same thing since symmetric)
        # sum_v sum_u adj[v,u]*labels[v]*labels[u] = labels @ adj @ labels
        # So cut = 0.5 * (2 * labels @ adj - 2 * labels @ adj @ labels)
        # = labels @ adj - labels @ adj @ labels
        l = labels.astype(np.float32)
        return float(l @ adj - l @ (adj @ l))
    
    def hill_climb(labels, max_iter=100):
        """Improve labels by flipping vertices that increase cut size."""
        labels = labels.copy()
        current_cut = cut_size(labels)
        l_float = labels.astype(np.float32)
        
        for it in range(max_iter):
            if time.time() > DEADLINE - 2:
                break
            improved = False
            # For each vertex, compute gain from flipping
            # If we flip vertex v from label c to 1-c:
            # New cut contribution for v: neighbors with label != 1-c = neighbors with label == c
            # Old cut contribution for v: neighbors with label != c = neighbors with label == 1-c
            # Gain = (neighbors_same_as_v_after_flip - neighbors_same_as_v_before_flip)
            # Wait, let me think again.
            # Cut size = sum over edges (u,v) where label[u] != label[v]
            # When we flip v, only edges incident to v are affected.
            # For each neighbor u of v:
            #   If label[u] == label[v] before flip: edge was NOT in cut, after flip it IS in cut -> +1
            #   If label[u] != label[v] before flip: edge WAS in cut, after flip it is NOT -> -1
            # So gain = (num neighbors with same label as v) - (num neighbors with different label as v)
            # = same_count - diff_count
            # But note: this counts each edge once from v's perspective, which is correct since
            # we're only changing v's label.
            
            # Compute for all vertices at once:
            # same_count[v] = sum_u adj[v,u] * (labels[u] == labels[v])
            # diff_count[v] = sum_u adj[v,u] * (labels[u] != labels[v])
            # gain[v] = same_count[v] - diff_count[v]
            
            # same_count = adj @ (labels == labels) per row... 
            # Actually: for vertex v with label c, same_count[v] = adj[v] @ (labels == c)
            # Let's compute: for label 0: adj @ (labels == 0), for label 1: adj @ (labels == 1)
            # same_count[v] = (adj @ (labels == 0))[v] if labels[v]==0 else (adj @ (labels == 1))[v]
            
            labels_eq_0 = (labels == 0).astype(np.float32)
            labels_eq_1 = (labels == 1).astype(np.float32)
            
            same_counts = np.where(labels == 0, adj @ labels_eq_0, adj @ labels_eq_1)
            total_deg = degrees
            diff_counts = total_deg - same_counts
            gains = same_counts - diff_counts
            
            # Find the vertex with maximum positive gain
            max_gain = np.max(gains)
            if max_gain > 1e-6:
                v = np.argmax(gains)
                labels[v] = 1 - labels[v]
                current_cut += gains[v]
                improved = True
                # Update: recompute gains would be ideal but expensive. 
                # For simplicity, we'll just do one pass at a time and recompute.
                # Actually, let's just do a full recomputation each iteration for correctness.
                # But that's O(n^2) per iteration. Let's limit iterations.
                continue
            else:
                break
        
        return labels, current_cut
    
    best_labels = None
    best_cut = -1
    
    # Strategy: multiple random restarts with hill climbing
    # Also try some structured initializations
    
    rng = np.random.default_rng(42)
    
    # Initializations to try:
    # 1. Random
    # 2. Alternating
    # 3. Based on degree (high degree in one group, low in other) - not obvious
    # 4. Random with bias
    
    num_restarts = 50
    restarts_done = 0
    
    for restart in range(num_restarts):
        if time.time() > DEADLINE - 5:
            break
        
        # Choose initialization
        if restart == 0:
            # Alternating
            labels = np.arange(n) % 2
        elif restart == 1:
            # Reverse alternating
            labels = 1 - (np.arange(n) % 2)
        elif restart == 2:
            # All zeros
            labels = np.zeros(n, dtype=int)
        elif restart == 3:
            # All ones
            labels = np.ones(n, dtype=int)
        else:
            # Random
            labels = rng.integers(0, 2, size=n)
        
        # Hill climb
        labels, cut = hill_climb(labels, max_iter=50)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        restarts_done += 1
    
    # If we have time, do more restarts
    extra_restarts = 0
    while time.time() < DEADLINE - 10 and extra_restarts < 200:
        labels = rng.integers(0, 2, size=n)
        labels, cut = hill_climb(labels, max_iter=30)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        extra_restarts += 1
    
    if best_labels is None:
        # Fallback: alternating
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
