"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    # labels is 0/1 array. Cut size = sum over edges (i,j) where labels[i] != labels[j]
    # For a boolean adjacency matrix A, cut = sum(A * (labels[:, None] != labels[None, :]))
    # But that's O(n^2). For n=100 it's fine.
    diff = labels[:, None] != labels[None, :]
    return int(np.sum(adjacency & diff))


def _greedy_polish(adjacency, labels, max_rounds=10):
    """Single-flip greedy local search until no improvement."""
    n = len(adjacency)
    labels = labels.copy()
    improved = True
    rounds = 0
    while improved and rounds < max_rounds:
        improved = False
        rounds += 1
        for v in range(n):
            # Count neighbors in same side vs other side
            same_side = (labels == labels[v])
            other_side = (labels != labels[v])
            same_count = int(np.sum(adjacency[v] & same_side))
            other_count = int(np.sum(adjacency[v] & other_side))
            if other_count > same_count:
                labels[v] = 1 - labels[v]
                improved = True
    return labels


def _double_flip_polish(adjacency, labels, n_samples=2000, n_rounds=10):
    """Try random pairs of flips to escape 2-flip local optima."""
    n = len(adjacency)
    labels = labels.copy()
    best_cut = _cut_size(adjacency, labels)
    
    for _ in range(n_rounds):
        improved = False
        for _ in range(n_samples):
            i, j = np.random.choice(n, 2, replace=False)
            # Try flipping both i and j
            if labels[i] == labels[j]:
                continue  # flipping two on same side is equivalent to no net change in cut for those two? 
                # Actually it changes edges from i to others and j to others. Let's just compute.
            
            old_labels = labels.copy()
            labels[i] = 1 - labels[i]
            labels[j] = 1 - labels[j]
            new_cut = _cut_size(adjacency, labels)
            
            if new_cut > best_cut:
                best_cut = new_cut
                improved = True
            else:
                labels = old_labels
        
        if not improved:
            break
    
    return labels


def _simulated_annealing(adjacency, n, start_temp=1.0, end_temp=0.01, 
                          n_iterations=None, deadline=None):
    """Simulated annealing for max cut."""
    labels = np.random.randint(0, 2, size=n)
    best_labels = labels.copy()
    best_cut = _cut_size(adjacency, labels)
    current_cut = best_cut
    
    if n_iterations is None:
        n_iterations = 5000
    
    t = start_temp
    cooling = (end_temp / start_temp) ** (1.0 / n_iterations)
    
    for it in range(n_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        v = np.random.randint(n)
        old_label = labels[v]
        labels[v] = 1 - old_label
        
        # Compute change in cut
        # When we flip vertex v, edges from v to same side become crossing, 
        # and edges from v to other side become non-crossing.
        same_side = (labels != old_label)  # After flip, same side means those now equal to new label
        other_side = (labels == old_label)  # After flip, other side means those not equal to new label
        
        # Wait, let me think more carefully.
        # Before flip: label of v is old_label.
        # After flip: label of v is 1 - old_label.
        # For each neighbor u of v:
        #   If labels[u] == old_label: before flip, edge (u,v) was same-side (not cut). After flip, it's cut.
        #   If labels[u] != old_label: before flip, edge (u,v) was cut. After flip, it's not cut.
        
        # So delta = (# neighbors with label == old_label) - (# neighbors with label != old_label)
        # But we need to exclude v itself from the count (no self-loops in simple graphs, but adjacency[v][v] should be 0)
        
        neighbors = adjacency[v]
        same_before = np.sum(neighbors & (labels == old_label))  # neighbors that were same side as old label
        other_before = np.sum(neighbors & (labels != old_label))  # neighbors that were other side
        
        # Wait, labels[v] has been changed. Let me use the state before the flip.
        # Actually, let's recompute properly.
        
        # Let's just compute the cut change directly.
        # Before flip: v had label old_label
        # After flip: v has label 1 - old_label
        # Edges from v: for each neighbor u, edge (u,v) is cut iff labels[u] != labels[v]
        
        # Before: cut contribution from v's edges = sum over u of (labels[u] != old_label)
        # After: cut contribution from v's edges = sum over u of (labels[u] != (1 - old_label))
        
        # But labels[u] for u != v hasn't changed.
        
        # Let me just compute both cuts for the edges incident to v.
        # This is simpler and correct for n=100.
        
        # Actually, let me restore and compute delta properly.
        labels[v] = old_label  # restore
        neighbors = adjacency[v]
        same_before = int(np.sum(neighbors & (labels == old_label)))
        other_before = int(np.sum(neighbors & (labels != old_label)))
        
        # After flipping v:
        # Neighbors that were same (labels[u] == old_label) now become different -> they add to cut
        # Neighbors that were different (labels[u] != old_label) now become same -> they subtract from cut
        # delta = same_before - other_before
        
        delta = same_before - other_before
        
        if delta > 0 or np.random.random() < np.exp(delta / t):
            labels[v] = 1 - old_label
            current_cut += delta
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        else:
            labels[v] = old_label  # revert
        
        t *= cooling
    
    return best_labels, best_cut


def partition(adjacency):
    n = len(adjacency)
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    best_labels = None
    best_cut = -1
    
    # Multiple restarts with simulated annealing + polish
    n_restarts = 0
    while time.time() < deadline - 5:  # Leave 5s for final polish
        n_restarts += 1
        
        # SA with random init
        sa_labels, sa_cut = _simulated_annealing(
            adjacency, n, 
            start_temp=1.0, end_temp=0.01,
            n_iterations=3000,
            deadline=deadline - 5
        )
        
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
        
        # Greedy polish
        polished = _greedy_polish(adjacency, sa_labels, max_rounds=5)
        polished_cut = _cut_size(adjacency, polished)
        if polished_cut > best_cut:
            best_cut = polished_cut
            best_labels = polished.copy()
        
        # Double-flip polish
        df_polished = _double_flip_polish(adjacency, polished, n_samples=1000, n_rounds=5)
        df_cut = _cut_size(adjacency, df_polished)
        if df_cut > best_cut:
            best_cut = df_cut
            best_labels = df_polished.copy()
    
    # Final polish with remaining time
    if best_labels is not None and time.time() < deadline - 1:
        final = _greedy_polish(adjacency, best_labels, max_rounds=10)
        final_cut = _cut_size(adjacency, final)
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final
        
        df_final = _double_flip_polish(adjacency, best_labels, n_samples=2000, n_rounds=10)
        df_final_cut = _cut_size(adjacency, df_final)
        if df_final_cut > best_cut:
            best_cut = df_final_cut
            best_labels = df_final
    
    if best_labels is None:
        # Fallback: alternating
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
