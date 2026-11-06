"""Maximum Cut via Spectral Initialization and Fast Local Search."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def partition(adjacency):
    """
    Assign each vertex a 0/1 label to maximize the number of cut edges.
    Uses spectral initialization (Fiedler vector) and fast 1-flip local search.
    """
    start_time = time.time()
    # Safety margin: 150s limit, keep 10s buffer
    deadline = start_time + 140.0

    adj = np.asarray(adjacency, dtype=np.int8)
    n = len(adj)

    if n <= 1:
        return np.zeros(n, dtype=np.int8)

    # Precompute degrees and neighbor lists for fast gain updates
    degrees = adj.sum(axis=1).astype(np.int32)
    neighbors = [np.where(adj[v] == 1)[0] for v in range(n)]

    def compute_cut(labels):
        """Compute the size of the cut given the labels."""
        in_0 = (labels == 0).astype(np.int32)
        in_1 = (labels == 1).astype(np.int32)
        return int(in_0 @ adj @ in_1)

    def spectral_init():
        """
        Initialize using the Fiedler vector (second smallest eigenvector of Laplacian).
        """
        try:
            D = np.diag(degrees)
            L = D - adj.astype(np.float64)
            L = (L + L.T) / 2.0
            
            # Use eigh for symmetric matrices
            eigvals, eigvecs = np.linalg.eigh(L)
            
            # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
            # Index 0 is the trivial eigenvalue 0
            fiedler = eigvecs[:, 1]
            
            # Partition based on sign
            labels = (fiedler > 0).astype(np.int8)
            
            # Ensure both partitions are non-empty
            if np.all(labels == 0) or np.all(labels == 1):
                # Fallback: split by median
                median_val = np.median(fiedler)
                labels = (fiedler > median_val).astype(np.int8)
                
            return labels
        except Exception:
            return None

    def local_search(labels, time_limit):
        """
        Fast 1-flip local search.
        Returns the improved labels and cut size.
        """
        labels = labels.copy()
        cut = compute_cut(labels)
        
        # Calculate initial gains for all vertices
        # gain[v] = deg[v] - 2 * (num neighbors in same group)
        gains = np.zeros(n, dtype=np.int32)
        for v in range(n):
            if labels[v] == 0:
                same_count = int(adj[v, labels == 0].sum())
            else:
                same_count = int(adj[v, labels == 1].sum())
            gains[v] = degrees[v] - 2 * same_count

        # Find best vertex to flip
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_gain = 0
            best_v = -1
            
            # Find the vertex with the maximum positive gain
            for v in range(n):
                if gains[v] > best_gain:
                    best_gain = gains[v]
                    best_v = v
            
            if best_v == -1:
                break
                
            # Flip vertex best_v
            v = best_v
            old_label = labels[v]
            labels[v] = 1 - old_label
            
            cut += best_gain
            
            # Update gains for neighbors of v
            # For a neighbor u:
            # If u and v were in same group, they are now in different groups.
            #   u lost a "same" neighbor -> gain increases by 2.
            # If u and v were in different groups, they are now in same group.
            #   u gained a "same" neighbor -> gain decreases by 2.
            
            for u in neighbors[v]:
                if labels[u] == labels[v]:
                    # Now in same group. Before flip, they were in different groups.
                    # So u gained a same neighbor.
                    gains[u] -= 2
                else:
                    # Now in different groups. Before flip, they were in same group.
                    # So u lost a same neighbor.
                    gains[u] += 2
            
            # Also update gain for v itself?
            # Actually, flipping v changes its own gain for the next iteration.
            # But since we only flip one vertex at a time and re-evaluate all gains implicitly via the loop,
            # we just need to ensure the gains array is consistent for the *next* iteration.
            # The update above handles neighbors. What about v?
            # If we flip v again, the gain would be -best_gain (since the state is reversed).
            # However, in 1-flip local search, we typically don't flip the same vertex back immediately
            # unless it becomes beneficial later.
            # The gain for v is currently best_gain. If we were to consider flipping it again,
            # the gain would be -best_gain.
            # To be safe and simple, we can just recompute gains for v or note that
            # the gain of v is now -best_gain relative to the new state.
            # Let's update gains[v] to reflect the new state.
            # If v was 0 and is now 1:
            #   Same neighbors for v (now 1) are those u with labels[u]==1.
            #   Previously, same neighbors for v (was 0) were those u with labels[u]==0.
            #   The change in "same" count for v is exactly -best_gain / 2 ? No.
            #   gain = deg - 2*same.
            #   New gain = deg - 2*new_same.
            #   We know cut increased by best_gain.
            #   Let's just recalculate gains[v] to be safe, or use the property:
            #   New gain of v = - (Old gain of v) IF no other changes affected v's neighbors' labels.
            #   But we just changed v's label, so v's "same" neighbors changed.
            #   Actually, the simplest correct update for v's own gain in the array for the NEXT iteration:
            #   The gain of flipping v back would be -best_gain.
            #   So gains[v] = -best_gain.
            gains[v] = -best_gain

        return labels, cut

    rng = np.random.default_rng(42)
    
    best_labels_global = None
    best_cut_global = -1

    # Generate initial candidates
    candidates = []
    
    # 1. Spectral
    spec_labels = spectral_init()
    if spec_labels is not None:
        candidates.append(spec_labels)
        
    # 2. Random
    for _ in range(3):
        candidates.append(rng.integers(0, 2, size=n, dtype=np.int8))

    # Process each candidate with Local Search
    # We allocate time per candidate to ensure we finish before deadline
    # Total time budget for search: deadline - start_time - 5s buffer
    search_budget = max(1.0, deadline - start_time - 5.0)
    time_per_candidate = search_budget / len(candidates)
    
    for init_labels in candidates:
        if time.time() > deadline - 5:
            break
            
        # Run local search with a specific time limit for this candidate
        # We give each candidate a fair share of the time
        ls_labels, ls_cut = local_search(init_labels, time.time() + time_per_candidate)
        
        if ls_cut > best_cut_global:
            best_cut_global = ls_cut
            best_labels_global = ls_labels.copy()

    if best_labels_global is None:
        best_labels_global = np.zeros(n, dtype=np.int8)

    return best_labels_global.astype(np.int8)

# EVOLVE-BLOCK-END
