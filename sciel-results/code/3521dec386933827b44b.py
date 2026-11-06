import time
import numpy as np
import heapq

def partition(adjacency):
    """
    Assign each vertex a 0/1 label to maximize the number of cut edges.
    Uses multi-restart local search with efficient incremental gain computation.
    """
    adj = np.asarray(adjacency, dtype=np.int8)
    n = len(adj)

    if n <= 1:
        return np.zeros(n, dtype=np.int8)

    # Precompute degrees and neighbor lists for fast access
    degrees = adj.sum(axis=1).astype(np.int32)
    neighbors = [np.where(adj[i] == 1)[0] for i in range(n)]
    
    # Remove self-loops from neighbor lists if any (adj[i,i] might be 1, but usually 0 in simple graphs)
    # Assuming simple graph, but safe to handle.
    for i in range(n):
        if adj[i, i] == 1:
            neighbors[i] = neighbors[i][neighbors[i] != i]

    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin

    def compute_initial_state(labels):
        """Compute 'same' counts and total cut size for a given labeling."""
        same = np.zeros(n, dtype=np.int32)
        cut = 0
        # Efficiently compute same counts
        # same[v] = sum(adj[v, u] for u where labels[u] == labels[v])
        in_0 = (labels == 0).astype(np.int32)
        in_1 = (labels == 1).astype(np.int32)
        
        same_0 = (adj @ in_0)
        same_1 = (adj @ in_1)
        
        same = np.where(labels == 0, same_0, same_1).astype(np.int32)
        
        # Cut size = total edges - sum(same)/2 ? No.
        # Cut size = sum over v in 0 of (deg[v] - same[v])
        # Or simply: sum(adj[0, 1])
        cut = int((in_0 @ adj @ in_1))
        
        return same, cut

    def local_search(labels, same, cut):
        """
        Perform local search using a priority queue of positive gain flips.
        Returns best labels, same, cut found.
        """
        labels = labels.copy()
        same = same.copy()
        best_cut = cut
        best_labels = labels.copy()
        best_same = same.copy()
        
        # Heap stores (-gain, vertex)
        heap = []
        for v in range(n):
            gain = 2 * same[v] - degrees[v]
            if gain > 0:
                heapq.heappush(heap, (-gain, v))
        
        # Counter to handle stale entries? Not strictly needed if we check validity.
        
        while heap and time.time() < deadline:
            neg_gain, v = heapq.heappop(heap)
            gain = -neg_gain
            
            # Check if gain is still valid
            current_gain = 2 * same[v] - degrees[v]
            if current_gain != gain or current_gain <= 0:
                continue
            
            # Flip vertex v
            old_label = labels[v]
            new_label = 1 - old_label
            labels[v] = new_label
            
            # Update cut
            cut += current_gain
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
                best_same = same.copy()
            
            # Update same[v]
            # New same[v] is the count of neighbors in the new group.
            # Previously, same[v] was count in old group.
            # Neighbors in new group = degrees[v] - (neighbors in old group excluding v? No, v is not its own neighbor).
            # Wait, same[v] = sum(adj[v, u] for u in same_group).
            # When v flips, its "same" neighbors become its "different" neighbors.
            # So new_same[v] = degrees[v] - old_same[v].
            same[v] = degrees[v] - same[v]
            
            # Update neighbors
            for u in neighbors[v]:
                if labels[u] == old_label:
                    # u was in same group as v (old), now v is different.
                    # u loses a same-neighbor.
                    same[u] -= 1
                else:
                    # u was in different group from v (old), now v is same.
                    # u gains a same-neighbor.
                    same[u] += 1
                
                # Check if u now has positive gain and push to heap
                u_gain = 2 * same[u] - degrees[u]
                if u_gain > 0:
                    heapq.heappush(heap, (-u_gain, u))
        
        return best_labels, best_same, best_cut

    def spectral_init():
        """Initialize using the top eigenvector of the adjacency matrix."""
        try:
            # Use scipy if available for better eigensolver, else numpy
            # For 100x100, numpy.linalg.eigh is fine
            vals, vecs = np.linalg.eigh(adj)
            # The largest eigenvalue corresponds to the last column (eigh returns ascending order)
            top_vec = vecs[:, -1]
            # Round to 0/1 based on sign
            labels = (top_vec >= 0).astype(np.int8)
            return labels
        except Exception:
            return np.zeros(n, dtype=np.int8)

    def greedy_init():
        """Greedy sequential initialization."""
        labels = np.zeros(n, dtype=np.int8)
        order = np.argsort(-degrees)
        for v in order:
            if len(neighbors[v]) == 0:
                continue
            # Count neighbors in 0 and 1
            n0 = np.sum(labels[neighbors[v]] == 0)
            n1 = np.sum(labels[neighbors[v]] == 1)
            if n1 > n0:
                labels[v] = 1
            else:
                labels[v] = 0
        return labels

    def random_init(rng):
        return rng.integers(0, 2, size=n, dtype=np.int8)

    best_labels = None
    best_cut = -1
    best_same = None

    rng = np.random.default_rng(42)

    # List of initializations
    inits = []
    
    # 1. Spectral
    inits.append(spectral_init())
    
    # 2. Greedy
    inits.append(greedy_init())
    
    # 3. Alternating
    inits.append(np.arange(n) % 2)
    
    # 4. Random restarts
    # We will generate more random restarts if time permits
    num_random = 20
    for _ in range(num_random):
        inits.append(random_init(rng))

    # Process initializations
    for init_labels in inits:
        if time.time() > deadline:
            break
            
        same, cut = compute_initial_state(init_labels)
        
        # Run local search
        res_labels, res_same, res_cut = local_search(init_labels, same, cut)
        
        if res_cut > best_cut:
            best_cut = res_cut
            best_labels = res_labels
            best_same = res_same

        # If we have significant time left, we can try more random restarts
        # or run a second pass on the best solution found so far (though local search is deterministic given start)
        # To escape local optima, we can perturb the best solution and run local search again.
        
    # If we still have time, perform "perturbation + local search" on the best solution
    # This is a form of Iterated Local Search (ILS)
    if best_labels is not None and time.time() < deadline - 5:
        current_best = best_labels.copy()
        current_cut = best_cut
        current_same = best_same.copy()
        
        # ILS loop
        while time.time() < deadline - 5:
            # Perturb: flip k random vertices
            k = max(1, n // 10) # 10% of nodes
            flip_indices = rng.choice(n, size=k, replace=False)
            perturbed = current_best.copy()
            perturbed[flip_indices] = 1 - perturbed[flip_indices]
            
            # Recompute state for perturbed
            p_same, p_cut = compute_initial_state(perturbed)
            
            # Run local search
            res_labels, res_same, res_cut = local_search(perturbed, p_same, p_cut)
            
            if res_cut > current_cut:
                current_best = res_labels
                current_cut = res_cut
                current_same = res_same
                best_cut = res_cut
                best_labels = res_labels
                best_same = res_same
            else:
                # Optionally accept worse solutions with probability? 
                # For simplicity and speed, just keep best found so far.
                # But to explore, we might accept slightly worse? 
                # Given time constraints, sticking to best-so-far is safer.
                pass

    if best_labels is None:
        best_labels = np.zeros(n, dtype=np.int8)

    return best_labels.astype(np.int8)
