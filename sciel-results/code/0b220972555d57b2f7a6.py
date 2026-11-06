"""Maximum Cut via Goemans-Williamson Rounding, Fast Local Search, and Simulated Annealing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
import math

def partition(adjacency):
    """
    Computes a maximum cut partition for the given adjacency matrix.
    Uses Goemans-Williamson style rounding, fast local search with heap, and simulated annealing.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    adj = np.asarray(adjacency, dtype=np.int8)
    n = len(adj)
    
    if n == 0:
        return np.array([], dtype=int)
    
    # Precompute neighbor lists for faster gain calculation
    neighbors = [np.nonzero(adj[i])[0] for i in range(n)]
    
    # Helper to calculate cut size
    def cut_size(labels):
        diff = labels[:, None] != labels[None, :]
        return int(np.sum(adj * diff))

    # Helper to compute gain of flipping vertex v
    def flip_gain(v, labels):
        lv = labels[v]
        same = 0
        diff = 0
        for u in neighbors[v]:
            if labels[u] == lv:
                same += 1
            else:
                diff += 1
        return same - diff  # gain if we flip v

    # Fast local search using a max-heap
    def local_search(labels, time_budget=5.0):
        labels = labels.copy()
        best_labels = labels.copy()
        best_cut = cut_size(best_labels)
        
        # Build initial heap with gains
        heap = []
        for v in range(n):
            g = flip_gain(v, labels)
            if g > 0:
                heapq.heappush(heap, (-g, v))
        
        # Track which vertices are in heap to avoid duplicates
        in_heap = [False] * n
        for v in range(n):
            g = flip_gain(v, labels)
            if g > 0:
                in_heap[v] = True
        
        local_start = time.time()
        while time.time() - local_start < time_budget:
            if not heap:
                break
            
            neg_g, v = heapq.heappop(heap)
            if not in_heap[v]:
                continue
            in_heap[v] = False
            
            # Verify gain is still positive (might have changed due to other flips)
            g = flip_gain(v, labels)
            if g <= 0:
                continue
            
            # Flip v
            labels[v] = 1 - labels[v]
            best_cut += g
            
            # Update gains for neighbors
            for u in neighbors[v]:
                if in_heap[u]:
                    # Remove old entry (lazy deletion)
                    in_heap[u] = False
                g_u = flip_gain(u, labels)
                if g_u > 0:
                    heapq.heappush(heap, (-g_u, u))
                    in_heap[u] = True
            
            if best_cut > cut_size(best_labels):
                best_labels = labels.copy()
        
        # Final check
        final_cut = cut_size(labels)
        if final_cut > best_cut:
            best_labels = labels.copy()
            best_cut = final_cut
        
        return best_labels, best_cut

    # Goemans-Williamson style initialization using eigenvectors
    def gw_rounding():
        """
        Use eigenvectors of the Laplacian to create a random hyperplane rounding.
        This is inspired by Goemans-Williamson but adapted for our problem.
        """
        degrees = np.sum(adj, axis=1)
        D = np.diag(degrees)
        L = D - adj.astype(float)
        
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
        except Exception:
            return np.random.randint(0, 2, n)
        
        # Use the first few eigenvectors (excluding the trivial one)
        # The Fiedler vector is index 1
        # We'll use a random linear combination of eigenvectors 1..k
        k = min(5, n - 1)  # Use up to 5 eigenvectors
        if k < 1:
            return np.random.randint(0, 2, n)
        
        # Create a random direction in the span of eigenvectors 1..k
        weights = np.random.randn(k)
        weights /= np.linalg.norm(weights)
        
        # Project onto the span
        v = np.zeros(n)
        for i in range(k):
            v += weights[i] * eigenvectors[:, 1 + i]
        
        # Round based on sign
        labels = (v >= 0).astype(int)
        
        # Ensure not all same
        if np.all(labels == labels[0]):
            labels = np.random.randint(0, 2, n)
        
        return labels

    # 1. Multiple initialization strategies
    best_labels = None
    best_cut = -1
    
    # Strategy 1: GW rounding
    labels_gw = gw_rounding()
    ls_labels, ls_cut = local_search(labels_gw, time_budget=10.0)
    if ls_cut > best_cut:
        best_cut = ls_cut
        best_labels = ls_labels.copy()
    
    # Strategy 2: Fiedler vector sign
    try:
        degrees = np.sum(adj, axis=1)
        D = np.diag(degrees)
        L = D - adj.astype(float)
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        fiedler = eigenvectors[:, 1]
        labels_fiedler = (fiedler >= 0).astype(int)
        if np.all(labels_fiedler == labels_fiedler[0]):
            labels_fiedler = np.random.randint(0, 2, n)
        ls_labels, ls_cut = local_search(labels_fiedler, time_budget=10.0)
        if ls_cut > best_cut:
            best_cut = ls_cut
            best_labels = ls_labels.copy()
    except Exception:
        pass
    
    # Strategy 3: Random restarts
    if time.time() < deadline - 20.0:
        for _ in range(5):
            if time.time() >= deadline - 15.0:
                break
            labels_rand = np.random.randint(0, 2, n)
            ls_labels, ls_cut = local_search(labels_rand, time_budget=8.0)
            if ls_cut > best_cut:
                best_cut = ls_cut
                best_labels = ls_labels.copy()
    
    # 2. Simulated Annealing to escape local minima
    if time.time() < deadline - 10.0:
        current_labels = best_labels.copy()
        current_cut = best_cut
        
        T_start = 20.0
        T_end = 0.5
        cooling_rate = 0.97
        T = T_start
        sa_deadline = deadline - 5.0
        
        # Precompute for faster SA
        while T > T_end and time.time() < sa_deadline:
            v = np.random.randint(0, n)
            gain = flip_gain(v, current_labels)
            
            if gain > 0 or (np.random.rand() < math.exp(-gain / T)):
                current_labels[v] = 1 - current_labels[v]
                current_cut += gain
                
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = current_labels.copy()
            
            T *= cooling_rate
        
        # Final local search polish
        if time.time() < deadline - 3.0:
            ls_labels, ls_cut = local_search(best_labels, time_budget=5.0)
            if ls_cut > best_cut:
                best_cut = ls_cut
                best_labels = ls_labels.copy()
    
    return best_labels

# EVOLVE-BLOCK-END
