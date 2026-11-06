"""
Maximum Cut solver using Spectral Initialization + Vectorized Local Search + SA Polish.
Strategy:
1. Compute the largest eigenvector of the adjacency matrix (or Fiedler vector) to get a structural initialization.
2. Use vectorized local search (multi-pass) to refine the partition.
3. Polish with Simulated Annealing to escape local optima.
4. Multi-start: try multiple spectral-based initializations (e.g., different eigenvectors or random perturbations of the spectral vector).
"""
import numpy as np
import time
from scipy.linalg import eigh

def partition(adjacency):
    start_time = time.time()
    DEADLINE = 115.0  # Leave 5s safety margin for 120s limit
    
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy float32 for faster ops
    adj = np.array(adjacency, dtype=np.float32)
    degrees = adj.sum(axis=1)
    
    # Helper: compute cut size
    def cut_size(labels):
        l = labels.astype(np.float32)
        # Cut size = 0.5 * sum_{u,v} A[u,v] * (1 - l[u]*l[v]) / 2? 
        # Actually, standard cut size = sum_{u<v, A[u,v]=1, l[u]!=l[v]} 1
        # Using: 0.5 * (sum(A) - sum(A * (l_u * l_v)))
        # But l is 0/1. Let's use: 0.5 * (total_edges - sum(A * (2*l - 1) outer (2*l - 1)))?
        # Simpler: 
        # Let s = 2*l - 1 (so s is +/-1)
        # Cut size = 0.25 * (n*E - s^T A s) ? No.
        # Let's stick to the reliable one:
        # cut = 0.5 * (sum(A) - sum(A * (l_u == l_v)))
        # But we can compute: 
        # l_float = l.astype(float)
        # cut = 0.5 * (l_float.T @ adj @ (1 - l_float) + (1 - l_float).T @ adj @ l_float) / 2?
        # Actually, the most efficient vectorized way:
        # cut = 0.5 * (sum(A) - sum(A * (l_u * l_v + (1-l_u)*(1-l_v))))
        # Let's use the s-vector method:
        s = 2.0 * l - 1.0  # +/-1
        # sum_{u,v} A[u,v] s[u] s[v] = s^T A s
        # This counts 2 for each edge in the same partition, 0 for crossing? No.
        # If s[u]=s[v], s[u]s[v]=1. If different, -1.
        # sum(A * s^T A s) = sum_{u,v} A[u,v] s[u] s[v]
        # = 2 * (sum_{same} 1 - sum_{diff} 1)
        # Let N_same = number of edges with same label, N_diff = number with diff.
        # Total edges E = N_same + N_diff.
        # s^T A s = 2 * (N_same - N_diff)
        # We want N_diff.
        # N_diff = (E - (s^T A s)/2) / 2? No.
        # N_same - N_diff = (s^T A s) / 2
        # N_same + N_diff = E
        # 2*N_diff = E - (s^T A s)/2
        # N_diff = 0.5 * E - 0.25 * (s^T A s)
        E = 0.5 * adj.sum()
        val = s @ (adj @ s)
        return 0.5 * E - 0.25 * val

    def vectorized_local_search(labels, max_passes=50):
        """
        Perform local search using vectorized gain computation.
        Gain for flipping vertex v:
        gain[v] = 2 * (sum_{u in N(v), label[u] != label[v]} 1) - degree[v]
        Actually, if we flip v, edges that were crossing become same, and same become crossing.
        Let S_v = sum of neighbors with same label as v.
        Let D_v = sum of neighbors with different label as v.
        Current contribution of v to cut = D_v.
        New contribution after flip = S_v.
        Change = S_v - D_v = S_v - (degree[v] - S_v) = 2*S_v - degree[v].
        We want to flip if Change > 0.
        """
        labels = labels.copy()
        for _ in range(max_passes):
            if time.time() > DEADLINE - 20:
                break
            # Compute S_v for all v at once
            # S_v = sum_{u} A[v,u] * (labels[u] == labels[v])
            # We can compute this by:
            # same_mask = (labels[:, None] == labels[None, :]) # n x n bool
            # S = adj @ same_mask # This is n x n, we want row sums? No, adj is n x n.
            # S[v] = sum_u adj[v,u] * (labels[u] == labels[v])
            # Let's compute:
            # l = labels.astype(float)
            # s = 2*l - 1
            # same_indicator = (s[:, None] * s[None, :]) > 0
            # S = adj @ same_indicator # This gives n x n, we want sum over u for each v?
            # Actually, adj @ same_indicator gives a matrix where M[v,u] = sum_w adj[v,w] * same_indicator[w,u].
            # We want sum_u adj[v,u] * same_indicator[v,u].
            # This is element-wise product then sum.
            # S = np.sum(adj * same_indicator, axis=1)
            
            s = 2.0 * labels.astype(np.float32) - 1.0
            # same_indicator[v,u] = 1 if s[v]*s[u] > 0
            # We can compute S without full n x n matrix if possible, but for n=100 it's fine.
            # However, creating n x n bool matrix each pass is O(n^2) memory and time.
            # Alternative:
            # S_v = sum_{u: label[u]==label[v]} adj[v,u]
            # Let's use the fact that:
            # sum_{u: label[u]==0} adj[v,u] = adj[v] @ (labels==0)
            # sum_{u: label[u]==1} adj[v,u] = adj[v] @ (labels==1)
            # S = np.where(labels==0, adj @ (labels==0), adj @ (labels==1))
            # This is O(n^2) per pass, which is fine.
            
            mask0 = (labels == 0).astype(np.float32)
            mask1 = (labels == 1).astype(np.float32)
            sum0 = adj @ mask0  # For each v, sum of adj[v,u] where label[u]==0
            sum1 = adj @ mask1  # For each v, sum of adj[v,u] where label[u]==1
            
            S = np.where(labels == 0, sum0, sum1)
            D = degrees - S
            gain = S - D  # 2*S - degree[v]
            
            # Flip vertices where gain > 0
            to_flip = gain > 1e-6
            if not np.any(to_flip):
                break
            labels[to_flip] = 1 - labels[to_flip]
            
        return labels

    def simulated_annealing(labels, start_temp=10.0, end_temp=0.1, max_iter=20000):
        labels = labels.copy()
        current_cut = cut_size(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        for it in range(max_iter):
            if time.time() > DEADLINE - 5:
                break
            t = start_temp * (end_temp / start_temp) ** (it / max_iter)
            v = np.random.randint(n)
            
            # Compute gain for flipping v
            if labels[v] == 0:
                same_count = adj[v] @ (labels == 0)
            else:
                same_count = adj[v] @ (labels == 1)
            diff_count = degrees[v] - same_count
            gain = same_count - diff_count
            
            if gain >= 0 or np.random.rand() < np.exp(gain / max(t, 1e-10)):
                labels[v] = 1 - labels[v]
                current_cut += gain
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
        
        return best_labels, best_cut

    best_labels = None
    best_cut = -1
    
    # Phase 1: Spectral Initialization
    # Compute largest eigenvector of adjacency matrix
    # Note: For connected graphs, the largest eigenvector of A is not necessarily the best cut.
    # The Fiedler vector (2nd smallest eigenvector of L) is often better for cuts.
    # Let's try both.
    
    # Option 1: Fiedler Vector
    try:
        D = np.diag(degrees)
        L = D - adj
        # Compute 2nd smallest eigenvalue and eigenvector
        # Use scipy.linalg.eigh for symmetric matrices
        vals, vecs = eigh(L, subset_by_index=[1, 1])
        fiedler = vecs[:, 0]
        # Threshold at 0
        init_fiedler = (fiedler > 0).astype(int)
        # Ensure both classes are non-empty
        if np.all(init_fiedler == 0) or np.all(init_fiedler == 1):
            # Fallback to median split
            med = np.median(fiedler)
            init_fiedler = (fiedler > med).astype(int)
    except Exception:
        init_fiedler = np.zeros(n, dtype=int)
        init_fiedler[:n//2] = 1

    # Option 2: Largest Eigenvector of A
    try:
        vals_a, vecs_a = eigh(adj, subset_by_index=[n-1, n-1])
        max_evec = vecs_a[:, 0]
        init_max = (max_evec > 0).astype(int)
        if np.all(init_max == 0) or np.all(init_max == 1):
            med = np.median(max_evec)
            init_max = (max_evec > med).astype(int)
    except Exception:
        init_max = np.zeros(n, dtype=int)
        init_max[:n//2] = 1

    # Phase 2: Local Search from Spectral Inits
    for init in [init_fiedler, init_max]:
        if time.time() > DEADLINE - 30:
            break
        improved = vectorized_local_search(init, max_passes=20)
        cut = cut_size(improved)
        if cut > best_cut:
            best_cut = cut
            best_labels = improved.copy()

    # Phase 3: Multi-start Random with Local Search
    rng = np.random.default_rng(42)
    while time.time() < DEADLINE - 30:
        labels = rng.integers(0, 2, size=n)
        improved = vectorized_local_search(labels, max_passes=10)
        cut = cut_size(improved)
        if cut > best_cut:
            best_cut = cut
            best_labels = improved.copy()

    # Phase 4: SA Polish
    if best_labels is not None and time.time() < DEADLINE - 5:
        polished, polished_cut = simulated_annealing(best_labels, start_temp=5.0, end_temp=0.05, max_iter=20000)
        if polished_cut > best_cut:
            best_cut = polished_cut
            best_labels = polished

    if best_labels is None:
        best_labels = np.arange(n) % 2

    return best_labels
