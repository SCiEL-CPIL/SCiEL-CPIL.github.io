"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int8)
    
    adjacency = np.asarray(adjacency, dtype=np.int8)
    
    # Precompute row sums (degrees) for fast gain calculation
    # gain of flipping vertex v from side s to 1-s:
    # Let S be the set of vertices on the same side as v.
    # Let O be the set of vertices on the opposite side.
    # Current contribution of v to cut: sum(adj[v, O])
    # New contribution: sum(adj[v, S])
    # Delta = sum(adj[v, S]) - sum(adj[v, O])
    # Note: sum(adj[v, :]) = degree[v]
    # sum(adj[v, O]) = degree[v] - sum(adj[v, S])
    # Delta = 2 * sum(adj[v, S]) - degree[v]
    
    def get_gain(v, labels):
        # labels[v] is the current side
        same_side_mask = (labels == labels[v])
        # Exclude v itself from the sum if adjacency is not symmetric or has self loops? 
        # Problem says "unweighted 100-node graphs", usually simple graphs, so A[v,v]=0.
        # Even if A[v,v]!=0, it cancels out in the difference? 
        # sum(adj[v, S]) includes adj[v,v] if v in S.
        # sum(adj[v, O]) does not include adj[v,v].
        # If we flip, v moves to O.
        # New cut edges from v: edges to new S (which is old O).
        # Old cut edges from v: edges to old O.
        # The edge (v,v) is never a cut edge.
        # So we should sum over neighbors u != v.
        # Since A is symmetric and A[v,v]=0 typically, we can just sum.
        
        sum_same = np.dot(adjacency[v], same_side_mask)
        # If A[v,v] is 0, this is fine. If not, we should subtract it?
        # Cut size definition: edges (u,v) with u!=v and labels[u]!=labels[v].
        # So self-loops don't count.
        # Let's assume standard simple graph.
        
        deg_v = adjacency[v].sum()
        # Delta = sum_same - (deg_v - sum_same) = 2 * sum_same - deg_v
        return 2 * sum_same - deg_v

    def local_search(labels):
        # Hill climbing: flip vertex if gain > 0
        improved = True
        while improved:
            improved = False
            for v in range(n):
                gain = get_gain(v, labels)
                if gain > 0:
                    labels[v] = 1 - labels[v]
                    improved = True
        return labels

    def greedy_init():
        # Start with empty sets? Or random?
        # Let's do a random start but then immediately optimize.
        labels = np.random.randint(0, 2, size=n).astype(np.int8)
        return local_search(labels)

    def perturbation(labels, k):
        # Randomly flip k vertices
        idx = np.random.choice(n, k, replace=False)
        labels[idx] = 1 - labels[idx]
        return labels

    best_labels = np.zeros(n, dtype=np.int8)
    best_score = -1
    
    def cut_score(labels):
        # Efficient cut score calculation
        # Sum of A[i,j] where labels[i] != labels[j]
        # = 0.5 * sum(A * (labels[:, None] != labels[None, :]))
        # Or: sum(A) - sum(A * (labels[:, None] == labels[None, :]))
        # Let's use the matrix multiplication trick for speed if n is small.
        # n=100, so 100x100 matrix is fine.
        diff = (labels[:, None] != labels[None, :]).astype(np.int8)
        return np.sum(adjacency * diff)

    # Initial run
    labels = greedy_init()
    score = cut_score(labels)
    best_labels = labels.copy()
    best_score = score

    # Multiple restarts with perturbation
    num_restarts = 50
    for i in range(num_restarts):
        # Perturb the current best or a random start?
        # Let's perturb the current best to explore its neighborhood.
        # Or start fresh?
        # Strategy: Mix of fresh starts and perturbations of the best.
        
        if i % 5 == 0:
            # Fresh start
            labels = greedy_init()
        else:
            # Perturb best
            k = np.random.randint(1, 10)
            labels = perturbation(best_labels.copy(), k)
            # Then local search
            labels = local_search(labels)
        
        score = cut_score(labels)
        if score > best_score:
            best_score = score
            best_labels = labels.copy()
            
    return best_labels
# EVOLVE-BLOCK-END
