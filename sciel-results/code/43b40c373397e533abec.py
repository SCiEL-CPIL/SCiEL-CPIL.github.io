# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


def priority(adjacency):
    """Score vertices for greedy independent set selection.
    
    Strategy: combine degree, clustering coefficient, and spectral peripherality
    to identify 'safe' vertices on the boundary of dense clusters.
    """
    start_time = time.time()
    DEADLINE = 120.0  # seconds
    
    adj = np.asarray(adjacency, dtype=np.float64)
    n = adj.shape[0]
    
    # Degree
    degree = adj.sum(axis=1)
    
    # Clustering coefficient per vertex
    # For each vertex, count edges among its neighbors / possible edges
    clustering = np.zeros(n)
    for i in range(n):
        if time.time() - start_time > DEADLINE - 5.0:
            break
        nbrs = np.where(adj[i] > 0.5)[0]
        d = len(nbrs)
        if d < 2:
            clustering[i] = 0.0
        else:
            sub = adj[np.ix_(nbrs, nbrs)]
            edges_among = sub.sum() / 2.0
            clustering[i] = edges_among / (d * (d - 1) / 2.0)
    
    # Spectral peripherality: use Fiedler vector (2nd eigenvector of Laplacian)
    # Vertices with extreme Fiedler values are on the "boundary"
    try:
        # Build Laplacian
        L = np.diag(degree) - adj
        # Use scipy for efficiency
        L_sparse = csr_matrix(L)
        # Get smallest eigenvalues (Fiedler vector is 2nd smallest)
        if n > 10:
            k = min(3, n - 1)
            vals, vecs = eigsh(L_sparse, k=k, sigma=0, which='LM')
            # Sort by eigenvalue
            idx = np.argsort(vals)
            fiedler = vecs[:, idx[1]]  # 2nd smallest eigenvalue's eigenvector
        else:
            eigvals, eigvecs = np.linalg.eigh(L)
            fiedler = eigvecs[:, 1]
        
        # Peripherality: distance from median of Fiedler values
        med = np.median(fiedler)
        peripherality = np.abs(fiedler - med)
        # Normalize
        if peripherality.max() > 0:
            peripherality = peripherality / peripherality.max()
        else:
            peripherality = np.zeros(n)
    except Exception:
        # Fallback: use degree-based peripherality
        peripherality = 1.0 - degree / (degree.max() + 1e-10)
    
    # Normalize all signals to [0, 1]
    def normalize(x):
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
    
    deg_norm = normalize(degree)
    clust_norm = normalize(clustering)
    periph_norm = normalize(peripherality)
    
    # Score: want LOW degree, LOW clustering, HIGH peripherality
    # Higher score = better to pick first
    score = -deg_norm * 0.5 + (1.0 - clust_norm) * 0.25 + periph_norm * 0.25
    
    # Additional: penalize vertices whose neighbors have high degree
    # (picking them blocks high-degree vertices which is bad)
    # Actually, blocking high-degree neighbors is GOOD because they'd block many others
    # Let's think: if I pick vertex v, I remove all neighbors. 
    # The "cost" is the independent-set potential lost from those neighbors.
    # A high-degree neighbor has high potential, so blocking it is costly.
    # But wait - we're building an independent set, so we WANT to pick vertices.
    # Picking v means we can't pick its neighbors. If a neighbor has high degree,
    # it would have blocked many other vertices if picked, so NOT picking it is good.
    # So picking a vertex adjacent to high-degree vertices is actually GOOD.
    
    # Neighbor degree sum (higher = better to pick, because you're "protecting"
    # the rest of the graph from high-degree vertices)
    nbr_deg_sum = adj @ degree  # sum of degrees of neighbors
    nbr_deg_norm = normalize(nbr_deg_sum)
    
    # Combine
    score = score + nbr_deg_norm * 0.15
    
    return score
# EVOLVE-BLOCK-END
