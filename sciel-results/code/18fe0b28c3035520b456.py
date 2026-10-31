import numpy as np
import itertools
import random
import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import random
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def construct():
    # 1. Generate all norm-2 integer vectors (squared norm 4)
    # Type A: one +/- 2
    # Type B: four +/- 1
    
    vectors = []
    
    # Type A
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            vectors.append(v)
    
    # Type B
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, val in zip(combo, signs):
                v[pos] = val
            vectors.append(v)
            
    N = len(vectors)
    
    # 2. Build conflict graph
    # Conflict if dot product > 2
    # We store adjacency list for the graph
    adj = [[] for _ in range(N)]
    
    # Use numpy for fast dot product calculation
    vec_arr = np.array(vectors, dtype=np.int8)
    dots = vec_arr @ vec_arr.T
    
    rows, cols = np.where(dots > 2)
    
    for r, c in zip(rows, cols):
        if r < c:
            adj[r].append(c)
            adj[c].append(r)
            
    # 3. Spectral Partitioning
    degrees = np.array([len(a) for a in adj], dtype=np.int32)
    
    data = np.ones(len(rows), dtype=np.int32)
    A_sparse = csr_matrix((data, (rows, cols)), shape=(N, N))
    
    D_sparse = csr_matrix(np.diag(degrees))
    L = D_sparse - A_sparse
    
    try:
        eigvals, eigvecs = eigsh(L, k=2, which='SM')
        idx = np.argsort(eigvals)
        fiedler_vec = eigvecs[:, idx[1]]
    except Exception:
        fiedler_vec = np.random.randn(N)
        
    # 4. Find Best Independent Set via Partitioning and Local Search
    best_set = set()
    best_size = 0
    
    # Try multiple partitions based on Fiedler vector with random perturbations
    for _ in range(5):
        # Perturb Fiedler vector
        if _ == 0:
            perturbed = fiedler_vec
        else:
            perturbed = fiedler_vec + np.random.randn(N) * 0.1
            
        P1 = [i for i in range(N) if perturbed[i] > 0]
        P2 = [i for i in range(N) if perturbed[i] <= 0]
        
        # Clean P1 and P2 to independent sets
        def clean(partition):
            # Greedy MIS on induced subgraph
            # Sort by degree descending
            sorted_part = sorted(partition, key=lambda x: len(adj[x]), reverse=True)
            s = set()
            for v in sorted_part:
                conflict = False
                for u in adj[v]:
                    if u in s:
                        conflict = True
                        break
                if not conflict:
                    s.add(v)
            return s
            
        S1 = clean(P1)
        S2 = clean(P2)
        
        # Pick larger
        current_set = S1 if len(S1) >= len(S2) else S2
        
        # Local Search: Add if possible
        in_set_bool = [False] * N
        for v in current_set:
            in_set_bool[v] = True
            
        improved = True
        while improved:
            improved = False
            for v in range(N):
                if not in_set_bool[v]:
                    can_add = True
                    for u in adj[v]:
                        if in_set_bool[u]:
                            can_add = False
                            break
                    if can_add:
                        in_set_bool[v] = True
                        current_set.add(v)
                        improved = True
                        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            
    # Construct final vectors
    final_vectors = []
    for i in range(N):
        if i in best_set:
            final_vectors.append(vectors[i])
            
    return np.array(final_vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
