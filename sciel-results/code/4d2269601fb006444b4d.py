"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh

D = 11

# EVOLVE-BLOCK-START

def generate_vectors(weight):
    vectors = []
    for support in itertools.combinations(range(D), weight):
        for signs in itertools.product([1, -1], repeat=weight):
            v = np.zeros(D, dtype=np.int8)
            for idx, val in zip(support, signs):
                v[idx] = val
            vectors.append(v)
    return np.array(vectors)

def construct():
    start_time = time.time()
    # Safety margin: 150s limit, aim for 130s max execution
    deadline = start_time + 130.0
    
    best_vectors = None
    best_count = 0
    
    # We primarily target weight 4, which offers the best balance of
    # graph density and potential code size for dimension 11.
    # Weight 3 is sparser but might allow slightly larger codes if the
    # greedy method gets stuck in local minima on the denser W=4 graph.
    
    for weight in [4, 3]:
        if time.time() > deadline - 10:
            break
            
        V = generate_vectors(weight)
        N = V.shape[0]
        
        if N == 0:
            continue
            
        # Condition for valid kissing configuration:
        # ||u - v||^2 >= ||u||^2 (for all u, v in code, u != v)
        # ||u||^2 + ||v||^2 - 2 u.v >= ||u||^2
        # 2w - 2 u.v >= w  =>  u.v <= w/2
        # For integer vectors, u.v is integer.
        # If w=4, limit = 2.
        # If w=3, limit = 1 (since 1.5 floors to 1 for integer check? No, u.v <= 1.5 means u.v <= 1 for integers).
        limit = weight // 2
        
        # Compute adjacency matrix for the conflict graph
        # Edge (i, j) exists if vectors i and j CANNOT be in the code together
        # i.e., if dot(i, j) > limit
        dots = V @ V.T
        # Conflict if dot > limit
        conflict_mask = (dots > limit)
        # Remove self-loops
        np.fill_diagonal(conflict_mask, False)
        
        # Convert to sparse matrix for spectral methods
        A = csr_matrix(conflict_mask)
        
        # Spectral Ordering Initialization
        # The Laplacian L = D - A.
        # The Fiedler vector (eigenvector of second smallest eigenvalue)
        # provides a 1D ordering of the graph that minimizes edge cuts.
        # Sorting vertices by their projection onto this vector tends to
        # cluster connected components or dense subgraphs, which can help
        # greedy algorithms pick a large independent set by processing
        # "dense" areas first or last depending on strategy.
        
        try:
            L = laplacian(A, normed=False)
            # Find eigenvector for lambda_1 (second smallest)
            # For large N, eigsh is efficient.
            # N=5280 for W=4.
            if N > 100:
                # k=2 to get lambda_0 (0) and lambda_1
                # sigma=0 to find smallest eigenvalues
                vals, vecs = eigsh(L, k=2, sigma=0, which='LM')
                # eigsh returns eigenvalues sorted by magnitude if using sigma?
                # Actually, with sigma=0 and which='LM' (largest magnitude of shifted), 
                # it finds eigenvalues closest to 0.
                # We need the second smallest.
                # Let's just get the smallest 2.
                idx = np.argsort(vals)
                fiedler_vec = vecs[:, idx[1]]
            else:
                vals, vecs = np.linalg.eigh(L.toarray())
                idx = np.argsort(vals)
                fiedler_vec = vecs[:, idx[1]]
                
            # Sort vertices by fiedler vector value
            order = np.argsort(fiedler_vec)
            
        except Exception:
            # Fallback to random or natural order
            order = np.random.permutation(N)
            
        # Greedy selection using spectral order
        # We maintain an 'active' set.
        # We iterate through 'order'.
        # If a vertex is active, we select it and deactivate all its neighbors.
        
        active = np.ones(N, dtype=bool)
        selected_indices = []
        
        # Precompute neighbor lists for faster updates?
        # With N=5280, boolean array operations are very fast in NumPy.
        # active &= ~A[order[i]].toarray().ravel() would be slow if done in python loop with dense conversion.
        # Let's stick to the dense boolean mask approach which was in the seed, 
        # but use the spectral order instead of random.
        
        # To make it faster, we can keep the conflict mask as a numpy boolean array
        # since N is small enough (5280x5280 ~ 25M bools ~ 25MB)
        conflict_bool = conflict_mask
        
        for idx in order:
            if time.time() > deadline - 5:
                break
            if active[idx]:
                selected_indices.append(idx)
                # Deactivate neighbors of idx
                # conflict_bool[idx] is True for neighbors that conflict
                active[conflict_bool[idx]] = False
                active[idx] = False # Mark self as inactive too
                
        current_count = len(selected_indices)
        
        # Local Search / Repair Phase
        # The greedy set is an independent set by construction (we removed neighbors).
        # However, we might have missed vertices that could have been added 
        # if we had chosen different neighbors earlier.
        # A common improvement is "Local Search for Independent Set":
        # 1. Try to add any vertex not in the set that has no neighbors in the set.
        # 2. If stuck, try swapping: remove a vertex v from set, add a set of vertices 
        #    that were blocked only by v.
        
        # Let's implement a simple "Add if possible" pass first, just in case 
        # (though greedy should have left no such vertices if it ran to completion on 'active').
        # Actually, the greedy algorithm deactivates neighbors. So any active vertex 
        # remaining is NOT a neighbor of any selected vertex. 
        # Wait, the loop iterates 'order'. If a vertex is active when we reach it, we pick it.
        # So at the end, all active vertices were skipped because they were deactivated 
        # by a previously selected vertex. 
        # Thus, the current set is maximal with respect to the order.
        
        # Can we improve?
        # Strategy: Randomized Local Search (Hill Climbing)
        # We can try to "swap" vertices.
        # But given the time constraint and that spectral ordering is usually quite good,
        # let's try a few random restarts of the greedy phase using the spectral order 
        # as a guide, but with slight perturbations, or just multiple random permutations 
        # if the spectral one wasn't best.
        
        # Actually, the seed used 500 random restarts. 
        # Spectral ordering is deterministic. 
        # Let's combine: Use spectral order for the first few, then random.
        
        best_local_indices = np.array(selected_indices, dtype=int)
        best_local_count = current_count
        
        # Try a few random restarts to see if we can beat the spectral result
        # The spectral result is a strong baseline.
        # Let's do 20 random restarts.
        
        for r in range(20):
            if time.time() > deadline - 5:
                break
                
            perm = np.random.permutation(N)
            active = np.ones(N, dtype=bool)
            sel = []
            
            for idx in perm:
                if active[idx]:
                    sel.append(idx)
                    active[conflict_bool[idx]] = False
                    active[idx] = False
                    
            if len(sel) > best_local_count:
                best_local_count = len(sel)
                best_local_indices = np.array(sel, dtype=int)
                
        if best_local_count > best_count:
            best_count = best_local_count
            best_vectors = V[best_local_indices]
            
    # Fallback
    if best_vectors is None or len(best_vectors) == 0:
        V2 = generate_vectors(2)
        if len(V2) > 0:
            best_vectors = V2
            best_count = len(V2)

    return best_vectors

# EVOLVE-BLOCK-END
