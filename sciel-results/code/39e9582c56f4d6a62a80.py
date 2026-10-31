import itertools
import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s limit
    
    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
    
    # Compute conflict graph: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Spectral initialization: use Fiedler vector of the conflict graph Laplacian
    degree = conflicts.sum(axis=1).astype(np.float64)
    L_dense = np.diag(degree) - conflicts.astype(np.float64)
    
    try:
        L_sparse = csr_matrix(L_dense)
        eigvals, eigvecs = eigsh(L_sparse, k=3, sigma=0, which='LM')
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        fiedler = eigvecs[:, 1]
    except Exception:
        rng = np.random.default_rng(42)
        fiedler = rng.standard_normal(N4)
    
    # Assign vertices to two sets based on sign of Fiedler vector
    # We want to select a maximum independent set, which is related to max-cut
    # For now, use this to guide a greedy selection
    
    # Greedy with spectral ordering
    # Order vertices by |fiedler[i]| descending, then by degree ascending
    abs_fiedler = np.abs(fiedler)
    deg_order = np.argsort(conflicts.sum(axis=1))
    
    # Combine: primary sort by abs_fiedler desc, secondary by degree asc
    # This is a heuristic ordering
    
    best_selected = np.zeros(N4, dtype=bool)
    best_count = 0
    
    # Multiple strategies:
    # 1. Pure spectral greedy
    # 2. Random restarts with degree-based greedy
    # 3. Local search refinement
    
    def greedy_select(order):
        selected = np.zeros(N4, dtype=bool)
        count = 0
        for idx in order:
            if time.time() > deadline - 5:
                break
            if not np.any(conflicts[idx, selected]):
                selected[idx] = True
                count += 1
        return selected, count
    
    # Strategy 1: Spectral ordering
    order_spec = np.argsort(-abs_fiedler)
    sel_spec, cnt_spec = greedy_select(order_spec)
    if cnt_spec > best_count:
        best_count = cnt_spec
        best_selected = sel_spec.copy()
    
    # Strategy 2: Random restarts with degree-based greedy
    rng = np.random.default_rng(42)
    n_restarts = 50
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        # Random perturbation of the order
        order = rng.permutation(N4)
        sel, cnt = greedy_select(order)
        
        if cnt > best_count:
            best_count = cnt
            best_selected = sel.copy()
    
    # Strategy 3: Local search refinement on the best solution
    # Try to add vertices that don't conflict, or swap to improve
    selected = best_selected.copy()
    count = best_count
    
    # Find all non-selected vertices that can be added
    for _ in range(5):  # Limit iterations for time
        if time.time() > deadline - 5:
            break
        
        # For each non-selected vertex, check if it can be added
        non_selected = np.where(~selected)[0]
        improved = False
        
        for idx in non_selected:
            if time.time() > deadline - 5:
                break
            if not np.any(conflicts[idx, selected]):
                selected[idx] = True
                count += 1
                improved = True
        
        if not improved:
            break
    
    if count > best_count:
        best_count = count
        best_selected = selected.copy()
    
    # Construct result
    result_v4 = vectors_v4[best_selected]
    
    # Add all V1 vectors (they're all compatible with V4 and each other)
    # Check: V1 vectors have norm 2, V4 vectors have norm 2.
    # Dot product between V1 (e.g., [2,0,...]) and V4 (e.g., [1,1,1,1,0,...]) is at most 2.
    # Conflict if dot > 2. So dot <= 2 is fine.
    # Max dot between V1 and V4: if V1 has 2 at position i, and V4 has 1 at position i, dot = 2.
    # If V4 has -1 at position i, dot = -2.
    # So no conflict between V1 and V4.
    # V1 vectors among themselves: [2,0,...] . [2,0,...] = 4 > 2. Conflict!
    # Wait, V1 vectors: [2,0,...] and [2,0,...] are the same? No, different signs or positions.
    # [2,0,...] and [-2,0,...] dot = -4.
    # [2,0,...] and [0,2,...] dot = 0.
    # So V1 vectors only conflict if they are identical? No, they are distinct vectors.
    # Actually, the set of V1 vectors is {2*e_i, -2*e_i for i=0..10}.
    # Dot between 2*e_i and 2*e_i is 4. But we only include each vector once.
    # Dot between 2*e_i and -2*e_i is -4.
    # Dot between 2*e_i and 2*e_j (i!=j) is 0.
    # So no two distinct V1 vectors conflict.
    
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors.astype(np.int64)
# EVOLVE-BLOCK-END
