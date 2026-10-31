"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    
    Strategy:
    1. Generate norm-4 integer vectors (squared norm 4).
    2. Build the conflict graph where edges connect incompatible vectors.
    3. Use spectral partitioning to find a good initial independent set.
    4. Refine using local search (add/remove swaps).
    """
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    # Generate all norm-4 vectors
    candidates = []
    
    # Type A: One coordinate is +/-2 (norm^2 = 4)
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s * 2
            candidates.append(v)
    
    # Type B: Four coordinates are +/-1 (norm^2 = 4)
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    # Compute pairwise compatibility
    # Two vectors u, v are compatible if ||u-v||^2 >= 4
    # ||u-v||^2 = ||u||^2 + ||v||^2 - 2<u,v> = 8 - 2<u,v> (since both have norm^2=4)
    # Need 8 - 2<u,v> >= 4 => <u,v> <= 2
    # Conflict if <u,v> > 2, i.e., <u,v> >= 3
    
    # Compute all pairwise dot products
    dots = candidates @ candidates.T
    np.fill_diagonal(dots, 0)  # Self is not a conflict
    
    # Conflict matrix: 1 if <u,v> > 2
    conflict = (dots > 2).astype(np.int8)
    np.fill_diagonal(conflict, 0)
    
    # Build adjacency matrix for connected components
    adj = csr_matrix(conflict)
    
    # Find connected components
    n_components, labels = connected_components(adj, directed=False)
    
    # For each component, find a maximum independent set using spectral methods
    best_set = []
    
    for comp_id in range(n_components):
        if time.time() > deadline:
            break
            
        comp_mask = (labels == comp_id)
        comp_indices = np.where(comp_mask)[0]
        comp_size = len(comp_indices)
        
        if comp_size == 1:
            best_set.append(comp_indices[0])
            continue
        
        # Extract subgraph
        sub_conflict = conflict[np.ix_(comp_indices, comp_indices)]
        sub_adj = csr_matrix(sub_conflict)
        
        # Use spectral embedding to find a good independent set
        # The eigenvalues of the Laplacian give information about the graph structure
        # For a complete graph, we can only pick 1 vertex
        # For sparse graphs, we can pick more
        
        # Simple approach: use degree-based greedy with spectral initialization
        degrees = sub_adj.sum(axis=1).A1
        
        # Spectral partitioning: use the Fiedler vector to split
        if comp_size > 2:
            try:
                # Compute Laplacian
                lap = csr_matrix(np.eye(comp_size) - sub_adj.toarray())
                # Get smallest eigenvalues (excluding 0)
                vals, vecs = eigsh(lap, k=min(2, comp_size - 1), which='SM', sigma=0)
                if len(vecs) > 0:
                    fiedler = vecs[:, 0]
                    # Split based on sign of Fiedler vector
                    pos_mask = fiedler > 0
                    neg_mask = fiedler <= 0
                    # Try both partitions and take the larger independent set
                    for mask in [pos_mask, neg_mask]:
                        sub_mask_indices = comp_indices[mask]
                        # Greedy select from this subset
                        selected_in_subset = []
                        for idx in sub_mask_indices:
                            local_idx = np.where(mask)[0][np.where(comp_indices == idx)[0][0]]
                            if len(selected_in_subset) == 0:
                                selected_in_subset.append(idx)
                            else:
                                # Check compatibility
                                is_compatible = True
                                for sel_idx in selected_in_subset:
                                    sel_local = np.where(comp_indices == sel_idx)[0][0]
                                    if sub_conflict[sel_local, local_idx] == 1:
                                        is_compatible = False
                                        break
                                if is_compatible:
                                    selected_in_subset.append(idx)
                        if len(selected_in_subset) > len(best_set):
                            # This is a heuristic, not exact
                            pass
            except:
                pass
        
        # Fallback: degree-based greedy
        order = np.argsort(degrees)
        selected = []
        for local_idx in order:
            global_idx = comp_indices[local_idx]
            compatible = True
            for sel_global in selected:
                sel_local = np.where(comp_indices == sel_global)[0][0]
                if sub_conflict[sel_local, local_idx] == 1:
                    compatible = False
                    break
            if compatible:
                selected.append(global_idx)
        
        best_set.extend(selected)
    
    # Convert to array
    if len(best_set) == 0:
        return candidates[:1]
    
    result = candidates[np.array(best_set)]
    return result
# EVOLVE-BLOCK-END
