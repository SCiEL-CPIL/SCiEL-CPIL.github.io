"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import itertools
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit (or 150s)
    
    # Generate all nonzero integer vectors with entries in {-1, 0, 1}
    # We focus on weights 1-4 which gives a manageable candidate pool
    # Weight 1: 22, Weight 2: 220, Weight 3: 1320, Weight 4: 5280
    # Total: 6842 candidates
    
    candidates = []
    
    # Weight 1
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
            
    # Weight 2
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                candidates.append(v)
                
    # Weight 3
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i], v[j], v[k] = si, sj, sk
                    candidates.append(v)
                    
    # Weight 4
    for i, j, k, l in itertools.combinations(range(D), 4):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i], v[j], v[k], v[l] = si, sj, sk, sl
                        candidates.append(v)
                        
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    norms_sq = np.sum(candidates**2, axis=1)
    
    # Compute dot products matrix
    dots = np.dot(candidates, candidates.T)
    
    # For a given set of vectors, the constraint is:
    # For all distinct i, j in selected set: ||v_i - v_j||^2 >= max_norm_sq
    # where max_norm_sq is the maximum squared norm in the selected set.
    # dist_sq[i,j] = norms_sq[i] + norms_sq[j] - 2*dots[i,j]
    
    # Strategy:
    # 1. Use Spectral Relaxation on the "conflict graph" for a specific max_norm target M.
    #    Conflict edge (i,j) exists if dist_sq[i,j] < M.
    #    We want a large independent set in this graph.
    #    The leading eigenvector of the Laplacian (or adjacency) can guide partitioning.
    #    Specifically, for Max-Cut on the conflict graph, the cut defines two sets.
    #    But we want an Independent Set. 
    #    A common spectral heuristic for Independent Set:
    #    Compute the smallest non-trivial eigenvector of the Laplacian (Fiedler vector).
    #    Partition vertices based on the sign of this vector.
    #    One of the partitions is likely to be a large independent set (or close to one).
    #    Then refine with local search.
    
    # However, the conflict graph depends on M.
    # Let's try a few candidate M values (max norm squared).
    # Possible max norms: 1, 2, 3, 4.
    # If M=4, we can include weight 4 vectors.
    # If M=3, we can include up to weight 3.
    # Generally, larger M allows more vectors but stricter distance constraint.
    
    best_selected = []
    best_score = 0
    
    # We will iterate over possible M values
    # For each M, build the conflict graph and apply spectral + greedy refinement.
    
    distinct_norms = np.unique(norms_sq)
    
    for M in distinct_norms:
        if time.time() > deadline - 10.0:
            break
            
        M_int = int(M)
        
        # Only consider candidates with norm_sq <= M
        mask = norms_sq <= M
        cand_indices = np.where(mask)[0]
        n_subset = len(cand_indices)
        
        if n_subset == 0:
            continue
            
        # Extract submatrix for this subset
        sub_dots = dots[np.ix_(cand_indices, cand_indices)]
        sub_norms = norms_sq[cand_indices]
        
        # Compute dist_sq for this subset
        dist_sq_sub = sub_norms[:, None] + sub_norms[None, :] - 2 * sub_dots
        np.fill_diagonal(dist_sq_sub, 0)
        
        # Conflict if dist_sq < M
        conflicts = dist_sq_sub < M_int
        
        # Build adjacency matrix for spectral methods
        # We want to find a large independent set.
        # Let's use the Laplacian of the conflict graph.
        # L = D - A
        # A is the conflict adjacency matrix.
        
        # Convert to sparse for efficiency
        A = csr_matrix(conflicts.astype(np.float64))
        degrees = np.array(A.sum(axis=1)).flatten()
        D_mat = csr_matrix(np.diag(degrees))
        L = D_mat - A
        
        # Find smallest eigenvalues
        # k=1 is trivial (0). We want the next few.
        k_eig = min(20, n_subset - 1)
        if k_eig < 2:
            # Too small, just greedy
            selected_idx_local = _greedy_select(candidates[cand_indices], norms_sq[cand_indices], M_int, start_time, deadline)
            if len(selected_idx_local) > best_score:
                best_score = len(selected_idx_local)
                best_selected = [cand_indices[i] for i in selected_idx_local]
            continue
            
        try:
            # which='SM' finds smallest magnitude
            # We need to skip the zero eigenvalue.
            # eigsh with which='SM' might return 0 first.
            # Let's get k+1 eigenvectors to be safe.
            evals, evecs = eigsh(L, k=k_eig, which='SM')
            
            # Sort by eigenvalue
            idx_sort = np.argsort(evals)
            evals = evals[idx_sort]
            evecs = evecs[:, idx_sort]
            
            # Try each non-trivial eigenvector
            for eig_idx in range(1, k_eig):
                if time.time() > deadline - 5.0:
                    break
                    
                vec = evecs[:, eig_idx]
                
                # Partition based on sign
                # Positive side
                pos_mask = vec > 0
                neg_mask = vec < 0
                
                # Try both sides
                for side_mask in [pos_mask, neg_mask]:
                    if time.time() > deadline - 5.0:
                        continue
                        
                    # Get indices in this side
                    side_indices_local = np.where(side_mask)[0]
                    
                    # This side is a candidate for independent set, but might have internal conflicts.
                    # Apply greedy refinement to remove conflicts.
                    # We want to select a subset of side_indices_local that is an independent set.
                    # Greedy: iterate through side_indices_local, add if no conflict with already added.
                    
                    # To speed up, we can check conflicts within the subset.
                    # But doing it one by one is O(N^2).
                    # Let's just do a simple greedy pass.
                    
                    selected_local = []
                    # Order by norm descending? Or random?
                    # Let's try norm descending first (heavier vectors are harder to place, maybe prioritize them?)
                    # Or norm ascending?
                    # Let's try a few orderings.
                    
                    orderings = [
                        np.argsort(-sub_norms[side_indices_local]), # Norm desc
                        np.argsort(sub_norms[side_indices_local]),  # Norm asc
                        np.random.permutation(len(side_indices_local)) # Random
                    ]
                    
                    for order in orderings:
                        if time.time() > deadline - 5.0:
                            continue
                            
                        sel = []
                        selected_vecs = []
                        
                        for i in order:
                            orig_idx_in_side = side_indices_local[i]
                            v = candidates[cand_indices[orig_idx_in_side]]
                            
                            if not selected_vecs:
                                sel.append(orig_idx_in_side)
                                selected_vecs.append(v)
                            else:
                                # Check distance to all selected
                                # dist_sq = ||v - u||^2
                                # We need dist_sq >= M_int
                                # Since we are in the same side of the spectral cut, they *might* be compatible.
                                # But we must verify.
                                
                                diffs = v - np.array(selected_vecs)
                                dist_sqs = np.sum(diffs**2, axis=1)
                                
                                if np.min(dist_sqs) >= M_int:
                                    sel.append(orig_idx_in_side)
                                    selected_vecs.append(v)
                        
                        if len(sel) > best_score:
                            best_score = len(sel)
                            best_selected = [cand_indices[i] for i in sel]
                            
        except Exception:
            pass
            
    # If spectral didn't find anything good, fallback to pure greedy on the full set
    # with M=4 (max possible for our candidates)
    if best_score == 0:
        # Try M=4 greedy
        mask = norms_sq <= 4
        cand_indices = np.where(mask)[0]
        selected_idx_local = _greedy_select(candidates[cand_indices], norms_sq[cand_indices], 4, start_time, deadline)
        if len(selected_idx_local) > best_score:
            best_score = len(selected_idx_local)
            best_selected = [cand_indices[i] for i in selected_idx_local]
            
    if not best_selected:
        # Fallback: return a single vector
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        return np.array([v], dtype=np.int64)
        
    return candidates[best_selected]

def _greedy_select(candidates_subset, norms_subset, M, start_time, deadline):
    """
    Greedy selection from candidates_subset with max norm constraint M.
    Returns indices into candidates_subset.
    """
    n = len(candidates_subset)
    if n == 0:
        return []
        
    # Precompute dots for subset? Might be expensive if n is large.
    # n is at most 6842. 6842^2 is ~47M. Dot product is fast.
    dots = np.dot(candidates_subset, candidates_subset.T)
    dist_sq = norms_subset[:, None] + norms_subset[None, :] - 2 * dots
    np.fill_diagonal(dist_sq, 0)
    
    available = set(range(n))
    selected = []
    
    # Order by norm descending to pack heavy vectors first?
    # Or random?
    # Let's try norm descending.
    order = np.argsort(-norms_subset)
    
    for idx in order:
        if time.time() > deadline:
            break
        if idx not in available:
            continue
            
        # Check if idx conflicts with any already selected
        # We can maintain a "blocked" set or just check against selected.
        # Checking against selected is O(k) per candidate, total O(n*k).
        # n=6842, k could be ~1000. 6M ops. Fast.
        
        compatible = True
        for sel_idx in selected:
            if dist_sq[idx, sel_idx] < M:
                compatible = False
                break
                
        if compatible:
            selected.append(idx)
            # Remove neighbors from available to speed up future checks?
            # Neighbors are those j where dist_sq[idx, j] < M.
            # We can just update available.
            # But 'selected' list is enough for verification.
            # Updating available helps skip candidates early.
            # Let's do it.
            neighbors = np.where(dist_sq[idx, :] < M)[0]
            for nb in neighbors:
                if nb in available:
                    available.discard(nb)
            available.discard(idx)
            
    return selected

# EVOLVE-BLOCK-END
