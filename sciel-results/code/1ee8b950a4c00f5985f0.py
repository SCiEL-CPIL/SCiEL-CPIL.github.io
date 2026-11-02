"""Spectral / Max-Flow style independent set search for kissing number in dimension 11.

Structural change from the previous greedy+swap program:
- Represent the problem as a maximum independent set on a conflict graph.
- Use a spectral relaxation (Laplacian eigenvector) to seed the independent set,
  then refine with a fast local search that supports 1-for-2, 1-for-3, and 2-for-3 moves.
- Exploit the symmetry of the candidate set (weight-2 and weight-4 integer vectors)
  by working on the full candidate pool but using vectorized NumPy operations for
  all pairwise checks.
- Multiple randomized restarts with different spectral seeds to escape local optima.
"""
import numpy as np
import itertools
import time


def _build_candidates(D=11):
    """Build the candidate pool: all weight-2 and weight-4 integer vectors in R^D."""
    cands = []
    # Weight 2 vectors: norm^2 = 2
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            cands.append(v)
    # Weight 4 vectors: norm^2 = 4
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            cands.append(v)
    return np.array(cands, dtype=np.int8)


def _conflict_matrix(candidates, max_norm_sq):
    """Build the conflict matrix. conflict[i,j]=1 if v_i and v_j cannot coexist."""
    c = candidates.astype(np.int32)
    norms_sq = np.sum(c ** 2, axis=1)
    dot = c @ c.T
    # Constraint: ||v_i - v_j||^2 >= max_norm_sq
    # => 2 * dot[i,j] <= norms_sq[i] + norms_sq[j] - max_norm_sq
    rhs = norms_sq[:, np.newaxis] + norms_sq[np.newaxis, :] - max_norm_sq
    conflict = (2 * dot > rhs).astype(np.uint8)
    np.fill_diagonal(conflict, 0)
    return conflict, norms_sq


def _spectral_seed(conflict, n, rng):
    """Use the Laplacian of the conflict graph to get an initial independent set seed."""
    # Laplacian L = D - A
    degree = conflict.sum(axis=1).astype(np.float64)
    # Build L as a sparse-like operation (but n is small enough for dense)
    L = np.diag(degree) - conflict.astype(np.float64)
    
    # Find the smallest non-trivial eigenvector of L
    # For large n, use scipy.sparse if available, else numpy
    try:
        from scipy.sparse import diags, csr_matrix
        from scipy.sparse.linalg import eigs
        L_sp = csr_matrix(L)
        # Find smallest eigenvalues
        vals, vecs = eigs(L_sp, k=min(5, n - 1), which='SM')
        # Use the eigenvector corresponding to the smallest non-zero eigenvalue
        # Find the one with smallest absolute value > 0
        nonzero_mask = np.abs(vals) > 1e-10
        if nonzero_mask.any():
            idx = np.where(nonzero_mask)[0]
            min_idx = idx[np.argmin(np.abs(vals[idx]))]
            x = np.real(vecs[:, min_idx])
        else:
            x = rng.standard_normal(n)
    except Exception:
        # Fallback: use numpy
        try:
            vals, vecs = np.linalg.eigh(L)
            # Skip the first (all-ones) eigenvector
            x = vecs[:, 1] if n > 1 else np.array([1.0])
        except Exception:
            x = rng.standard_normal(n)
    
    # Threshold: select vertices with positive x value
    # Then greedily remove conflicts
    selected = np.zeros(n, dtype=np.uint8)
    candidates_sorted = np.argsort(-x)
    remaining = np.ones(n, dtype=bool)
    
    for i in candidates_sorted:
        if not remaining[i]:
            continue
        if selected[i] == 1:
            continue
        # Check if i conflicts with any already selected
        # conflict[i, selected==1]
        if conflict[i].sum(axis=0) == 0 or (conflict[i] & selected).sum() == 0:
            selected[i] = 1
            remaining[i] = False
            remaining[conflict[i].astype(bool)] = False
    
    return selected


def _greedy_fill(conflict, selected, n, rng, time_limit):
    """Greedily add compatible vertices to the selected set."""
    sel_indices = np.where(selected == 1)[0]
    if len(sel_indices) == 0:
        # Just pick a random vertex
        selected[rng.integers(n)] = 1
        sel_indices = np.where(selected == 1)[0]
    
    while True:
        if time.time() > time_limit:
            break
        # Find all unselected vertices compatible with current set
        c_with_S = conflict[:, sel_indices].sum(axis=1).astype(np.int32)
        addable = np.where((selected == 0) & (c_with_S == 0))[0]
        if len(addable) == 0:
            break
        # Pick one with minimum degree in the remaining graph
        degs = conflict[addable].sum(axis=1)
        min_idx = addable[np.argmin(degs)]
        selected[min_idx] = 1
        sel_indices = np.append(sel_indices, min_idx)
    
    return selected


def _local_search(conflict, selected, n, rng, time_limit, max_moves=500):
    """Local search with 1-for-2, 1-for-3, and 2-for-3 moves."""
    improved = True
    move_count = 0
    while improved and move_count < max_moves and time.time() < time_limit:
        improved = False
        move_count += 1
        sel_indices = np.where(selected == 1)[0]
        if len(sel_indices) == 0:
            break
        
        # Shuffle the order of selected vertices
        order = rng.permutation(len(sel_indices))
        
        for i in order:
            if time.time() > time_limit:
                break
            
            v1 = sel_indices[i]
            
            # 1-for-2 swap: remove v1, add two compatible vertices
            c_with_S = conflict[:, sel_indices].sum(axis=1).astype(np.int32)
            c_with_v1 = conflict[:, v1].astype(np.int32)
            c_with_S_no_v1 = c_with_S - c_with_v1
            
            cand_mask = (selected == 0) & (c_with_S_no_v1 == 0)
            cand_indices = np.where(cand_mask)[0]
            
            if len(cand_indices) >= 2:
                # Find a compatible pair among candidates
                # Check the first candidate against others
                found = False
                u = cand_indices[0]
                compat = conflict[u, cand_indices[1:]] == 0
                if compat.any():
                    w = cand_indices[1:][np.argmax(compat)]
                    selected[v1] = 0
                    selected[u] = 1
                    selected[w] = 1
                    improved = True
                    found = True
                    break
                
                # Try a few more random candidates
                if not found and len(cand_indices) > 5:
                    for _ in range(5):
                        u = cand_indices[rng.integers(len(cand_indices))]
                        compat = conflict[u, cand_indices] == 0
                        compat[u] = False  # exclude self
                        if compat.any():
                            w = cand_indices[np.argmax(compat)]
                            if u != v1 and w != v1:
                                selected[v1] = 0
                                selected[u] = 1
                                selected[w] = 1
                                improved = True
                                found = True
                                break
            
            # 1-for-3 swap: remove v1, add three compatible vertices
            if not improved:
                if len(cand_indices) >= 3:
                    # Try to find three mutually compatible vertices
                    # This is more expensive, so only try a few
                    found3 = False
                    for _ in range(3):
                        u = cand_indices[rng.integers(len(cand_indices))]
                        compat_u = conflict[u, cand_indices] == 0
                        compat_u[u] = False
                        if compat_u.sum() < 2:
                            continue
                        v2 = cand_indices[np.argmax(compat_u)]
                        compat_v2 = conflict[v2, cand_indices] == 0
                        compat_v2[u] = False
                        compat_v2[v2] = False
                        if compat_v2.any():
                            v3 = cand_indices[np.argmax(compat_v2)]
                            if u != v1 and v2 != v1 and v3 != v1:
                                selected[v1] = 0
                                selected[u] = 1
                                selected[v2] = 1
                                selected[v3] = 1
                                improved = True
                                found3 = True
                                break
                    if found3:
                        break
        
        # 2-for-3 swap: remove two selected, add three
        if not improved and len(sel_indices) >= 2:
            # Try a few random pairs of selected vertices
            for _ in range(3):
                if time.time() > time_limit:
                    break
                i1 = rng.integers(len(sel_indices))
                i2 = rng.integers(len(sel_indices))
                if i1 == i2:
                    continue
                v1 = sel_indices[i1]
                v2 = sel_indices[i2]
                
                c_with_S = conflict[:, sel_indices].sum(axis=1).astype(np.int32)
                c_with_v1 = conflict[:, v1].astype(np.int32)
                c_with_v2 = conflict[:, v2].astype(np.int32)
                c_with_S_no_v1v2 = c_with_S - c_with_v1 - c_with_v2
                
                cand_mask = (selected == 0) & (c_with_S_no_v1v2 == 0)
                cand_indices = np.where(cand_mask)[0]
                
                if len(cand_indices) >= 3:
                    # Try to find three mutually compatible vertices
                    found3 = False
                    for _ in range(5):
                        u = cand_indices[rng.integers(len(cand_indices))]
                        compat_u = conflict[u, cand_indices] == 0
                        compat_u[u] = False
                        if compat_u.sum() < 2:
                            continue
                        v3 = cand_indices[np.argmax(compat_u)]
                        compat_v3 = conflict[v3, cand_indices] == 0
                        compat_v3[u] = False
                        compat_v3[v3] = False
                        if compat_v3.any():
                            v4 = cand_indices[np.argmax(compat_v3)]
                            if u != v1 and u != v2 and v3 != v1 and v3 != v2 and v4 != v1 and v4 != v2:
                                selected[v1] = 0
                                selected[v2] = 0
                                selected[u] = 1
                                selected[v3] = 1
                                selected[v4] = 1
                                improved = True
                                found3 = True
                                break
                    if found3:
                        break
    
    return selected


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    D = 11
    rng = np.random.default_rng(42)
    
    # Build candidates
    candidates = _build_candidates(D)
    n = len(candidates)
    
    # Max norm squared is 4 (from weight-4 vectors)
    max_norm_sq = 4
    
    # Build conflict matrix
    conflict, norms_sq = _conflict_matrix(candidates, max_norm_sq)
    
    best_selected = np.zeros(n, dtype=np.uint8)
    best_count = 0
    
    # Multiple restarts with different spectral seeds
    num_restarts = 5
    time_per_restart = (deadline - start_time - 10.0) / num_restarts
    
    for restart in range(num_restarts):
        restart_start = time.time()
        restart_deadline = restart_start + min(time_per_restart, deadline - restart_start - 5.0)
        
        # Spectral seed
        selected = _spectral_seed(conflict, n, rng)
        
        # Greedy fill
        selected = _greedy_fill(conflict, selected, n, rng, restart_deadline - 5.0)
        
        # Local search
        selected = _local_search(conflict, selected, n, rng, restart_deadline - 2.0, max_moves=200)
        
        # Final greedy fill
        selected = _greedy_fill(conflict, selected, n, rng, restart_deadline - 1.0)
        
        count = selected.sum()
        if count > best_count:
            best_count = count
            best_selected = selected.copy()
        
        if time.time() > deadline - 15.0:
            break
    
    # Final greedy fill on best
    sel_indices = np.where(best_selected == 1)[0]
    if time.time() < deadline - 3.0:
        while True:
            if time.time() > deadline - 3.0:
                break
            c_with_S = conflict[:, sel_indices].sum(axis=1).astype(np.int32)
            addable = np.where((best_selected == 0) & (c_with_S == 0))[0]
            if len(addable) == 0:
                break
            degs = conflict[addable].sum(axis=1)
            min_idx = addable[np.argmin(degs)]
            best_selected[min_idx] = 1
            sel_indices = np.append(sel_indices, min_idx)
    
    # Extract result
    sel_indices = np.where(best_selected == 1)[0]
    if len(sel_indices) == 0:
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)
    
    result = candidates[sel_indices.astype(np.int64)]
    return result.astype(np.int64)
