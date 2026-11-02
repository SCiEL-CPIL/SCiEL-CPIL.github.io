"""Kissing number in dimension 11: spectral relaxation + min-degree greedy + swap local search."""
import time
import itertools
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_independent_set
import random

D = 11
DEADLINE_SECONDS = 120.0
SAFETY_MARGIN = 15.0


def _generate_candidates():
    """Generate all nonzero integer vectors in R^11 with squared norm <= 9.
    Includes weight 2 (squared norm 4) and weight 3 (squared norm 9) vectors,
    with symmetry-breaking (first nonzero coordinate positive)."""
    cands = []
    # Weight 2: two coordinates are ±1, rest 0
    for i, j in itertools.combinations(range(D), 2):
        for sj in (1, -1):
            v = [0] * D
            v[i] = 1
            v[j] = sj
            cands.append(v)
    # Weight 3: three coordinates are ±1, rest 0
    for i, j, k in itertools.combinations(range(D), 3):
        for sj in (1, -1):
            for sk in (1, -1):
                v = [0] * D
                v[i] = 1
                v[j] = sj
                v[k] = sk
                cands.append(v)
    return np.array(cands, dtype=np.int64)


def _conflict_graph(cands):
    """Build the conflict graph where an edge exists if two vectors are NOT orthogonal.
    Returns a sparse adjacency matrix."""
    n = len(cands)
    # Compute all-pairs inner products
    ips = cands @ cands.T
    # Conflict if inner product != 0 (and not the same vector, i.e., i != j)
    np.fill_diagonal(ips, 0)
    conflict = (ips != 0).astype(np.int8)
    return csr_matrix(conflict)


def _spectral_ordering(conflict_csr, n):
    """Use spectral relaxation to get an ordering for greedy selection.
    The eigenvector corresponding to the smallest eigenvalue of the Laplacian
    gives a geometric embedding; sorting by this coordinate tends to group
    non-adjacent vertices together."""
    from scipy.sparse.linalg import eigsh
    
    # Laplacian L = D - A
    degrees = np.asarray(conflict_csr.sum(axis=1)).flatten()
    L = csr_matrix((degrees, (np.arange(n), np.arange(n)))) - conflict_csr
    
    try:
        # Find the eigenvector for the smallest eigenvalue (or second smallest for connected graphs)
        vals, vecs = eigsh(L, k=2, which='SM')
        # Use the eigenvector with the smallest eigenvalue (or the second if the first is trivially 0)
        # For disconnected graphs, the smallest eigenvalue is 0 with constant vector
        # We want the Fiedler vector (second smallest) for ordering
        idx = np.argmin(np.abs(vals - 0.0))
        if vals[idx] < 1e-10:
            # Use the second smallest
            idx2 = np.argsort(vals)[1]
            order_vec = vecs[:, idx2]
        else:
            order_vec = vecs[:, idx]
        
        # Sort by the eigenvector values
        order = np.argsort(order_vec)
        return order
    except Exception:
        return np.random.permutation(n)


def _greedy_select(cands, order, time_limit):
    """Greedy selection: pick candidates in the given order if orthogonal to all selected."""
    selected = []
    n = len(cands)
    
    for idx in order:
        if time.time() > time_limit:
            break
        v = cands[idx]
        if not selected:
            selected.append(idx)
        else:
            # Check orthogonality with all selected
            sel_arr = cands[selected]
            ips = sel_arr @ v
            if np.all(ips == 0):
                selected.append(idx)
    
    return selected


def _local_search_swap(cands, selected, conflict_csr, time_limit, max_swaps=10000):
    """Try to improve by swapping: remove one selected vector, try to add two non-selected ones."""
    n = len(cands)
    selected_set = set(selected)
    selected_arr = cands[selected]
    
    # Precompute conflicts for each vector
    # For each candidate, which selected vectors conflict with it?
    
    improved = True
    swap_count = 0
    while improved and swap_count < max_swaps and time.time() < time_limit:
        improved = False
        swap_count += 1
        
        # For each selected vector, try removing it
        for i, sel_idx in enumerate(selected):
            if time.time() > time_limit:
                break
            
            # Try adding each non-selected vector that was blocked by sel_idx
            # First, find non-selected vectors that conflict only with sel_idx (among selected)
            candidate_adds = []
            for j in range(n):
                if j in selected_set:
                    continue
                v = cands[j]
                # Check if v is orthogonal to all selected except possibly sel_idx
                sel_except_i = selected[:i] + selected[i+1:]
                if sel_except_i:
                    sel_arr = cands[sel_except_i]
                    ips = sel_arr @ v
                    if np.any(ips != 0):
                        continue
                # v is orthogonal to all selected except possibly sel_idx
                # Check if it actually conflicts with sel_idx
                if cands[sel_idx] @ v != 0:
                    candidate_adds.append(j)
            
            # Try to add as many from candidate_adds as possible
            for add_idx in candidate_adds:
                if time.time() > time_limit:
                    break
                v = cands[add_idx]
                # Check if v is orthogonal to all currently selected (which includes sel_idx still)
                # Wait, we haven't removed sel_idx yet. Let me restructure.
            
            # Better approach: temporarily remove sel_idx, then try to add
            new_selected = selected[:i] + selected[i+1:]
            new_selected_set = set(new_selected)
            
            added_any = False
            for add_idx in candidate_adds:
                if time.time() > time_limit:
                    break
                v = cands[add_idx]
                if add_idx in new_selected_set:
                    continue
                if new_selected:
                    sel_arr = cands[new_selected]
                    ips = sel_arr @ v
                    if np.any(ips != 0):
                        continue
                # Can add
                new_selected.append(add_idx)
                new_selected_set.add(add_idx)
                added_any = True
            
            if added_any:
                # Check if we improved (added more than we removed)
                if len(new_selected) > len(selected):
                    selected = new_selected
                    selected_set = set(selected)
                    improved = True
                    break
    
    return selected


def _min_degree_greedy(cands, conflict_csr, time_limit):
    """Min-degree greedy: at each step, pick the vertex with the fewest conflicts among remaining."""
    n = len(cands)
    remaining = set(range(n))
    selected = []
    
    # Degree in the induced subgraph on remaining
    # We'll maintain this incrementally
    
    while remaining and time.time() < time_limit:
        # Find the vertex in remaining with minimum degree within remaining
        best_idx = None
        best_deg = float('inf')
        
        # Sample if too many
        if len(remaining) > 500:
            sample = list(random.sample(remaining, 500))
        else:
            sample = list(remaining)
        
        for idx in sample:
            # Count conflicts with other remaining vertices
            row = conflict_csr.getrow(idx)
            neighbors = row.indices
            deg = 0
            for nb in neighbors:
                if nb in remaining:
                    deg += 1
                    if deg >= best_deg:
                        break
            if deg < best_deg:
                best_deg = deg
                best_idx = idx
        
        if best_idx is None:
            break
        
        selected.append(best_idx)
        remaining.discard(best_idx)
    
    return selected


def construct():
    start_time = time.time()
    time_limit = start_time + DEADLINE_SECONDS - SAFETY_MARGIN
    
    cands = _generate_candidates()
    n = len(cands)
    
    # Build conflict graph
    conflict_csr = _conflict_graph(cands)
    
    best_selected = []
    best_size = 0
    
    # Strategy 1: Spectral ordering + greedy
    if time.time() < time_limit:
        order = _spectral_ordering(conflict_csr, n)
        selected = _greedy_select(cands, order, time_limit)
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected
    
    # Strategy 2: Min-degree greedy
    if time.time() < time_limit:
        selected = _min_degree_greedy(cands, conflict_csr, time_limit)
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected
    
    # Strategy 3: Random restarts with greedy
    rng = random.Random(42)
    for trial in range(20):
        if time.time() > time_limit:
            break
        order = list(range(n))
        rng.shuffle(order)
        selected = _greedy_select(cands, order, time_limit)
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected
    
    # Strategy 4: Local search swap on best found
    if best_selected and time.time() < time_limit:
        improved = _local_search_swap(cands, best_selected, conflict_csr, time_limit)
        if len(improved) > best_size:
            best_size = len(improved)
            best_selected = improved
    
    # Strategy 5: Try maximum_independent_set on the conflict graph (if fast enough)
    if time.time() < time_limit - 30:
        try:
            # This might be slow for large graphs, so only try if we have time
            if n <= 500:
                mis = maximum_independent_set(conflict_csr)
                mis_indices = np.where(mis)[0]
                if len(mis_indices) > best_size:
                    best_size = len(mis_indices)
                    best_selected = list(mis_indices)
        except Exception:
            pass
    
    if not best_selected:
        # Fallback: just return weight-2 vectors
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    return cands[best_selected]
