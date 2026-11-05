"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    d = 7
    n = 2 ** d
    
    # Precompute all edges
    all_edges = []
    edge_to_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y))
                edge_to_idx[(x, y)] = idx
    
    m = len(all_edges)  # 448 edges
    
    # Build conflict graph: edge i conflicts with edge j if they are parallel in some 4-face
    # Two edges are parallel in a 4-face if they are the "opposite" edges of that face
    # Edge (u,v) with direction i, and edge (u^j, v^j) with direction i are parallel in face (i,j)
    # Also, edge (u,v) direction i and edge (u^j, u^j^i) direction i are parallel
    # Wait, let me reconsider. In a 4-face with directions i,j:
    # Vertices: u, u^i, u^j, u^i^j
    # Edges: (u,u^i) dir i, (u^j, u^i^j) dir i -- these are parallel
    #         (u,u^j) dir j, (u^i, u^i^j) dir j -- these are parallel
    # So edge (u, u^i) conflicts with edge (u^j, u^i^j) for each j != i
    
    conflict = [[] for _ in range(m)]
    for idx, (x, y) in enumerate(all_edges):
        diff = x ^ y
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            # The parallel edge in face (i,j) is (x^j, y^j)
            c = x ^ (1 << j)
            e = y ^ (1 << j)
            e_par = (min(c, e), max(c, e))
            idx_par = edge_to_idx.get(e_par)
            if idx_par is not None and idx_par != idx:
                if idx_par not in conflict[idx]:
                    conflict[idx].append(idx_par)
    
    # Spectral initialization using Fiedler vector
    # Build adjacency matrix of conflict graph
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import laplacian
    from scipy.sparse.linalg import eigsh
    
    row = []
    col = []
    data = []
    for i in range(m):
        for j in conflict[i]:
            if i < j:  # symmetric
                row.append(i)
                col.append(j)
                data.append(1.0)
                row.append(j)
                col.append(i)
                data.append(1.0)
    
    A = csr_matrix((data, (row, col)), shape=(m, m))
    L = laplacian(A, normed=True)
    
    try:
        # Find Fiedler vector (second smallest eigenvalue)
        eigenvalues, eigenvectors = eigsh(L, k=3, sigma=0, which='LM')
        # eigsh with sigma=0 finds eigenvalues closest to 0
        # Sort by eigenvalue
        idx_sort = np.argsort(eigenvalues)
        fiedler = eigenvectors[:, idx_sort[1]]  # second smallest
    except Exception:
        fiedler = np.random.randn(m)
        fiedler -= fiedler.mean()
    
    # Initial partition: edges with positive fiedler value are "selected"
    # But we need to ensure C4-freeness, so use greedy completion
    chosen = np.zeros(m, dtype=bool)
    for idx in range(m):
        if fiedler[idx] > 0:
            # Check if adding this edge creates a C4
            ok = True
            for (idx1, idx2, idx3) in _get_forbidden_triples(idx, all_edges, edge_to_idx, d):
                if chosen[idx1] and chosen[idx2] and chosen[idx3]:
                    ok = False
                    break
            if ok:
                chosen[idx] = True
    
    count = chosen.sum()
    best_count = count
    best_chosen = chosen.copy()
    
    # Local search phases
    def check_add(idx, chosen):
        """Check if edge idx can be added without creating a C4."""
        for (i1, i2, i3) in _get_forbidden_triples(idx, all_edges, edge_to_idx, d):
            if chosen[i1] and chosen[i2] and chosen[i3]:
                return False
        return True
    
    def check_remove(idx, chosen):
        """After removing idx, check if any previously-forbidden edge can now be added."""
        # This is used in swap moves
        pass
    
    # Phase 1: Greedy completion - try to add all unselected edges
    for idx in range(m):
        if not chosen[idx]:
            if check_add(idx, chosen):
                chosen[idx] = True
                count += 1
    if count > best_count:
        best_count = count
        best_chosen = chosen.copy()
    
    # Phase 2: Local search with 1-opt (add/remove) and 2-opt (swap)
    T = 3.0
    T_min = 0.05
    cooling = 0.99995
    no_improve = 0
    max_no_improve = 50000
    
    while time.time() < deadline and T > T_min:
        # Random 1-opt: try to add or remove a random edge
        idx = random.randint(0, m - 1)
        
        if not chosen[idx]:
            # Try adding
            if check_add(idx, chosen):
                chosen[idx] = True
                count += 1
                if count > best_count:
                    best_count = count
                    best_chosen = chosen.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Can't add directly; try a 2-opt swap
                # Find a conflicting edge to remove
                conflicts = []
                for (i1, i2, i3) in _get_forbidden_triples(idx, all_edges, edge_to_idx, d):
                    if chosen[i1] and chosen[i2] and chosen[i3]:
                        # Any of i1, i2, i3 could be removed
                        conflicts.extend([i1, i2, i3])
                
                if conflicts:
                    # Try removing a random conflicting edge and adding idx
                    remove_idx = random.choice(conflicts)
                    if not chosen[remove_idx]:
                        continue
                    chosen[remove_idx] = False
                    count -= 1
                    if check_add(idx, chosen):
                        chosen[idx] = True
                        count += 1
                        if count > best_count:
                            best_count = count
                            best_chosen = chosen.copy()
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        # Revert
                        chosen[idx] = False
                        chosen[remove_idx] = True
                        count += 1
                        no_improve += 1
                else:
                    no_improve += 1
        else:
            # Try removing (less likely to help, but allows exploration)
            chosen[idx] = False
            count -= 1
            # Accept with probability exp(-delta/T) where delta = -1
            # For removal, delta = -1, so always accept at reasonable T
            # But only keep if it leads to improvement later
            # Use SA: accept with prob exp(1/T) which is > 1, so always accept
            # Actually, for removal, we lose 1 edge, so delta = -1
            # Accept with prob min(1, exp(-(-1)/T)) = min(1, exp(1/T)) = 1
            # So always accept removals in SA? That would destroy the solution.
            # Better: only remove if it enables adding 2+ edges
            # For now, revert removals unless they enable improvements
            # Try to add back or find better edges
            # Simpler: revert most removals
            if random.random() < 0.7:  # 70% chance to revert
                chosen[idx] = True
                count += 1
            else:
                # Keep removed, try to add other edges
                for other_idx in range(m):
                    if not chosen[other_idx]:
                        if check_add(other_idx, chosen):
                            chosen[other_idx] = True
                            count += 1
                if count > best_count:
                    best_count = count
                    best_chosen = chosen.copy()
                    no_improve = 0
                else:
                    no_improve += 1
        
        T *= cooling
        
        if no_improve > max_no_improve:
            # Reset to best and restart with higher temperature
            chosen = best_chosen.copy()
            count = best_count
            T = max(T, 2.0)
            no_improve = 0
    
    # Final greedy polish
    for idx in range(m):
        if not chosen[idx]:
            if check_add(idx, chosen):
                chosen[idx] = True
                count += 1
    if count > best_count:
        best_count = count
        best_chosen = chosen.copy()
    
    # Convert to output
    out = []
    for idx in range(m):
        if best_chosen[idx]:
            x, y = all_edges[idx]
            out.append([x, y])
    
    return out


def _get_forbidden_triples(idx, all_edges, edge_to_idx, d):
    """Get the triples of edges that, together with edge idx, form a C4."""
    x, y = all_edges[idx]
    diff = x ^ y
    i = diff.bit_length() - 1
    triples = []
    for j in range(d):
        if j == i:
            continue
        c = x ^ (1 << j)
        e = y ^ (1 << j)
        e1 = (min(x, c), max(x, c))
        e2 = (min(y, e), max(y, e))
        e3 = (min(c, e), max(c, e))
        idx1 = edge_to_idx.get(e1)
        idx2 = edge_to_idx.get(e2)
        idx3 = edge_to_idx.get(e3)
        if idx1 is not None and idx2 is not None and idx3 is not None:
            triples.append((idx1, idx2, idx3))
    return triples
# EVOLVE-BLOCK-END
