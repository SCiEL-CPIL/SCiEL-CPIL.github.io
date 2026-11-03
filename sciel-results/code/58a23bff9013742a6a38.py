<<<<<<< SEARCH
    # Initial solution: rank matching
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_col_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.astype(np.int64)
=======
    # Initial solution: spectral bisection-based partitioning
    # Build graph where facilities are connected by flow weights and locations by distance weights
    # Use spectral bisection to find a good initial partition
    
    # Create flow graph (facilities) - use negative flow as edge weights for min-cut
    flow_graph = -a.copy()
    np.fill_diagonal(flow_graph, 0)
    
    # Create distance graph (locations) - use distance as edge weights
    dist_graph = b.copy()
    np.fill_diagonal(dist_graph, 0)
    
    # Compute Laplacians
    flow_deg = flow_graph.sum(axis=1)
    dist_deg = dist_graph.sum(axis=1)
    
    L_flow = flow_graph.copy()
    for i in range(n):
        L_flow[i, i] = flow_deg[i]
        
    L_dist = dist_graph.copy()
    for i in range(n):
        L_dist[i, i] = dist_deg[i]
    
    # Get second eigenvector of location Laplacian for bisection
    try:
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import csr_matrix
        
        # Use sparse eigsh for efficiency
        L_dist_sparse = csr_matrix(L_dist)
        # Get smallest eigenvalues (eigsh with which='SM')
        vals, vecs = eigsh(L_dist_sparse, k=3, which='SM')
        # Second smallest eigenvector (Fiedler vector)
        fiedler_vec = vecs[:, 1]
        
        # Bisection: split locations based on sign of Fiedler vector
        loc_partition = (fiedler_vec >= 0).astype(np.int64)
        
        # For facilities, use flow graph structure
        L_flow_sparse = csr_matrix(L_flow)
        vals_f, vecs_f = eigsh(L_flow_sparse, k=3, which='SM')
        fiedler_vec_f = vecs_f[:, 1]
        fac_partition = (fiedler_vec_f >= 0).astype(np.int64)
        
        # Assign facilities to locations respecting partitions
        p = np.empty(n, dtype=np.int64)
        
        # Match within each partition
        for part in [0, 1]:
            facs = np.where(fac_partition == part)[0]
            locs = np.where(loc_partition == part)[0]
            
            if len(facs) == 0 or len(locs) == 0:
                continue
                
            # Within each partition, use rank matching based on connectivity
            if part == 0:
                # High-flow facilities to low-distance locations
                fac_scores = a[facs, :].sum(axis=1)
                loc_scores = b[locs, :].sum(axis=1)
                fac_order = np.argsort(-fac_scores, kind="stable")
                loc_order = np.argsort(loc_scores, kind="stable")
            else:
                # Low-flow facilities to high-distance locations (or vice versa)
                fac_scores = a[facs, :].sum(axis=1)
                loc_scores = b[locs, :].sum(axis=1)
                fac_order = np.argsort(fac_scores, kind="stable")
                loc_order = np.argsort(-loc_scores, kind="stable")
            
            for fi, li in zip(facs[fac_order], locs[loc_order]):
                p[fi] = li
        
        # Handle any unassigned facilities/locations
        assigned_locs = set(p[p >= 0])
        unassigned_facs = np.where(p < 0)[0]
        unassigned_locs = np.array([i for i in range(n) if i not in assigned_locs])
        
        if len(unassigned_facs) > 0 and len(unassigned_locs) > 0:
            # Use rank matching for remaining
            fac_scores = a[unassigned_facs, :].sum(axis=1)
            loc_scores = b[unassigned_locs, :].sum(axis=1)
            fac_order = np.argsort(-fac_scores, kind="stable")
            loc_order = np.argsort(loc_scores, kind="stable")
            for fi, li in zip(unassigned_facs[fac_order], unassigned_locs[loc_order]):
                p[fi] = li
        
        # Fallback: if anything is still -1, use random assignment
        if np.any(p < 0):
            remaining_locs = np.array([i for i in range(n) if i not in set(p[p >= 0])])
            remaining_facs = np.where(p < 0)[0]
            for fi, li in zip(remaining_facs, remaining_locs):
                p[fi] = li
        
    except Exception:
        # Fallback to rank matching if spectral method fails
        a_row_sums = a.sum(axis=1)
        b_col_sums = b.sum(axis=1)
        heavy = np.argsort(-a_row_sums, kind="stable")
        central = np.argsort(b_col_sums, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        p = p.astype(np.int64)
>>>>>>> REPLACE
