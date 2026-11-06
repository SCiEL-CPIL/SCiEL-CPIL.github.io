def is_valid_5_free(adj, n):
    # adj is list of bitmasks
    # Check K5
    for i in range(n):
        for j in range(i+1, n):
            if not (adj[i] >> j) & 1: continue
            common = adj[i] & adj[j]
            # common has bits for vertices connected to both i and j
            # We need to find a K3 in common
            # Iterate k in common
            # For each k, check if there is a l in common & adj[k] such that l is connected to k
            # This is getting complex.
            pass
