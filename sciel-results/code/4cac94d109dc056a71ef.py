    # We have 400 edges. We need 401.
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    missing = [e for e in all_possible if e not in edge_set]
