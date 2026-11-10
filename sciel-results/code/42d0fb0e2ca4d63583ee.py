    degrees = [len(adj_active[i]) for i in range(M)]
    order = sorted(range(M), key=lambda x: degrees[x]) # Ascending
