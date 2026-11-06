    adj = np.array(adjacency, dtype=bool)
    adj_int = adj.astype(int)
    # ...
    labels, cut = _simulated_annealing(adj, adj_int, n, start_time, deadline)
