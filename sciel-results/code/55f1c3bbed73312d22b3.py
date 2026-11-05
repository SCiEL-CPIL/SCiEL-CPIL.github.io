    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 0:
                increase = count_cycles_using(i, j, adj)
                if increase < best_c4_increase:
                    best_c4_increase = increase
                    best_edge = [i, j]
    