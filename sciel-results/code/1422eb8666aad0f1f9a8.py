        # Cover remaining internal edges with 2-cliques (edges)
        # This is always valid and fast.
        for edge in E_rem:
            partition.append([edge[0], edge[1]])
