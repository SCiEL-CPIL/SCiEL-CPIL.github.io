        # Second pass
        for u, v in all_edges:
            if u not in adj[v]:
                if not has_path_len_5(u, v, adj):
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append([u, v])
