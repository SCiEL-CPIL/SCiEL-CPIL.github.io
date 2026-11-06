    def greedy_add(adj):
        missing = [(u, v) for u in range(n) for v in range(u+1, n) if v not in adj[u]]
        random.shuffle(missing)
        for u, v in missing:
            if v not in adj[u]:
                if try_add_edge(adj, u, v):
                    adj[u].add(v)
                    adj[v].add(u)
