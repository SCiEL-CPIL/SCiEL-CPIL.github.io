    def embed_csp(tree_edges, k, used_edges, rng, node_limit=200000, fixed_map=None):
        host_deg = [0] * N
        for (x, y) in used_edges:
            host_deg[x] += 1; host_deg[y] += 1
        host_rem = [N - 1 - host_deg[h] for h in range(N)]
        
        mapping = [-1] * k
        used_hosts = set()
        
        if k == 1:
            if fixed_map and 0 in fixed_map:
                return [fixed_map[0]]
            # Pick any available host, prefer high remaining degree
            cands = [h for h in range(N) if h not in used_hosts] # used_hosts is empty
            cands.sort(key=lambda h: -host_rem[h])
            if cands:
                return [cands[0]]
            return None

        # ... rest of the code ...
