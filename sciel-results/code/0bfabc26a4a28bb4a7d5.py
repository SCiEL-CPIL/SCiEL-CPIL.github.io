import time

def construct():
    n = 16
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    m = len(all_edges)
    
    best_count = 0
    best_edges = []
    
    # Seed with Grötzsch extension
    def get_seed():
        edges = [
            (0,1), (1,2), (2,3), (3,4), (4,0),
            (5,6), (6,7), (7,8), (8,9), (9,5),
            (0,6), (0,9),
            (1,5), (1,7),
            (2,6), (2,8),
            (3,7), (3,9),
            (4,5), (4,8),
            (10,5), (10,6), (10,7), (10,8), (10,9)
        ]
        return edges

    seed_edges = get_seed()
    
    def greedy_extend(initial_edges):
        adj = [0] * n
        for u, v in initial_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)) and (adj[u] & adj[v] == 0):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        count = sum(bin(adj[i]).count('1') for i in range(n)) // 2
        res = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    res.append([u, v])
        return count, res

    sc, se = greedy_extend(seed_edges)
    best_count = sc
    best_edges = se

    start_time = time.time()
    timeout = 115

    def is_3_colorable(adj):
        color = [-1] * n
        def bt(v):
            if v == n: return True
            for c in range(3):
                ok = True
                nb = adj[v]
                for u in range(n):
                    if (nb >> u) & 1 and color[u] == c:
                        ok = False
                        break
                if ok:
                    color[v] = c
                    if bt(v+1): return True
                    color[v] = -1
            return False
        return bt(0)

    def can_extend_to_non_3(adj, rem_mask):
        if not is_3_colorable(adj):
            return True
        # Check bipartite
        col = [-1] * n
        bip = True
        for i in range(n):
            if col[i] == -1:
                col[i] = 0
                st = [i]
                while st and bip:
                    u = st.pop()
                    for v in range(n):
                        if (adj[u] >> v) & 1:
                            if col[v] == -1:
                                col[v] = col[u] ^ 1
                                st.append(v)
                            elif col[v] == col[u]:
                                bip = False
                                break
        if bip:
            mask = rem_mask
            while mask:
                bit = mask & -mask
                idx = bit.bit_length() - 1
                u, v = all_edges[idx]
                if col[u] == col[v]:
                    return True
                mask ^= bit
            return False
        return True

    def solve(idx, adj, curr_edges, rem_mask):
        nonlocal best_count, best_edges
        if time.time() - start_time > timeout:
            return
        if len(curr_edges) + rem_mask.bit_count() <= best_count:
            return
        
        if not is_3_colorable(adj):
            # Greedy fill
            adj_c = adj[:]
            added = 0
            mask = rem_mask
            while mask:
                bit = mask & -mask
                idx_e = bit.bit_length() - 1
                u, v = all_edges[idx_e]
                if not (adj_c[u] & (1 << v)) and (adj_c[u] & adj_c[v] == 0):
                    adj_c[u] |= (1 << v)
                    adj_c[v] |= (1 << u)
                    added += 1
                mask ^= bit
            total = len(curr_edges) + added
            if total > best_count:
                best_count = total
                best_edges = [[u, v] for u in range(n) for v in range(u+1, n) if adj_c[u] & (1 << v)]
            return

        if not can_extend_to_non_3(adj, rem_mask):
            return

        if idx == m:
            return
        
        u, v = all_edges[idx]
        
        # Include
        if (adj[u] & adj[v]) == 0:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            curr_edges.append((u, v))
            solve(idx + 1, adj, curr_edges, rem_mask & ~(1 << idx))
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            curr_edges.pop()
        
        # Exclude
        solve(idx + 1, adj, curr_edges, rem_mask & ~(1 << idx))

    solve(0, [0]*n, [], (1 << m) - 1)
    return best_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found graph with {len(edges)} edges")
