<<<<<<< SEARCH
def _quick_girth_check(n, adj, g, num_starts):
    """Quick girth check using BFS from a few starting points."""
    import random as _random
    _random.seed(123)
    starts = _random.sample(range(n), min(num_starts, n))
    
    for start in starts:
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        q_head = 0
        while q_head < len(queue):
            u = queue[q_head]
            q_head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif parent[u] != v and parent[v] != u:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True
=======
def _quick_girth_check(n, adj, g, num_starts):
    """Quick girth check using BFS from a few starting points."""
    import random as _random
    _random.seed(123)
    starts = _random.sample(range(n), min(num_starts, n))
    
    for start in starts:
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        q_head = 0
        while q_head < len(queue):
            u = queue[q_head]
            q_head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif parent[u] != v and parent[v] != u:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _voltage_girth_check(n, k, m, voltages, g):
    """
    Check girth of a voltage graph over K_{k,k} more efficiently.
    A cycle in the lift corresponds to a closed walk in K_{k,k} with voltage sum = 0 mod m.
    For girth >= g, no non-backtracking walk of length < g should close with zero voltage.
    """
    # For each starting vertex (side, i, x), do BFS up to depth (g-1)//2
    # Track: (side, node, x_offset) where x_offset is the accumulated voltage
    # Actually, we can track the voltage sum directly
    
    # Simpler: just do BFS on the actual graph
    # But we can be smarter: a walk in K_{k,k} of length L corresponds to a sequence
    # of nodes i0, i1, ..., iL where consecutive nodes are on opposite sides.
    # The voltage sum is v[i0][i1] + v[i1][i2] + ... + v[iL-1][iL] mod m (with signs for direction)
    
    # For the lift, vertex (side, i, x) connects to (1-side, j, (x + v[i][j]) % m) if side=0
    # or (1-side, j, (x - v[j][i]) % m) if side=1
    
    # Let's just do BFS on the constructed graph for correctness
    adj = [[] for _ in range(n)]
    for i in range(k):
        for x in range(m):
            u = i * m + x
            for j in range(k):
                v_val = voltages[i][j]
                y = (x + v_val) % m
                v = k * m + j * m + y
                adj[u].append(v)
                adj[v].append(u)
    
    # BFS from a few starting points
    import random as _random
    _random.seed(123)
    starts = _random.sample(range(n), min(5, n))
    
    for start in starts:
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        q_head = 0
        while q_head < len(queue):
            u = queue[q_head]
            q_head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif parent[u] != v and parent[v] != u:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True
>>>>>>> REPLACE

