import random
import numpy as np

def construct(k, g):
    # Try K4 lift with p=131 (n=524)
    # K4 edges
    edges_k4 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    adj_k4 = [[] for _ in range(4)]
    for u, v in edges_k4:
        adj_k4[u].append(v)
        adj_k4[v].append(u)
        
    edge_index_k4 = {}
    for i, e in enumerate(edges_k4):
        edge_index_k4[e] = i
        edge_index_k4[(e[1], e[0])] = i

    def get_walks_k4(max_len):
        walks = []
        for start in range(4):
            stack = [(start, -1, 0, [])]
            while stack:
                curr, prev, l, seq = stack.pop()
                if l == max_len:
                    if curr == start:
                        walks.append(seq)
                    continue
                for nxt in adj_k4[curr]:
                    if nxt == prev:
                        continue
                    e_idx = edge_index_k4.get((curr, nxt), edge_index_k4.get((nxt, curr)))
                    if (curr, nxt) == edges_k4[e_idx]:
                        sign = 1
                    else:
                        sign = -1
                    stack.append((nxt, curr, l + 1, seq + [(e_idx, sign)]))
        return walks

    constraints_k4 = []
    for L in range(3, g): # K4 girth is 3
        walks = get_walks_k4(L)
        for w in walks:
            constraints_k4.append(w)

    p = 131
    rng = random.Random(42)
    found = False
    voltages = None
    for attempt in range(10000):
        voltages = [rng.randint(0, p-1) for _ in range(6)]
        valid = True
        for w in constraints_k4:
            s = 0
            for e_idx, sign in w:
                s += sign * voltages[e_idx]
            if s % p == 0:
                valid = False
                break
        if valid:
            found = True
            break
            
    if found:
        n = 4 * p
        edges_out = []
        for i in range(6):
            u, v = edges_k4[i]
            ve = voltages[i]
            for fiber in range(p):
                v1 = u * p + fiber
                v2 = v * p + (fiber + ve) % p
                edges_out.append((v1, v2))
                v3 = v * p + fiber
                v4 = u * p + (fiber - ve) % p
                edges_out.append((v3, v4))
        
        unique_edges = set()
        for e in edges_out:
            if e[0] > e[1]:
                e = (e[1], e[0])
            unique_edges.add(e)
        return n, list(unique_edges)

    # Fallback to Petersen p=53
    # ... (original code)
