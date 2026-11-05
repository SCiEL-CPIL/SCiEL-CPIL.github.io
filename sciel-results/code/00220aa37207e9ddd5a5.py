import time
from random import Random
from collections import defaultdict

def instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def decompose(edges):
    if not edges:
        return []
    
    adj = defaultdict(list)
    for e in edges:
        u, v = e
        key = tuple(sorted((u, v)))
        adj[u].append((v, key))
        adj[v].append((u, key))
    
    odd_vertices = [v for v in adj if len(adj[v]) % 2 == 1]
    pairs = []
    for i in range(0, len(odd_vertices), 2):
        if i + 1 < len(odd_vertices):
            pairs.append((odd_vertices[i], odd_vertices[i+1]))
    
    adj2 = defaultdict(list)
    eid_counter = 0
    
    for e in edges:
        u, v = e
        key = tuple(sorted((u, v)))
        eid = ('r', eid_counter)
        eid_counter += 1
        adj2[u].append((v, eid, False))
        adj2[v].append((u, eid, False))
    
    for (a, b) in pairs:
        eid = ('v', eid_counter)
        eid_counter += 1
        adj2[a].append((b, eid, True))
        adj2[b].append((a, eid, True))
    
    adj_list = {v: list(nbrs) for v, nbrs in adj2.items()}
    
    circuit = []
    stack = []
    start = next(iter(adj_list)) if adj_list else 0
    stack.append(start)
    
    while stack:
        v = stack[-1]
        if adj_list[v]:
            nbr, eid, is_virtual = adj_list[v].pop()
            for i, (n, k, iv) in enumerate(adj_list[nbr]):
                if k == eid:
                    adj_list[nbr].pop(i)
                    break
            stack.append(nbr)
        else:
            circuit.append(stack.pop())
    
    circuit.reverse()
    
    # Determine which consecutive pairs in circuit are virtual edges
    # Build a set of virtual edge endpoints for quick lookup
    virtual_edge_set = set()
    for (a, b) in pairs:
        virtual_edge_set.add((a, b))
        virtual_edge_set.add((b, a))
    
    paths = []
    current_path = [circuit[0]]
    
    for i in range(1, len(circuit)):
        u = circuit[i-1]
        v = circuit[i]
        if (u, v) in virtual_edge_set:
            if len(current_path) > 1:
                paths.append(current_path)
            current_path = [v]
        else:
            current_path.append(v)
    
    if len(current_path) > 1:
        paths.append(current_path)
    
    return paths

def construct():
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
    return results
