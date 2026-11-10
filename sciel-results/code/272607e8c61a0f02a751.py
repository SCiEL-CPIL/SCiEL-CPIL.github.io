import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    all_8sets = list(combinations(range(n), 8))
    
    # Precompute for each edge, the list of 8-sets containing it
    edge_to_8sets = {}
    for i, j in all_edges:
        edge_to_8sets[(i, j)] = []
    for combo in all_8sets:
        for i in range(8):
            for j in range(i+1, 8):
                edge_to_8sets[(combo[i], combo[j])].append(combo)
    
    def check_q3(verts, adj):
        deg = [0]*8
        edge_count = 0
        for i in range(8):
            for j in range(i+1, 8):
                u, v = verts[i], verts[j]
                if (u, v) in adj or (v, u) in adj:
                    edge_count += 1
                    deg[i] += 1
                    deg[j] += 1
        if edge_count != 12:
            return False
        if any(d != 3 for d in deg):
            return False
        visited = [False]*8
        visited[0] = True
        queue = [0]
        while queue:
            i = queue.pop(0)
            for j in range(8):
                if j != i and not visited[j]:
                    u, v = verts[i], verts[j]
                    if (u, v) in adj or (v, u) in adj:
                        visited[j] = True
                        queue.append(j)
        if not all(visited):
            return False
        color = [-1]*8
        color[0] = 0
        queue = [0]
        while queue:
            i = queue.pop(0)
            for j in range(8):
                if i != j:
                    u, v = verts[i], verts[j]
                    if (u, v) in adj or (v, u) in adj:
                        if color[j] == -1:
                            color[j] = 1 - color[i]
                            queue.append(j)
                        elif color[j] == color[i]:
                            return False
        if sum(1 for c in color if c == 0) != 4:
            return False
        return True
    
    # Start with K16, remove edges to eliminate Q3s
    adj = set(all_edges)
    
    while time.time() < deadline - 50:
        q3 = None
        for combo in all_8sets:
            if check_q3(combo, adj):
                q3 = combo
                break
        if q3 is None:
            break
        # Remove an edge from q3
        removed = False
        for i in range(8):
            for j in range(i+1, 8):
                u, v = q3[i], q3[j]
                if (u, v) in adj:
                    adj.discard((u, v))
                    removed = True
                    break
            if removed:
                break
    
    best_adj = set(adj)
    best_count = len(adj)
    
    # Local search: try to add back edges
    for _ in range(3):
        if time.time() > deadline:
            break
        adj = set(best_adj)
        for _ in range(200):
            if time.time() > deadline:
                break
            missing = [e for e in all_edges if e not in adj]
            if not missing:
                break
            edge = random.choice(missing)
            adj.add(edge)
            # Check only Q3s containing this edge
            has_q3 = False
            for combo in edge_to_8sets.get(edge, []):
                if check_q3(combo, adj):
                    has_q3 = True
                    break
            if not has_q3:
                if len(adj) > best_count:
                    best_count = len(adj)
                    best_adj = set(adj)
            else:
                adj.discard(edge)
    
    return [[u, v] for u, v in best_adj]
