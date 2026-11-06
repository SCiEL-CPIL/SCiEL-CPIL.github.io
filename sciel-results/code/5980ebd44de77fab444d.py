"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    deadline = t_start + 140.0
    n = 40
    best_edges = []
    best_count = 0
    
    def has_path_len_5(adj, u, v):
        """Check if there's a simple path of length 5 from u to v: u-a-b-c-d-v."""
        # u-a, a-b, b-c, c-d, d-v, all 6 vertices distinct
        for a in adj[u]:
            if a == v:
                continue
            for b in adj[a]:
                if b == u or b == v or b == a:
                    continue
                for c in adj[b]:
                    if c == u or c == v or c == a or c == b:
                        continue
                    for d in adj[c]:
                        if d == u or d == v or d == a or d == b or d == c:
                            continue
                        if v in adj[d]:
                            return True
        return False
    
    def init_and_prune(rng):
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < 0.5:
                    adj[i].add(j)
                    adj[j].add(i)
        
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in sorted(adj[i]):
                    if j <= i:
                        continue
                    adj[i].discard(j)
                    adj[j].discard(i)
                    if has_path_len_5(adj, i, j):
                        changed = True
                    else:
                        adj[i].add(j)
                        adj[j].add(i)
        
        edges = []
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    edges.append([i, j])
        return edges, adj
    
    def local_search(edges, adj, rng, dl):
        edge_set = set()
        for e in edges:
            edge_set.add((min(e[0], e[1]), max(e[0], e[1])))
        
        candidates = [(i, j) for i in range(n) for j in range(i+1, n) if (i,j) not in edge_set]
        rng.shuffle(candidates)
        
        improved = True
        while improved and time.time() < dl:
            improved = False
            for (i, j) in candidates:
                if (i, j) in edge_set:
                    continue
                if not has_path_len_5(adj, i, j):
                    adj[i].add(j)
                    adj[j].add(i)
                    edge_set.add((i, j))
                    edges.append([i, j])
                    improved = True
        
        return edges, adj
    
    for start_idx in range(30):
        if time.time() > deadline - 20:
            break
        
        rng = random.Random(42 + start_idx * 104729)
        edges, adj = init_and_prune(rng)
        cnt = len(edges)
        
        if cnt > best_count:
            best_count = cnt
            best_edges = edges[:]
        
        edges, adj = local_search(edges, adj, rng, deadline - 20)
        cnt = len(edges)
        
        if cnt > best_count:
            best_count = cnt
            best_edges = edges[:]
    
    return best_edges
# EVOLVE-BLOCK-END
