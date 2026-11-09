import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for 90s limit
    
    n = 15
    
    def is_valid(adj):
        """Check if graph has no 3-edge path with same-degree endpoints."""
        degs = [len(adj[i]) for i in range(n)]
        groups = {}
        for i in range(n):
            d = degs[i]
            if d not in groups:
                groups[d] = []
            groups[d].append(i)
        
        for d, group in groups.items():
            if len(group) < 2:
                continue
            for i in range(len(group)):
                s = group[i]
                for j in range(i+1, len(group)):
                    t = group[j]
                    # Check if simple 3-edge path s-u-v-t exists
                    # s,u,v,t all distinct
                    for u in adj[s]:
                        if u == t:
                            continue
                        for v in adj[t]:
                            if v == s or v == u:
                                continue
                            if u in adj[v]:
                                return False
        return True
    
    def edge_count(adj):
        return sum(len(adj[i]) for i in range(n)) // 2
    
    # Start with K_{7,8} bipartite graph: vertices 0-6 in A, 7-14 in B
    # All A vertices degree 8, all B vertices degree 7. Valid since 8 != 7.
    adj = [set() for _ in range(n)]
    for i in range(7):
        for j in range(7, 15):
            adj[i].add(j)
            adj[j].add(i)
    
    best_adj = [set(adj[i]) for i in range(n)]
    best_count = edge_count(best_adj)  # 56
    
    # Try to add edges (intra-part) via local search
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Try adding each missing edge
    for trial in range(50):
        if time.time() > deadline - 5:
            break
        
        random.seed(trial)
        cur_adj = [set(best_adj[i]) for i in range(n)]
        cur_count = best_count
        
        # Get missing edges
        missing = [e for e in all_possible if e[1] not in cur_adj[e[0]]]
        random.shuffle(missing)
        
        for u, v in missing:
            if time.time() > deadline - 3:
                break
            cur_adj[u].add(v)
            cur_adj[v].add(u)
            if is_valid(cur_adj):
                cur_count += 1
            else:
                cur_adj[u].discard(v)
                cur_adj[v].discard(u)
        
        if cur_count > best_count:
            best_count = cur_count
            best_adj = [set(cur_adj[i]) for i in range(n)]
        
        # Try removing some edges and re-adding for diversity
        if time.time() > deadline - 3:
            break
        cur_adj = [set(best_adj[i]) for i in range(n)]
        edges_in = [(min(u,v), max(u,v)) for u in range(n) for v in cur_adj[u] if u < v]
        random.shuffle(edges_in)
        # Remove 3-5 random edges
        for e in edges_in[:random.randint(3, 5)]:
            cur_adj[e[0]].discard(e[1])
            cur_adj[e[1]].discard(e[0])
        
        # Re-add missing edges greedily
        missing = [e for e in all_possible if e[1] not in cur_adj[e[0]]]
        random.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline - 3:
                break
            cur_adj[u].add(v)
            cur_adj[v].add(u)
            if is_valid(cur_adj):
                pass
            else:
                cur_adj[u].discard(v)
                cur_adj[v].discard(u)
        
        cur_count = edge_count(cur_adj)
        if cur_count > best_count:
            best_count = cur_count
            best_adj = [set(cur_adj[i]) for i in range(n)]
    
    edges = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                edges.append([int(u), int(v)])
    
    return edges
