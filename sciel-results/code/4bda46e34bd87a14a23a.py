import time
import random

def construct():
    deadline = time.time() + 80
    n = 15
    
    def check_validity(adj):
        deg = [len(adj[i]) for i in range(n)]
        for u in range(n):
            for v in list(adj[u]):
                if v > u:
                    for s in adj[u]:
                        if s == v:
                            continue
                        for t in adj[v]:
                            if t == u:
                                continue
                            if s == t:
                                continue
                            if deg[s] == deg[t]:
                                return False
        return True
    
    def edge_count(adj):
        return sum(len(adj[i]) for i in range(n)) // 2
    
    # Start with K_{7,8}
    adj = [set() for _ in range(n)]
    # Part A: 0-6 (7 vertices), Part B: 7-14 (8 vertices)
    for i in range(7):
        for j in range(7, 15):
            adj[i].add(j)
            adj[j].add(i)
    
    # Verify base is valid
    assert check_validity(adj), "Base bipartite should be valid"
    
    best_adj = [set(s) for s in adj]
    best_count = edge_count(best_adj)
    
    # Try to add intra-part edges
    def try_add_intra(adj):
        deg = [len(adj[i]) for i in range(n)]
        improved = True
        while improved:
            improved = False
            # Try adding edges within part A and within part B
            candidates = []
            for i in range(7):
                for j in range(i+1, 7):
                    if j not in adj[i]:
                        candidates.append((i,j))
            for i in range(7, 15):
                for j in range(i+1, 15):
                    if j not in adj[i]:
                        candidates.append((i,j))
            
            random.shuffle(candidates)
            for i,j in candidates:
                adj[i].add(j)
                adj[j].add(i)
                if check_validity(adj):
                    improved = True
                    break
                else:
                    adj[i].discard(j)
                    adj[j].discard(i)
        return adj
    
    adj = try_add_intra(adj)
    count = edge_count(adj)
    if count > best_count:
        best_count = count
        best_adj = [set(s) for s in adj]
    
    # Try more aggressive local search with multiple restarts
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Start fresh from K_{7,8}
        adj = [set() for _ in range(n)]
        for i in range(7):
            for j in range(7, 15):
                adj[i].add(j)
                adj[j].add(i)
        
        # Add random intra-part edges first
        for i in range(7):
            for j in range(i+1, 7):
                if random.random() < 0.3:
                    adj[i].add(j)
                    adj[j].add(i)
        for i in range(7, 15):
            for j in range(i+1, 15):
                if random.random() < 0.3:
                    adj[i].add(j)
                    adj[j].add(i)
        
        # Now try to add more edges
        adj = try_add_intra(adj)
        
        # Also try adding cross edges that were removed? No, we have all cross edges.
        # Try edge swaps: remove one intra edge, add another
        for _ in range(100):
            if time.time() > deadline:
                break
            
            # Try adding any missing edge
            added = False
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i+1, n):
                    if j not in adj[i]:
                        adj[i].add(j)
                        adj[j].add(i)
                        if check_validity(adj):
                            added = True
                            break
                        else:
                            adj[i].discard(j)
                            adj[j].discard(i)
                if added:
                    break
            
            if not added:
                break
        
        count = edge_count(adj)
        if count > best_count:
            best_count = count
            best_adj = [set(s) for s in adj]
    
    # Convert to edge list
    edges = []
    for i in range(n):
        for j in best_adj[i]:
            if i < j:
                edges.append([int(i), int(j)])
    
    return edges
