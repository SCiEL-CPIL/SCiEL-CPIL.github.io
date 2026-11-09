import random
import time

def construct(k, g):
    if k != 3:
        # Fallback for other k
        # The task is specifically for k=3, g=16
        # But we should handle others gracefully
        return _construct_generic(k, g)

    start_time = time.time()
    
    def try_construct(N):
        # Build a random tree
        adj = [[] for _ in range(N)]
        for i in range(1, N):
            p = random.randint(0, i - 1)
            adj[p].append(i)
            adj[i].append(p)
            
        deg = [len(adj[i]) for i in range(N)]
        active = [i for i in range(N) if deg[i] < 3]
        
        def get_dist(u, v):
            if u == v: return 0
            visited = [-1] * N
            visited[u] = 0
            queue = [u]
            while queue:
                curr = queue.pop(0)
                for nb in adj[curr]:
                    if visited[nb] == -1:
                        visited[nb] = visited[curr] + 1
                        if nb == v:
                            return visited[nb]
                        queue.append(nb)
            return -1
            
        edges_to_add = sum(3 - deg[i] for i in range(N)) // 2
        
        for _ in range(edges_to_add):
            if time.time() - start_time > 115:
                return None
                
            found = False
            # We need to find a pair (u, v) in active such that dist(u, v) >= g - 1
            # To speed up, we can just check random pairs
            # If N is small, checking all is fine.
            # If N is large, random sampling is better.
            
            # Let's try a few random pairs first
            for _ in range(100):
                if len(active) < 2:
                    break
                i = random.randint(0, len(active) - 1)
                j = random.randint(0, len(active) - 1)
                if i == j: continue
                u = active[i]
                v = active[j]
                if deg[u] < 3 and deg[v] < 3:
                    d = get_dist(u, v)
                    if d >= g - 1:
                        adj[u].append(v)
                        adj[v].append(u)
                        deg[u] += 1
                        deg[v] += 1
                        if deg[u] == 3:
                            active.remove(u)
                        if deg[v] == 3:
                            active.remove(v)
                        found = True
                        break
                        
            if not found:
                # If random sampling fails, try exhaustive (slower)
                # This is a fallback for small N or stuck states
                for i in range(len(active)):
                    for j in range(i + 1, len(active)):
                        u = active[i]
                        v = active[j]
                        if deg[u] < 3 and deg[v] < 3:
                            d = get_dist(u, v)
                            if d >= g - 1:
                                adj[u].append(v)
                                adj[v].append(u)
                                deg[u] += 1
                                deg[v] += 1
                                if deg[u] == 3:
                                    active.remove(u)
                                if deg[v] == 3:
                                    active.remove(v)
                                found = True
                                break
                    if found:
                        break
                if not found:
                    return None
        
        if all(deg[i] == 3 for i in range(N)):
            edges = []
            for u in range(N):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return N, edges
        return None

    # Try N values
    # Moore bound 512
    for N in range(512, 950, 16):
        res = try_construct(N)
        if res:
            return res
            
    # If failed, try a larger N
    res = try_construct(1000)
    if res:
        return res
        
    # Last resort: return a known valid graph if possible, or fail
    # Since we can't fail, we try N=1024
    res = try_construct(1024)
    if res:
        return res
        
    return 0, []

def _construct_generic(k, g):
    # Simple fallback
    return 0, []
