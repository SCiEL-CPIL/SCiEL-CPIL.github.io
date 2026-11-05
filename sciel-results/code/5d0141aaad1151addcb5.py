import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    def is_c4_free(adj):
        n = len(adj)
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if len(common) >= 2:
                    return False
        return True

    def count_c4s(adj):
        n = len(adj)
        count = 0
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if len(common) >= 2:
                    count += 1
        return count

    def min_degree(adj):
        return min(len(adj[i]) for i in range(len(adj)))

    def find_c4(adj):
        n = len(adj)
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if len(common) >= 2:
                    w, x = list(common)[:2]
                    return (u, v, w, x)
        return None

    # Try to find a C4-free graph with n=46 and min degree >= 7
    n = 46
    min_deg_target = n - 39  # 7
    
    random.seed(42)
    
    for _ in range(100):
        if time.time() > deadline - 5:
            break
        adj = [set() for _ in range(n)]
        # Start with a random graph with ~161 edges
        for _ in range(161):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v and v not in adj[u]:
                adj[u].add(v)
                adj[v].add(u)
        
        # Local search to remove C4s
        for step in range(1000):
            if time.time() > deadline - 5:
                break
            c4 = find_c4(adj)
            if c4 is None:
                if min_degree(adj) >= min_deg_target:
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append([u, v])
                    return {'n': n, 'edges': edges}
                else:
                    break
            u, v, w, x = c4
            # Remove a random edge from the C4
            edge = random.choice([(u, v), (v, w), (w, x), (x, u)])
            a, b = edge
            adj[a].remove(b)
            adj[b].remove(a)
            
            # Try to add an edge if min degree is low
            if min_degree(adj) < min_deg_target:
                # Find a vertex with low degree
                low_v = min(range(n), key=lambda i: len(adj[i]))
                # Try to add an edge to low_v
                candidates = [i for i in range(n) if i != low_v and i not in adj[low_v]]
                random.shuffle(candidates)
                added = False
                for cand in candidates:
                    # Check if adding edge creates a C4
                    adj[low_v].add(cand)
                    adj[cand].add(low_v)
                    if is_c4_free(adj):
                        added = True
                        break
                    else:
                        adj[low_v].remove(cand)
                        adj[cand].remove(low_v)
                if not added:
                    break

    # Fallback: return a valid graph with n=39
    return {'n': 39, 'edges': []}
