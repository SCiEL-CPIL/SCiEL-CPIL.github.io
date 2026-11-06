import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin

    def has_tri(adj, u, v):
        # adj is list of sets
        return bool(adj[u] & adj[v])

    def greedy_mis(adj, n):
        """Greedy Max Independent Set."""
        avail = set(range(n))
        mis = []
        while avail:
            # Pick vertex with min degree in induced subgraph
            v = min(avail, key=lambda x: len(adj[x] & avail))
            mis.append(v)
            avail -= {v}
            avail -= adj[v]
        return mis

    def try_build(n, seed):
        random.seed(seed)
        adj = [set() for _ in range(n)]
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)

        # Phase 1: Random maximal triangle-free
        for u, v in all_edges:
            if not has_tri(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)

        # Phase 2: Repair
        for _ in range(500):
            if time.time() > deadline:
                return None
            
            mis = greedy_mis(adj, n)
            if len(mis) < 10:
                return adj
            
            # Try to add edge in MIS
            added = False
            # Shuffle MIS to try different pairs
            mis_copy = mis[:]
            random.shuffle(mis_copy)
            for i in range(len(mis_copy)):
                for j in range(i + 1, len(mis_copy)):
                    u, v = mis_copy[i], mis_copy[j]
                    if v not in adj[u] and not has_tri(adj, u, v):
                        adj[u].add(v)
                        adj[v].add(u)
                        added = True
                        break
                if added:
                    break
            
            if not added:
                # Stuck, try to remove a random edge to allow new ones?
                # Or just break and restart
                break
        
        return None

    # Try n=39
    for s in range(500):
        if time.time() > deadline:
            break
        res = try_build(39, s)
        if res is not None:
            edges = [[i, j] for i in range(39) for j in sorted(res[i]) if i < j]
            return {'n': 39, 'edges': edges}

    # Fallback: C7 blow-up with m=3 -> n=21, alpha=9
    n = 21
    adj = [set() for _ in range(n)]
    for g1 in range(7):
        for g2 in range(g1 + 1, 7):
            if (g2 - g1) % 7 in (1, 6):
                for u in range(g1 * 3, (g1 + 1) * 3):
                    for v in range(g2 * 3, (g2 + 1) * 3):
                        adj[u].add(v)
                        adj[v].add(u)
    edges = [[i, j] for i in range(n) for j in sorted(adj[i]) if i < j]
    return {'n': n, 'edges': edges}

if __name__ == '__main__':
    res = construct()
    print(res['n'])
