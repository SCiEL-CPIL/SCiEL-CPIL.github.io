import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    best_edges = []
    best_count = 0
    n = 20
    
    # Precompute all pairs
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_safe_to_add(adj, u, v):
        """
        Check if adding edge (u, v) keeps the graph K2,3-free.
        adj: list of sets
        """
        # 1. Check pairs involving u or v
        # Pair (x, u): N(x) intersect N(u)
        # If v in N(x), v is added to N(x) intersect N(u) (since v in N(u) now)
        # So if v in N(x) and |N(x) intersect N(u)| >= 2, then new size >= 3 -> Invalid
        for x in range(n):
            if x == u or x == v:
                continue
            # Check pair (x, u)
            if v in adj[x]:
                # v will be in N(x) and N(u) (since we add u-v)
                # Current common neighbors of x and u
                common_xu = len(adj[x].intersection(adj[u]))
                if common_xu >= 2:
                    return False
            
            # Check pair (x, v)
            if u in adj[x]:
                common_xv = len(adj[x].intersection(adj[v]))
                if common_xv >= 2:
                    return False
        
        # 2. Check pairs (x, y) not involving u or v
        # Adding (u, v) increases |N(x) intersect N(y)| by 1 if:
        # u is in N(x) intersect N(y) AND v is in N(x) intersect N(y)
        # i.e., x~u, y~u, x~v, y~v
        # If so, check if current |N(x) intersect N(y)| >= 2
        # We can iterate x, y where x < y, x,y not u,v
        # Optimization: Only check if u and v are "close"
        for x in range(n):
            if x == u or x == v:
                continue
            if u not in adj[x] or v not in adj[x]:
                continue
            for y in range(x + 1, n):
                if y == u or y == v:
                    continue
                if u not in adj[y] or v not in adj[y]:
                    continue
                # x and y are both connected to u and v
                # So u and v are common neighbors of x and y
                common_xy = len(adj[x].intersection(adj[y]))
                # If common_xy >= 2, then adding u-v makes it >= 3?
                # Wait. u and v are ALREADY common neighbors?
                # No. u is in N(x) and N(y). v is in N(x) and N(y).
                # So u and v are in N(x) intersect N(y).
                # So |N(x) intersect N(y)| already includes u and v.
                # Adding (u, v) does NOT change N(x) or N(y).
                # It only changes N(u) and N(v).
                # So this case is irrelevant for the intersection of N(x) and N(y).
                # The intersection N(x) intersect N(y) consists of vertices z such that x~z and y~z.
                # Adding (u, v) adds v to N(u) and u to N(v).
                # It does NOT add u or v to N(x) or N(y).
                # So N(x) and N(y) are unchanged.
                # Thus, pairs not involving u or v are unaffected.
                pass
        
        return True

    def greedy_add(adj):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Shuffle to avoid bias? Or just iterate.
            # Iterating in fixed order might be faster but less random.
            for u, v in all_pairs:
                if (u, v) not in adj[u]: # Check if edge exists
                    if is_safe_to_add(adj, u, v):
                        adj[u].add(v)
                        adj[v].add(u)
                        changed = True
        return adj

    def repair(adj):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Find a violating pair
            violating = None
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj[u].intersection(adj[v])
                    if len(common) >= 3:
                        violating = (u, v, list(common))
                        break
                if violating:
                    break
            
            if violating:
                u, v, commons = violating
                # Pick a random common neighbor
                w = random.choice(commons)
                # Randomly remove (u, w) or (v, w)
                if random.random() < 0.5:
                    adj[u].discard(w)
                    adj[w].discard(u)
                else:
                    adj[v].discard(w)
                    adj[w].discard(v)
                changed = True
        return adj

    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges

    # Main Loop
    while time.time() < deadline:
        # Initialization
        p = random.uniform(0.3, 0.6)
        adj = [set() for _ in range(n)]
        for u, v in all_pairs:
            if random.random() < p:
                adj[u].add(v)
                adj[v].add(u)
        
        # Repair
        adj = repair(adj)
        
        # Add
        adj = greedy_add(adj)
        
        # Local Search Kick (optional, if time permits)
        if time.time() < deadline - 5:
            # Remove a random edge to potentially escape local optimum
            if any(adj[u] for u in range(n)):
                # Pick random edge
                u = random.randint(0, n - 1)
                if adj[u]:
                    v = random.choice(list(adj[u]))
                    adj[u].discard(v)
                    adj[v].discard(u)
                    # Re-add greedily
                    adj = greedy_add(adj)
        
        # Update Best
        current_edges = adj_to_edges(adj)
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges
            
    return best_edges
