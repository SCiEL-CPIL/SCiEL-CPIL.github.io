import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s safety margin before 150s kill
    n = 20
    
    def check_k33(adj):
        """Check if graph has K3,3. adj is list of sets."""
        # For every triple (a,b,c), check if |common neighbors| >= 3
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    common = adj[a] & adj[b] & adj[c]
                    if len(common) >= 3:
                        return True, (a, b, c, list(common)[:3])
        return False, None
    
    def count_common3(adj):
        """Count number of (triple, common-neighbor-triple) violations."""
        violations = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    common = adj[a] & adj[b] & adj[c]
                    if len(common) >= 3:
                        violations += 1
        return violations
    
    def try_add_edge(adj, u, v):
        """Try to add edge (u,v); return True if valid."""
        if v in adj[u]:
            return True
        # Adding u-v: check if any triple containing u and v (or just u, or just v) 
        # now has >=3 common neighbors
        # A triple (a,b,c) is violated if all three share >=3 common neighbors
        # After adding u-v, new violations involve triples where u and v are both in the triple,
        # OR where u (or v) is in the triple and the edge u-v creates a new common neighbor
        # Actually: common neighbors of triple {a,b,c} = N(a)∩N(b)∩N(c)
        # Adding u-v means u∈N(v) and v∈N(u)
        # So for triple {u,v,x}: common = N(u)∩N(v)∩N(x)
        #   Before: N(u)∩N(x) ∩ N(v) (without v in N(u))
        #   After: N(u)∩N(v)∩N(x) which includes u? No, u is not in N(u)
        #   Actually N(u)∩N(v)∩N(x) - the new common neighbors gained are those in N(v)∩N(x) that were newly added to N(u)'s perspective... 
        #   Simpler: just check all triples involving u or v
        for x in range(n):
            if x == u or x == v:
                continue
            # Triple {u, v, x}
            common = adj[u] & adj[v] & adj[x]
            # u is in adj[v] after addition, v in adj[u]
            # But u not in adj[u], v not in adj[v]
            # So common doesn't include u or v
            if len(common) >= 3:
                return False
            # Triple {u, x, y} for y != u,v,x: adding u-v doesn't change N(u)∩N(x)∩N(y) unless v is in all three
            # v in N(u) (new), v in N(x)? v in N(y)?
            # This gets complex. Let's just do full check for small n.
        # For efficiency with n=20, check all triples involving u or v
        for b in range(n):
            if b == u:
                continue
            for c in range(n):
                if c == u or c == b:
                    continue
                common = adj[u] & adj[b] & adj[c]
                if len(common) >= 3:
                    return False
        for b in range(n):
            if b == v:
                continue
            for c in range(n):
                if c == v or c == b:
                    continue
                common = adj[v] & adj[b] & adj[c]
                if len(common) >= 3:
                    return False
        return True
    
    def greedy_construct(seed):
        random.seed(seed)
        adj = [set() for _ in range(n)]
        # Start with a base structure: complete bipartite K_{10,10} is K3,3-free? 
        # K_{10,10} has K3,3 as subgraph! So that's bad.
        # K_{2,18} works (score 1.0 baseline = 36 edges)
        # Let's try: random greedy
        edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(edges)
        for u, v in edges:
            if try_add_edge(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
        return adj
    
    def local_search(adj, max_time):
        """Try to improve via edge additions and swaps."""
        best_adj = [set(s) for s in adj]
        best_count = sum(len(s) for s in best_adj) // 2
        improved = True
        while improved and time.time() < max_time:
            improved = False
            # Try adding missing edges
            for u in range(n):
                for v in range(u+1, n):
                    if v not in adj[u]:
                        if try_add_edge(adj, u, v):
                            adj[u].add(v)
                            adj[v].add(u)
                            count = sum(len(s) for s in adj) // 2
                            if count > best_count:
                                best_count = count
                                best_adj = [set(s) for s in adj]
                                improved = True
            # Try swaps: remove one edge, add another (net zero or +1)
            # Actually just try all addable edges again after removals
            # Try removing an edge and adding two
            for u in range(n):
                for v in list(adj[u]):
                    if u < v:
                        # Temporarily remove u-v
                        adj[u].remove(v)
                        adj[v].remove(u)
                        # Try to add edges
                        added = 0
                        for x in range(n):
                            for y in range(x+1, n):
                                if y not in adj[x] and try_add_edge(adj, x, y):
                                    adj[x].add(y)
                                    adj[y].add(x)
                                    added += 1
                        count = sum(len(s) for s in adj) // 2
                        if count > best_count:
                            best_count = count
                            best_adj = [set(s) for s in adj]
                            improved = True
                        # Restore: remove the added edges, re-add u-v
                        # This is getting complex; skip swaps for now
                        # Re-add u-v
                        adj[u].add(v)
                        adj[v].add(u)
                        # Remove edges we added
                        # We don't track them easily; skip swap optimization
                        break
                else:
                    continue
                break
        return best_adj, best_count
    
    # Phase 1: Try multiple greedy constructions
    best_adj = None
    best_count = 0
    
    for seed in range(50):
        if time.time() > deadline - 30:
            break
        adj = greedy_construct(seed)
        count = sum(len(s) for s in adj) // 2
        if count > best_count:
            best_count = count
            best_adj = adj
    
    if best_adj is None:
        # Fallback to K_{2,18}
        return [[a, b] for a in range(2) for b in range(2, 20)]
    
    # Phase 2: Local search improvement
    if time.time() < deadline - 10:
        best_adj, best_count = local_search(best_adj, deadline - 10)
    
    # Convert to edge list
    edges = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                edges.append([u, v])
    
    # Final validation
    adj_check = [set() for _ in range(n)]
    for u, v in edges:
        adj_check[u].add(v)
        adj_check[v].add(u)
    has_k33, _ = check_k33(adj_check)
    if has_k33:
        # Fallback
        return [[a, b] for a in range(2) for b in range(2, 20)]
    
    return edges
