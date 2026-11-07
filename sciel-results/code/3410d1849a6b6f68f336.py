import random
import time
from collections import defaultdict

# EVOLVE-BLOCK-START

def _check_valid(adj, n):
    """
    Check if the graph is valid.
    Valid: No simple path s-u-v-t has d(s) == d(t).
    """
    deg = [len(adj[i]) for i in range(n)]
    
    # Group vertices by degree
    deg_groups = defaultdict(list)
    for i in range(n):
        deg_groups[deg[i]].append(i)
        
    # For each group with size > 1, check all pairs
    for d_val, vertices in deg_groups.items():
        if len(vertices) < 2:
            continue
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                s = vertices[i]
                t = vertices[j]
                # Check if there is a path s-u-v-t
                # u in N(s), v in N(t), (u,v) in E, u!=v, u!=t, v!=s
                # Since s!=t, u!=t and v!=s are handled by simple path definition mostly, 
                # but we must ensure u != v.
                # Also u cannot be t (since t is not in N(s) if s,t not adjacent? 
                # Wait, if s,t are adjacent, t is in N(s). 
                # Path s-t-v-t? No, simple path.
                # Path s-u-v-t.
                # u in N(s). If u==t, then path is s-t-v-t. Not simple (t repeated).
                # So u != t.
                # v in N(t). If v==s, path is s-u-s-t. Not simple.
                # So v != s.
                
                found = False
                # Optimize: iterate over smaller neighborhood
                if len(adj[s]) < len(adj[t]):
                    for u in adj[s]:
                        if u == t:
                            continue
                        for v in adj[t]:
                            if v == s or v == u:
                                continue
                            if v in adj[u]:
                                found = True
                                break
                        if found:
                            break
                else:
                    for v in adj[t]:
                        if v == s:
                            continue
                        for u in adj[s]:
                            if u == t or u == v:
                                continue
                            if u in adj[v]:
                                found = True
                                break
                        if found:
                            break
                if found:
                    return False
    return True

def _count_edges(adj):
    return sum(len(s) for s in adj) // 2

def _construct_bipartite(n, rng):
    """Construct a complete bipartite graph K_{k, n-k}."""
    # Try different splits
    best_adj = None
    best_count = -1
    
    for k in range(1, n):
        adj = [set() for _ in range(n)]
        # Partition: 0..k-1 and k..n-1
        for i in range(k):
            for j in range(k, n):
                adj[i].add(j)
                adj[j].add(i)
        
        cnt = _count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = [set(s) for s in adj]
            
    return best_adj, best_count

def _local_search(adj, n, rng, deadline):
    """
    Local search to maximize edges while maintaining validity.
    """
    cur_adj = [set(s) for s in adj]
    cur_count = _count_edges(cur_adj)
    best_adj = [set(s) for s in cur_adj]
    best_count = cur_count
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    no_improve = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            break
            
        # Try to add a random missing edge
        missing = [(i, j) for i, j in all_edges if j not in cur_adj[i]]
        if not missing:
            break
            
        rng.shuffle(missing)
        improved = False
        
        for (i, j) in missing:
            if time.time() > deadline:
                return best_adj, best_count
                
            # Try adding (i, j)
            cur_adj[i].add(j)
            cur_adj[j].add(i)
            
            if _check_valid(cur_adj, n):
                cur_count += 1
                improved = True
                if cur_count > best_count:
                    best_count = cur_count
                    best_adj = [set(s) for s in cur_adj]
                    no_improve = 0
                break
            else:
                # Revert
                cur_adj[i].discard(j)
                cur_adj[j].discard(i)
        
        if not improved:
            no_improve += 1
            # Try a swap move: remove one edge, try to add another
            # This helps escape local minima where no single addition works
            existing = [(i, j) for i, j in all_edges if j in cur_adj[i]]
            if existing:
                rng.shuffle(existing)
                for (a, b) in existing[:10]: # Limit swap attempts
                    if time.time() > deadline:
                        return best_adj, best_count
                        
                    cur_adj[a].discard(b)
                    cur_adj[b].discard(a)
                    temp_count = cur_count - 1
                    
                    # Try to add a different edge
                    missing_after_removal = [(i, j) for i, j in all_edges if j not in cur_adj[i]]
                    rng.shuffle(missing_after_removal)
                    swap_improved = False
                    for (c, d) in missing_after_removal:
                        if (c == a and d == b) or (c == b and d == a):
                            continue
                        cur_adj[c].add(d)
                        cur_adj[d].add(c)
                        if _check_valid(cur_adj, n):
                            # Keep the swap
                            cur_count = temp_count + 1
                            swap_improved = True
                            if cur_count > best_count:
                                best_count = cur_count
                                best_adj = [set(s) for s in cur_adj]
                                no_improve = 0
                            break
                        else:
                            cur_adj[c].discard(d)
                            cur_adj[d].discard(c)
                    
                    if not swap_improved:
                        # Revert removal
                        cur_adj[a].add(b)
                        cur_adj[b].add(a)
                        cur_count = temp_count + 1
                    else:
                        improved = True
                        break
                        
        if improved:
            no_improve = 0

    return best_adj, best_count

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 45.0 # Safety margin for 60s limit
    
    rng = random.Random(42)
    
    best_adj = None
    best_count = -1
    
    # 1. Start with best bipartite graph
    bip_adj, bip_count = _construct_bipartite(n, rng)
    best_adj = bip_adj
    best_count = bip_count
    
    # 2. Local search from bipartite
    ls_adj, ls_count = _local_search(bip_adj, n, rng, deadline - 10.0)
    if ls_count > best_count:
        best_count = ls_count
        best_adj = ls_adj
        
    # 3. Random restarts with dense random graphs
    # Try a few random dense graphs to see if they yield better local optima
    attempts = 0
    while time.time() < deadline - 5.0 and attempts < 20:
        attempts += 1
        adj = [set() for _ in range(n)]
        # Random dense graph (p=0.8)
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.8:
                    adj[i].add(j)
                    adj[j].add(i)
                    
        # If invalid, it's a bad start, but local search might fix it?
        # Better to just run local search which adds/edges.
        # Actually, if it starts invalid, _local_search might get stuck or behave weirdly.
        # Let's ensure we start valid or handle invalid starts.
        # For now, skip if invalid to save time?
        # Or just let local search try to add edges (which might not fix invalidity).
        # The current _local_search only adds edges. It doesn't remove to fix invalidity.
        # So we need a valid start.
        
        # Let's try a greedy construction for random restarts
        adj_greedy = [set() for _ in range(n)]
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(edges)
        for i, j in edges:
            if time.time() > deadline - 5.0:
                break
            adj_greedy[i].add(j)
            adj_greedy[j].add(i)
            if not _check_valid(adj_greedy, n):
                adj_greedy[i].discard(j)
                adj_greedy[j].discard(i)
                
        g_count = _count_edges(adj_greedy)
        if g_count > best_count:
            best_count = g_count
            best_adj = adj_greedy
            
        # Refine
        ls_adj2, ls_count2 = _local_search(adj_greedy, n, rng, deadline - 2.0)
        if ls_count2 > best_count:
            best_count = ls_count2
            best_adj = ls_adj2

    # Build edge list
    edges = []
    for i in range(n):
        for j in best_adj[i]:
            if i < j:
                edges.append([int(i), int(j)])
                
    return edges

# EVOLVE-BLOCK-END
