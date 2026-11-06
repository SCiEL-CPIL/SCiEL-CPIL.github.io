import time
import random
import itertools

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s
    n = 20
    best_edges = None
    best_count = 0
    
    def count_edges(edges):
        return len(edges)
    
    def is_k23_free(adj):
        # Check that no two vertices have >= 3 common neighbors
        for u in range(n):
            for v in range(u+1, n):
                common = adj[u] & adj[v]
                if len(common) >= 3:
                    return False
        return True
    
    def get_edge_list(adj):
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges
    
    def try_greedy_add(adj, edge_order):
        """Try to add edges in given order if K2,3-free."""
        for u, v in edge_order:
            if u not in adj[v] and v not in adj[u]:
                # Check if adding (u,v) would create K2,3
                # After adding, check all pairs that might now have >=3 common neighbors
                # The new edge (u,v) means u's neighbors gain v, v's neighbors gain u
                # Pairs to check: (u, x) for x in N(u) - but actually we need to check
                # if any pair now has >=3 common neighbors
                # Adding (u,v): for any pair (a,b), common neighbors could increase
                # if a=u or b=u or a=v or b=v
                # Specifically: pairs involving u: (u, w) for w != v: common = N(u)∩N(w) gains nothing new from v unless v in N(w)
                # Actually: adding edge (u,v) means v becomes a neighbor of u and u becomes neighbor of v.
                # For pair (u, w): common neighbors = N(u) ∩ N(w). After adding, N(u) gains v.
                #   So new common = (N(u)∩N(w)) ∪ ({v} if v in N(w) else {})
                #   This increases by 1 iff v in N(w) and v not already in N(u)∩N(w)
                #   i.e., v in N(w) and v not in N(u) (which is true since we're adding)
                #   So if v in N(w), then |N(u)∩N(w)| increases by 1
                # Similarly for pair (v, w): |N(v)∩N(w)| increases by 1 iff u in N(w)
                # For pair (u,v): they were not adjacent, now they are. Common = N(u)∩N(v) (unchanged by adding the edge itself)
                #   Wait, N(u) and N(v) don't include each other in the adjacency for common neighbor count?
                #   Actually in simple graph, N(u) doesn't include u. When we add (u,v), v enters N(u) and u enters N(v).
                #   But common neighbors of u and v: vertices w such that w in N(u) and w in N(v).
                #   Before: N(u) and N(v) didn't include each other. After: N(u) has v, N(v) has u.
                #   But v is not a neighbor of v, and u is not a neighbor of u, so v not in N(v) and u not in N(u).
                #   So common neighbors of (u,v) unchanged.
                # 
                # So we need to check:
                # For each w in N(v) (excluding u): if u in N(w), then pair (u,w) gains common neighbor v
                #   Wait, I need to re-examine.
                # 
                # Let me just check all pairs involving u or v after the addition.
                
                # Save state
                adj[u].add(v)
                adj[v].add(u)
                
                # Check all pairs involving u or v
                ok = True
                for w in range(n):
                    if w == u or w == v:
                        continue
                    # Check pair (u, w)
                    if len(adj[u] & adj[w]) >= 3:
                        ok = False
                        break
                    # Check pair (v, w)
                    if len(adj[v] & adj[w]) >= 3:
                        ok = False
                        break
                
                if not ok:
                    adj[u].discard(v)
                    adj[v].discard(u)
                # else: keep the edge
        return adj
    
    def greedy_build(seed_val):
        """Build a K2,3-free graph greedily."""
        rng = random.Random(seed_val)
        adj = [set() for _ in range(n)]
        all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        rng.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() > deadline:
                break
            if v not in adj[u]:
                adj[u].add(v)
                adj[v].add(u)
                # Check if K2,3-free
                ok = True
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if len(adj[u] & adj[w]) >= 3 or len(adj[v] & adj[w]) >= 3:
                        ok = False
                        break
                if not ok:
                    adj[u].discard(v)
                    adj[v].discard(u)
        
        return adj
    
    def local_search(adj, seed_val, max_iter=5000):
        """Local search: try to add edges, remove+add pairs."""
        rng = random.Random(seed_val)
        current_adj = [set(s) for s in adj]
        current_edges = sum(len(s) for s in current_adj) // 2
        best_local = current_edges
        best_local_adj = [set(s) for s in current_adj]
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            action = rng.random()
            
            if action < 0.4:
                # Try to add a random non-edge
                non_edges = []
                for u in range(n):
                    for v in range(u+1, n):
                        if v not in current_adj[u]:
                            non_edges.append((u, v))
                if not non_edges:
                    continue
                u, v = rng.choice(non_edges)
                current_adj[u].add(v)
                current_adj[v].add(u)
                # Check
                ok = True
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if len(current_adj[u] & current_adj[w]) >= 3 or len(current_adj[v] & current_adj[w]) >= 3:
                        ok = False
                        break
                if ok:
                    current_edges += 1
                    if current_edges > best_local:
                        best_local = current_edges
                        best_local_adj = [set(s) for s in current_adj]
                else:
                    current_adj[u].discard(v)
                    current_adj[v].discard(u)
            
            elif action < 0.7:
                # Try to remove an edge and add another
                edges_list = []
                for u in range(n):
                    for v in current_adj[u]:
                        if u < v:
                            edges_list.append((u, v))
                if not edges_list:
                    continue
                u1, v1 = rng.choice(edges_list)
                current_adj[u1].discard(v1)
                current_adj[v1].discard(u1)
                current_edges -= 1
                
                non_edges = []
                for u in range(n):
                    for v in range(u+1, n):
                        if v not in current_adj[u]:
                            non_edges.append((u, v))
                if non_edges:
                    u2, v2 = rng.choice(non_edges)
                    current_adj[u2].add(v2)
                    current_adj[v2].add(u2)
                    current_edges += 1
                    ok = True
                    for w in range(n):
                        if w == u2 or w == v2:
                            continue
                        if len(current_adj[u2] & current_adj[w]) >= 3 or len(current_adj[v2] & current_adj[w]) >= 3:
                            ok = False
                            break
                    if not ok:
                        current_adj[u2].discard(v2)
                        current_adj[v2].discard(u2)
                        current_edges -= 1
                        # Re-add removed edge
                        current_adj[u1].add(v1)
                        current_adj[v1].add(u1)
                        current_edges += 1
                    else:
                        if current_edges > best_local:
                            best_local = current_edges
                            best_local_adj = [set(s) for s in current_adj]
                else:
                    current_adj[u1].add(v1)
                    current_adj[v1].add(u1)
                    current_edges += 1
            
            else:
                # Try to remove an edge (if it helps to add others later)
                edges_list = []
                for u in range(n):
                    for v in current_adj[u]:
                        if u < v:
                            edges_list.append((u, v))
                if edges_list:
                    u1, v1 = rng.choice(edges_list)
                    current_adj[u1].discard(v1)
                    current_adj[v1].discard(u1)
                    current_edges -= 1
                    # Try to add a random non-edge
                    non_edges = []
                    for u in range(n):
                        for v in range(u+1, n):
                            if v not in current_adj[u]:
                                non_edges.append((u, v))
                    if non_edges:
                        u2, v2 = rng.choice(non_edges)
                        current_adj[u2].add(v2)
                        current_adj[v2].add(u2)
                        current_edges += 1
                        ok = True
                        for w in range(n):
                            if w == u2 or w == v2:
                                continue
                            if len(current_adj[u2] & current_adj[w]) >= 3 or len(current_adj[v2] & current_adj[w]) >= 3:
                                ok = False
                                break
                        if not ok:
                            current_adj[u2].discard(v2)
                            current_adj[v2].discard(u2)
                            current_edges -= 1
                    if current_edges > best_local:
                        best_local = current_edges
                        best_local_adj = [set(s) for s in current_adj]
        
        return best_local_adj, best_local
    
    # Phase 1: Greedy builds with multiple seeds
    for seed in range(50):
        if time.time() > deadline:
            break
        adj = greedy_build(seed)
        edges = get_edge_list(adj)
        cnt = len(edges)
        if cnt > best_count:
            best_count = cnt
            best_edges = edges
    
    # Phase 2: Local search from best
    if best_edges is not None:
        # Reconstruct adjacency from best_edges
        adj = [set() for _ in range(n)]
        for u, v in best_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        for ls_seed in range(30):
            if time.time() > deadline:
                break
            best_adj, best_cnt = local_search(adj, ls_seed, max_iter=3000)
            if best_cnt > best_count:
                best_count = best_cnt
                best_edges = get_edge_list(best_adj)
    
    # Phase 3: More greedy + local search
    for seed in range(50, 200):
        if time.time() > deadline:
            break
        adj = greedy_build(seed)
        best_adj, best_cnt = local_search(adj, seed, max_iter=2000)
        if best_cnt > best_count:
            best_count = best_cnt
            best_edges = get_edge_list(best_adj)
    
    if best_edges is None:
        # Fallback: 5 disjoint K4's
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges
    
    return best_edges
