"""Dense K2,3-free graph on 20 vertices via greedy + local search."""
import time
import random

N = 20
DEADLINE_MARGIN = 5.0  # seconds to stay under the 150s wall-clock kill

def _common_neighbours_count(adj, u, v):
    """Return the number of common neighbours of u and v using adjacency sets."""
    return len(adj[u] & adj[v])

def _is_valid(adj):
    """Check no two vertices have 3+ common neighbours."""
    for u in range(N):
        for v in range(u + 1, N):
            if len(adj[u] & adj[v]) >= 3:
                return False
    return True

def _count_edges(adj):
    return sum(len(adj[u]) for u in range(N)) // 2

def _greedy_build(seed):
    """Greedy construction: add edges in random order if valid."""
    rng = random.Random(seed)
    adj = [set() for _ in range(N)]
    # Generate all possible edges
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    rng.shuffle(all_edges)
    
    for u, v in all_edges:
        # Check if adding edge (u,v) would create a K2,3
        # Adding (u,v) increases common neighbours for pairs (u,w) and (v,w) for all w != u,v
        # Specifically, for any w, if w is common neighbour of u and v already, that's fine.
        # The constraint is: for any pair (a,b), |N(a) ∩ N(b)| <= 2.
        # Adding edge (u,v):
        # - For pair (u, v): their common neighbours are N(u) ∩ N(v). After adding, v is in N(u) and u is in N(v).
        #   New common neighbours of u and v: (N(u) ∩ N(v)) ∪ {nothing new from the edge itself unless we consider the edge creates new commonality}
        #   Actually, N(u) and N(v) change: v is added to N(u), u is added to N(v).
        #   So new N(u) = N(u) ∪ {v}, new N(v) = N(v) ∪ {u}.
        #   New common neighbours of u and v = (N(u) ∪ {v}) ∩ (N(v) ∪ {u})
        #   = (N(u) ∩ N(v)) ∪ (N(u) ∩ {u}) ∪ ({v} ∩ N(v)) ∪ ({v} ∩ {u})
        #   = (N(u) ∩ N(v)) ∪ ({v} if v in N(v) else empty) ... wait, v is not in N(v) in simple graph.
        #   Let's just check the condition directly.
        
        # Potential violations:
        # 1. Pair (u, v): common neighbours become |N(u) ∩ N(v)|. Note that u is not in N(u), v is not in N(v).
        #    After adding edge, N'(u) = N(u) ∪ {v}, N'(v) = N(v) ∪ {u}.
        #    N'(u) ∩ N'(v) = (N(u) ∪ {v}) ∩ (N(v) ∪ {u})
        #    = (N(u) ∩ N(v)) ∪ (N(u) ∩ {u}) ∪ ({v} ∩ N(v)) ∪ ({v} ∩ {u})
        #    Since N(u) doesn't contain u, N(v) doesn't contain v, and u!=v:
        #    = N(u) ∩ N(v)
        #    So the number of common neighbours of u and v remains |N(u) ∩ N(v)|.
        #    We need |N(u) ∩ N(v)| <= 2.
        
        # 2. For any w != u, v:
        #    Pair (u, w): N'(u) = N(u) ∪ {v}, N'(w) = N(w).
        #    Common = (N(u) ∪ {v}) ∩ N(w) = (N(u) ∩ N(w)) ∪ ({v} ∩ N(w))
        #    Increase by 1 if v is in N(w).
        #    So we need |N(u) ∩ N(w)| + (1 if v in N(w) else 0) <= 2.
        #    i.e., if v in N(w), then |N(u) ∩ N(w)| <= 1.
        
        #    Pair (v, w): similarly, if u in N(w), then |N(v) ∩ N(w)| <= 1.
        
        # 3. Pair (w1, w2) where w1, w2 != u, v: no change.
        
        # So check:
        # a) |N(u) ∩ N(v)| <= 2
        # b) For all w in N(v) (w != u): |N(u) ∩ N(w)| <= 1
        # c) For all w in N(u) (w != v): |N(v) ∩ N(w)| <= 1
        
        if len(adj[u] & adj[v]) >= 3:
            continue
        
        valid = True
        # Check condition b: for w in adj[v], w != u
        for w in adj[v]:
            if w == u:
                continue
            if len(adj[u] & adj[w]) >= 2:
                valid = False
                break
        if not valid:
            continue
        
        # Check condition c: for w in adj[u], w != v
        for w in adj[u]:
            if w == v:
                continue
            if len(adj[v] & adj[w]) >= 2:
                valid = False
                break
        if not valid:
            continue
        
        # Add edge
        adj[u].add(v)
        adj[v].add(u)
    
    return adj

def _local_search(adj, seed, max_iter=2000):
    """Local search: try to add edges, and if stuck, try swapping (remove one, add one)."""
    rng = random.Random(seed)
    adj = [set(s) for s in adj]  # deep copy
    current_edges = _count_edges(adj)
    best_edges = current_edges
    best_adj = [set(s) for s in adj]
    
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    
    def can_add(u, v):
        if v in adj[u]:
            return True
        if len(adj[u] & adj[v]) >= 3:
            return False
        for w in adj[v]:
            if w != u and len(adj[u] & adj[w]) >= 2:
                return False
        for w in adj[u]:
            if w != v and len(adj[v] & adj[w]) >= 2:
                return False
        return True
    
    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)
    
    def remove_edge(u, v):
        adj[u].discard(v)
        adj[v].discard(u)
    
    for iteration in range(max_iter):
        if time.time() > deadline - DEADLINE_MARGIN:
            break
        
        # Try to add a random missing edge
        improved = False
        missing = [(u, v) for u, v in all_edges if v not in adj[u]]
        if not missing:
            break
        
        rng.shuffle(missing)
        for u, v in missing[:50]:  # Try up to 50 random missing edges
            if can_add(u, v):
                add_edge(u, v)
                current_edges += 1
                improved = True
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = [set(s) for s in adj]
                break
        
        if not improved:
            # Try a swap: remove a random edge, then try to add two edges
            # Or just try to find any addable edge after a random removal
            present = [(u, v) for u, v in all_edges if v in adj[u]]
            if not present:
                break
            rng.shuffle(present)
            for u_rm, v_rm in present[:10]:
                remove_edge(u_rm, v_rm)
                current_edges -= 1
                
                # Try to add edges
                added = 0
                missing_now = [(u, v) for u, v in all_edges if v not in adj[u]]
                rng.shuffle(missing_now)
                for u_add, v_add in missing_now[:30]:
                    if can_add(u_add, v_add):
                        add_edge(u_add, v_add)
                        current_edges += 1
                        added += 1
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj = [set(s) for s in adj]
                        if added >= 1:
                            break
                
                if added == 0:
                    # Revert
                    add_edge(u_rm, v_rm)
                    current_edges += 1
    
    return best_adj, best_edges

def construct():
    global deadline
    deadline = time.time() + 150.0
    
    best_adj = None
    best_edges = 0
    
    # Start with greedy builds with different seeds
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111]
    
    for seed in seeds:
        if time.time() > deadline - DEADLINE_MARGIN:
            break
        adj = _greedy_build(seed)
        edges = _count_edges(adj)
        if edges > best_edges:
            best_edges = edges
            best_adj = adj
    
    if best_adj is None:
        # Fallback: 5 copies of K4
        adj = [set() for _ in range(N)]
        for k in range(5):
            for a in range(4):
                for b in range(a + 1, 4):
                    u = 4 * k + a
                    v = 4 * k + b
                    adj[u].add(v)
                    adj[v].add(u)
        best_adj = adj
        best_edges = _count_edges(adj)
    
    # Local search on best
    if time.time() < deadline - DEADLINE_MARGIN:
        best_adj, best_edges = _local_search(best_adj, 42, max_iter=5000)
    
    # Convert to edge list
    edges = []
    for u in range(N):
        for v in adj[u]:
            if u < v:
                edges.append([u, v])
    
    return edges

# Pre-compute deadline for local search
deadline = 0
