import time
import random

N = 20
ALL_EDGES = [(i, j) for i in range(N) for j in range(i+1, N)]
ALL_EDGES_SET = set(ALL_EDGES)

# Precompute: for each vertex w, list of (triple_idx, x, y) where x,y are the other two in a triple containing w
# A triple is a set of 3 vertices {a,b,c} with a<b<c
triples = []
triples_for_vertex = [[] for _ in range(N)]
for a in range(N):
    for b in range(a+1, N):
        for c in range(b+1, N):
            t_idx = len(triples)
            triples.append((a, b, c))
            triples_for_vertex[a].append((t_idx, b, c))
            triples_for_vertex[b].append((t_idx, a, c))
            triples_for_vertex[c].append((t_idx, a, b))

def is_valid_add(u, v, adj):
    """
    Check if adding edge (u, v) to graph represented by adj (list of bitmasks)
    results in a K3,3-free graph.
    
    A K3,3 is formed if there exist two disjoint triples {a1,a2,a3} and {b1,b2,b3}
    such that all 9 edges between them exist.
    
    When adding edge (u,v), the new K3,3 must use this edge. So u and v are in
    opposite parts. WLOG u is in part A, v is in part B.
    Then we need: two more vertices a1,a2 in A (adjacent to v and to b1,b2 in B)
    and two more vertices b1,b2 in B (adjacent to u and to a1,a2 in A).
    
    Equivalent check: for pairs (x,y) in N(u) (i.e., x,y adjacent to u),
    check if |N(x) ∩ N(y) ∩ N(v)| >= 2.
    """
    nu = adj[u]
    nv = adj[v]
    
    # Iterate over vertices x in N(u)
    # For each pair (x, y) both in N(u), check common neighbors in N(v)
    x_list = []
    for x in range(N):
        if x != u and x != v and (nu >> x) & 1:
            x_list.append(x)
    
    for i in range(len(x_list)):
        x = x_list[i]
        ax = adj[x]
        for j in range(i+1, len(x_list)):
            y = x_list[j]
            common = ax & adj[y] & nv
            if common.bit_count() >= 2:
                return False
    
    return True

def add_edge(u, v, adj):
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)

def remove_edge(u, v, adj):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)

def count_edges(adj):
    return sum(adj[i].bit_count() for i in range(N)) // 2

def get_missing_edges(adj):
    missing = []
    for u, v in ALL_EDGES:
        if not ((adj[u] >> v) & 1):
            missing.append((u, v))
    return missing

def get_current_edges(adj):
    current = []
    for u, v in ALL_EDGES:
        if (adj[u] >> v) & 1:
            current.append((u, v))
    return current

def hill_climbing(adj, deadline):
    """
    Hill climbing with add, 1-swap (remove 1 add 2), and 2-swap (remove 2 add 3).
    Returns improved adj.
    """
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add any missing edge
        missing = get_missing_edges(adj)
        random.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline:
                return adj
            if is_valid_add(u, v, adj):
                add_edge(u, v, adj)
                improved = True
                break
        
        if time.time() > deadline:
            return adj
        
        if improved:
            continue
        
        # Phase 2: Try 1-swap (remove 1 edge, add 2 edges)
        current = get_current_edges(adj)
        random.shuffle(current)
        found = False
        for u1, v1 in current:
            if time.time() > deadline:
                return adj
            
            saved_adj = adj[:]
            remove_edge(u1, v1, adj)
            
            missing = get_missing_edges(adj)
            random.shuffle(missing)
            added = []
            for u2, v2 in missing:
                if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                    continue
                if is_valid_add(u2, v2, adj):
                    add_edge(u2, v2, adj)
                    added.append((u2, v2))
                    if len(added) == 2:
                        break
            
            if len(added) == 2:
                improved = True
                found = True
                break
            else:
                # Restore
                for u2, v2 in added:
                    remove_edge(u2, v2, adj)
                add_edge(u1, v1, adj)
        
        if time.time() > deadline:
            return adj
        if found:
            continue
        
        # Phase 3: Try 2-swap (remove 2 edges, add 3 edges)
        current = get_current_edges(adj)
        random.shuffle(current)
        found = False
        for i in range(len(current)):
            if time.time() > deadline:
                return adj
            u1, v1 = current[i]
            for j in range(i+1, len(current)):
                if time.time() > deadline:
                    return adj
                u2, v2 = current[j]
                
                # Check they don't share a vertex (to keep it simple, allow sharing too)
                saved_adj = adj[:]
                remove_edge(u1, v1, adj)
                remove_edge(u2, v2, adj)
                
                missing = get_missing_edges(adj)
                random.shuffle(missing)
                added = []
                for u3, v3 in missing:
                    if (u3, v3) in [(u1, v1), (v1, u1), (u2, v2), (v2, u2)]:
                        continue
                    if is_valid_add(u3, v3, adj):
                        add_edge(u3, v3, adj)
                        added.append((u3, v3))
                        if len(added) == 3:
                            break
                
                if len(added) == 3:
                    improved = True
                    found = True
                    break
                else:
                    # Restore
                    for u3, v3 in added:
                        remove_edge(u3, v3, adj)
                    add_edge(u1, v1, adj)
                    add_edge(u2, v2, adj)
            
            if found:
                break
        
        if time.time() > deadline:
            return adj

    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin (evaluator kills at 150s)
    
    best_adj = None
    best_count = 0
    
    # Try multiple random restarts
    restart = 0
    while time.time() < deadline - 5:
        restart += 1
        
        # Greedy construction with random edge order
        adj = [0] * N
        shuffled_edges = ALL_EDGES[:]
        random.shuffle(shuffled_edges)
        
        for u, v in shuffled_edges:
            if time.time() > deadline - 5:
                break
            if is_valid_add(u, v, adj):
                add_edge(u, v, adj)
        
        # Hill climbing
        adj = hill_climbing(adj, deadline - 5)
        
        current_count = count_edges(adj)
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
        
        # Early stop if we hit a high threshold
        if best_count >= 80:
            break

    if best_adj is None:
        return []
        
    # Convert adj back to list of edges
    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
