import time
import random
import math

N = 20
ALL_EDGES = [(i, j) for i in range(N) for j in range(i+1, N)]
ALL_EDGES_SET = set(ALL_EDGES)

def is_valid_add(u, v, adj):
    """
    Check if adding edge (u, v) keeps the graph K3,3-free.
    K3,3 formed iff exists x,y in N(u) with |N(x) ∩ N(y) ∩ N(v)| >= 2.
    """
    nu = adj[u]
    nv = adj[v]
    
    # Collect neighbors of u excluding v
    x_list = []
    for x in range(N):
        if x != u and x != v and (nu >> x) & 1:
            x_list.append(x)
    
    if len(x_list) < 2:
        return True
    
    # For each pair (x,y) in N(u), check if they share >= 2 common neighbors in N(v)
    for i in range(len(x_list)):
        x = x_list[i]
        ax_nv = adj[x] & nv  # neighbors of x that are also in N(v)
        for j in range(i+1, len(x_list)):
            y = x_list[j]
            # Count bits in (ax_nv & adj[y])
            common = ax_nv & adj[y]
            # Quick check: if common has >= 2 bits
            # Use: common & (common - 1) != 0 means >= 2 bits
            if common & (common - 1):
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

def local_search(adj, deadline):
    """
    Local search: add edges, then try improving swaps (remove k, add k+1).
    """
    best_adj = adj[:]
    best_count = count_edges(adj)
    current_count = best_count
    
    while time.time() < deadline:
        # Phase 1: Add any valid missing edge
        missing = get_missing_edges(adj)
        random.shuffle(missing)
        added = False
        for u, v in missing:
            if time.time() > deadline:
                break
            if is_valid_add(u, v, adj):
                add_edge(u, v, adj)
                current_count += 1
                added = True
                if current_count > best_count:
                    best_count = current_count
                    best_adj = adj[:]
                break
        
        if added:
            continue
        
        if time.time() > deadline:
            break
        
        # Phase 2: Try 1-for-2 swap (remove 1 edge, add 2)
        current = get_current_edges(adj)
        random.shuffle(current)
        found = False
        for u1, v1 in current:
            if time.time() > deadline:
                break
            
            saved = adj[:]
            remove_edge(u1, v1, adj)
            
            missing = get_missing_edges(adj)
            random.shuffle(missing)
            added_edges = []
            for u2, v2 in missing:
                if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                    continue
                if is_valid_add(u2, v2, adj):
                    add_edge(u2, v2, adj)
                    added_edges.append((u2, v2))
                    if len(added_edges) == 2:
                        break
            
            if len(added_edges) == 2:
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_adj = adj[:]
                found = True
                break
            else:
                for u2, v2 in added_edges:
                    remove_edge(u2, v2, adj)
                add_edge(u1, v1, adj)
                adj[:] = saved
        
        if found:
            continue
        
        # Phase 3: Try 2-for-3 swap (remove 2 edges, add 3)
        current = get_current_edges(adj)
        random.shuffle(current)
        found = False
        for i in range(len(current)):
            if time.time() > deadline:
                break
            u1, v1 = current[i]
            for j in range(i+1, len(current)):
                if time.time() > deadline:
                    break
                u2, v2 = current[j]
                
                saved = adj[:]
                remove_edge(u1, v1, adj)
                remove_edge(u2, v2, adj)
                
                missing = get_missing_edges(adj)
                random.shuffle(missing)
                added_edges = []
                for u3, v3 in missing:
                    if (u3, v3) in [(u1, v1), (v1, u1), (u2, v2), (v2, u2)]:
                        continue
                    if is_valid_add(u3, v3, adj):
                        add_edge(u3, v3, adj)
                        added_edges.append((u3, v3))
                        if len(added_edges) == 3:
                            break
                
                if len(added_edges) == 3:
                    current_count += 1
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = adj[:]
                    found = True
                    break
                else:
                    for u3, v3 in added_edges:
                        remove_edge(u3, v3, adj)
                    add_edge(u1, v1, adj)
                    add_edge(u2, v2, adj)
                    adj[:] = saved
            
            if found:
                break
        
        if found:
            continue
        
        # Phase 4: Try 1-for-1 neutral swap to escape local optima
        current = get_current_edges(adj)
        random.shuffle(current)
        made_move = False
        for u1, v1 in current:
            if time.time() > deadline:
                break
            
            saved = adj[:]
            remove_edge(u1, v1, adj)
            
            missing = get_missing_edges(adj)
            random.shuffle(missing)
            for u2, v2 in missing:
                if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                    continue
                if is_valid_add(u2, v2, adj):
                    add_edge(u2, v2, adj)
                    made_move = True
                    break
            
            if not made_move:
                add_edge(u1, v1, adj)
                adj[:] = saved
                continue
            else:
                break
        
        if not made_move:
            # Stuck; break out
            break
    
    return best_adj

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_adj = None
    best_count = 0
    
    restart = 0
    while time.time() < deadline - 10:
        restart += 1
        
        # Greedy construction with random edge order
        adj = [0] * N
        shuffled_edges = ALL_EDGES[:]
        random.shuffle(shuffled_edges)
        
        for u, v in shuffled_edges:
            if time.time() > deadline - 10:
                break
            if is_valid_add(u, v, adj):
                add_edge(u, v, adj)
        
        # Local search
        adj = local_search(adj, deadline - 5)
        
        current_count = count_edges(adj)
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
        
        # Early stop if we hit a high threshold
        if best_count >= 85:
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
