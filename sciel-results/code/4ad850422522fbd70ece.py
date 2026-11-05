import time
import random

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    adj = [0] * N
    adj_list = [[] for _ in range(N)]
    edges_set = set()
    
    def add_edge(u, v):
        if u == v: return
        if u > v: u, v = v, u
        if (u, v) in edges_set: return
        edges_set.add((u, v))
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        adj_list[u].append(v)
        adj_list[v].append(u)
        
    def remove_edge(u, v):
        if u > v: u, v = v, u
        if (u, v) not in edges_set: return
        edges_set.remove((u, v))
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        adj_list[u].remove(v)
        adj_list[v].remove(u)

    def creates_c6(u, v):
        # Check for path of length 5
        # u-a-b-c-d-e-v
        # a in N(u), e in N(v)
        # b in N(a), d in N(e)
        # c in N(b) & N(d)
        
        # Optimization: If u and v are already connected, we don't check (but we only call for non-edges)
        
        for a in adj_list[u]:
            for e in adj_list[v]:
                # Check path length 3 between a and e
                # b in N(a), d in N(e), c in N(b) & N(d)
                # Iterate b in N(a)
                for b in adj_list[a]:
                    nb = adj[b]
                    # Check if any d in N(e) has common neighbor with b
                    # i.e., nb & adj[d] != 0
                    for d in adj_list[e]:
                        if nb & adj[d]:
                            return True
        return False

    # Initialize
    all_edges = [(u, v) for u in range(N) for v in range(u+1, N)]
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        if time.time() > deadline - 5: break
        if not creates_c6(u, v):
            add_edge(u, v)
            
    best_edges = len(edges_set)
    best_edges_set = set(edges_set)
    
    non_edges = [e for e in all_edges if e not in edges_set]
    random.shuffle(non_edges)
    
    while time.time() < deadline - 2:
        if not non_edges: break
        
        u, v = non_edges.pop()
        
        if not creates_c6(u, v):
            add_edge(u, v)
            if len(edges_set) > best_edges:
                best_edges = len(edges_set)
                best_edges_set = set(edges_set)
        else:
            # Try swap
            if edges_set:
                # Pick random edge to remove
                # Using list(edges_set) is O(E). E~100.
                # To speed up, maintain a list of edges
                # But for simplicity, let's just use random.choice on a list
                # We can keep a list `current_edges_list`
                # For now, let's just pick random u,v from all_edges and check if in set
                # Or just use list(edges_set)
                e_out = random.choice(list(edges_set))
                remove_edge(e_out[0], e_out[1])
                
                if not creates_c6(u, v):
                    add_edge(u, v)
                    non_edges.append(e_out)
                else:
                    add_edge(e_out[0], e_out[1])

    return [[u, v] for u, v in best_edges_set]
