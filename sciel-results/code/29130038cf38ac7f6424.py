import random
import time

def construct():
    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_index = {e: i for i, e in enumerate(all_edges)}
    
    best_edges = set()
    deadline = time.time() + 140  # Safety margin
    
    def is_c4_free(edge_set):
        # Check if adding an edge creates a C4
        # This function is for checking a specific edge addition to a set
        pass

    def can_add(edge_set, adj, u, v):
        # Check if (u,v) can be added to edge_set (represented by adj)
        # without creating a 4-cycle
        # A 4-cycle u-a-b-v-u is created if exists a in N(u), b in N(v)
        # such that (a,b) is an edge.
        # Note: a != b, a != v, b != u.
        # Since graph is simple, u,v not in N(u) etc.
        # We need to check if there is a path of length 2 from u to v?
        # No, path u-a-b-v means (u,a), (a,b), (b,v) are edges.
        # So for each b in N(v), check if N(u) intersects N(b).
        for b in adj[v]:
            # Check if any neighbor of u is also a neighbor of b
            # If so, there is a path u-a-b-v
            # This forms a 4-cycle u-a-b-v-u
            if adj[u].intersection(adj[b]):
                return False
        return True

    def build_greedy():
        edges_set = set()
        adj = [set() for _ in range(n)]
        edges_list = all_edges[:]
        random.shuffle(edges_list)
        
        for u, v in edges_list:
            if can_add(edges_set, adj, u, v):
                edges_set.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
        return edges_set, adj

    def local_search(edges_set, adj):
        # Try to improve by swaps
        # We maintain edges_set and adj
        # We want to increase size, so we first try to add.
        # If stuck, we swap.
        
        # Pre-compute non-edges?
        # Just pick random non-edges
        
        for _ in range(500): # Fixed number of local search steps per restart
            if time.time() > deadline:
                break
                
            # 1. Try to add a random non-edge
            # Pick random u, v
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            if u > v: u, v = v, u
            
            if (u, v) not in edges_set:
                if can_add(edges_set, adj, u, v):
                    edges_set.add((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
                    continue
            
            # 2. Try a swap
            # Pick a random edge e in E
            if not edges_set: continue
            e = random.choice(list(edges_set))
            e_u, e_v = e
            
            # Remove e
            edges_set.remove(e)
            adj[e_u].remove(e_v)
            adj[e_v].remove(e_u)
            
            # Try to add a random non-edge e'
            # To be efficient, we can try to add the e' we failed to add earlier?
            # Or just random.
            # Let's try to add a random non-edge
            added = False
            for _ in range(5): # Try a few candidates
                u2 = random.randint(0, n-1)
                v2 = random.randint(0, n-1)
                if u2 == v2: continue
                if u2 > v2: u2, v2 = v2, u2
                
                if (u2, v2) not in edges_set:
                    if can_add(edges_set, adj, u2, v2):
                        edges_set.add((u2, v2))
                        adj[u2].add(v2)
                        adj[v2].add(u2)
                        added = True
                        break
            
            if not added:
                # Restore e
                edges_set.add(e)
                adj[e_u].add(e_v)
                adj[e_v].add(e_u)
                
        return edges_set, adj

    # Main loop
    while time.time() < deadline:
        edges_set, adj = build_greedy()
        edges_set, adj = local_search(edges_set, adj)
        
        if len(edges_set) > len(best_edges):
            best_edges = set(edges_set)
            # Optional: print(len(best_edges))
            
    # Color all edges with 0
    result = []
    for u, v in best_edges:
        result.append([u, v, 0])
        
    return result
