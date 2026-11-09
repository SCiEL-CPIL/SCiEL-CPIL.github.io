import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 40
    target_edges = 401
    
    def compute_max_book(edge_set, adj):
        """Compute max book size. adj: dict of vertex -> set of neighbors"""
        max_book = 0
        for u, v in edge_set:
            # Number of triangles containing edge (u,v) = |N(u) ∩ N(v)|
            common = len(adj[u] & adj[v])
            if common > max_book:
                max_book = common
        return max_book
    
    def build_adj(edge_set):
        adj = {i: set() for i in range(n)}
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def edge_key(e):
        return (min(e), max(e))
    
    # Start with a balanced bipartite-like structure
    # Partition into two sets of 20
    left = list(range(20))
    right = list(range(20, 40))
    
    # Start with K_{20,20} minus one edge, plus one intra-left edge
    # This gives 399 + 1 = 400... need 401
    # K_{20,20} = 400 edges. Remove 0, add 1 intra = 401.
    
    edge_set = set()
    for u in left:
        for v in right:
            edge_set.add(edge_key((u, v)))
    
    # Now we have 400 edges. Add one intra-partition edge.
    # To minimize max book, add an edge where the endpoints have low common neighbors.
    # In K_{20,20}, two vertices in left share all 20 right neighbors.
    # So any intra-left edge creates 20 triangles.
    # Instead, let's remove some bipartite edges and add intra edges strategically.
    
    # Better approach: Start with random graph near 401 edges, then local search.
    
    # Initialize: random graph with 401 edges
    random.seed(42)
    all_possible = list(itertools.combinations(range(n), 2))
    random.shuffle(all_possible)
    edge_set = set(edge_key(e) for e in all_possible[:target_edges])
    
    adj = build_adj(edge_set)
    current_max = compute_max_book(edge_set, adj)
    best_edge_set = set(edge_set)
    best_adj = adj
    best_max = current_max
    
    # Local search: try edge swaps (remove one, add another) to reduce max book
    # Also try single edge additions/removals to maintain 401 edges
    
    iteration = 0
    while time.time() < deadline - 5:
        iteration += 1
        
        # Pick a random edge to remove
        edges_list = list(edge_set)
        if not edges_list:
            break
        u_remove, v_remove = random.choice(edges_list)
        
        # Pick a random non-edge to add
        # Generate candidates more efficiently
        candidates = []
        for _ in range(20):  # Try a few random candidates
            u_add = random.randint(0, n-1)
            v_add = random.randint(0, n-1)
            if u_add == v_add:
                continue
            ek = edge_key((u_add, v_add))
            if ek not in edge_set:
                candidates.append(ek)
        
        if not candidates:
            continue
            
        u_add, v_add = random.choice(candidates)
        
        # Make the swap
        edge_set.remove(edge_key((u_remove, v_remove)))
        edge_set.add(edge_key((u_add, v_add)))
        
        # Update adjacency
        adj[u_remove].discard(v_remove)
        adj[v_remove].discard(u_remove)
        adj[u_add].add(v_add)
        adj[v_add].add(u_add)
        
        new_max = compute_max_book(edge_set, adj)
        
        if new_max < best_max:
            best_max = new_max
            best_edge_set = set(edge_set)
            best_adj = {k: set(v) for k, v in adj.items()}
        elif new_max < current_max:
            # Accept (greedy improvement)
            current_max = new_max
        else:
            # Revert
            edge_set.remove(edge_key((u_add, v_add)))
            edge_set.add(edge_key((u_remove, v_remove)))
            adj[u_add].discard(v_add)
            adj[v_add].discard(u_add)
            adj[u_remove].add(v_remove)
            adj[v_remove].add(u_remove)
            # current_max stays the same
        
        # Occasionally do a 2-swap for better exploration
        if iteration % 50 == 0 and time.time() < deadline - 10:
            # Try removing 2 edges and adding 2 different edges
            edges_list2 = list(edge_set)
            if len(edges_list2) >= 2:
                e1 = random.choice(edges_list2)
                edges_list2.remove(e1)
                e2 = random.choice(edges_list2)
                
                cand1, cand2 = None, None
                for _ in range(10):
                    u1 = random.randint(0, n-1)
                    v1 = random.randint(0, n-1)
                    if u1 != v1:
                        ck1 = edge_key((u1, v1))
                        if ck1 not in edge_set and ck1 != e1 and ck1 != e2:
                            cand1 = ck1
                            break
                for _ in range(10):
                    u2 = random.randint(0, n-1)
                    v2 = random.randint(0, n-1)
                    if u2 != v2:
                        ck2 = edge_key((u2, v2))
                        if ck2 not in edge_set and ck2 != e1 and ck2 != e2 and (cand1 is None or ck2 != cand1):
                            cand2 = ck2
                            break
                
                if cand1 is not None and cand2 is not None:
                    # Apply 2-swap
                    u1r, v1r = e1
                    u2r, v2r = e2
                    u1a, v1a = cand1
                    u2a, v2a = cand2
                    
                    edge_set.remove(e1)
                    edge_set.remove(e2)
                    edge_set.add(cand1)
                    edge_set.add(cand2)
                    
                    adj[u1r].discard(v1r)
                    adj[v1r].discard(u1r)
                    adj[u2r].discard(v2r)
                    adj[v2r].discard(u2r)
                    adj[u1a].add(v1a)
                    adj[v1a].add(u1a)
                    adj[u2a].add(v2a)
                    adj[v2a].add(u2a)
                    
                    new_max2 = compute_max_book(edge_set, adj)
                    
                    if new_max2 < best_max:
                        best_max = new_max2
                        best_edge_set = set(edge_set)
                        best_adj = {k: set(v) for k, v in adj.items()}
                    elif new_max2 < current_max:
                        current_max = new_max2
                    else:
                        # Revert
                        edge_set.remove(cand1)
                        edge_set.remove(cand2)
                        edge_set.add(e1)
                        edge_set.add(e2)
                        adj[u1a].discard(v1a)
                        adj[v1a].discard(u1a)
                        adj[u2a].discard(v2a)
                        adj[v2a].discard(u2a)
                        adj[u1r].add(v1r)
                        adj[v1r].add(u1r)
                        adj[u2r].add(v2r)
                        adj[v2r].add(u2r)
    
    # Use the best found
    result = [[u, v] for u, v in best_edge_set]
    
    # Verify exactly 401 edges
    if len(result) != target_edges:
        # Fallback: use a simple valid construction
        edge_set_fb = set()
        left = list(range(20))
        right = list(range(20, 40))
        for u in left:
            for v in right:
                edge_set_fb.add(edge_key((u, v)))
        # 400 edges, add one more
        edge_set_fb.add(edge_key((0, 1)))
        result = [[u, v] for u, v in edge_set_fb]
    
    return result
