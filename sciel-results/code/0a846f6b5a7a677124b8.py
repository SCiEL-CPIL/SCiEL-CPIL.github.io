import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    
    n = 16
    all_vertices = set(range(n))
    
    # Q3 edge list (8 vertices, 12 edges) - standard cube
    q3_edges = [
        (0,1), (1,3), (3,2), (2,0),
        (4,5), (5,7), (7,6), (6,4),
        (0,4), (1,5), (3,7), (2,6)
    ]
    
    def has_q3_with_edge(adj_mask, u, v):
        """Check if adding edge u-v creates a Q3. adj_mask is list of int bitmasks."""
        neu = [x for x in range(n) if x != u and x != v and (adj_mask[u] >> x) & 1]
        nev = [x for x in range(n) if x != v and x != u and (adj_mask[v] >> x) & 1]
        
        if len(neu) < 2 or len(nev) < 2:
            return False
        
        for pos2 in neu:
            for pos4 in neu:
                if pos2 == pos4:
                    continue
                for pos3 in nev:
                    if pos3 == pos2 or pos3 == pos4:
                        continue
                    for pos5 in nev:
                        if pos5 == pos2 or pos5 == pos3 or pos5 == pos4:
                            continue
                        
                        if not ((adj_mask[pos2] >> pos3) & 1): continue
                        if not ((adj_mask[pos4] >> pos5) & 1): continue
                        
                        excluded = {u, v, pos2, pos3, pos4, pos5}
                        cand_pos6 = []
                        for x in range(n):
                            if x in excluded: continue
                            if ((adj_mask[pos2] >> x) & 1) and ((adj_mask[pos4] >> x) & 1):
                                cand_pos6.append(x)
                        
                        cand_pos7 = []
                        for x in range(n):
                            if x in excluded: continue
                            if ((adj_mask[pos3] >> x) & 1) and ((adj_mask[pos5] >> x) & 1):
                                cand_pos7.append(x)
                        
                        for pos6 in cand_pos6:
                            for pos7 in cand_pos7:
                                if pos6 == pos7: continue
                                if ((adj_mask[pos6] >> pos7) & 1) and \
                                   ((adj_mask[pos3] >> pos7) & 1) and \
                                   ((adj_mask[pos5] >> pos7) & 1):
                                    return True
        return False
    
    def count_edges(adj_mask):
        return sum(bin(adj_mask[i]).count('1') for i in range(n)) // 2
    
    def verify_no_q3(adj_mask):
        """Verify the graph has no Q3 subgraph."""
        for u in range(n):
            for v in range(u+1, n):
                if (adj_mask[u] >> v) & 1:
                    if has_q3_with_edge(adj_mask, u, v):
                        return False
        return True
    
    def is_valid(adj_mask):
        """Check if adj_mask represents a Q3-free graph."""
        return verify_no_q3(adj_mask)
    
    def try_add_edge(adj_mask, u, v):
        """Try to add edge u-v, return True if successful (no Q3 created)."""
        if (adj_mask[u] >> v) & 1:
            return True  # Already present
        
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
        if has_q3_with_edge(adj_mask, u, v):
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            return False
        return True
    
    def try_remove_edge(adj_mask, u, v):
        """Remove edge u-v from adj_mask."""
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
    
    def greedy_fill(adj_mask, max_time=None):
        """Greedily add edges to adj_mask until no more can be added or time runs out."""
        if max_time is None:
            max_time = deadline - 2
        
        current_edges = count_edges(adj_mask)
        
        while time.time() < max_time:
            progress = False
            # Get missing edges
            for i in range(n):
                for j in range(i+1, n):
                    if not ((adj_mask[i] >> j) & 1):
                        if try_add_edge(adj_mask, i, j):
                            current_edges += 1
                            progress = True
            
            if not progress:
                break
        
        return current_edges

    def local_search_swaps(adj_mask, num_iterations=200, max_time=None):
        """Local search with edge swaps: remove one edge, try to add up to k others."""
        if max_time is None:
            max_time = deadline - 2
        
        best_adj = adj_mask[:]
        best_edges = count_edges(adj_mask)
        
        for _ in range(num_iterations):
            if time.time() > max_time:
                break
            
            current_adj = best_adj[:]
            current_edges = count_edges(current_adj)
            
            # Collect all edges
            current_edge_list = []
            for i in range(n):
                for j in range(i+1, n):
                    if (current_adj[i] >> j) & 1:
                        current_edge_list.append((i, j))
            
            if not current_edge_list:
                break
            
            # Try removing a random edge and greedily refilling
            rem_u, rem_v = random.choice(current_edge_list)
            try_remove_edge(current_adj, rem_u, rem_v)
            current_edges -= 1
            
            # Greedy fill
            current_edges = greedy_fill(current_adj, max_time)
            
            if current_edges > best_edges:
                best_adj = current_adj[:]
                best_edges = current_edges
        
        return best_adj, best_edges
    
    def local_search_with_kicks(adj_mask, num_kicks=5, kick_size=3):
        """Perform local search with random kicks and edge swaps to escape local optima."""
        best_adj = adj_mask[:]
        best_edges = count_edges(adj_mask)
        
        for kick_iter in range(num_kicks):
            if time.time() > deadline - 3:
                break
            
            # Random kick: remove some random edges
            current_adj = best_adj[:]
            current_edges = count_edges(current_adj)
            
            # Remove kick_size random edges
            edges_to_remove = []
            for i in range(n):
                for j in range(i+1, n):
                    if (current_adj[i] >> j) & 1:
                        edges_to_remove.append((i, j))
            
            if edges_to_remove:
                to_remove = random.sample(edges_to_remove, min(kick_size, len(edges_to_remove)))
                for u, v in to_remove:
                    try_remove_edge(current_adj, u, v)
                    current_edges -= 1
            
            # Greedy fill
            current_edges = greedy_fill(current_adj)
            
            # Edge swap local search
            current_adj, current_edges = local_search_swaps(current_adj, num_iterations=50)
            
            # If improved, accept
            if current_edges > best_edges:
                best_adj = current_adj[:]
                best_edges = current_edges
        
        return best_adj, best_edges
    
    best_edge_count = 0
    best_adj_mask = None
    
    random.seed(42)
    
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Pre-define starting structures
    def make_k3_13():
        adj = [0] * n
        for u in range(3):
            for v in range(3, 16):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj
    
    def make_k8_8():
        adj = [0] * n
        for u in range(8):
            for v in range(8, 16):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj
    
    def make_k3_13_cycle():
        adj = [0] * n
        for u in range(3):
            for v in range(3, 16):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        for i in range(13):
            u = 3 + i
            v = 3 + ((i + 1) % 13)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def make_k4_12():
        adj = [0] * n
        for u in range(4):
            for v in range(4, 16):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        adj[0] |= (1 << 1)
        adj[1] |= (1 << 0)
        adj[0] |= (1 << 2)
        adj[2] |= (1 << 0)
        adj[0] |= (1 << 3)
        adj[3] |= (1 << 0)
        return adj
    
    def make_random(density):
        adj = [0] * n
        for i, j in all_pairs:
            if random.random() < density:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        return adj
    
    def clean_adj(adj):
        """Remove edges that are part of Q3 subgraphs."""
        changed = True
        while changed and time.time() < deadline - 3:
            changed = False
            for u in range(n):
                for v in range(u+1, n):
                    if (adj[u] >> v) & 1:
                        if has_q3_with_edge(adj, u, v):
                            try_remove_edge(adj, u, v)
                            changed = True
        return adj
    
    # Phase 1: Try different initializations with greedy fill
    initializations = [
        make_k3_13,
        make_k8_8,
        make_k3_13_cycle,
        make_k4_12,
        lambda: make_random(0.3),
        lambda: make_random(0.4),
        lambda: make_random(0.5),
        lambda: make_random(0.2),
    ]
    
    for init_func in initializations:
        if time.time() > deadline - 10:
            break
        
        adj = init_func()
        adj = clean_adj(adj)
        edges = greedy_fill(adj)
        
        if is_valid(adj):
            if edges > best_edge_count:
                best_edge_count = edges
                best_adj_mask = adj[:]
    
    # Phase 2: Local search with kicks from best solution
    if best_adj_mask is not None and time.time() < deadline - 15:
        improved_adj, improved_edges = local_search_with_kicks(best_adj_mask, num_kicks=20, kick_size=2)
        if improved_edges > best_edge_count:
            best_edge_count = improved_edges
            best_adj_mask = improved_adj
    
    # Phase 3: More random restarts with local search
    for restart in range(80):
        if time.time() > deadline - 10:
            break
        
        density = random.uniform(0.15, 0.55)
        adj = make_random(density)
        adj = clean_adj(adj)
        edges = greedy_fill(adj)
        
        if is_valid(adj):
            # Local search with kicks
            adj, edges = local_search_with_kicks(adj, num_kicks=3, kick_size=2)
            
            if edges > best_edge_count:
                best_edge_count = edges
                best_adj_mask = adj[:]
    
    # Phase 4: Final polish - edge swap search on best solution
    if best_adj_mask is not None and time.time() < deadline - 5:
        final_adj, final_edges = local_search_swaps(best_adj_mask, num_iterations=300)
        if final_edges > best_edge_count:
            best_edge_count = final_edges
            best_adj_mask = final_adj
    
    # If no valid graph found, return empty
    if best_adj_mask is None:
        best_adj_mask = [0] * n
        best_edge_count = 0
    
    # Return edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj_mask[i] >> j) & 1:
                edges.append([i, j])
    
    return edges
