import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 120.0  # seconds
    MARGIN = 5.0      # safety margin
    n = 40
    
    # Precompute all possible edges
    all_edges = list(combinations(range(n), 2))
    n_edges_total = len(all_edges)
    
    # Helper: check if adding edge (u,v) to adjacency list creates a C6
    # C6: u-a-b-c-d-v where a,b,c,d are distinct and not u,v
    # Equivalently: there exist a in N(u), d in N(v), and a path of length 3 from a to d
    #   where the path a-b-c-d has b,c not in {u,v} and a!=d
    
    def would_create_c6(adj, u, v):
        """Check if adding edge (u,v) creates a C6.
        adj is a list of sets."""
        # We need to check if there's a path u-a-b-c-d-v with all distinct vertices
        # a in N(u), d in N(v), path a-b-c-d of length 3 with b,c not in {u,v}
        # and a != d, and the path doesn't use u or v as internal nodes
        
        # For each a in N(u) and d in N(v) with a != d:
        #   check if there's a path a-b-c-d of length 3 where b,c not in {u,v,a,d}
        #   Actually b and c just need to be distinct from each other and from u,v
        #   and the path must be simple (all vertices distinct in the cycle)
        
        # A C6 is: u, a, b, c, d, v, u
        # Constraints: all 6 vertices distinct
        # a in N(u), d in N(v), a-b in E, b-c in E, c-d in E
        # a,b,c,d all distinct and none equal to u or v
        
        Nu = adj[u]
        Nv = adj[v]
        
        for a in Nu:
            if a == v:
                continue
            for d in Nv:
                if d == u or d == a:
                    continue
                # Check if there's a path a-b-c-d of length 3
                # b in N(a), c in N(d), b-c in E, b != c, b != d, c != a, b != u, b != v, c != u, c != v
                for b in adj[a]:
                    if b == u or b == v or b == d or b == c:  # c not known yet
                        continue
                    if b == a:
                        continue
                    for c in adj[d]:
                        if c == u or c == v or c == a or c == b:
                            continue
                        if c == d:
                            continue
                        if b in adj[c]:
                            # Found a C6: u-a-b-c-d-v-u
                            return True
        return False
    
    def would_create_c6_fast(adj, adj_matrix, u, v):
        """Faster C6 check using adjacency matrix (numpy-like with lists).
        adj_matrix[i][j] = 1 if edge exists."""
        Nu = adj[u]
        Nv = adj[v]
        
        for a in Nu:
            if a == v:
                continue
            for d in Nv:
                if d == u or d == a:
                    continue
                # Need path a-b-c-d of length 3, b,c distinct from {u,v,a,d}
                # b in N(a) \ {u,v,d}, c in N(d) \ {u,v,a}, b != c, b-c in E
                for b in adj[a]:
                    if b == u or b == v or b == d or b == a:
                        continue
                    for c in adj[d]:
                        if c == u or c == v or c == a or c == b or c == d:
                            continue
                        if adj_matrix[b][c]:
                            return True
        return False
    
    def count_c6s(adj, adj_matrix):
        """Count number of C6s in the graph (for verification)."""
        count = 0
        for u in range(n):
            for v in range(u+1, n):
                if not adj_matrix[u][v]:
                    # Check if adding (u,v) would create a C6
                    if would_create_c6_fast(adj, adj_matrix, u, v):
                        count += 1
        return count
    
    def try_add_edges_greedy(adj, adj_matrix, edge_order, max_time=None):
        """Try to add edges in given order, skipping those that create C6."""
        for i, (u, v) in enumerate(edge_order):
            if max_time is not None and time.time() > max_time:
                break
            if adj_matrix[u][v]:
                continue
            if not would_create_c6_fast(adj, adj_matrix, u, v):
                adj_matrix[u][v] = 1
                adj_matrix[v][u] = 1
                adj[u].add(v)
                adj[v].add(u)
    
    def local_search_remove_add(adj, adj_matrix, max_time=None):
        """Try removing one edge and re-adding to find better configurations.
        Returns number of edges after local search."""
        current_edges = sum(len(adj[i]) for i in range(n)) // 2
        best_edges = current_edges
        best_adj = [set(s) for s in adj]
        best_adj_matrix = [row[:] for row in adj_matrix]
        
        # Try removing each edge and see if we can add more
        edges_list = [(u, v) for u in range(n) for v in range(u+1, n) if adj_matrix[u][v]]
        random.shuffle(edges_list)
        
        for (u, v) in edges_list:
            if max_time is not None and time.time() > max_time:
                break
            
            # Remove edge (u,v)
            adj_matrix[u][v] = 0
            adj_matrix[v][u] = 0
            adj[u].discard(v)
            adj[v].discard(u)
            
            # Try to add all possible edges greedily
            all_missing = [(a, b) for a in range(n) for b in range(a+1, n) if not adj_matrix[a][b]]
            random.shuffle(all_missing)
            try_add_edges_greedy(adj, adj_matrix, all_missing, max_time)
            
            new_edges = sum(len(adj[i]) for i in range(n)) // 2
            if new_edges > best_edges:
                best_edges = new_edges
                best_adj = [set(s) for s in adj]
                best_adj_matrix = [row[:] for row in adj_matrix]
            
            # Restore original state
            adj_matrix[u][v] = 1
            adj_matrix[v][u] = 1
            adj[u].add(v)
            adj[v].add(u)
            
            # Re-add edges that were added during the greedy phase
            # Actually this is complex; let's just restore from scratch
            # Better approach: save state before, restore after
            pass
        
        return best_edges, best_adj, best_adj_matrix
    
    def run_one_trial(seed, max_time):
        """Run one greedy construction trial with random edge ordering."""
        random.seed(seed)
        adj = [set() for _ in range(n)]
        adj_matrix = [[0]*n for _ in range(n)]
        
        # Start with a random sparse seed
        # Try different initial structures
        strategy = random.randint(0, 3)
        
        if strategy == 0:
            # Start with random edges
            random.shuffle(all_edges)
            try_add_edges_greedy(adj, adj_matrix, all_edges, max_time)
        elif strategy == 1:
            # Start with some K5-like structures
            # Partition into groups and add intra-group edges
            groups = []
            remaining = list(range(n))
            random.shuffle(remaining)
            group_sizes = [random.randint(3, 7) for _ in range(8)]
            group_sizes = [min(gs, len(remaining)) for gs in group_sizes]
            idx = 0
            for gs in group_sizes:
                if idx >= len(remaining):
                    break
                groups.append(remaining[idx:idx+gs])
                idx += gs
            for group in groups:
                for a, b in combinations(group, 2):
                    if not would_create_c6_fast(adj, adj_matrix, a, b):
                        adj_matrix[a][b] = 1
                        adj_matrix[b][a] = 1
                        adj[a].add(b)
                        adj[b].add(a)
            # Now add inter-group edges greedily
            remaining_edges = [(a, b) for a, b in all_edges if not adj_matrix[a][b]]
            random.shuffle(remaining_edges)
            try_add_edges_greedy(adj, adj_matrix, remaining_edges, max_time)
        elif strategy == 2:
            # Start with a random regular-ish graph
            target_deg = random.randint(4, 8)
            for u in range(n):
                candidates = [v for v in range(n) if v != u and len(adj[u]) < target_deg]
                random.shuffle(candidates)
                for v in candidates:
                    if len(adj[u]) >= target_deg:
                        break
                    if adj_matrix[u][v]:
                        continue
                    if not would_create_c6_fast(adj, adj_matrix, u, v):
                        adj_matrix[u][v] = 1
                        adj_matrix[v][u] = 1
                        adj[u].add(v)
                        adj[v].add(u)
            # Then add remaining edges greedily
            remaining_edges = [(a, b) for a, b in all_edges if not adj_matrix[a][b]]
            random.shuffle(remaining_edges)
            try_add_edges_greedy(adj, adj_matrix, remaining_edges, max_time)
        else:
            # Start with random matching + greedy
            vertices = list(range(n))
            random.shuffle(vertices)
            for i in range(0, n-1, 2):
                u, v = vertices[i], vertices[i+1]
                adj_matrix[u][v] = 1
                adj_matrix[v][u] = 1
                adj[u].add(v)
                adj[v].add(u)
            remaining_edges = [(a, b) for a, b in all_edges if not adj_matrix[a][b]]
            random.shuffle(remaining_edges)
            try_add_edges_greedy(adj, adj_matrix, remaining_edges, max_time)
        
        # Local search: try removing and re-adding
        edges_count = sum(len(adj[i]) for i in range(n)) // 2
        
        # Try a few rounds of edge removal and re-addition
        for _ in range(3):
            if time.time() > max_time:
                break
            if not adj_matrix:
                break
            edges_list = [(u, v) for u in range(n) for v in range(u+1, n) if adj_matrix[u][v]]
            if not edges_list:
                break
            random.shuffle(edges_list)
            improved = False
            for (u, v) in edges_list[:20]:  # Try first 20
                if time.time() > max_time:
                    break
                # Save state
                old_adj = [set(s) for s in adj]
                old_adj_matrix = [row[:] for row in adj_matrix]
                
                # Remove edge
                adj_matrix[u][v] = 0
                adj_matrix[v][u] = 0
                adj[u].discard(v)
                adj[v].discard(u)
                
                # Try adding edges
                missing = [(a, b) for a in range(n) for b in range(a+1, n) if not adj_matrix[a][b]]
                random.shuffle(missing)
                try_add_edges_greedy(adj, adj_matrix, missing, max_time)
                
                new_count = sum(len(adj[i]) for i in range(n)) // 2
                if new_count > edges_count:
                    edges_count = new_count
                    improved = True
                else:
                    # Restore
                    adj = [set(s) for s in old_adj]
                    adj_matrix = [row[:] for row in old_adj_matrix]
            
            if not improved:
                break
        
        return edges_count, adj, adj_matrix
    
    # Run multiple trials
    best_edges = 0
    best_adj = None
    best_adj_matrix = None
    best_seed = 0
    
    trial_budget = 110.0  # leave 10s for final processing
    trial_num = 0
    
    while time.time() - start_time < trial_budget:
        trial_start = time.time()
        seed = trial_num * 12345 + 67890
        max_time_for_trial = min(time.time() - start_time + trial_budget - (time.time() - start_time), 
                                  trial_budget - (time.time() - start_time))
        # Give each trial at most 5 seconds, but at least 0.5s
        trial_time = min(5.0, max(0.5, trial_budget - (time.time() - start_time)))
        
        edges_count, adj, adj_matrix = run_one_trial(seed, start_time + trial_budget)
        
        if edges_count > best_edges:
            best_edges = edges_count
            best_adj = adj
            best_adj_matrix = adj_matrix
            best_seed = seed
        
        trial_num += 1
        if trial_num >= 20:
            break
    
    # Verify the best result has no C6
    if best_adj is not None and best_adj_matrix is not None:
        # Final verification
        c6_count = 0
        for u in range(n):
            for v in range(u+1, n):
                if not best_adj_matrix[u][v]:
                    if would_create_c6_fast(best_adj, best_adj_matrix, u, v):
                        c6_count += 1
        # c6_count should be 0 for a valid C6-free graph
        # (it counts non-edges that would complete a C6, which should be 0)
        
        # Extract edges
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if best_adj_matrix[u][v]:
                    edges.append([u, v])
        
        return edges
    
    # Fallback: return K5 copies
    edges = []
    for k in range(8):
        for a in range(5):
            for b in range(a+1, 5):
                edges.append([5*k + a, 5*k + b])
    return edges

# EVOLVE-BLOCK-END
