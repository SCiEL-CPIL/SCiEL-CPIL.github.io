import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 15

    def is_valid(adj):
        degs = [len(adj[i]) for i in range(n)]
        # For each vertex u, map degree -> set of neighbors with that degree
        nb_by_deg = [defaultdict(set) for _ in range(n)]
        for u in range(n):
            for v in adj[u]:
                nb_by_deg[u][degs[v]].add(v)
        
        for u in range(n):
            for v in adj[u]:
                if u >= v:
                    continue
                all_d = set(nb_by_deg[u].keys()) | set(nb_by_deg[v].keys())
                for d in all_d:
                    su = nb_by_deg[u].get(d, set())
                    if v in su:
                        su = su - {v}
                    tv = nb_by_deg[v].get(d, set())
                    if u in tv:
                        tv = tv - {u}
                    if not su or not tv:
                        continue
                    if su == tv and len(su) == 1:
                        continue
                    return False
        return True
    
    # Start from K_{7,8} bipartite graph (provably valid, 56 edges)
    def make_bipartite_base():
        adj = [set() for _ in range(n)]
        left = list(range(7))
        right = list(range(7, 15))
        for u in left:
            for v in right:
                adj[u].add(v)
                adj[v].add(u)
        return adj
    
    # Degree-sequence-based greedy edge insertion
    def greedy_degree_sequence(adj, max_time):
        """
        Greedy approach: prioritize adding edges between vertices with largest degree gaps.
        This helps maintain the constraint that no 3-path connects equal-degree endpoints.
        """
        adj = [set(x) for x in adj]
        
        while time.time() < max_time:
            degs = [len(adj[i]) for i in range(n)]
            # Find all missing edges, sorted by degree gap (largest first)
            missing_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if v not in adj[u]:
                        gap = abs(degs[u] - degs[v])
                        missing_edges.append((gap, u, v))
            
            if not missing_edges:
                break
            
            # Sort by degree gap descending, then by vertex index for determinism
            missing_edges.sort(key=lambda x: (-x[0], x[1], x[2]))
            
            improved = False
            for gap, u, v in missing_edges:
                if time.time() > max_time:
                    break
                adj[u].add(v)
                adj[v].add(u)
                if is_valid(adj):
                    improved = True
                else:
                    adj[u].discard(v)
                    adj[v].discard(u)
            
            if not improved:
                break
        
        return adj
    
    # Local search: try to add intra-part edges
    def local_search_add(adj, max_time):
        adj = [set(x) for x in adj]
        improved = True
        while improved and time.time() < max_time:
            improved = False
            # Try adding all missing edges
            for u in range(n):
                for v in range(u + 1, n):
                    if time.time() > max_time:
                        return adj
                    if v not in adj[u]:
                        adj[u].add(v)
                        adj[v].add(u)
                        if is_valid(adj):
                            improved = True
                        else:
                            adj[u].discard(v)
                            adj[v].discard(u)
        return adj
    
    # More efficient local search: try swapping edges
    def local_search_swap_efficient(adj, max_time):
        adj = [set(x) for x in adj]
        current_count = sum(len(adj[i]) for i in range(n)) // 2
        improved = True
        while improved and time.time() < max_time:
            improved = False
            edges_to_remove = [(u, v) for u in range(n) for v in adj[u] if u < v]
            random.shuffle(edges_to_remove)
            for (u1, v1) in edges_to_remove:
                if time.time() > max_time:
                    return adj
                # Remove edge (u1, v1)
                adj[u1].discard(v1)
                adj[v1].discard(u1)
                
                # Try adding a new edge (only missing edges)
                for u2 in range(n):
                    for v2 in range(u2 + 1, n):
                        if time.time() > max_time:
                            adj[u1].add(v1)
                            adj[v1].add(u1)
                            return adj
                        if (u2, v2) == (u1, v1):
                            continue
                        if v2 not in adj[u2]:
                            adj[u2].add(v2)
                            adj[v2].add(u2)
                            if is_valid(adj):
                                improved = True
                                break
                            else:
                                adj[u2].discard(v2)
                                adj[v2].discard(u2)
                    if improved:
                        break
                if not improved:
                    adj[u1].add(v1)
                    adj[v1].add(u1)
        return adj
    
    # Local search: try swapping edges (remove one, add another)
    def local_search_swap(adj, max_time):
        return local_search_swap_efficient(adj, max_time)
    
    # Multiple restarts with different strategies
    random.seed(42)
    best_adj = None
    best_count = 0
    
    # First, start from bipartite base and do greedy degree-sequence search
    base_adj = make_bipartite_base()
    base_count = sum(len(base_adj[i]) for i in range(n)) // 2
    best_adj = [set(x) for x in base_adj]
    best_count = base_count
    
    if is_valid(base_adj):
        # Try greedy degree-sequence approach first
        improved_adj = greedy_degree_sequence(base_adj, deadline - 50)
        improved_count = sum(len(improved_adj[i]) for i in range(n)) // 2
        if improved_count > best_count:
            best_adj = improved_adj
            best_count = improved_count
        
        # Then try standard local search
        improved_adj2 = local_search_add(base_adj, deadline - 40)
        improved_count2 = sum(len(improved_adj2[i]) for i in range(n)) // 2
        if improved_count2 > best_count:
            best_adj = improved_adj2
            best_count = improved_count2
    
    for trial in range(100):
        if time.time() > deadline - 15:
            break
        
        # Random initial graph: start with bipartite and randomly add some edges
        adj = [set(x) for x in base_adj]
        
        # Randomly try to add some intra-part edges
        left = list(range(7))
        right = list(range(7, 15))
        random.shuffle(left)
        random.shuffle(right)
        
        added = 0
        for u in left:
            for v in left:
                if u < v and random.random() < 0.2:
                    adj[u].add(v)
                    adj[v].add(u)
        for u in right:
            for v in right:
                if u < v and random.random() < 0.3:
                    adj[u].add(v)
                    adj[v].add(u)
        # Check validity, if invalid, remove problematic edges
        if not is_valid(adj):
            changed = True
            while changed and time.time() < deadline - 15:
                changed = False
                edges_list = [(u, v) for u in range(n) for v in adj[u] if u < v]
                random.shuffle(edges_list)
                for (u, v) in edges_list:
                    if time.time() > deadline - 15:
                        break
                    adj[u].discard(v)
                    adj[v].discard(u)
                    if is_valid(adj):
                        changed = True
                        break
        
        # Use greedy degree-sequence first, then local search
        adj = greedy_degree_sequence(adj, deadline - 20)
        adj = local_search_add(adj, deadline - 15)
        cnt = sum(len(adj[i]) for i in range(n)) // 2
        
        if cnt > best_count:
            best_adj = [set(x) for x in adj]
            best_count = cnt
    
    # Final improvement: try more swaps
    if best_adj and time.time() < deadline - 5:
        improved = local_search_swap(best_adj, deadline - 5)
        cnt = sum(len(improved[i]) for i in range(n)) // 2
        if cnt > best_count:
            best_adj = improved
            best_count = cnt
    
    # Build result
    if best_adj is None:
        # Fallback: star graph
        edges = [[0, v] for v in range(1, n)]
    else:
        edges = []
        for u in range(n):
            for v in best_adj[u]:
                if u < v:
                    edges.append([u, v])
    
    # Verify final result
    if best_adj:
        if not is_valid(best_adj):
            # Fallback to bipartite
            edges = []
            left = list(range(7))
            right = list(range(7, 15))
            for u in left:
                for v in right:
                    edges.append([u, v])
    
    # Ensure types are correct
    result = []
    for e in edges:
        result.append([int(e[0]), int(e[1])])
    
    return result
