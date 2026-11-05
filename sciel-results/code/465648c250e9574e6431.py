import time
import random
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 140  # safety margin before 150s limit
    
    N = 20
    all_pairs = list(combinations(range(N), 2))
    num_pairs = len(all_pairs)  # 190
    
    def common_neighbors_count(adj, u, v):
        """Count common neighbors of u and v."""
        cnt = 0
        for w in range(N):
            if w != u and w != v and adj[u][w] and adj[v][w]:
                cnt += 1
        return cnt
    
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) would create a K2,3 violation."""
        for w in range(N):
            if w == u or w == v:
                continue
            if adj[w][v] and common_neighbors_count(adj, u, w) >= 2:
                return False
            if adj[u][w] and common_neighbors_count(adj, v, w) >= 2:
                return False
        return True
    
    def has_k23_violation(adj):
        """Check if any pair has >= 3 common neighbors. Return (u,v) if found."""
        for u in range(N):
            for v in range(u+1, N):
                if common_neighbors_count(adj, u, v) >= 3:
                    return (u, v)
        return None
    
    def num_edges(adj):
        cnt = 0
        for u in range(N):
            for v in range(u+1, N):
                if adj[u][v]:
                    cnt += 1
        return cnt
    
    def edge_list(adj):
        edges = []
        for u in range(N):
            for v in range(u+1, N):
                if adj[u][v]:
                    edges.append([u, v])
        return edges
    
    def greedy_fill(adj):
        """Try to add all possible edges greedily. Returns number of edges added."""
        added = 0
        changed = True
        while changed:
            changed = False
            for (u, v) in all_pairs:
                if not adj[u][v] and can_add_edge(adj, u, v):
                    adj[u][v] = adj[v][u] = True
                    added += 1
                    changed = True
        return added
    
    def local_search(adj, current_edges, max_iters=100):
        """Local search: try single edge additions and 1-for-1 swaps."""
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try to add a single edge
            improved = False
            for (u, v) in random.sample(all_pairs, min(190, num_pairs)):
                if adj[u][v]:
                    continue
                if can_add_edge(adj, u, v):
                    adj[u][v] = adj[v][u] = True
                    current_edges += 1
                    improved = True
                    break
            
            if not improved:
                # Try 1-for-1 swaps
                swapped = False
                for (u1, v1) in random.sample(all_pairs, min(100, num_pairs)):
                    if not adj[u1][v1]:
                        continue
                    adj[u1][v1] = adj[v1][u1] = False
                    current_edges -= 1
                    
                    for (u2, v2) in random.sample(all_pairs, min(80, num_pairs)):
                        if adj[u2][v2] or (u2 == u1 and v2 == v1):
                            continue
                        if can_add_edge(adj, u2, v2):
                            adj[u2][v2] = adj[v2][u2] = True
                            current_edges += 1
                            swapped = True
                            break
                    
                    if not swapped:
                        adj[u1][v1] = adj[v1][u1] = True
                        current_edges += 1
            
            if time.time() > deadline:
                break
        
        return current_edges
    
    def try_2for1_swap(adj, current_edges):
        """Try removing 1 edge and adding 2 edges (net +1)."""
        for (u1, v1) in random.sample(all_pairs, min(50, num_pairs)):
            if not adj[u1][v1]:
                continue
            adj[u1][v1] = adj[v1][u1] = False
            current_edges -= 1
            
            added = 0
            for (u2, v2) in random.sample(all_pairs, min(60, num_pairs)):
                if adj[u2][v2] or (u2 == u1 and v2 == v1):
                    continue
                if can_add_edge(adj, u2, v2):
                    adj[u2][v2] = adj[v2][u2] = True
                    current_edges += 1
                    added += 1
                    if added >= 2:
                        break
            
            if added < 2:
                # Restore
                adj[u1][v1] = adj[v1][u1] = True
                current_edges += 1
                # Remove the edges we added
                for (u2, v2) in all_pairs:
                    if added > 0 and adj[u2][v2] and (u2, v2) != (u1, v1):
                        pass  # This is tricky; let's just not do 2-for-1 swaps for simplicity
        
        return current_edges
    
    random.seed(42)
    
    best_edges = 0
    best_adj = None
    
    # Phase 1: Multiple diverse initializations
    for attempt in range(80):
        if time.time() > deadline:
            break
        
        adj = [[False] * N for _ in range(N)]
        
        # Different initialization strategies based on attempt number
        if attempt < 40:
            # Random greedy with shuffled edge order
            edge_order = list(all_pairs)
            random.shuffle(edge_order)
            for (u, v) in edge_order:
                if can_add_edge(adj, u, v):
                    adj[u][v] = adj[v][u] = True
        elif attempt < 60:
            # Structured: circulant-like initial graph
            # Start with a circulant graph pattern
            for i in range(N):
                for j in [1, 2, 3, 4]:
                    v = (i + j) % N
                    if i < v:
                        if can_add_edge(adj, i, v):
                            adj[i][v] = adj[v][i] = True
                    elif v < i:
                        if can_add_edge(adj, i, v):
                            adj[i][v] = adj[v][i] = True
            # Then greedy fill
            greedy_fill(adj)
        else:
            # Random with varying density
            density = 0.3 + 0.5 * (attempt - 60) / 20.0
            for (u, v) in all_pairs:
                if random.random() < density:
                    if can_add_edge(adj, u, v):
                        adj[u][v] = adj[v][u] = True
            # Then greedy fill
            greedy_fill(adj)
        
        ne = num_edges(adj)
        if ne > best_edges:
            best_edges = ne
            best_adj = [row[:] for row in adj]
    
    if best_adj is None:
        # Fallback to seed
        edges = []
        for k in range(5):
            edges += [[4*k+a, 4*k+b] for a in range(4) for b in range(a+1, 4)]
        return edges
    
    adj = best_adj
    current_edges = best_edges
    
    # Phase 2: Local search on the best solution
    current_edges = local_search(adj, current_edges, max_iters=300)
    
    # Phase 3: Restart with perturbation from the best solution
    for attempt in range(40):
        if time.time() > deadline:
            break
        
        trial_adj = [row[:] for row in adj]
        trial_edges = current_edges
        
        # Perturbation: remove 5-15 random edges
        num_remove = random.randint(5, 15)
        for _ in range(num_remove):
            (u, v) = random.choice(all_pairs)
            if trial_adj[u][v]:
                trial_adj[u][v] = trial_adj[v][u] = False
                trial_edges -= 1
        
        # Greedy fill
        greedy_fill(trial_adj)
        trial_edges = num_edges(trial_adj)
        
        # Local search
        trial_edges = local_search(trial_adj, trial_edges, max_iters=50)
        
        if trial_edges > current_edges:
            adj = trial_adj
            current_edges = trial_edges
    
    # Phase 4: Final polish - try all possible single edge additions
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for (u, v) in all_pairs:
            if not adj[u][v] and can_add_edge(adj, u, v):
                adj[u][v] = adj[v][u] = True
                current_edges += 1
                changed = True
    
    # Final validation
    violation = has_k23_violation(adj)
    if violation:
        changed = True
        while changed and time.time() < deadline:
            changed = False
            v = has_k23_violation(adj)
            if v is None:
                break
            u, w = v
            common = [x for x in range(N) if x != u and x != w and adj[u][x] and adj[w][x]]
            if common:
                cn = common[0]
                adj[u][cn] = adj[cn][u] = False
                adj[w][cn] = adj[cn][w] = False
                current_edges -= 2
                changed = True
            else:
                break
    
    return edge_list(adj)
