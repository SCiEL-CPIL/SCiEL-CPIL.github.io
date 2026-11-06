import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin (evaluator kills at 150s)
    
    n = 20
    
    def is_k33_free(adj):
        """Check if graph is K3,3-free using the common-neighbor condition."""
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    # Count common neighbors of {a,b,c}
                    count = 0
                    for w in range(n):
                        if adj[a][w] and adj[b][w] and adj[c][w]:
                            count += 1
                            if count >= 3:
                                return False
        return True
    
    def count_edges(adj):
        return sum(sum(row) for row in adj) // 2
    
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) keeps the graph K3,3-free."""
        if adj[u][v]:
            return False
        
        # Adding (u,v) affects triples containing u or v
        # For triples {u, a, b}: new common neighbor count increases by 1 if v is common neighbor of a and b
        # We need the new count to be <= 2
        
        # Check triples {u, a, b} where a, b != u, v
        for a in range(n):
            if a == u or a == v:
                continue
            for b in range(a+1, n):
                if b == u or b == v:
                    continue
                # If v is a common neighbor of a and b, then adding u-v increases the count
                if adj[a][v] and adj[b][v]:
                    # Count current common neighbors of {u, a, b}
                    count = 0
                    for w in range(n):
                        if adj[u][w] and adj[a][w] and adj[b][w]:
                            count += 1
                    if count >= 2:
                        return False
        
        # Check triples {v, a, b} where a, b != u, v
        for a in range(n):
            if a == u or a == v:
                continue
            for b in range(a+1, n):
                if b == u or b == v:
                    continue
                # If u is a common neighbor of a and b, then adding u-v increases the count
                if adj[a][u] and adj[b][u]:
                    # Count current common neighbors of {v, a, b}
                    count = 0
                    for w in range(n):
                        if adj[v][w] and adj[a][w] and adj[b][w]:
                            count += 1
                    if count >= 2:
                        return False
        
        return True
    
    def greedy_build(seed):
        """Build a graph using greedy edge addition with random ordering."""
        random.seed(seed)
        adj = [[0]*n for _ in range(n)]
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if time.time() > deadline:
                break
            if can_add_edge(adj, u, v):
                adj[u][v] = 1
                adj[v][u] = 1
        
        return adj
    
    def local_search_add(adj):
        """Try to add more edges to the graph."""
        adj = [row[:] for row in adj]
        pairs = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
        random.shuffle(pairs)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for u, v in pairs:
                if not adj[u][v] and can_add_edge(adj, u, v):
                    adj[u][v] = 1
                    adj[v][u] = 1
                    improved = True
        
        return adj
    
    def edge_swap_search(adj, max_iterations=500):
        """
        Edge swap local search: try removing one edge and adding up to two new edges.
        This allows escaping local optima of add-only search.
        """
        adj = [row[:] for row in adj]
        best_adj = [row[:] for row in adj]
        best_count = count_edges(adj)
        
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        for _ in range(max_iterations):
            if time.time() > deadline - 10:
                break
            
            # Try removing a random edge
            if best_count == 0:
                break
            
            existing_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i][j]]
            if not existing_edges:
                break
            
            rem_u, rem_v = random.choice(existing_edges)
            
            # Try removing this edge
            trial_adj = [row[:] for row in adj]
            trial_adj[rem_u][rem_v] = 0
            trial_adj[rem_v][rem_u] = 0
            
            # Now try to add edges to the trial graph
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if not trial_adj[i][j]]
            random.shuffle(missing)
            
            added = 0
            for u, v in missing:
                if time.time() > deadline - 10:
                    break
                if can_add_edge(trial_adj, u, v):
                    trial_adj[u][v] = 1
                    trial_adj[v][u] = 1
                    added += 1
                    if added >= 2:  # We removed 1, so adding 2 gives net +1
                        break
            
            trial_count = count_edges(trial_adj)
            if trial_count > best_count:
                best_count = trial_count
                best_adj = [row[:] for row in trial_adj]
                adj = [row[:] for row in trial_adj]
            elif trial_count == best_count:
                # Accept equal moves with some probability to explore
                if random.random() < 0.3:
                    adj = [row[:] for row in trial_adj]
        
        return best_adj, best_count
    
    def spectral_init(seed):
        """
        Spectral initialization: create a dense random-like starting point.
        Use a random symmetric matrix, threshold at a level that gives roughly 60-70% edge density,
        then prune to make it K3,3-free.
        """
        random.seed(seed)
        
        # Start with a random dense graph (roughly 65% edge probability)
        adj = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.65:
                    adj[i][j] = 1
                    adj[j][i] = 1
        
        # Prune to make it K3,3-free: repeatedly find violations and remove edges
        for _ in range(200):
            if time.time() > deadline:
                break
            
            # Find a K3,3 violation
            found_violation = False
            for a in range(n):
                if found_violation:
                    break
                for b in range(a+1, n):
                    if found_violation:
                        break
                    for c in range(b+1, n):
                        if found_violation:
                            break
                        # Count common neighbors
                        common = []
                        for w in range(n):
                            if adj[a][w] and adj[b][w] and adj[c][w]:
                                common.append(w)
                        if len(common) >= 3:
                            # Remove an edge to break this violation
                            # Remove a random edge incident to one of the common neighbors
                            w = random.choice(common)
                            # Remove edge from w to one of a, b, c
                            target = random.choice([a, b, c])
                            adj[w][target] = 0
                            adj[target][w] = 0
                            found_violation = True
                            break
            
            if not found_violation:
                break
        
        return adj
    
    # Main search
    best_adj = None
    best_edges = 0
    
    # Phase 1: Spectral initialization (dense starting points)
    num_spectral = 30
    for i in range(num_spectral):
        if time.time() > deadline - 30:
            break
        
        adj = spectral_init(i)
        adj = local_search_add(adj)
        edges = count_edges(adj)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj
    
    # Phase 2: Greedy construction with local search
    num_greedy = 150
    for i in range(num_greedy):
        if time.time() > deadline - 30:
            break
        
        adj = greedy_build(1000 + i)
        adj = local_search_add(adj)
        edges = count_edges(adj)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj
    
    # Phase 3: Edge swap search on the best solution
    if best_adj is not None and time.time() < deadline - 20:
        adj, edges = edge_swap_search(best_adj, max_iterations=300)
        if edges > best_edges:
            best_edges = edges
            best_adj = adj
        
        # Another round of edge swaps
        if time.time() < deadline - 15:
            adj, edges = edge_swap_search(best_adj, max_iterations=200)
            if edges > best_edges:
                best_edges = edges
                best_adj = adj
    
    # Phase 4: Final local search polish
    if best_adj is not None and time.time() < deadline - 5:
        adj = local_search_add(best_adj)
        edges = count_edges(adj)
        if edges > best_edges:
            best_edges = edges
            best_adj = adj
    
    # Convert to edge list
    if best_adj is None:
        return []
    
    result_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                result_edges.append([i, j])
    
    return result_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
