import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    n = 20
    
    # adj[u] is a bitmask of neighbors of u
    # bit i is set if edge (u, i) exists
    
    def is_valid_addition(adj, u, v):
        """
        Checks if adding edge (u, v) violates the K3,3-free condition.
        Condition: For any triple T, |N(T)| <= 2.
        """
        if adj[u] & (1 << v):
            return False
            
        # Check triples containing u: {u, a, b}
        # If v is in N(a) and N(b), then v is added to N({u, a, b}).
        # We need old |N({u, a, b})| <= 1.
        # N({u, a, b}) = N(u) & N(a) & N(b)
        
        for a in range(n):
            if a == u or a == v:
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                
                # Check if v is a common neighbor of a and b
                if (adj[a] & (1 << v)) and (adj[b] & (1 << v)):
                    # v is in N(a) ∩ N(b)
                    # So adding u-v adds v to N({u, a, b})
                    # We must ensure current |N({u, a, b})| <= 1
                    count = (adj[u] & adj[a] & adj[b]).bit_count()
                    if count > 1:
                        return False
                        
                # Check if u is a common neighbor of a and b
                # This corresponds to triples {v, a, b} where u is added to N({v, a, b})
                if (adj[a] & (1 << u)) and (adj[b] & (1 << u)):
                    count = (adj[v] & adj[a] & adj[b]).bit_count()
                    if count > 1:
                        return False
                        
        return True

    def count_edges(adj):
        return sum(x.bit_count() for x in adj) // 2

    def greedy_construct(random_seed):
        random.seed(random_seed)
        adj = [0] * n
        
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if time.time() > deadline:
                break
            if is_valid_addition(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
        return adj

    def spectral_initialize(seed):
        """
        Initialize using a spectral-like heuristic.
        Generate a random symmetric matrix, threshold at mean to create a dense
        random-like starting point, then prune invalid edges.
        """
        random.seed(seed)
        # Generate random weights for each pair
        weights = {}
        for i in range(n):
            for j in range(i+1, n):
                weights[(i, j)] = random.random()
        
        # Sort by weight descending, try to add in that order
        # This creates a denser initial structure than pure random order
        sorted_pairs = sorted(weights.items(), key=lambda x: -x[1])
        
        adj = [0] * n
        for (u, v), _ in sorted_pairs:
            if time.time() > deadline:
                break
            if is_valid_addition(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        return adj

    def local_search(adj, max_steps=500):
        current_adj = adj[:]
        current_edges = count_edges(current_adj)
        
        # Phase 1: Try to add edges in random order
        pairs = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] & (1 << j))]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if time.time() > deadline:
                break
            if is_valid_addition(current_adj, u, v):
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_edges += 1
                
        # Phase 2: Edge swap local search - remove an edge, try to add two
        # This allows escaping local optima where no single edge can be added
        improved = True
        swap_iterations = 0
        max_swap_iterations = 200
        while improved and swap_iterations < max_swap_iterations:
            if time.time() > deadline:
                break
            improved = False
            swap_iterations += 1
            
            # Get all existing edges and missing edges
            existing_edges = [(i, j) for i in range(n) for j in range(i+1, n) if current_adj[i] & (1 << j)]
            missing_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] & (1 << j))]
            random.shuffle(existing_edges)
            random.shuffle(missing_edges)
            
            for u, v in existing_edges:
                if time.time() > deadline:
                    break
                # Try removing edge (u, v) and adding two missing edges
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
                
                added_count = 0
                for a, b in missing_edges:
                    if time.time() > deadline:
                        break
                    if (a == u or a == v or b == u or b == v):
                        continue
                    if is_valid_addition(current_adj, a, b):
                        current_adj[a] |= (1 << b)
                        current_adj[b] |= (1 << a)
                        added_count += 1
                        if added_count >= 2:
                            break
                
                if added_count >= 1:
                    # Net gain or neutral, keep the swap
                    current_edges += added_count - 1
                    improved = True
                    # Update missing_edges to reflect new state
                    missing_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] & (1 << j))]
                else:
                    # Revert the swap
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    
        # Phase 3: Final pass to add any remaining edges
        pairs = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] & (1 << j))]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if time.time() > deadline:
                break
            if is_valid_addition(current_adj, u, v):
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_edges += 1
                
        return current_adj, current_edges

    best_adj = [0] * n
    best_edges = 0
    
    # Generate initial candidates using both greedy and spectral initialization
    # We can run many more restarts now due to speedup
    num_restarts = 2000 
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Alternate between greedy and spectral initialization
        if i % 2 == 0:
            adj = greedy_construct(i)
        else:
            adj = spectral_initialize(i)
        
        edges = count_edges(adj)
        
        # Local search improvement
        adj, edges = local_search(adj)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
            
    # If we have time, try to improve the best solution further
    if time.time() < deadline - 5:
        adj, edges = local_search(best_adj)
        if edges > best_edges:
            best_edges = edges
            best_adj = adj
    
    # Additional polish: run local search multiple times on the best solution
    if time.time() < deadline - 10:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            adj, edges = local_search(best_adj)
            if edges > best_edges:
                best_edges = edges
                best_adj = adj
            else:
                break
            
    # Convert bitmask to list of edges
    result_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result_edges.append([i, j])
                
    return result_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
