import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    n = 20
    
    # adj[u] is a bitmask of neighbors of u
    # I[a][b] is a bitmask of common neighbors of a and b
    # Note: I[a][b] == I[b][a]
    
    def is_valid_addition(adj, I, u, v):
        """
        Checks if adding edge (u, v) violates the K3,3-free condition.
        Condition: For any triple T, |N(T)| <= 2.
        
        Adding edge (u, v) affects:
        1. Triples containing u: {u, a, b}. N({u,a,b}) gains v if v in N(a)∩N(b).
           We need old |N({u,a,b})| <= 1 in that case.
           N({u,a,b}) = N(u) ∩ N(a) ∩ N(b) = N(u) ∩ I[a][b].
           So if v in I[a][b], we need (N(u) & I[a][b]).bit_count() <= 1.
        2. Triples containing v: {v, a, b}. N({v,a,b}) gains u if u in N(a)∩N(b).
           We need old |N({v,a,b})| <= 1 in that case.
           N({v,a,b}) = N(v) ∩ N(a) ∩ N(b) = N(v) ∩ I[a][b].
           So if u in I[a][b], we need (N(v) & I[a][b]).bit_count() <= 1.
        """
        if adj[u] & (1 << v):
            return False
            
        nu = adj[u]
        nv = adj[v]
        
        # Iterate over all pairs (a, b) with a < b
        for a in range(n):
            if a == u or a == v:
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                Iab = I[a][b]
                
                # Check if v is a common neighbor of a and b
                if (Iab >> v) & 1:
                    count = (nu & Iab).bit_count()
                    if count > 1:
                        return False
                        
                # Check if u is a common neighbor of a and b
                if (Iab >> u) & 1:
                    count = (nv & Iab).bit_count()
                    if count > 1:
                        return False
                        
        return True

    def add_edge(adj, I, u, v):
        """Adds edge (u, v) to adj and updates I."""
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        # Update I[u][v] and I[v][u]
        I[u][v] = adj[u] & adj[v]
        I[v][u] = I[u][v]
        
        # Update I[u][x] and I[v][x] for all x != u, v
        for x in range(n):
            if x == u or x == v:
                continue
            if adj[x] & (1 << v):
                I[u][x] |= (1 << v)
                I[x][u] |= (1 << v)
            if adj[x] & (1 << u):
                I[v][x] |= (1 << u)
                I[x][v] |= (1 << u)

    def remove_edge(adj, I, u, v):
        """Removes edge (u, v) from adj and updates I."""
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
        # Update I[u][v] and I[v][u]
        I[u][v] = adj[u] & adj[v]
        I[v][u] = I[u][v]
        
        # Update I[u][x] and I[v][x] for all x != u, v
        for x in range(n):
            if x == u or x == v:
                continue
            if adj[x] & (1 << v):
                I[u][x] &= ~(1 << v)
                I[x][u] &= ~(1 << v)
            if adj[x] & (1 << u):
                I[v][x] &= ~(1 << u)
                I[x][v] &= ~(1 << u)

    def count_edges(adj):
        return sum(x.bit_count() for x in adj) // 2

    def init_structures():
        adj = [0] * n
        I = [[0] * n for _ in range(n)]
        return adj, I

    def greedy_fill(adj, I):
        """Add all possible edges greedily."""
        current_adj = adj[:]
        # Deep copy I
        current_I = [row[:] for row in I]
        
        pairs = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] & (1 << j))]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if time.time() > deadline:
                break
            if is_valid_addition(current_adj, current_I, u, v):
                add_edge(current_adj, current_I, u, v)
                
        return current_adj, current_I

    def kick(adj, I, num_removals=2):
        """Randomly remove some edges to escape local optima."""
        current_adj = adj[:]
        current_I = [row[:] for row in I]
        
        existing_edges = [(i, j) for i in range(n) for j in range(i+1, n) if current_adj[i] & (1 << j)]
        random.shuffle(existing_edges)
        
        for u, v in existing_edges[:num_removals]:
            remove_edge(current_adj, current_I, u, v)
            
        return current_adj, current_I

    def local_search(adj, I, max_kick_iterations=20):
        current_adj = adj[:]
        current_I = [row[:] for row in I]
        current_edges = count_edges(current_adj)

        best_adj = current_adj[:]
        best_I = [row[:] for row in current_I]
        best_edges = current_edges

        # Phase 1: Greedy fill
        current_adj, current_I = greedy_fill(current_adj, current_I)
        current_edges = count_edges(current_adj)
        
        if current_edges > best_edges:
            best_adj = current_adj[:]
            best_I = [row[:] for row in current_I]
            best_edges = current_edges

        # Phase 2: Kick + greedy fill cycles
        for _ in range(max_kick_iterations):
            if time.time() > deadline:
                break

            kicked_adj, kicked_I = kick(current_adj, current_I, num_removals=2)
            filled_adj, filled_I = greedy_fill(kicked_adj, kicked_I)
            filled_edges = count_edges(filled_adj)

            if filled_edges > best_edges:
                best_adj = filled_adj[:]
                best_I = [row[:] for row in filled_I]
                best_edges = filled_edges
                current_adj = best_adj[:]
                current_I = [row[:] for row in best_I]
            else:
                # Try a larger kick
                kicked_adj, kicked_I = kick(current_adj, current_I, num_removals=4)
                filled_adj, filled_I = greedy_fill(kicked_adj, kicked_I)
                filled_edges = count_edges(filled_adj)
                
                if filled_edges > best_edges:
                    best_adj = filled_adj[:]
                    best_I = [row[:] for row in filled_I]
                    best_edges = filled_edges
                    current_adj = best_adj[:]
                    current_I = [row[:] for row in best_I]
                else:
                    break

        current_adj = best_adj
        current_I = best_I
        current_edges = best_edges

        # Phase 3: Final greedy fill
        current_adj, current_I = greedy_fill(current_adj, current_I)
        current_edges = count_edges(current_adj)

        if current_edges > best_edges:
            best_adj = current_adj[:]
            best_I = [row[:] for row in current_I]
            best_edges = current_edges

        return best_adj, best_I, best_edges

    best_adj = [0] * n
    best_I = [[0] * n for _ in range(n)]
    best_edges = 0
    
    # Generate initial candidates
    num_restarts = 500 
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        random.seed(i)
        adj, I = init_structures()
        
        # Random initialization: add random edges
        pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if is_valid_addition(adj, I, u, v):
                add_edge(adj, I, u, v)
        
        # Local search improvement
        adj, I, edges = local_search(adj, I)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
            best_I = [row[:] for row in I]
            
    # If we have time, try to improve the best solution further
    if best_adj is not None and time.time() < deadline - 10:
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            adj, I, edges = local_search(best_adj, best_I)
            if edges > best_edges:
                best_edges = edges
                best_adj = adj[:]
                best_I = [row[:] for row in I]
            else:
                # No improvement, maybe break early to save time for other strategies?
                # Or just continue with smaller kicks?
                pass

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
