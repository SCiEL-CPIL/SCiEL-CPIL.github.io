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
        
        Adding edge (u, v) affects:
        1. Triples containing u: {u, a, b}. N({u,a,b}) gains v if v in N(a)∩N(b).
           We need old |N({u,a,b})| <= 1 in that case.
           N({u,a,b}) = N(u) ∩ N(a) ∩ N(b).
        2. Triples containing v: {v, a, b}. N({v,a,b}) gains u if u in N(a)∩N(b).
           We need old |N({v,a,b})| <= 1 in that case.
           N({v,a,b}) = N(v) ∩ N(a) ∩ N(b).
        """
        if adj[u] & (1 << v):
            return False
            
        # Precompute neighbors of u and v as bitmasks for fast intersection
        nu = adj[u]
        nv = adj[v]
        
        # Iterate over all pairs (a, b) with a < b, a,b != u,v
        # We can iterate a from 0 to n-1, b from a+1 to n-1
        for a in range(n):
            if a == u or a == v:
                continue
            na = adj[a]
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                nb = adj[b]
                
                # Check if v is a common neighbor of a and b
                # If yes, adding u-v increases |N({u,a,b})| by 1.
                # We must ensure current |N({u,a,b})| <= 1.
                if (na & (1 << v)) and (nb & (1 << v)):
                    # N({u,a,b}) = nu & na & nb
                    count = (nu & na & nb).bit_count()
                    if count > 1:
                        return False
                        
                # Check if u is a common neighbor of a and b
                # If yes, adding u-v increases |N({v,a,b})| by 1.
                # We must ensure current |N({v,a,b})| <= 1.
                if (na & (1 << u)) and (nb & (1 << u)):
                    # N({v,a,b}) = nv & na & nb
                    count = (nv & na & nb).bit_count()
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

    def local_search(adj, max_swap_iterations=50):
        current_adj = adj[:]
        current_edges = count_edges(current_adj)
        
        # Phase 1: Greedy addition of any valid edges
        # We do this in a random order to break symmetries
        pairs = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] & (1 << j))]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if time.time() > deadline:
                return current_adj, current_edges
            if is_valid_addition(current_adj, u, v):
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_edges += 1
                
        # Phase 2: Edge swap local search (2-opt style)
        # Try removing an edge and adding two new ones to escape local optima
        improved = True
        swap_iterations = 0
        while improved and swap_iterations < max_swap_iterations:
            if time.time() > deadline:
                return current_adj, current_edges
            improved = False
            swap_iterations += 1
            
            # Get all existing edges and missing edges
            existing_edges = [(i, j) for i in range(n) for j in range(i+1, n) if current_adj[i] & (1 << j)]
            missing_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] & (1 << j))]
            random.shuffle(existing_edges)
            random.shuffle(missing_edges)
            
            # Try each existing edge for removal
            for u, v in existing_edges:
                if time.time() > deadline:
                    return current_adj, current_edges
                
                # Temporarily remove edge (u, v)
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
                
                added_count = 0
                added_edges = []
                
                # Try to add missing edges
                for a, b in missing_edges:
                    if time.time() > deadline:
                        # Revert before returning? We return current state, but it's invalid if we don't revert.
                        # Better to break out cleanly.
                        current_adj[u] |= (1 << v)
                        current_adj[v] |= (1 << u)
                        return current_adj, current_edges
                    
                    # Skip edges involving u or v if they were already there? 
                    # Actually, we can add edges involving u or v if they are missing.
                    # But we just removed (u,v). We can try to add (u, x) or (v, y).
                    
                    if is_valid_addition(current_adj, a, b):
                        current_adj[a] |= (1 << b)
                        current_adj[b] |= (1 << a)
                        added_count += 1
                        added_edges.append((a, b))
                        if added_count >= 2:
                            break
                
                if added_count >= 1:
                    # Net gain or neutral, keep the swap
                    current_edges += added_count - 1
                    improved = True
                    # Update missing_edges to reflect new state for next iteration of swap loop?
                    # The outer while loop will re-fetch them.
                else:
                    # Revert the swap
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    
        # Phase 3: Final pass to add any remaining edges after swaps
        pairs = [(i, j) for i in range(n) for j in range(i+1, n) if not (current_adj[i] & (1 << j))]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if time.time() > deadline:
                return current_adj, current_edges
            if is_valid_addition(current_adj, u, v):
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_edges += 1
                
        return current_adj, current_edges

    best_adj = [0] * n
    best_edges = 0
    
    # Generate initial candidates with many restarts due to speed
    num_restarts = 3000 
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        adj = greedy_construct(i)
        edges = count_edges(adj)
        
        # Local search improvement
        adj, edges = local_search(adj)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
            
    # If we have time, try to improve the best solution further with more aggressive swaps
    if best_adj is not None and time.time() < deadline - 10:
        # Run local search on the best solution multiple times with different random seeds for swaps
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            adj, edges = local_search(best_adj, max_swap_iterations=100)
            if edges > best_edges:
                best_edges = edges
                best_adj = adj
            else:
                # If no improvement, break early? Or keep trying?
                # Let's keep trying a few times to ensure we escape local optima
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
