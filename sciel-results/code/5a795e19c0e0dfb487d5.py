"""Optimized book-size minimization for n=40, m=401."""
import random
import time
import sys

def construct():
    deadline = time.time() + 145.0  # safety margin before 150s kill
    n = 40
    target_m = 401
    
    # --- Helper: compute book sizes for all edges using bitsets ---
    # For each vertex v, neighbors[v] is a bitmask of its neighbors
    # For each edge (u,v), book size = popcount(neighbors[u] & neighbors[v])
    # We track the max book size over all edges.
    
    def compute_max_book(neighbors):
        """neighbors: list of int bitmasks, one per vertex."""
        max_book = 0
        for u in range(n):
            nu = neighbors[u]
            for v in range(u + 1, n):
                if (nu >> v) & 1:
                    book = (nu & neighbors[v]).bit_count()
                    if book > max_book:
                        max_book = book
        return max_book
    
    def count_edges(neighbors):
        total = 0
        for u in range(n):
            total += neighbors[u].bit_count()
        return total // 2
    
    # --- Initial construction: balanced random graph ---
    # Start with a random graph close to the target density
    # m/n^2 = 401/1600 ≈ 0.25, so ~1/4 density
    
    def random_init():
        neighbors = [0] * n
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.25:
                    neighbors[u] |= (1 << v)
                    neighbors[v] |= (1 << u)
                    edges.append((u, v))
        # Adjust to exactly 401 edges
        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_pairs)
        
        # Add random edges until we have 401
        while len(edges) < target_m:
            u, v = all_pairs.pop()
            if not ((neighbors[u] >> v) & 1):
                neighbors[u] |= (1 << v)
                neighbors[v] |= (1 << u)
                edges.append((u, v))
        
        # Remove random edges if we have too many
        while len(edges) > target_m:
            u, v = edges.pop(random.randrange(len(edges)))
            neighbors[u] &= ~(1 << v)
            neighbors[v] &= ~(1 << u)
        
        return neighbors, edges
    
    # --- Local search: edge swap to reduce max book size ---
    def local_search(neighbors, edges, time_limit):
        current_max = compute_max_book(neighbors)
        best_neighbors = neighbors[:]
        best_max = current_max
        best_edges = edges[:]
        
        # Precompute for speed: for each vertex, list of neighbors
        def get_neighbor_list(neighbors):
            return [[v for v in range(n) if (neighbors[u] >> v) & 1] for u in range(n)]
        
        # Try edge swaps: remove one edge, add another, if max book decreases
        # Or: remove an edge that's in a large book, add a non-edge that doesn't create large books
        
        while time.time() < time_limit:
            improved = False
            
            # Find edges with large book sizes
            # For efficiency, just try random swaps
            
            for _ in range(200):  # attempts per outer iteration
                if time.time() > time_limit:
                    break
                
                # Pick a random edge to remove
                u, v = edges[random.randrange(len(edges))]
                if not ((neighbors[u] >> v) & 1):
                    continue
                
                # Remove edge (u,v)
                neighbors[u] &= ~(1 << v)
                neighbors[v] &= ~(1 << u)
                edges.remove((u, v))
                
                # Find a non-edge to add that minimizes book size impact
                # Try a few random non-edges
                best_add = None
                best_add_book = current_max + 1
                
                for _ in range(20):
                    a = random.randrange(n)
                    b = random.randrange(n)
                    if a == b:
                        continue
                    if (neighbors[a] >> b) & 1:
                        continue
                    # Check book size if we add (a,b)
                    # book = popcount(neighbors[a] & neighbors[b])
                    book = (neighbors[a] & neighbors[b]).bit_count()
                    if book < best_add_book:
                        best_add_book = book
                        best_add = (a, b)
                
                if best_add is None:
                    # Revert
                    neighbors[u] |= (1 << v)
                    neighbors[v] |= (1 << u)
                    edges.append((u, v))
                    continue
                
                a, b = best_add
                neighbors[a] |= (1 << b)
                neighbors[b] |= (1 << a)
                edges.append((a, b))
                
                # Check if max book size improved
                new_max = compute_max_book(neighbors)
                if new_max < current_max:
                    current_max = new_max
                    if new_max < best_max:
                        best_max = new_max
                        best_neighbors = neighbors[:]
                        best_edges = edges[:]
                    improved = True
                    break  # restart with new state
                else:
                    # Revert
                    neighbors[a] &= ~(1 << b)
                    neighbors[b] &= ~(1 << a)
                    edges.remove((a, b))
                    neighbors[u] |= (1 << v)
                    neighbors[v] |= (1 << u)
                    edges.append((u, v))
            
            if not improved:
                # Stuck, try a bigger perturbation or break
                break
        
        return best_neighbors, best_edges, best_max
    
    # --- Multi-start local search ---
    best_overall_max = float('inf')
    best_overall_neighbors = None
    best_overall_edges = None
    
    # Try a few random inits
    num_starts = 5
    for start in range(num_starts):
        if time.time() > deadline - 10:
            break
        
        neighbors, edges = random_init()
        cur_max = compute_max_book(neighbors)
        
        if cur_max < best_overall_max:
            best_overall_max = cur_max
            best_overall_neighbors = neighbors[:]
            best_overall_edges = edges[:]
        
        # Local search with time budget
        remaining = deadline - time.time() - 5.0
        if remaining > 5:
            time_per_start = remaining / (num_starts - start)
            best_neighbors, best_edges, best_max = local_search(
                neighbors[:], edges[:], time.time() + time_per_start
            )
            if best_max < best_overall_max:
                best_overall_max = best_max
                best_overall_neighbors = best_neighbors
                best_overall_edges = best_edges
    
    # Convert back to list of [u, v] pairs
    result = [[u, v] for u, v in best_overall_edges]
    
    # Verify: exactly 401 distinct edges on vertices 0..39
    assert len(result) == 401, f"Expected 401 edges, got {len(result)}"
    assert len(set(tuple(sorted(e)) for e in result)) == 401, "Duplicate edges"
    for e in result:
        assert 0 <= e[0] <= 39 and 0 <= e[1] <= 39 and e[0] != e[1], f"Invalid edge {e}"
    
    return result
