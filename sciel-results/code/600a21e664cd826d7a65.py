import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    def is_regular(mask, graph, n):
        """Check if induced subgraph on 'mask' is regular."""
        # Get vertices
        # Instead of extracting list, we can just iterate bits?
        # But we need to check equality of degrees.
        # Extracting list is O(k).
        verts = [i for i in range(n) if (mask >> i) & 1]
        if len(verts) < 6:
            return False
        first_deg = -1
        for v in verts:
            deg = bin(graph[v] & mask).count('1')
            if first_deg == -1:
                first_deg = deg
            elif deg != first_deg:
                return False
        return True

    def count_regular_subgraphs(graph, n):
        """Count all regular induced subgraphs of size >= 6."""
        count = 0
        # We can iterate over all masks from 0 to 2^n - 1
        # But that's 4M masks.
        # It's faster to iterate combinations?
        # combinations(range(n), 6) -> 74k
        # combinations(range(n), 7) -> 170k
        # Total ~1M.
        # Let's iterate combinations.
        for size in range(6, n + 1):
            for comb in combinations(range(n), size):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                if is_regular(mask, graph, n):
                    count += 1
        return count

    def try_find(n, time_limit):
        """Try to find a valid graph on n vertices within time_limit."""
        if time.time() > deadline - time_limit:
            return None
        
        # Initialize random graph
        graph = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    graph[i] |= (1 << j)
                    graph[j] |= (1 << i)
        
        current_cost = count_regular_subgraphs(graph, n)
        if current_cost == 0:
            return graph
        
        best_graph = graph
        best_cost = current_cost
        
        while time.time() < deadline - time_limit:
            # Pick random edge to flip
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            
            # Flip edge (i, j)
            bit = 1 << j
            if graph[i] & bit:
                graph[i] ^= bit
                graph[j] ^= (1 << i)
            else:
                graph[i] |= bit
                graph[j] |= (1 << i)
            
            new_cost = count_regular_subgraphs(graph, n)
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_graph = graph[:]
                if new_cost == 0:
                    return best_graph
            else:
                # Revert
                graph[i] ^= bit
                graph[j] ^= (1 << i)
        
        if best_cost == 0:
            return best_graph
        return None

    # Try n=22, 21, 20
    for n in [22, 21, 20]:
        # Allocate remaining time
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        # Give more time to larger n? Or equal?
        # Let's give 40s for 22, 40s for 21, 40s for 20?
        # Total 120s.
        t_limit = min(40.0, remaining)
        
        graph = try_find(n, t_limit)
        if graph:
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if graph[i] & (1 << j):
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
        
    # Fallback
    return {'n': 5, 'edges': []}
