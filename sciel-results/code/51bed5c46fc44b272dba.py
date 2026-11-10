import time
import random
import sys

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def max_clique_tomita(adj, n, order):
        """
        Tomita's algorithm for Maximum Weight Clique.
        Here weights are 1.
        adj: list of bitmasks for neighbors.
        n: number of vertices.
        order: list of vertices in the order to process.
        Returns the max clique as a bitmask.
        """
        # We use a recursive search with bounding.
        # To speed up, we use the "color sort" heuristic from Tomita.
        # However, implementing full color sorting is complex.
        # A simpler effective heuristic is to use the greedy coloring bound.
        
        best_clique = [0] # bitmask
        best_size = [0]
        
        # Pre-compute degrees or use adj directly.
        # We need to be careful with the order.
        # Standard Tomita uses a specific ordering.
        # We will implement a simplified version that uses bitmasks and a greedy coloring bound.
        
        def color_sort(P, order_subset):
            """
            Greedily color the subgraph induced by P.
            Returns a list of (vertex, color) sorted by color descending, then by vertex.
            This helps in bounding: if a vertex has color c, the max clique size 
            in the remaining graph is at most c (if we process in order).
            """
            # For simplicity and speed in Python, we might skip full color sorting 
            # and just use a simple bound: max degree in P + 1.
            # Or we can use a rough bound.
            # Let's try a simple recursive search with a bound based on |P|.
            # Bound: current_size + |P| <= best_size?
            
            # Actually, for n=64, a simple branch and bound with 
            # bound = current_size + popcount(P) is often sufficient if the graph is dense.
            # In dense graphs, max clique is large, so |P| shrinks fast.
            
            pass

        def expand(R, P, X):
            # R: current clique (bitmask)
            # P: candidates (bitmask)
            # X: excluded (bitmask)
            
            if P == 0:
                if R.bit_count() > best_size[0]:
                    best_size[0] = R.bit_count()
                    best_clique[0] = R
                return
            
            # Bound: if |R| + |P| <= best_size, prune
            if R.bit_count() + P.bit_count() <= best_size[0]:
                return
            
            # Choose a pivot u from P | X with max degree in P
            pivot_mask = P | X
            if pivot_mask:
                best_pivot = -1
                max_deg = -1
                pm = pivot_mask
                while pm:
                    lsb = pm & -pm
                    u = lsb.bit_length() - 1
                    deg = (adj[u] & P).bit_count()
                    if deg > max_deg:
                        max_deg = deg
                        best_pivot = u
                    pm ^= lsb
                candidates = P & ~adj[best_pivot]
            else:
                candidates = P
            
            c = candidates
            while c:
                lsb = c & -c
                v = lsb.bit_length() - 1
                
                # Branch
                new_R = R | lsb
                new_P = P & adj[v]
                new_X = X & adj[v]
                
                expand(new_R, new_P, new_X)
                
                P ^= lsb
                X |= lsb
                c ^= lsb
                
                # Update bound check in loop? 
                # If R.bit_count() + P.bit_count() <= best_size, break?
                # Yes, because we are removing vertices from P, so |P| decreases.
                if R.bit_count() + P.bit_count() <= best_size[0]:
                    break

        # Initialize
        all_vertices = (1 << n) - 1
        expand(0, all_vertices, 0)
        
        return best_clique[0]

    def greedy_partition(adj, n, order, deadline):
        graph = adj[:]
        cliques = []
        
        while True:
            has_edges = False
            for u in range(n):
                if graph[u]:
                    has_edges = True
                    break
            if not has_edges:
                break
            
            # Find max clique
            # Note: Tomita implementation above is basic. 
            # For better performance, we should use a more optimized version.
            # But let's try this first.
            
            mask = max_clique_tomita(graph, n, order)
            
            if mask.bit_count() < 2:
                # Fallback: find any edge
                found = False
                for u in range(n):
                    if graph[u]:
                        lsb = graph[u] & -graph[u]
                        v = lsb.bit_length() - 1
                        mask = (1 << u) | (1 << v)
                        found = True
                        break
                if not found:
                    break
            
            verts = []
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                verts.append(v)
                m ^= lsb
            
            if len(verts) >= 2:
                cliques.append(verts)
                for i in range(len(verts)):
                    u = verts[i]
                    for j in range(i + 1, len(verts)):
                        v = verts[j]
                        graph[u] &= ~(1 << v)
                        graph[v] &= ~(1 << u)
            
            if time.time() > deadline:
                break
        
        # Safety: cover remaining edges
        for u in range(n):
            mask = graph[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if v > u:
                    cliques.append([u, v])
                    graph[u] &= ~(1 << v)
                    graph[v] &= ~(1 << u)
                mask ^= lsb
        
        return cliques

    start_time = time.time()
    deadline = start_time + 145
    
    adj_1 = instance(32, 0.6, 601)
    adj_2 = instance(64, 0.55, 602)
    
    best_cliques_1 = None
    best_count_1 = float('inf')
    
    best_cliques_2 = None
    best_count_2 = float('inf')
    
    # Multi-start for Instance 1
    # We have 32 vertices. Tomita should be fast.
    # Try 20 random orderings.
    for i in range(20):
        if time.time() > deadline - 5: break
        order = list(range(32))
        random.shuffle(order)
        cliques = greedy_partition(adj_1, 32, order, deadline - 5)
        if len(cliques) < best_count_1:
            best_count_1 = len(cliques)
            best_cliques_1 = cliques
            
    # If we didn't find any, do a default
    if best_cliques_1 is None:
        best_cliques_1 = greedy_partition(adj_1, 32, list(range(32)), deadline - 5)

    # Multi-start for Instance 2
    # We have 64 vertices. Tomita might be slower.
    # Try 10 random orderings.
    for i in range(10):
        if time.time() > deadline - 5: break
        order = list(range(64))
        random.shuffle(order)
        cliques = greedy_partition(adj_2, 64, order, deadline - 5)
        if len(cliques) < best_count_2:
            best_count_2 = len(cliques)
            best_cliques_2 = cliques
            
    if best_cliques_2 is None:
        best_cliques_2 = greedy_partition(adj_2, 64, list(range(64)), deadline - 5)

    return [best_cliques_1, best_cliques_2]

if __name__ == "__main__":
    res = construct()
    print(f"Instance 1: {len(res[0])} cliques")
    print(f"Instance 2: {len(res[1])} cliques")
    print(f"Total: {len(res[0]) + len(res[1])}")
