import time
import random
import sys

def construct():
    n = 20
    deadline = time.time() + 110.0
    
    # Total possible edges: 20*19/2 = 190
    # We map each edge (u, v) to an index 0..189
    edge_indices = []
    edge_map = {}
    idx = 0
    for u in range(n):
        for v in range(u+1, n):
            edge_map[(u, v)] = idx
            edge_indices.append((u, v))
            idx += 1
            
    num_edges = len(edge_indices)
    
    # Precompute bitmasks for vertices
    # adj_mask[v] is a bitmask of length num_edges where bit i is 1 if edge i is incident to v
    vertex_edge_masks = [0] * n
    for i, (u, v) in enumerate(edge_indices):
        bit = 1 << i
        vertex_edge_masks[u] |= bit
        vertex_edge_masks[v] |= bit
        
    # Precompute common neighbor masks for all pairs? 
    # common_mask[u][v] = bitmask of edges incident to common neighbors of u and v?
    # No, we need to check if adding edge (u,v) creates a K2,3.
    # Condition: For any pair (a,b), the number of common neighbors <= 2.
    # If we have a graph G, and we add edge (u,v), we need to check:
    # 1. Do u and v have > 2 common neighbors in G?
    # 2. For any neighbor w of u (in G), do v and w have > 1 common neighbor in G? (Since v is now neighbor of u, v becomes a common neighbor of u and w? No. u-w is edge. v-u is edge. v-w is edge? If v-w is edge, then v is common neighbor of u and w. Wait.
    # Let's stick to the standard check:
    # A graph is K2,3-free iff for every pair of vertices {u, v}, |N(u) ∩ N(v)| <= 2.
    
    # To check if a graph (bitmask) is valid:
    # We can iterate over all pairs (u, v) and count common neighbors.
    # N(u) is the set of vertices connected to u.
    # We can store the graph as a list of vertex masks?
    # Let's store the graph as a list of 20 integers, where graph[u] is a bitmask of neighbors (0..19).
    # This is much more efficient for common neighbor checks.
    
    # Let's switch representation to:
    # graph = [mask_0, mask_1, ..., mask_19]
    # mask_u has bit v set if u-v is an edge.
    
    # Edge index to (u, v)
    # To check validity of a graph:
    def is_valid(graph):
        # graph is list of 20 ints
        for u in range(n):
            for v in range(u+1, n):
                common = graph[u] & graph[v]
                if common.bit_count() > 2:
                    return False
        return True
        
    def count_edges(graph):
        # Sum of bits in all masks / 2
        return sum(m.bit_count() for m in graph) // 2
        
    def clone(graph):
        return graph[:]
        
    def add_edge(graph, u, v):
        graph[u] |= (1 << v)
        graph[v] |= (1 << u)
        
    def remove_edge(graph, u, v):
        graph[u] &= ~(1 << v)
        graph[v] &= ~(1 << u)
        
    # Initialization: Clebsch Graph + 4 vertices
    # Vertices 0..15 are Clebsch
    # Clebsch graph on 16 vertices (folded Q5)
    # Q5 vertices 0..31. Fold x ~ x^31.
    # Reps: 0..15.
    def get_rep(x):
        return min(x, x ^ 31)
        
    clebsch_edges = set()
    for u in range(32):
        for b in range(5):
            v = u ^ (1 << b)
            if u < v:
                cu = get_rep(u)
                cv = get_rep(v)
                if cu != cv:
                    clebsch_edges.add((cu, cv))
                    
    base_graph = [0] * n
    for u, v in clebsch_edges:
        add_edge(base_graph, u, v)
        
    # Add vertices 16, 17, 18, 19
    # Connect them to random adjacent pairs in Clebsch to maintain validity?
    # Actually, we can just add them and let the GA fix/optimize.
    # But starting valid is better.
    # Let's try to add them greedily to the base graph.
    
    for new_v in range(16, 20):
        # Find a pair of adjacent vertices in the current graph that are not connected to new_v (trivially true)
        # and such that adding edges (new_v, u) and (new_v, v) keeps the graph valid.
        # We need |N(new_v) ∩ N(u)| <= 2 and |N(new_v) ∩ N(v)| <= 2 etc.
        # Since new_v is new, N(new_v) is empty initially.
        # If we add edges to u and v:
        # N(new_v) = {u, v}
        # Check pair (new_v, u): Common neighbors are N(u) ∩ {v}. If u-v is edge, count=1. Else 0. OK.
        # Check pair (new_v, v): Common neighbors are N(v) ∩ {u}. If u-v is edge, count=1. Else 0. OK.
        # Check pair (u, v): New common neighbor is new_v. Old common neighbors + 1 <= 2.
        # So we need u, v to be adjacent in Clebsch AND have <= 1 common neighbor in Clebsch.
        # In Clebsch, every edge is in exactly 1 triangle? 
        # Clebsch is strongly regular (16, 5, 0, 2). lambda=0, mu=2.
        # Adjacent vertices have 0 common neighbors.
        # So adding new_v connected to any edge (u,v) in Clebsch is valid.
        
        if clebsch_edges:
            u, v = random.choice(list(clebsch_edges))
            add_edge(base_graph, new_v, u)
            add_edge(base_graph, new_v, v)
            
    # Verify base graph
    if not is_valid(base_graph):
        # Fallback to empty graph
        base_graph = [0] * n

    # Genetic Algorithm
    pop_size = 50
    population = [clone(base_graph) for _ in range(pop_size)]
    # Diversify initial population
    for i in range(1, pop_size):
        g = clone(population[0])
        # Randomly remove some edges
        for _ in range(random.randint(5, 15)):
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v and g[u] & (1 << v):
                remove_edge(g, u, v)
        # Try to add random edges
        for _ in range(random.randint(5, 15)):
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v and not (g[u] & (1 << v)):
                # Check if adding is valid
                # Save state
                old_u = g[u]
                old_v = g[v]
                add_edge(g, u, v)
                if not is_valid(g):
                    g[u] = old_u
                    g[v] = old_v
        population[i] = g
        
    best_graph = clone(base_graph)
    best_score = count_edges(best_graph)
    
    # Crossover
    def crossover(parent1, parent2):
        # Create child by taking union of edges, then pruning invalid ones
        # Start with parent1
        child = clone(parent1)
        
        # Get edges in parent2 not in child
        # Iterate over all possible edges
        for i in range(num_edges):
            u, v = edge_indices[i]
            if (parent2[u] & (1 << v)) and not (child[u] & (1 << v)):
                # Try to add edge (u, v) to child
                old_u = child[u]
                old_v = child[v]
                add_edge(child, u, v)
                if is_valid(child):
                    # Keep it
                    pass
                else:
                    # Revert
                    child[u] = old_u
                    child[v] = old_v
                    
        return child
        
    # Mutation
    def mutate(graph):
        # Flip 1-3 edges
        num_flips = random.randint(1, 3)
        for _ in range(num_flips):
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            if graph[u] & (1 << v):
                # Remove
                remove_edge(graph, u, v)
            else:
                # Add
                old_u = graph[u]
                old_v = graph[v]
                add_edge(graph, u, v)
                if not is_valid(graph):
                    graph[u] = old_u
                    graph[v] = old_v
                    
        return graph
        
    # Selection: Tournament
    def select(pop):
        # Pick 3 random, return best
        candidates = random.sample(pop, 3)
        return max(candidates, key=count_edges)
        
    while time.time() < deadline:
        new_population = []
        
        # Elitism: keep best
        best_in_pop = max(population, key=count_edges)
        if count_edges(best_in_pop) > best_score:
            best_score = count_edges(best_in_pop)
            best_graph = clone(best_in_pop)
            
        new_population.append(clone(best_in_pop))
        
        while len(new_population) < pop_size:
            p1 = select(population)
            p2 = select(population)
            
            child = crossover(p1, p2)
            
            if random.random() < 0.3:
                child = mutate(child)
                
            new_population.append(child)
            
        population = new_population
        
    # Final polish: Local search on best_graph
    # Try to add any missing edge
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(num_edges):
            u, v = edge_indices[i]
            if not (best_graph[u] & (1 << v)):
                old_u = best_graph[u]
                old_v = best_graph[v]
                add_edge(best_graph, u, v)
                if is_valid(best_graph):
                    improved = True
                    best_score = count_edges(best_graph)
                else:
                    best_graph[u] = old_u
                    best_graph[v] = old_v
                    
    # Convert best_graph to list of edges
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_graph[u] & (1 << v):
                edges.append([u, v])
                
    return edges

if __name__ == "__main__":
    edges = construct()
    print(f"Edges: {len(edges)}")
