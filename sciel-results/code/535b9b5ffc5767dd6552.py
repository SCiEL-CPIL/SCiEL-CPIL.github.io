import time
import random

def construct():
    start_time = time.time()
    
    def get_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u+1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges = get_graph(n, p, seed)
        
        # Build adjacency sets for G
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # The problem is equivalent to coloring the complement graph \bar{G}
        # Vertices u, v are adjacent in \bar{G} if (u,v) is NOT in edges (and u!=v)
        
        # Greedy Coloring of \bar{G}
        # We want to assign colors such that if u and v have the same color, 
        # they are NOT adjacent in \bar{G}, meaning they ARE connected in G (or u=v).
        # So each color class must be a clique in G.
        
        # Heuristic: Process vertices in order of increasing degree in G?
        # Or decreasing degree in \bar{G}?
        # Degree in \bar{G} for u is (n-1) - deg_G(u).
        # So decreasing degree in \bar{G} is increasing degree in G? No.
        # deg_bar(u) = n - 1 - deg_G(u).
        # High deg_bar means low deg_G.
        # Welsh-Powell typically sorts by degree in the graph being colored in descending order.
        # So sort by deg_bar(u) descending, which is deg_G(u) ascending.
        
        # Let's sort vertices by their degree in G ascending.
        # This prioritizes vertices with few neighbors in G (many non-neighbors), 
        # which are "harder" to place in small cliques? 
        # Actually, let's just use the standard greedy coloring on the complement.
        
        vertices = list(range(n))
        # Sort by degree in complement (descending) == degree in G (ascending)
        vertices.sort(key=lambda x: len(adj[x]))
        
        color_of_vertex = [-1] * n
        cliques = [] # list of lists of vertices
        
        for u in vertices:
            # Find the smallest color that is not used by any neighbor in \bar{G}
            # Neighbors in \bar{G} are vertices w such that (u,w) not in edges (and u!=w)
            # If w has color c, and w is a neighbor in \bar{G}, then u cannot have color c.
            # So u can have color c if for all w with color c, (u,w) IS in edges.
            
            used_colors = set()
            for w in range(n):
                if w == u:
                    continue
                if (min(u,w), max(u,w)) not in edges:
                    # w is a neighbor in \bar{G}
                    if color_of_vertex[w] != -1:
                        used_colors.add(color_of_vertex[w])
            
            # Find smallest available color
            c = 0
            while c in used_colors:
                c += 1
                
            color_of_vertex[u] = c
            if c >= len(cliques):
                cliques.append([])
            cliques[c].append(u)
            
        # The cliques are the color classes.
        # We need to ensure every edge is in exactly one clique.
        # By construction, if u and v are in the same clique, they have the same color.
        # In greedy coloring, we ensured that if u and v have the same color, 
        # they are NOT adjacent in \bar{G}, which means they ARE adjacent in G.
        # Thus each clique is a valid clique in G.
        # Every edge (u,v) in G: u and v are not adjacent in \bar{G}.
        # Do they have the same color? 
        # No, the coloring only guarantees that adjacent vertices in \bar{G} have DIFFERENT colors.
        # It does NOT guarantee that non-adjacent vertices in \bar{G} have the SAME color.
        # Wait. 
        # If u and v are connected in G, they are NOT connected in \bar{G}.
        # The greedy coloring assigns colors. 
        # If u and v end up with different colors, the edge (u,v) is NOT covered by a single clique.
        # The requirement is "every edge must lie in exactly one listed clique".
        # This means the partition of vertices into cliques must cover all edges.
        # This is exactly the definition of a clique partition: a partition of V into cliques such that 
        # every edge is within one of the cliques.
        # This is equivalent to a proper coloring of the complement graph.
        # In a proper coloring of \bar{G}, if two vertices have the same color, they are not adjacent in \bar{G}, 
        # so they are adjacent in G (or same vertex).
        # If two vertices have different colors, they might be adjacent in G or not.
        # But the edges are only "covered" if both endpoints are in the same clique.
        # So yes, finding a proper coloring of \bar{G} gives a set of cliques (the color classes) 
        # such that every edge of G is contained in exactly one clique (the one corresponding to the color of its endpoints).
        # Wait, if u and v are connected in G, they are not adjacent in \bar{G}. 
        # They CAN have different colors. 
        # If they have different colors, the edge (u,v) is NOT in any single clique of the partition?
        # Let's re-read carefully: "every edge must lie in exactly one listed clique".
        # This implies that for every edge (u,v), u and v must be in the same clique.
        # This means the partition must be such that no edge crosses between two different cliques.
        # This is exactly a partition into independent sets in \bar{G} (which are cliques in G).
        # And it must cover ALL edges.
        # If u and v are in different cliques, the edge (u,v) is not in any clique.
        # So we require that for every edge (u,v) in G, u and v are in the same part of the partition.
        # This is equivalent to saying that the partition is a valid coloring of \bar{G}?
        # No. A valid coloring of \bar{G} ensures that if u and v are adjacent in \bar{G}, they are in DIFFERENT parts.
        # This means that if u and v are NOT adjacent in \bar{G} (i.e., adjacent in G), they CAN be in the same part.
        # But they don't HAVE to be.
        # If they are in different parts, the edge (u,v) is "broken".
        # The problem statement says: "every edge must lie in exactly one listed clique".
        # This implies that the cliques form a partition of the EDGES? 
        # "Clique partitions ... every listed set must be a clique ... and every edge must lie in exactly one listed clique."
        # This is a partition of the edge set into cliques.
        # This is different from a partition of the vertex set.
        # A clique partition of edges means we group edges into sets, each set forms a clique, and every edge is in exactly one set.
        # This is NOT the same as coloring the complement graph.
        # Coloring the complement graph partitions the VERTICES into independent sets of \bar{G} (cliques of G).
        # If we partition vertices into cliques C1, C2, ..., Ck, then the edges within Ci form a clique.
        # But what about edges between Ci and Cj?
        # If there is an edge (u,v) with u in Ci and v in Cj (i!=j), then this edge is not in any Ci.
        # So a vertex partition into cliques only works if the graph is a disjoint union of cliques.
        # But our graph is dense and connected.
        # So the "clique partition" here must refer to a partition of the EDGE SET.
        
        # Let's re-evaluate the task description.
        # "Clique partitions of two fixed dense random graphs"
        # "construct() -> two lists of cliques (vertex lists)..."
        # "Every listed set must be a clique with at least two vertices and every edge must lie in exactly one listed clique."
        
        # This confirms: The output is a list of vertex lists (cliques).
        # The union of edges in these cliques must be exactly E(G).
        # And no edge can be in two cliques.
        # This is an **Edge Clique Partition** (ECP).
        # The goal is to minimize the number of cliques.
        
        # The seed program does exactly this:
        # 1. Pick a maximal clique C.
        # 2. Remove all edges in C from the remaining edge pool.
        # 3. Repeat.
        
        # So my previous "complement graph coloring" idea was for Vertex Clique Partition (which doesn't cover all edges unless the graph is a union of cliques).
        # The task is Edge Clique Partition.
        
        # The seed program's greedy approach is:
        # - Find a large clique.
        # - Remove its edges.
        # - Repeat.
        
        # To improve this, we need to find larger cliques or cliques that "block" fewer future edges.
        # A common heuristic for ECP is to pick the edge with the fewest common neighbors (to force it into a small clique if necessary) or the edge with the most common neighbors (to form a large clique).
        # Actually, picking an edge (u,v) and finding the maximum clique containing (u,v) is a good strategy.
        # If we pick the edge with the MINIMUM number of common neighbors, we might be forced into a small clique, which is bad.
        # If we pick the edge with the MAXIMUM number of common neighbors, we can form a large clique, removing many edges at once.
        
        # Let's try a heuristic:
        # 1. Maintain a set of remaining edges.
        # 2. Find an edge (u,v) in the remaining set.
        #    Heuristic: Choose (u,v) such that the size of the maximum clique containing (u,v) is maximized?
        #    Or simply: Choose (u,v) that has the most common neighbors in the remaining graph.
        # 3. Find a maximal clique C containing (u,v).
        #    We can use a greedy expansion:
        #    Start with C = {u, v}.
        #    Candidates = common neighbors of u and v in the remaining graph.
        #    While candidates is not empty:
        #        Pick a candidate w (e.g., one with max degree in remaining graph).
        #        If w is connected to all vertices in C, add w to C.
        #        Remove w from candidates.
        #        Update candidates to be intersection of neighbors of all vertices in C.
        # 4. Remove all edges in C from the remaining set.
        # 5. Repeat until no edges left.
        
        # This is essentially what the seed program does, but it picks a vertex with max degree, not an edge.
        # Picking an edge might be better because it ensures the clique has at least 2 vertices and covers that specific edge.
        
        # Let's implement the edge-based greedy heuristic.
        
        if not edges:
            return []
            
        remaining_edges = set(edges)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        cliques = []
        
        while remaining_edges:
            # Heuristic: Find an edge (u, v) that is part of the largest potential clique.
            # Approximation: Find edge (u, v) with the maximum number of common neighbors in the remaining graph.
            best_edge = None
            max_common = -1
            
            # To speed up, we can sample or just iterate all edges if n is small.
            # n=64, edges ~ 64*63/2 * 0.55 ~ 1100. Iterating all is fine.
            for u, v in remaining_edges:
                # Common neighbors in the remaining graph
                # common = adj[u] & adj[v]
                # But we only care about neighbors that are connected in the remaining graph.
                # Since adj is updated, adj[u] contains only neighbors connected by remaining edges.
                common = adj[u] & adj[v]
                if len(common) > max_common:
                    max_common = len(common)
                    best_edge = (u, v)
                    
            if best_edge is None:
                break
                
            u, v = best_edge
            C = [u, v]
            
            # Greedy expansion
            # Candidates are common neighbors of all vertices in C.
            candidates = adj[u] & adj[v]
            
            while candidates:
                # Pick candidate with highest degree in remaining graph
                # This helps form a dense clique
                w = max(candidates, key=lambda x: len(adj[x]))
                
                # Check if w is connected to all in C
                # Since candidates is intersection of adj of all in C, this is true by definition.
                # But we must ensure w is in candidates.
                C.append(w)
                candidates = candidates & adj[w]
                
            # Add clique to result
            cliques.append(C)
            
            # Remove edges in C
            for i in range(len(C)):
                for j in range(i+1, len(C)):
                    x, y = C[i], C[j]
                    if x > y:
                        x, y = y, x
                    edge = (x, y)
                    if edge in remaining_edges:
                        remaining_edges.remove(edge)
                        adj[x].remove(y)
                        adj[y].remove(x)
                        
            # Time check
            if time.time() - start_time > 110:
                # If we run out of time, we might have an invalid partition (missing edges).
                # The problem requires a valid partition.
                # We should probably just return what we have if it's valid, but if edges remain, it's invalid.
                # However, with n=64, this should be very fast.
                pass
                
        return cliques

    results = []
    results.append(solve_instance(32, 0.6, 601))
    results.append(solve_instance(64, 0.55, 602))
    
    return results

# Test locally
if __name__ == "__main__":
    res = construct()
    # Verify validity
    def verify(n, p, seed, cliques):
        edges = set()
        rng = random.Random(seed)
        for u in range(n):
            for v in range(u+1, n):
                if rng.random() < p:
                    edges.add((u, v))
        
        covered = set()
        for clique in cliques:
            if len(clique) < 2:
                return False
            # Check clique validity
            for i in range(len(clique)):
                for j in range(i+1, len(clique)):
                    u, v = clique[i], clique[j]
                    if u > v: u, v = v, u
                    if (u, v) not in edges:
                        return False
                    if (u, v) in covered:
                        return False
                    covered.add((u, v))
                    
        return covered == edges

    print("Instance 1 valid:", verify(32, 0.6, 601, res[0]))
    print("Instance 2 valid:", verify(64, 0.55, 602, res[1]))
    print("Num cliques 1:", len(res[0]))
    print("Num cliques 2:", len(res[1]))
