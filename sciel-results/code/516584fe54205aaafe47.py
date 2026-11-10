import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s buffer
    
    n = 18
    best_score = 12  # Initial guess from K6,6,6
    best_edges = None
    
    # Helper to check if adding an edge creates a K4
    def is_k4_free(edges, u, v):
        # Check if there is a triangle containing u and v
        # If yes, adding (u,v) might create a K4 if there's a common neighbor w 
        # such that (u,w) and (v,w) exist. Wait, a K4 is 4 vertices.
        # If we add (u,v), we form a K4 if there exist x, y such that
        # {u,v,x,y} is a K4.
        # This requires u-v, u-x, u-y, v-x, v-y, x-y all present.
        # So we need a triangle (u, x, y) and (v, x, y) and (x, y).
        # Essentially, u and v must have at least two common neighbors x, y that are connected.
        
        common_neighbors = [w for w in range(n) if (min(u,w), max(u,w)) in edges and (min(v,w), max(v,w)) in edges]
        for i in range(len(common_neighbors)):
            for j in range(i+1, len(common_neighbors)):
                x = common_neighbors[i]
                y = common_neighbors[j]
                if (min(x,y), max(x,y)) in edges:
                    return False
        return True

    def get_largest_triangle_free_induced_size(edges):
        # Greedy heuristic: remove vertices from triangles
        # This provides an upper bound on the size of the triangle-free set found
        # But we want the MAX size.
        # A better heuristic: Start with all vertices, iteratively remove one vertex from each triangle.
        # The remaining set is triangle-free.
        # To find the LARGEST, this greedy might not be optimal, but it's fast.
        
        # Actually, for n=18, we can try a simple branch and bound or just check all subsets of size > current_best
        # But that's slow.
        
        # Let's use the greedy removal.
        remaining = set(range(n))
        edge_set = set(edges)
        
        # Find triangles
        triangles = []
        for u, v in edge_set:
            common = [w for w in range(n) if (min(u,w), max(u,w)) in edge_set and (min(v,w), max(v,w)) in edge_set]
            for w in common:
                if w > v and v > u: # Avoid duplicates
                    triangles.append((u, v, w))
        
        # Remove vertices to break triangles
        # Prioritize removing vertices that are in the most triangles
        while triangles:
            # Count triangle incidence
            counts = {}
            for t in triangles:
                for v in t:
                    counts[v] = counts.get(v, 0) + 1
            
            # Pick vertex with max count
            if not counts:
                break
            v_to_remove = max(counts, key=counts.get)
            remaining.remove(v_to_remove)
            
            # Update triangles
            new_triangles = []
            for t in triangles:
                if v_to_remove not in t:
                    new_triangles.append(t)
            triangles = new_triangles
            
            # Recalculate triangles? No, just removing v breaks all triangles containing v.
            # But new triangles might form? No, we are removing vertices, so edges only disappear.
            # So we just need to filter out triangles containing v.
            
        return len(remaining)

    # Local Search
    for iteration in range(200):
        if time.time() > deadline:
            break
            
        # Initialize with random 3-partite graph with some noise
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        edges = set()
        
        # Add cross edges with probability 0.8
        for i in range(3):
            for j in range(i+1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        if random.random() < 0.8:
                            edges.add((min(a,b), max(a,b)))
        
        # Ensure K4-free by removing edges if necessary
        # This is complex, so we start with a known K4-free structure: K6,6,6
        # And then perturb
        edges = set()
        for i in range(3):
            for j in range(i+1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        edges.add((min(a,b), max(a,b)))
                        
        # Perturb: randomly remove some cross edges
        for _ in range(20):
            e = random.choice(list(edges))
            edges.remove(e)
            
        # Try to add intra-part edges
        for _ in range(30):
            part_idx = random.randint(0, 2)
            part = parts[part_idx]
            u, v = random.sample(part, 2)
            if (min(u,v), max(u,v)) not in edges:
                if is_k4_free(edges, u, v):
                    edges.add((min(u,v), max(u,v)))
        
        # Evaluate
        size = get_largest_triangle_free_induced_size(list(edges))
        
        if size < best_score:
            best_score = size
            best_edges = list(edges)
            
        # If we find a very good score, we can stop early or refine
        if best_score <= 10:
            break

    # Convert back to list of lists
    if best_edges is None:
        # Fallback to K6,6,6
        parts = [range(0, 6), range(6, 12), range(12, 18)]
        return [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
        
    return [[u, v] for u, v in best_edges]
