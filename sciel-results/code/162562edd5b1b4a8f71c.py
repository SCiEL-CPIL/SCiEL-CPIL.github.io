import random
import math
from collections import defaultdict

def construct():
    n = 16
    target_edges = 65
    
    # Start with a balanced structure: try to create a graph where
    # degrees are as equal as possible and triangles have low degree sums
    
    random.seed(42)
    
    # Initial construction: start with a near-regular graph
    # Target degree: 65*2/16 = 8.125, so degrees around 8
    
    def initial_graph():
        edges = set()
        # Create a structure based on two parts but with some mixing
        # Part A: 0-7, Part B: 8-15
        # Add all cross edges except a few, and add some within-part edges
        
        # Start with K_{8,8}
        for u in range(8):
            for v in range(8, 16):
                edges.add(frozenset((u, v)))
        
        # We have 64 edges, need 1 more
        # Instead of adding one within-part edge, let's try a different approach:
        # Remove a few cross edges and add within-part edges to balance degrees
        
        # Try: remove 3 cross edges, add 3 within-part edges (net +0, but we need +1)
        # Actually, let's just add one within-part edge but choose it carefully
        
        # Add edge (0,1) in part A
        edges.add(frozenset((0, 1)))
        
        return [list(e) for e in edges]
    
    def build_adjacency(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def find_triangles(n, adj):
        triangles = []
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    for w in adj[v]:
                        if w > v and w in adj[u]:
                            triangles.append((u, v, w))
        return triangles
    
    def max_triangle_degree_sum(n, adj):
        if n <= 2:
            return 0
        triangles = find_triangles(n, adj)
        if not triangles:
            return 0
        max_sum = 0
        for (u, v, w) in triangles:
            s = len(adj[u]) + len(adj[v]) + len(adj[w])
            if s > max_sum:
                max_sum = s
        return max_sum
    
    def get_worst_triangles(n, adj, top_k=5):
        """Get the k triangles with highest degree sum."""
        triangles = find_triangles(n, adj)
        if not triangles:
            return []
        scored = []
        for (u, v, w) in triangles:
            s = len(adj[u]) + len(adj[v]) + len(adj[w])
            scored.append((s, (u, v, w)))
        scored.sort(key=lambda x: -x[0])
        return scored[:top_k]
    
    edges = initial_graph()
    adj = build_adjacency(n, edges)
    current_score = max_triangle_degree_sum(n, adj)
    
    best_score = current_score
    best_edges = edges[:]
    
    # Local search with targeted moves
    max_iters = 30000
    no_improve = 0
    max_no_improve = 5000
    
    for it in range(max_iters):
        if no_improve > max_no_improve:
            break
        
        # Try to improve by modifying edges near the worst triangles
        worst = get_worst_triangles(n, adj, top_k=3)
        
        if not worst:
            break
        
        improved = False
        
        # For each worst triangle, try to reduce its degree sum
        for score, (u, v, w) in worst:
            # Try removing one edge from the triangle and adding a different edge
            for edge_to_remove in [(u, v), (u, w), (v, w)]:
                a, b = edge_to_remove
                edge_key = frozenset((a, b))
                
                if edge_key not in set(frozenset(e) for e in edges):
                    continue
                
                # Try adding a new edge that doesn't create worse triangles
                all_pairs = [(x, y) for x in range(n) for y in range(x+1, n)]
                current_edge_set = set(frozenset(e) for e in edges)
                
                # Try a few random new edges
                for _ in range(10):
                    new_a, new_b = random.choice(all_pairs)
                    new_edge_key = frozenset((new_a, new_b))
                    
                    if new_edge_key in current_edge_set and new_edge_key != edge_key:
                        continue
                    if new_edge_key == edge_key:
                        continue
                    
                    new_edges_set = set(current_edge_set)
                    new_edges_set.remove(edge_key)
                    new_edges_set.add(new_edge_key)
                    
                    if len(new_edges_set) != target_edges:
                        continue
                    
                    new_edges = [list(e) for e in new_edges_set]
                    new_adj = build_adjacency(n, new_edges)
                    new_score = max_triangle_degree_sum(n, new_adj)
                    
                    if new_score < current_score:
                        current_score = new_score
                        edges = new_edges
                        adj = new_adj
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_edges = edges[:]
                        break
                    elif new_score == current_score and random.random() < 0.1:
                        # Accept equal with some probability to escape local minima
                        current_score = new_score
                        edges = new_edges
                        adj = new_adj
                        improved = True
                        break
        
        if not improved:
            no_improve += 1
        else:
            no_improve = 0
    
    return {'n': n, 'edges': best_edges}
