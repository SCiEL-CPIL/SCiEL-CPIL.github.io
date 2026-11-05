"""Polarity graph C4-free core + optimized extra edge + local search."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n = 31
    m = 91
    
    # Construct polarity graph on PG(2,5)
    # Points of PG(2,5) are 1D subspaces of F_5^3
    # There are (5^3 - 1)/(5 - 1) = 31 points
    # A polarity graph connects points if they are orthogonal under the polarity
    
    # Generate all 31 points in PG(2,5)
    points = []
    for a in range(5):
        for b in range(5):
            for c in range(5):
                if a == 0 and b == 0 and c == 0:
                    continue
                # Normalize: make first non-zero coordinate 1
                vec = [a, b, c]
                for i in range(3):
                    if vec[i] != 0:
                        inv = pow(vec[i], 3, 5)  # x^3 = x^-1 mod 5
                        vec = [(v * inv) % 5 for v in vec]
                        break
                if tuple(vec) not in [tuple(p) for p in points]:
                    points.append(tuple(vec))
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Define polarity: symmetric bilinear form
    # Use the standard polarity: x1*y3 + x2*y2 + x3*y1 = 0 (or similar)
    # For PG(2,q) with q odd, a polarity has (q+1)/2 absolute points
    # Let's use: x1*y1 + x2*y2 + x3*y3 = 0 (orthogonality in F_5^3)
    
    def is_orthogonal(p, q):
        return (p[0]*q[0] + p[1]*q[1] + p[2]*q[2]) % 5 == 0
    
    # Build the polarity graph: edge between p and q if orthogonal
    edges = set()
    for i, p in enumerate(points):
        for j, q in enumerate(points):
            if i < j and is_orthogonal(p, q):
                edges.add((i, j))
    
    # The polarity graph should have exactly 90 edges (C4-free, ex(31;C4)=90)
    # Actually, for a polarity on PG(2,5), the number of edges is:
    # Each point is orthogonal to q+1 = 6 other points (including itself if absolute)
    # Let me count more carefully
    
    # Actually, let me just compute and see how many edges we get
    # If it's not exactly 90, I'll adjust
    
    edge_list = list(edges)
    
    # If we have exactly 90 edges, add one more optimally
    # If we have a different count, we need to adjust
    
    # For now, let's use this as our base and add edges to reach 91
    # If len(edge_list) < 91, add edges that create fewest C4s
    # If len(edge_list) > 91, remove edges that eliminate most C4s
    
    # Let me reconsider: the polarity graph on PG(2,5) might not have exactly 90 edges
    # Let me just count and adapt
    
    if len(edge_list) == 90:
        # Add one edge that minimizes C4 creation
        best_extra_edge = None
        best_c4_increase = float('inf')
        
        # Build adjacency for quick common neighbor counting
        adj = [set() for _ in range(n)]
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)
        
        # Try all non-edges
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) not in edges and (v, u) not in edges:
                    # Count common neighbors
                    common = len(adj[u] & adj[v])
                    c4_increase = common * (common - 1) // 2
                    if c4_increase < best_c4_increase:
                        best_c4_increase = c4_increase
                        best_extra_edge = (u, v)
        
        if best_extra_edge:
            edge_list.append(best_extra_edge)
            edges.add(best_extra_edge)
            adj[best_extra_edge[0]].add(best_extra_edge[1])
            adj[best_extra_edge[1]].add(best_extra_edge[0])
    
    elif len(edge_list) < 91:
        # Need to add more edges
        adj = [set() for _ in range(n)]
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)
        
        while len(edge_list) < 91:
            best_extra_edge = None
            best_c4_increase = float('inf')
            
            for u in range(n):
                for v in range(u + 1, n):
                    if (u, v) not in edges and (v, u) not in edges:
                        common = len(adj[u] & adj[v])
                        c4_increase = common * (common - 1) // 2
                        if c4_increase < best_c4_increase:
                            best_c4_increase = c4_increase
                            best_extra_edge = (u, v)
            
            if best_extra_edge:
                edge_list.append(best_extra_edge)
                edges.add(best_extra_edge)
                adj[best_extra_edge[0]].add(best_extra_edge[1])
                adj[best_extra_edge[1]].add(best_extra_edge[0])
            else:
                break
    
    else:
        # Too many edges, remove some
        # This case shouldn't happen for a proper polarity, but handle it
        pass
    
    # Now do local search to minimize C4 count
    def count_c4s(edge_set):
        """Count number of C4s in the graph."""
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        c4_count = 0
        for u, v in combinations(range(n), 2):
            common = len(adj[u] & adj[v])
            if common >= 2:
                c4_count += common * (common - 1) // 2
        return c4_count
    
    def try_swap(edge_set, max_time=None):
        """Try to improve by swapping one edge for another."""
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        current_c4 = count_c4s(edge_set)
        improved = True
        swap_count = 0
        
        while improved and (max_time is None or time.time() < max_time):
            improved = False
            # Try all possible swaps: remove edge e1, add edge e2
            # To be efficient, only consider swaps where e1 and e2 are in some sense related
            # For simplicity, try removing each edge and adding the best non-edge
            
            for e1_idx, e1 in enumerate(list(edge_set)):
                u1, v1 = e1
                # Remove e1
                adj[u1].remove(v1)
                adj[v1].remove(u1)
                edge_set.discard(e1)
                if (v1, u1) in edge_set:
                    edge_set.discard((v1, u1))
                
                c4_after_remove = count_c4s(edge_set)
                
                # Find best edge to add
                best_add = None
                best_c4 = float('inf')
                for u2 in range(n):
                    for v2 in range(u2 + 1, n):
                        if (u2, v2) not in edge_set and (v2, u2) not in edge_set:
                            common = len(adj[u2] & adj[v2])
                            c4_increase = common * (common - 1) // 2
                            total_c4 = c4_after_remove + c4_increase
                            if total_c4 < best_c4:
                                best_c4 = total_c4
                                best_add = (u2, v2)
                
                # Restore e1
                adj[u1].add(v1)
                adj[v1].add(u1)
                edge_set.add(e1)
                
                if best_add and best_c4 < current_c4:
                    # Perform the swap
                    edge_set.discard(e1)
                    if (v1, u1) in edge_set:
                        edge_set.discard((v1, u1))
                    adj[u1].remove(v1)
                    adj[v1].remove(u1)
                    
                    u2, v2 = best_add
                    edge_set.add(best_add)
                    adj[u2].add(v2)
                    adj[v2].add(u2)
                    
                    current_c4 = best_c4
                    improved = True
                    swap_count += 1
                    if time.time() > deadline - 5:
                        return edge_set, current_c4
        
        return edge_set, current_c4
    
    # Convert to set for easy manipulation
    edge_set = set(edge_list)
    
    # Run local search
    edge_set, c4_count = try_swap(edge_set, max_time=deadline - 5)
    
    # Return as list of lists
    result = [[u, v] for u, v in sorted(edge_set)]
    
    # Verify we have exactly 91 edges
    if len(result) != 91:
        # Fallback: return first 91 lexicographic
        pairs = [[i, j] for i in range(31) for j in range(i + 1, 31)]
        return pairs[:91]
    
    return result
# EVOLVE-BLOCK-END
