import time
import random
import numpy as np

def construct():
    n = 31
    target_edges = 91
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    
    # Precompute all possible edges as indices
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    num_pairs = len(all_pairs)
    
    # Map edge index to (u, v)
    edge_to_uv = {idx: pair for idx, pair in enumerate(all_pairs)}
    uv_to_idx = {pair: idx for idx, pair in enumerate(all_pairs)}
    
    # Precompute for each edge, which other edges share a vertex with it
    # This helps in quick C4 detection
    
    # Function to count C4s in a graph given as a set of edge indices
    def count_c4s(edge_set):
        if len(edge_set) < 4:
            return 0
        
        # Build adjacency list
        adj = [set() for _ in range(n)]
        for idx in edge_set:
            u, v = edge_to_uv[idx]
            adj[u].add(v)
            adj[v].add(u)
        
        # Count C4s: for each pair of vertices (u, v), count common neighbors
        # A C4 is counted twice (once for each pair of opposite vertices)
        # Actually, each C4 has 2 pairs of opposite vertices, so we count each C4 twice
        # Wait, let me think again.
        # A C4 is u-a-v-b-u. The pairs of opposite vertices are (u,v) and (a,b).
        # So for each pair (u,v), the number of common neighbors c(u,v) means there are C(c(u,v), 2) C4s
        # where u and v are opposite. But each C4 is counted exactly once this way? No.
        # Each C4 u-a-v-b-u: opposite pairs are (u,v) and (a,b).
        # When we consider pair (u,v), we count the common neighbors a and b, giving 1 C4.
        # When we consider pair (a,b), we count the common neighbors u and v, giving 1 C4.
        # So each C4 is counted exactly twice in the sum over all pairs of C(common_neighbors, 2).
        
        total = 0
        for u in range(n):
            for v in range(u + 1, n):
                if u in adj[v]:
                    continue  # Skip if u and v are adjacent, since they can't be opposite in a C4
                # Actually, u and v CAN be adjacent and still be opposite in a different C4?
                # No, in a C4 u-a-v-b-u, u and v are not adjacent (otherwise it would be a triangle or something else).
                # Wait, if u and v are adjacent, can they be opposite vertices of a C4?
                # A C4 is a cycle of length 4. If u and v are adjacent, they are consecutive, not opposite.
                # So for opposite vertices in a C4, they must NOT be adjacent.
                # But actually, the standard formula is:
                # Number of C4s = (1/4) * sum_{u,v} C(d(u,v), 2) where d(u,v) is the number of common neighbors
                # But this counts each C4 exactly 4 times? Let me verify.
                # C4: 1-2-3-4-1.
                # Opposite pairs: (1,3) and (2,4).
                # Common neighbors of 1 and 3: {2, 4} -> C(2,2) = 1
                # Common neighbors of 2 and 4: {1, 3} -> C(2,2) = 1
                # Sum = 2. Each C4 contributes 2 to the sum over all pairs.
                # So total C4s = sum / 2? No wait, we only sum over u < v.
                # Pairs (1,3) and (2,4) are both counted. So sum = 2, and there is 1 C4.
                # So total C4s = sum / 2.
                
                common = len(adj[u] & adj[v])
                if common >= 2:
                    total += common * (common - 1) // 2
        
        return total // 2
    
    # Better C4 count using matrix multiplication
    def count_c4s_matrix(edge_set):
        if len(edge_set) < 4:
            return 0
        
        # Build adjacency matrix
        A = np.zeros((n, n), dtype=np.int32)
        for idx in edge_set:
            u, v = edge_to_uv[idx]
            A[u, v] = 1
            A[v, u] = 1
        
        # Number of C4s = (1/8) * (trace(A^4) - 4*m - 2*sum(d_i^2))
        # Actually, the standard formula:
        # trace(A^4) = 8 * (# of C4s) + 4 * m + 2 * sum_{i} d_i^2
        # Wait, let me recall.
        # (A^2)_{ij} = number of length-2 paths from i to j
        # (A^4)_{ii} = number of closed walks of length 4 starting and ending at i
        # A closed walk of length 4 can be:
        # 1. A C4 (4 vertices, 4 edges): contributes to trace. Each C4 is counted 8 times in trace(A^4)
        #    (4 starting vertices * 2 directions)
        # 2. A triangle with one edge traversed twice: i-j-k-j-i. Counted for each triangle.
        #    Each triangle contributes 6 to trace(A^4)? Let's see.
        #    For triangle i-j-k-i:
        #    Closed walks of length 4:
        #    i-j-i-j-i, i-j-k-j-i, i-j-i-k-i, etc.
        #    This is getting complicated. Let me use a different approach.
        
        # Simpler: count C4s by iterating over pairs of vertices and counting common neighbors
        # This is O(n^3) which for n=31 is 31^3 = 29791, very fast.
        
        adj = [set() for _ in range(n)]
        for idx in edge_set:
            u, v = edge_to_uv[idx]
            adj[u].add(v)
            adj[v].add(u)
        
        total = 0
        for u in range(n):
            for v in range(u + 1, n):
                common = len(adj[u] & adj[v])
                if common >= 2:
                    total += common * (common - 1) // 2
        
        return total // 2
    
    # Greedy construction: build a graph with 91 edges minimizing C4s
    # Strategy: start with no edges, add edges one by one, choosing the edge that adds the fewest C4s
    
    best_c4_count = float('inf')
    best_edges = None
    
    # Try multiple random orderings
    for trial in range(200):
        if time.time() > deadline:
            break
        
        random.seed(trial)
        
        # Randomly order all edges
        edge_order = list(range(num_pairs))
        random.shuffle(edge_order)
        
        # Greedy: add edges in order, but only if they don't create too many C4s
        # Actually, let's do a smarter greedy: at each step, pick the edge that minimizes the increase in C4 count
        
        current_edges = set()
        current_adj = [set() for _ in range(n)]
        
        for _ in range(target_edges):
            if time.time() > deadline:
                break
            
            best_edge = None
            best_c4_increase = float('inf')
            
            # Sample a subset of candidate edges to keep it fast
            candidates = edge_order[:500]  # Consider first 500 unadded edges in random order
            # Actually, let's just iterate through all non-added edges but limit the check
            
            checked = 0
            for idx in edge_order:
                if idx in current_edges:
                    continue
                if checked >= 200:  # Limit checks per step
                    break
                
                u, v = edge_to_uv[idx]
                if u in current_adj[v]:
                    continue  # Edge already exists
                
                # Count how many C4s would be created by adding this edge
                # A new C4 is created if there is a path u-a-b-v of length 3
                # i.e., a is neighbor of u, b is neighbor of v, and a is neighbor of b
                # But we need to count the number of such paths, as each one creates a C4
                
                c4_increase = 0
                for a in current_adj[u]:
                    for b in current_adj[v]:
                        if a == b:
                            continue
                        if b in current_adj[a]:
                            c4_increase += 1
                
                if c4_increase < best_c4_increase:
                    best_c4_increase = c4_increase
                    best_edge = idx
                    if c4_increase == 0:
                        break  # Can't get better than 0
                
                checked += 1
            
            if best_edge is not None:
                current_edges.add(best_edge)
                u, v = edge_to_uv[best_edge]
                current_adj[u].add(v)
                current_adj[v].add(u)
            else:
                # No suitable edge found, just pick a random one
                for idx in edge_order:
                    if idx not in current_edges:
                        current_edges.add(idx)
                        u, v = edge_to_uv[idx]
                        current_adj[u].add(v)
                        current_adj[v].add(u)
                        break
        
        if len(current_edges) < target_edges:
            continue
        
        # Count total C4s
        c4_count = count_c4s_matrix(current_edges)
        
        if c4_count < best_c4_count:
            best_c4_count = c4_count
            best_edges = list(current_edges)
            
            if best_c4_count <= 1:
                break
    
    if best_edges is None:
        # Fallback: return a simple construction
        best_edges = list(range(target_edges))
    
    # Convert edge indices to [u, v] format
    result = []
    for idx in best_edges:
        u, v = edge_to_uv[idx]
        result.append([u, v])
    
    return result
