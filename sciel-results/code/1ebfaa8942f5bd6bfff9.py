"""Minimize 4-cycles in a 31-vertex, 91-edge graph using polarity graph + optimal extra edge."""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n = 31
    m = 91
    
    # Enumerate points of PG(2,5) as 1D subspaces of F_5^3
    # Represent each point by a canonical representative vector
    points = []
    seen = set()
    for a in range(5):
        for b in range(5):
            for c in range(5):
                if a == 0 and b == 0 and c == 0:
                    continue
                # Canonical form: first non-zero entry is 1
                vec = [a, b, c]
                # Find first non-zero
                for i in range(3):
                    if vec[i] != 0:
                        inv = pow(vec[i], 3, 5)  # inverse mod 5: x^3 = x^-1 mod 5
                        vec = [(x * inv) % 5 for x in vec]
                        break
                key = tuple(vec)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Define polarity: u^T A v = 0 where A is a symmetric matrix
    # Use A = [[0,1,0],[1,0,0],[0,0,1]]
    # u^T A v = u0*v1 + u1*v0 + u2*v2
    def orthogonal(u, v):
        return (u[0]*v[1] + u[1]*v[0] + u[2]*v[2]) % 5 == 0
    
    # Build adjacency for polarity graph
    adj = [[False]*n for _ in range(n)]
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if orthogonal(points[i], points[j]):
                adj[i][j] = adj[j][i] = True
                edges.append([i, j])
    
    polarity_edge_count = len(edges)
    
    # For q=5, the polarity graph should have a specific number of edges
    # Let's see what we get and proceed
    # If we have exactly 90 edges, we need to add 1
    # If we have fewer, we need to add more
    
    # Count C4s in the polarity graph
    def count_c4s(adj_matrix, n_vertices):
        count = 0
        # A 4-cycle is a set of 4 distinct vertices a,b,c,d with edges ab, bc, cd, da
        # Count using: for each pair (a,c), count common neighbors, then sum over unordered pairs
        # Number of C4s = (1/8) * sum over all unordered pairs {a,c} of (common_neighbors(a,c) choose 2)
        # Wait, that counts each C4 multiple times. Let me think...
        # Actually: number of C4s = (1/2) * sum over unordered pairs {a,c} of C(common_neighbors(a,c), 2)
        # No, let me be more careful.
        # A 4-cycle a-b-c-d-a has diagonals (a,c) and (b,d).
        # For a fixed pair (a,c), the number of 4-cycles with a and c as opposite vertices
        # equals C(common_neighbors(a,c), 2)
        # Each 4-cycle is counted twice (once for each pair of opposite vertices)
        # So total C4s = (1/2) * sum over all unordered pairs {a,c} of C(common_neighbors(a,c), 2)
        
        for a in range(n_vertices):
            for c in range(a+1, n_vertices):
                common = 0
                for b in range(n_vertices):
                    if b != a and b != c and adj_matrix[a][b] and adj_matrix[b][c]:
                        common += 1
                count += common * (common - 1) // 2
        return count // 2
    
    base_c4s = count_c4s(adj, n)
    
    # If we already have 91 edges, we're done
    if polarity_edge_count == m:
        return edges
    
    # If we have fewer than 91 edges, we need to add edges
    # If we have more, we need to remove edges (shouldn't happen for a proper polarity)
    
    # Let's check: for the polarity [[0,1,0],[1,0,0],[0,0,1]] on PG(2,5)
    # Each point P=(x,y,z) has polar line: x*y' + y*x' + z*z' = 0
    # P is on its polar line iff x*y + y*x + z*z = 2xy + z^2 = 0
    # Absolute points satisfy 2xy + z^2 = 0 mod 5
    
    # The number of edges depends on the specific polarity
    # Let's just work with what we have
    
    if polarity_edge_count > m:
        # Need to remove edges - this is problematic, should not happen
        # Just return first m edges as fallback
        return edges[:m]
    
    # Need to add (m - polarity_edge_count) edges
    num_to_add = m - polarity_edge_count
    
    # Find all non-edges
    non_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i][j]:
                non_edges.append((i, j))
    
    # For each candidate edge to add, compute how many C4s it creates
    # Adding edge (u,v) creates a C4 for each pair of common neighbors (a,b) of u and v
    # where a and b are both adjacent to both u and v, AND a-b is an edge
    # Wait, no. Adding edge (u,v) creates a 4-cycle u-a-v-b-u if:
    #   a is adjacent to u, b is adjacent to v, a is adjacent to v (no, a is not adjacent to v yet)
    # Let me reconsider.
    # A 4-cycle containing the new edge (u,v) is: u-v-b-a-u
    # This requires: v-b exists, b-a exists, a-u exists
    # So for each pair (a,b) where a~u, b~v, and a~b, we get a C4
    # But we also need u,v,a,b all distinct
    
    def c4s_created_by_edge(u, v, adj_matrix, n_vertices):
        count = 0
        for a in range(n_vertices):
            if a == u or a == v:
                continue
            if not adj_matrix[u][a]:
                continue
            for b in range(n_vertices):
                if b == u or b == v or b == a:
                    continue
                if not adj_matrix[v][b]:
                    continue
                if not adj_matrix[a][b]:
                    continue
                count += 1
        return count
    
    # Greedy: add the edge that creates the fewest C4s
    current_edges = list(edges)
    current_adj = [row[:] for row in adj]
    
    for _ in range(num_to_add):
        best_edge = None
        best_c4s = float('inf')
        
        # Check time
        if time.time() > deadline - 5:
            break
        
        for (u, v) in non_edges:
            if current_adj[u][v]:
                continue
            c4s = c4s_created_by_edge(u, v, current_adj, n)
            if c4s < best_c4s:
                best_c4s = c4s
                best_edge = (u, v)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        current_adj[u][v] = current_adj[v][u] = True
        current_edges.append([u, v])
    
    # Now do local search: try swapping edges to reduce C4s
    # Time check
    if time.time() > deadline - 10:
        # Return current best
        return current_edges
    
    # Local search: try to remove an edge and add another to reduce C4s
    # First, let's compute current C4 count
    current_c4s = count_c4s(current_adj, n)
    
    # Try single-edge swaps: remove edge e1, add edge e2
    # This is expensive, so let's be strategic
    # For each edge in the graph, compute how many C4s it's in
    # For each non-edge, compute how many C4s adding it would create
    # If removing an edge destroys more C4s than adding a non-edge creates, do the swap
    
    def c4s_containing_edge(u, v, adj_matrix, n_vertices):
        """Count C4s that contain the edge (u,v)."""
        count = 0
        # A C4 containing edge (u,v) is: u-v-b-a-u
        # Requires: v~b, b~a, a~u, and u,v,a,b distinct
        for a in range(n_vertices):
            if a == u or a == v:
                continue
            if not adj_matrix[u][a]:
                continue
            for b in range(n_vertices):
                if b == u or b == v or b == a:
                    continue
                if not adj_matrix[v][b]:
                    continue
                if not adj_matrix[a][b]:
                    continue
                count += 1
        return count
    
    # Build edge list from current_adj
    edge_set = set()
    for i in range(n):
        for j in range(i+1, n):
            if current_adj[i][j]:
                edge_set.add((i, j))
    
    non_edge_set = set()
    for i in range(n):
        for j in range(i+1, n):
            if not current_adj[i][j]:
                non_edge_set.add((i, j))
    
    # Try swaps with time limit
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        best_swap = None
        best_gain = 0
        
        # Sample some edges to try removing (not all, to save time)
        edge_list = list(edge_set)
        random.seed(42)
        # Try a subset
        sample_size = min(len(edge_list), 200)
        sampled_edges = random.sample(edge_list, sample_size)
        
        for e1 in sampled_edges:
            u1, v1 = e1
            # C4s destroyed by removing e1
            destroyed = c4s_containing_edge(u1, v1, current_adj, n)
            if destroyed <= 0:
                continue
            
            # Try adding some non-edges
            non_edge_list = list(non_edge_set)
            non_sample_size = min(len(non_edge_list), 100)
            sampled_non_edges = random.sample(non_edge_list, non_sample_size)
            
            for e2 in sampled_non_edges:
                u2, v2 = e2
                # C4s created by adding e2
                created = c4s_created_by_edge(u2, v2, current_adj, n)
                gain = destroyed - created
                if gain > best_gain:
                    best_gain = gain
                    best_swap = (e1, e2)
        
        if best_swap is not None:
            e1, e2 = best_swap
            u1, v1 = e1
            u2, v2 = e2
            current_adj[u1][v1] = current_adj[v1][u1] = False
            current_adj[u2][v2] = current_adj[v2][u2] = True
            edge_set.remove(e1)
            non_edge_set.remove(e2)
            edge_set.add(e2)
            non_edge_set.add(e1)
            improved = True
    
    # Convert back to list of edges
    result = []
    for (u, v) in edge_set:
        result.append([u, v])
    
    return result
