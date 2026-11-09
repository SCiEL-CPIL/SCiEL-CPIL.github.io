import numpy as np
import time
import random

N = 31
E = 91
DEADLINE_SECONDS = 115.0

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE_SECONDS
    
    # 1. Generate points of PG(2,5)
    points = []
    for x in range(5):
        for y in range(5):
            points.append((x, y, 1))
    for x in range(5):
        points.append((x, 1, 0))
    points.append((1, 0, 0))
    
    # 2. Construct Polarity Graph (Orthogonality Graph)
    # Edge if dot == 0
    adj = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(i + 1, N):
            p1 = points[i]
            p2 = points[j]
            dot = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % 5
            if dot == 0:
                adj[i, j] = 1
                adj[j, i] = 1
                
    # Current edge count
    current_edges = int(np.sum(adj) // 2)
    
    # Helper to count C4s
    def count_c4(adj):
        A2 = adj @ adj
        i, j = np.triu_indices(N, k=1)
        common_counts = A2[i, j]
        terms = common_counts * (common_counts - 1) // 2
        total = np.sum(terms)
        return int(total) // 2
        
    current_c4 = count_c4(adj)
    
    # 3. Adjust edge count to E=91
    # If current_edges > 91, remove edges that are part of the most C4s.
    # If current_edges < 91, add edges that create the fewest new C4s.
    
    def get_edges(adj):
        edges = []
        for i in range(N):
            for j in range(i+1, N):
                if adj[i,j] == 1:
                    edges.append((i,j))
        return edges
        
    def add_edge(adj, u, v):
        adj[u, v] = 1
        adj[v, u] = 1
        
    def remove_edge(adj, u, v):
        adj[u, v] = 0
        adj[v, u] = 0
        
    # Greedy adjustment
    while current_edges > E:
        # Find edge in most C4s
        # Calculate C4 count contribution for each edge
        # An edge (u,v) is in C(u,v) C4s, where C(u,v) = common_neighbors(u,v) * (common_neighbors(u,v)-1) / 2 is wrong.
        # Actually, removing (u,v) reduces C4s by the number of C4s containing (u,v).
        # A C4 containing (u,v) is u-w-v-x-u or similar.
        # The number of C4s containing edge (u,v) is the number of pairs (w,x) such that w is neighbor of u and v, x is neighbor of u and v... no.
        # C4s containing (u,v): u-a-b-v-u. a is neighbor of u and v? No.
        # Path u-a-b-v. a neighbor of u, b neighbor of v, a-b edge.
        # So for edge (u,v), count is sum over a in N(u) of |N(a) ∩ N(v)| ? No.
        # Let's just use the brute force delta for small N.
        
        best_edge = None
        max_reduction = -1
        
        # Precompute common neighbors for all pairs?
        # For N=31, N^2 is small.
        # Let's just iterate edges and calculate delta.
        for i in range(N):
            for j in range(i+1, N):
                if adj[i,j] == 1:
                    # Calculate C4s removed by removing (i,j)
                    # C4s containing (i,j) correspond to paths of length 3 between i and j not using edge (i,j)
                    # Path i-a-b-j. a ~ i, b ~ j, a ~ b.
                    # And a != j, b != i.
                    # Count = sum_{a in N(i)} sum_{b in N(j)} (1 if a~b and a!=j and b!=i else 0)
                    # Wait, if a~b, then i-a-b-j-i is a C4.
                    # We need to be careful not to count triangles or other things.
                    # A C4 is a cycle of length 4.
                    # The number of C4s containing edge (i,j) is the number of length-3 paths from i to j that do not use edge (i,j).
                    # Since (i,j) is an edge, any length 3 path i-a-b-j with a~i, b~j, a~b forms a C4 i-a-b-j-i.
                    # We must ensure a != j and b != i (which is true if simple graph) and a != b (simple graph).
                    
                    count = 0
                    # N(i) and N(j)
                    # Using bitsets or lists
                    # For speed, let's use numpy dot product or similar?
                    # N=31 is small.
                    
                    for a in range(N):
                        if a == i or a == j:
                            continue
                        if adj[i, a] == 1:
                            for b in range(N):
                                if b == i or b == j or b == a:
                                    continue
                                if adj[j, b] == 1 and adj[a, b] == 1:
                                    count += 1
                                    
                    # Each C4 is counted once for each of its 4 edges? No, we are counting C4s containing (i,j).
                    # The path i-a-b-j defines a C4.
                    # Is it possible that the same C4 is counted twice for the same edge (i,j)?
                    # i-a-b-j-i. The reverse is i-j-b-a-i.
                    # The path i-a-b-j is distinct from i-b-a-j?
                    # If we swap a and b, we get i-b-a-j.
                    # This is a different path unless the graph has symmetries.
                    # But a C4 is a set of 4 vertices.
                    # For a fixed edge (i,j), the other two vertices are {a,b}.
                    # The set {a,b} determines the C4 uniquely (if a~b).
                    # So we should count pairs {a,b}.
                    # My loop counts ordered pairs (a,b).
                    # So divide by 2?
                    # Let's check: i-a-b-j. a~i, b~j, a~b.
                    # If we swap a and b: i-b-a-j. b~i, a~j, b~a.
                    # This requires b~i and a~j.
                    # In a C4 i-a-b-j-i, a~i, a~b, b~j, j~i.
                    # Does b~i? No, b is not neighbor of i (unless triangle, but C4 is induced? No, subgraph).
                    # Wait, C4 is a cycle.
                    # If i-a-b-j-i is a C4, then edges are (i,a), (a,b), (b,j), (j,i).
                    # Non-edges are (i,b) and (a,j).
                    # So b is NOT neighbor of i. a is NOT neighbor of j.
                    # So the condition for (b,a) to be a valid path start is b~i, which is false.
                    # So each C4 containing (i,j) is counted exactly once?
                    # Let's re-verify.
                    # Path i-a-b-j.
                    # Conditions: a~i, b~j, a~b.
                    # And we need to form a cycle i-a-b-j-i.
                    # This requires j~i (true) and a~b (true).
                    # Does it require a~j or b~i? No.
                    # So if a~j is true, then i-a-j-i is a triangle, and i-a-b-j-i is a C4 plus a diagonal?
                    # The problem asks for "four-cycles (subgraphs)".
                    # Usually this means induced C4s or just C4 subgraphs?
                    # "number of four-cycles (subgraphs)" usually means the number of distinct sets of 4 vertices that form a C4.
                    # Or the number of cyclic sequences?
                    # Standard definition: number of C4 subgraphs.
                    # If i-a-j is a triangle, and i-b-j is a triangle, and a-b is edge...
                    # Let's stick to the standard count: number of pairs of edges (u,v) and (w,x) with u~v, w~x, u~w, v~x, u!=w, v!=x, etc.
                    # The formula A2[i,j]*(A2[i,j]-1)/2 counts pairs of common neighbors.
                    # Each pair of common neighbors (a,b) for non-adjacent i,j forms a C4 i-a-j-b-i.
                    # If i,j are adjacent, they form a triangle with a, and a C4 with b?
                    # No, if i~j, and a,b are common neighbors.
                    # i-a-j-b-i.
                    # Edges: (i,a), (a,j), (j,b), (b,i).
                    # This is a C4.
                    # So the formula counts C4s where i,j are opposite vertices?
                    # Yes, i and j are opposite.
                    # Total C4s = sum over all pairs (i,j) (adjacent or not) of C(common(i,j), 2) / 2 ?
                    # No.
                    # Each C4 has 2 pairs of opposite vertices.
                    # So sum over all pairs (i,j) of C(common(i,j), 2) counts each C4 twice.
                    # So Total = 1/2 * sum_{i<j} C(common(i,j), 2).
                    # This works for all pairs (i,j).
                    
                    # So for edge removal (i,j), we need to calculate how many C4s contain (i,j).
                    # A C4 contains edge (i,j) if i,j are adjacent in the C4.
                    # In the sum formula, i,j are opposite.
                    # So C4s containing edge (i,j) are NOT counted in the term for pair (i,j).
                    # They are counted in terms for pairs (i, neighbor_of_j) and (j, neighbor_of_i)?
                    # Let's trace a C4: v1-v2-v3-v4-v1.
                    # Opposite pairs: (v1,v3) and (v2,v4).
                    # Common neighbors of v1,v3 are v2,v4. C(2,2)=1.
                    # Common neighbors of v2,v4 are v1,v3. C(2,2)=1.
                    # Sum = 2. Divide by 2 = 1 C4.
                    # So the formula works.
                    
                    # To find delta for removing edge (u,v):
                    # We need to count C4s that use edge (u,v).
                    # These C4s have u,v as adjacent vertices.
                    # Let the C4 be u-a-b-v-u.
                    # Opposite pairs: (u,b) and (a,v).
                    # So this C4 is counted in term (u,b) if a,v are common neighbors of u,b.
                    # And in term (a,v) if u,b are common neighbors of a,v.
                    # This is getting complicated to calculate delta directly from the formula.
                    
                    # Alternative: Just calculate count_c4(adj) and count_c4(adj_removed).
                    # N=31, E~90.
                    # count_c4 is O(N^3) or O(N^2 * avg_deg).
                    # It's fast enough for 31 nodes.
                    
                    pass # Placeholder, we will use the slow method for now to ensure correctness, then optimize if needed.
                    
                    # Actually, let's just use the slow method in the loop.
                    # It might be slow but N=31 is small.
                    
                    # Let's try to optimize count_c4 or the delta calculation.
                    # For now, let's just do the brute force delta.
                    
                    # Save adj, remove, count, restore.
                    # This is O(N^3) per edge. 90 edges * 31^3 ~ 2.7M ops.
                    # Python is slow. 2.7M * constant.
                    # Might be too slow for 115s if we do many iterations.
                    
                    # Let's stick to the heuristic or a faster delta.
                    
                    # Heuristic: Degree sum.
                    pass

        # Let's use the heuristic from the original code but maybe improved.
        # Or just random removal?
        # The original code used degree sum.
        
        degrees = adj.sum(axis=1)
        max_sum = -1
        u, v = 0, 1
        for i in range(N):
            for j in range(i+1, N):
                if adj[i,j] == 1:
                    s = degrees[i] + degrees[j]
                    if s > max_sum:
                        max_sum = s
                        u, v = i, j
        remove_edge(adj, u, v)
        current_edges -= 1
        current_c4 = count_c4(adj)
        
    while current_edges < E:
        # Find non-edge with min common neighbors
        best_edge = None
        min_common = float('inf')
        for i in range(N):
            for j in range(i+1, N):
                if adj[i,j] == 0:
                    common = np.sum(adj[i] & adj[j])
                    if common < min_common:
                        min_common = common
                        best_edge = (i, j)
                        if min_common == 0:
                            break
            if min_common == 0:
                break
        if best_edge:
            add_edge(adj, best_edge[0], best_edge[1])
            current_edges += 1
            current_c4 = count_c4(adj)
            
    # 4. Local Search to minimize C4s
    # Swap an existing edge with a non-edge if it reduces C4s
    best_c4 = current_c4
    no_improvement_count = 0
    
    # Precompute list of all possible edges
    all_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    while time.time() < deadline:
        improved = False
        # Randomize order to escape local minima
        edges = get_edges(adj)
        random.shuffle(edges)
        
        for u, v in edges:
            if time.time() > deadline:
                break
            # Try removing (u,v)
            remove_edge(adj, u, v)
            
            # Try adding a random non-edge
            non_edges = [e for e in all_edges if adj[e[0], e[1]] == 0]
            
            if not non_edges:
                add_edge(adj, u, v)
                break
                
            # Pick a few candidates to add
            random.shuffle(non_edges)
            candidates = non_edges[:20] # Increase candidates
            
            best_add = None
            best_c4_after = float('inf')
            
            for a, b in candidates:
                add_edge(adj, a, b)
                c4_after = count_c4(adj)
                remove_edge(adj, a, b)
                if c4_after < best_c4_after:
                    best_c4_after = c4_after
                    best_add = (a, b)
                    
            if best_add and best_c4_after < best_c4:
                # Commit the swap
                add_edge(adj, best_add[0], best_add[1])
                best_c4 = best_c4_after
                improved = True
                no_improvement_count = 0
            else:
                # Restore (u,v)
                add_edge(adj, u, v)
                
        if not improved:
            no_improvement_count += 1
            if no_improvement_count > 50:
                break
                
    return get_edges(adj)

if __name__ == "__main__":
    edges = construct()
    print(f"Edges: {len(edges)}")
    adj = np.zeros((N, N), dtype=int)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    def count_c4_final(adj):
        A2 = adj @ adj
        i, j = np.triu_indices(N, k=1)
        common_counts = A2[i, j]
        terms = common_counts * (common_counts - 1) // 2
        total = np.sum(terms)
        return int(total) // 2
    print(f"C4s: {count_c4_final(adj)}")
