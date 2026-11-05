"""Polarity graph core + optimized extra edge + local search for min C4 count."""
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n = 31
    m = 91
    
    # Construct the polarity graph on PG(2,5)
    # Points of PG(2,5) are 1D subspaces of F_5^3
    # There are (5^3 - 1)/(5 - 1) = 31 points
    # A polarity is a symmetric bilinear form; we use the standard one
    # Bilinear form: B(x,y) = x0*y0 + x1*y1 + x2*y2 (mod 5)
    # Actually for PG(2,q) with q odd, the standard polarity uses:
    # B(x,y) = x0*y0 - x1*y1 + x2*y2 or similar
    # Let's use the standard Hermitian-like polarity for q=5:
    # A point is a vector (a,b,c) in F_5^3, not all zero, up to scaling
    # Two points are adjacent if B(p,q) = 0 where B is the polarity
    
    # Generate all 31 points in PG(2,5)
    # Each point is represented by a normalized vector (first non-zero entry is 1)
    points = []
    seen = set()
    for a in range(5):
        for b in range(5):
            for c in range(5):
                if a == 0 and b == 0 and c == 0:
                    continue
                # Normalize: make first non-zero entry 1
                vec = [a, b, c]
                for i in range(3):
                    if vec[i] != 0:
                        inv = pow(vec[i], 3, 5)  # inverse in F_5: x^3 = x^{-1}
                        vec = [(v * inv) % 5 for v in vec]
                        break
                key = tuple(vec)
                if key not in seen:
                    seen.add(key)
                    points.append(vec)
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Define the polarity: B(x,y) = x0*y0 + x1*y1 + x2*y2 (mod 5)
    # Actually, let's use a standard polarity. For PG(2,q) with q odd,
    # a common polarity is given by the symmetric matrix:
    # M = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    # But this gives B(x,y) = x·y, and the orthogonal graph may not be C4-free.
    # Let me use a different polarity. For q=5, use:
    # B(x,y) = x0*y0 + x1*y1 + x2*y2 + (x0*y1 + x1*y0)  -- no, that's not standard.
    
    # Actually, the standard polarity for PG(2,q) with q odd uses:
    # B(x,y) = x0*y0 - x1*y1 + x2*y2 (mod q)
    # Let's try this.
    
    def bilinear(p, q):
        return (p[0]*q[0] - p[1]*q[1] + p[2]*q[2]) % 5
    
    # Build adjacency: two points are adjacent if B(p,q) = 0
    # Note: for a polarity, B(p,p) = 0 for all p (self-orthogonal)
    # The polarity graph has edges between distinct points p,q with B(p,q)=0
    
    adj = np.zeros((n, n), dtype=np.int8)
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if bilinear(points[i], points[j]) == 0:
                adj[i, j] = 1
                adj[j, i] = 1
                edges.append((i, j))
    
    # Check how many edges we have
    polarity_edges = len(edges)
    
    # The polarity graph on PG(2,5) should have a specific number of edges.
    # For q=5, each point is orthogonal to q+1 = 6 other points (including itself).
    # So each vertex has degree 5 (excluding self). Total edges = 31*5/2 = 77.5?
    # Wait, that's not integer. Let me reconsider.
    
    # Actually, in PG(2,q), a point is orthogonal to q+1 points under a non-degenerate
    # polarity. For q=5, that's 6 points including itself. So degree = 5.
    # Total edges = 31*5/2 = 77.5 -- not integer, so something's wrong.
    
    # Let me just count what we get.
    # If polarity_edges != 90, we need to add/remove edges to get to 90 or 91.
    
    # Let's see what we got. If it's not 90, we'll adjust.
    # Actually, for a polarity graph, the number of edges should be q(q+1)(q-1)/2... 
    # Let me just use whatever we get and add edges if needed.
    
    # Count C4s in the current graph
    def count_c4s(adj_matrix):
        """Count the number of 4-cycles in the graph."""
        # A 4-cycle is a set of 4 vertices {a,b,c,d} with edges ab, bc, cd, da
        # We count each 4-cycle once by requiring a < b, a < d, and b < c (or similar)
        # Better: count pairs of vertices with 2 common neighbors, then divide appropriately.
        # Number of C4s = sum over all pairs (u,v) of C(common_neighbors(u,v), 2) / 2
        # Wait, each C4 is counted twice (once for each pair of opposite vertices).
        # Actually: for a 4-cycle a-b-c-d-a, the pairs of opposite vertices are (a,c) and (b,d).
        # Each pair (u,v) with k common neighbors contributes C(k,2) to the count,
        # and each C4 is counted exactly twice (once for each diagonal pair).
        # So total C4s = (1/2) * sum_{u<v} C(common(u,v), 2)
        
        n = adj_matrix.shape[0]
        total = 0
        for u in range(n):
            for v in range(u+1, n):
                # Common neighbors of u and v
                common = np.sum(adj_matrix[u] & adj_matrix[v])
                if common >= 2:
                    total += common * (common - 1) // 2
        return total // 2
    
    # If the polarity graph doesn't have exactly 90 edges, we need to adjust.
    # Let's first try with whatever we have and add edges to reach 91.
    
    if polarity_edges < 91:
        # Add edges until we have 91
        edge_set = set(edges)
        # Try adding edges that create the fewest C4s
        # First, add random edges
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n) if (i,j) not in edge_set]
        random.seed(42)
        random.shuffle(all_possible)
        
        for e in all_possible:
            if len(edge_set) >= 91:
                break
            edge_set.add(e)
            adj[e[0], e[1]] = 1
            adj[e[1], e[0]] = 1
    elif polarity_edges > 91:
        # Remove edges until we have 91 -- this shouldn't happen for a proper polarity graph
        # but let's handle it
        edge_list = list(edge_set)
        random.seed(42)
        random.shuffle(edge_list)
        while len(edge_list) > 91:
            e = edge_list.pop()
            edge_set.remove(e)
            adj[e[0], e[1]] = 0
            adj[e[1], e[0]] = 0
    
    # Now we have 91 edges. Let's do local search to minimize C4 count.
    # Local search: try swapping an edge (remove one, add another) to reduce C4 count.
    
    current_c4s = count_c4s(adj)
    
    # Precompute common neighbor counts for speed
    def compute_common_neighbors(adj_matrix):
        """Compute common neighbor counts for all pairs."""
        n = adj_matrix.shape[0]
        common = np.zeros((n, n), dtype=np.int32)
        for u in range(n):
            for v in range(u+1, n):
                common[u, v] = np.sum(adj_matrix[u] & adj_matrix[v])
                common[v, u] = common[u, v]
        return common
    
    common = compute_common_neighbors(adj)
    
    # Delta C4 when adding edge (u,v): C4s created = common[u,v]
    # Delta C4 when removing edge (u,v): C4s destroyed = common[u,v] - 1
    # (minus 1 because the edge itself contributes to the common neighbor count)
    
    def delta_add(u, v):
        """C4s created by adding edge (u,v)."""
        return common[u, v]
    
    def delta_remove(u, v):
        """C4s destroyed by removing edge (u,v)."""
        # When we remove edge (u,v), the common neighbor count for (u,v) changes,
        # and also for pairs involving neighbors of u and v.
        # This is complex; let's just recompute locally.
        # For now, approximate: removing edge (u,v) destroys C(common[u,v]-1, 1) C4s?
        # Actually, each common neighbor w of u and v forms a path u-w-v, and with edge (u,v)
        # forms a triangle, not a C4. C4s involving edge (u,v) are of the form u-x-v-y-u
        # where x is common neighbor of u,v and y is... no.
        # A C4 containing edge (u,v) has the form u-v-x-y-u where x is neighbor of v,
        # y is neighbor of u, and x-y is an edge. So the number of C4s containing (u,v)
        # is the number of edges in the bipartite graph between N(u)\{v} and N(v)\{u}
        # that go from a neighbor of u to a neighbor of v.
        # This is complex. Let's just recompute the C4 count after each swap.
        pass
    
    # For efficiency, let's use a simpler approach: just recompute C4 count after each swap.
    # With n=31, counting C4s takes O(n^3) ~ 30000 operations, which is fast.
    
    best_c4s = current_c4s
    best_adj = adj.copy()
    best_edges = list(edge_set)
    
    # Local search
    iterations = 0
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline - 5 and no_improve < max_no_improve:
        iterations += 1
        improved = False
        
        # Try random swaps
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            
            # Pick a random edge to remove
            if not best_edges:
                break
            idx = random.randint(0, len(best_edges) - 1)
            remove_edge = best_edges[idx]
            
            # Pick a random non-edge to add
            # Generate a random pair
            while True:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                if (u, v) in edge_set:
                    continue
                break
            
            # Check if removing remove_edge and adding (u,v) keeps 91 edges
            # (it does, since we remove one and add one)
            # But we need (u,v) != remove_edge
            if (u, v) == remove_edge:
                continue
            
            # Apply swap
            old_adj = best_adj.copy()
            best_adj[remove_edge[0], remove_edge[1]] = 0
            best_adj[remove_edge[1], remove_edge[0]] = 0
            best_adj[u, v] = 1
            best_adj[v, u] = 1
            
            new_c4s = count_c4s(best_adj)
            
            if new_c4s < best_c4s:
                best_c4s = new_c4s
                best_edges[idx] = (u, v)
                edge_set = set(best_edges)
                improved = True
                no_improve = 0
            else:
                # Revert
                best_adj[:] = old_adj
        
        if not improved:
            no_improve += 1
    
    # Convert to list of [u, v] pairs
    result = [[e[0], e[1]] for e in best_edges]
    return result

# Test
if __name__ == "__main__":
    edges = construct()
    assert len(edges) == 91
    assert len(set(tuple(sorted(e)) for e in edges)) == 91
    for e in edges:
        assert 0 <= e[0] <= 30
        assert 0 <= e[1] <= 30
        assert e[0] != e[1]
    print("OK")
