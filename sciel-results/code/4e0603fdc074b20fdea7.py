"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n = 20
    colors = list(range(n))
    
    # We'll represent the graph as a dict: (u,v) -> color, with u < v
    # Start with a complete graph K20 and try to assign colors
    # such that every 4-cycle is rainbow.
    
    # Strategy: Use a greedy/local search approach.
    # Key insight: In a proper edge-coloring of K_n with n colors (when n is even),
    # each color class is a perfect matching. For n=20, we have 20 perfect matchings.
    # The constraint is that no 4-cycle uses fewer than 4 distinct colors.
    
    # A 4-cycle u-v-w-x-u uses edges (u,v), (v,w), (w,x), (x,u).
    # These four edges must have distinct colors.
    
    # Let's try to construct a valid coloring of K20.
    # One known construction: use the cyclic group Z_n for even n.
    # For n=20, label vertices 0..19.
    # Color class c: edges (i, i+1+c mod 20) for i in some set... 
    # Actually, for complete graph K_{2m}, a 1-factorization exists.
    # The standard 1-factorization of K_{2m} uses colors 0..2m-1.
    # Color c (0 <= c < 2m-1): pairs (i, j) where i+j ≡ c (mod 2m-1), 
    # plus the edge (2m-1, ...). Let me use a simpler formulation.
    
    # Standard 1-factorization of K_{2m}:
    # Vertices 0, 1, ..., 2m-2, ∞
    # For color c (0 <= c < 2m-1):
    #   Edge (∞, c)
    #   For i in 0..2m-2: edge (c+i mod (2m-1), c-i mod (2m-1))
    
    # Let's implement this for m=10, so 2m=20.
    # Vertices: 0..18 and ∞=19
    # Colors: 0..18 (19 colors for the 1-factorization of K_19 part) 
    # Wait, K_20 has 190 edges and needs 19 colors for a proper edge-coloring? 
    # No, K_{2m} has chromatic index 2m-1. So K_20 needs 19 colors for proper edge-coloring.
    # But we have 20 colors available. That's fine.
    
    # Actually, let me reconsider. The task says colours 0..19, so 20 colors.
    # K_20 has 190 edges. A proper edge-coloring uses 19 colors (each a perfect matching of 10 edges).
    # We can use 19 of the 20 colors.
    
    # Let's build the standard 1-factorization and check if it satisfies the 4-cycle rainbow condition.
    
    def build_1_factorization(n):
        """Build a 1-factorization of K_n where n is even."""
        if n % 2 != 0:
            return None
        m = n // 2
        # Vertices 0..n-2 and vertex n-1 (infinity)
        inf = n - 1
        colors_used = n - 1  # 0..n-2
        edge_color = {}
        
        for c in range(colors_used):
            # Edge (inf, c)
            u, v = min(inf, c), max(inf, c)
            edge_color[(u, v)] = c
            
            # For i in 1..m-1: pair (c+i mod (n-1), c-i mod (n-1))
            for i in range(1, m):
                u1 = (c + i) % (n - 1)
                v1 = (c - i) % (n - 1)
                u, v = min(u1, v1), max(u1, v1)
                edge_color[(u, v)] = c
        
        return edge_color
    
    edge_color = build_1_factorization(n)
    
    # Now check: does this coloring satisfy the rainbow 4-cycle condition?
    # A 4-cycle is u-v-w-x-u where {u,v,w,x} are distinct and all four edges exist.
    # In K20, all edges exist, so we just need to check all 4-cycles.
    # A 4-cycle is determined by 4 distinct vertices {a,b,c,d} and the ordering.
    # Actually, a 4-cycle in a complete graph: pick 4 vertices, there are 3 distinct 4-cycles (up to rotation/reflection).
    # For vertices a,b,c,d, the 4-cycles are:
    # a-b-c-d-a, a-b-d-c-a, a-c-b-d-a (and their reverses, which are the same cycle)
    
    def is_valid_4cycle_coloring(edge_color, n):
        """Check if all 4-cycles have rainbow colors."""
        vertices = list(range(n))
        for a, b, c, d in itertools.combinations(vertices, 4):
            # Three distinct 4-cycles on {a,b,c,d}:
            # Cycle 1: a-b-c-d-a
            edges1 = [(a,b), (b,c), (c,d), (d,a)]
            colors1 = [edge_color.get((min(u,v), max(u,v))) for u,v in edges1]
            if len(set(colors1)) < 4:
                return False
            
            # Cycle 2: a-b-d-c-a
            edges2 = [(a,b), (b,d), (d,c), (c,a)]
            colors2 = [edge_color.get((min(u,v), max(u,v))) for u,v in edges2]
            if len(set(colors2)) < 4:
                return False
            
            # Cycle 3: a-c-b-d-a
            edges3 = [(a,c), (c,b), (b,d), (d,a)]
            colors3 = [edge_color.get((min(u,v), max(u,v))) for u,v in edges3]
            if len(set(colors3)) < 4:
                return False
        
        return True
    
    # Check validity
    if is_valid_4cycle_coloring(edge_color, n):
        # Convert to output format
        result = []
        for (u, v), c in sorted(edge_color.items()):
            result.append([u, v, c])
        return result
    
    # If the standard 1-factorization doesn't work, we need to modify it.
    # Let's try local search: start with the 1-factorization and try to fix violations
    # by swapping colors or reassigning edges.
    
    # Actually, let's think about this differently. The constraint is quite strong.
    # Let's try a different construction or use simulated annealing.
    
    # Alternative: try to find a valid subgraph of K20 (not necessarily complete)
    # that maximizes edges while satisfying the rainbow 4-cycle condition.
    
    # Let's start with the 1-factorization and use local search to fix violations.
    # Represent state as edge_color dict.
    
    # First, let's count violations in the current coloring.
    def count_violations(edge_color, n):
        """Count the number of 4-cycles that are not rainbow."""
        vertices = list(range(n))
        violations = 0
        for a, b, c, d in itertools.combinations(vertices, 4):
            # Cycle 1: a-b-c-d-a
            edges1 = [(a,b), (b,c), (c,d), (d,a)]
            colors1 = [edge_color.get((min(u,v), max(u,v))) for u,v in edges1]
            if len(set(colors1)) < 4:
                violations += 1
            
            # Cycle 2: a-b-d-c-a
            edges2 = [(a,b), (b,d), (d,c), (c,a)]
            colors2 = [edge_color.get((min(u,v), max(u,v))) for u,v in edges2]
            if len(set(colors2)) < 4:
                violations += 1
            
            # Cycle 3: a-c-b-d-a
            edges3 = [(a,c), (c,b), (b,d), (d,a)]
            colors3 = [edge_color.get((min(u,v), max(u,v))) for u,v in edges3]
            if len(set(colors3)) < 4:
                violations += 1
        
        return violations
    
    # Let's try a different approach: use a random proper edge-coloring and check.
    # Or better, let's try to construct a valid graph incrementally.
    
    # Given time constraints, let's try a few different 1-factorizations and pick the best.
    # Or use a randomized local search.
    
    # Let's try: start with the 1-factorization, and if invalid, try swapping edges
    # between color classes to reduce violations.
    
    # For now, let's try a simple approach: use the 1-factorization but with a different
    # labeling of vertices to see if any permutation works.
    
    # Actually, let's just try to find ANY valid graph with many edges.
    # Start with a sparse graph and add edges while maintaining validity.
    
    # Greedy approach: start with no edges, add edges one by one if valid.
    # But checking validity after each addition is expensive.
    
    # Let's try: start with the 1-factorization, and if it has violations,
    # remove edges that are part of violating 4-cycles until valid, then try to re-add.
    
    # This is getting complex. Let's try a simpler heuristic:
    # Use a random proper edge-coloring (1-factorization) and check.
    
    # Try multiple random 1-factorizations
    best_edges = 0
    best_result = None
    
    for trial in range(50):
        if time.time() > deadline - 30:
            break
        
        # Random permutation of vertices
        perm = list(range(n))
        random.shuffle(perm)
        
        # Build 1-factorization with permuted vertices
        edge_color_perm = {}
        inf_idx = perm[19]  # last vertex is infinity
        for c in range(19):
            inf_vertex = inf_idx
            c_vertex = perm[c]
            u, v = min(inf_vertex, c_vertex), max(inf_vertex, c_vertex)
            edge_color_perm[(u, v)] = c
            
            for i in range(1, 10):
                u1_idx = (c + i) % 19
                v1_idx = (c - i) % 19
                u1 = perm[u1_idx]
                v1 = perm[v1_idx]
                u, v = min(u1, v1), max(u1, v1)
                edge_color_perm[(u, v)] = c
        
        # Check validity
        if is_valid_4cycle_coloring(edge_color_perm, n):
            result = []
            for (u, v), c in sorted(edge_color_perm.items()):
                result.append([u, v, c])
            if len(result) > best_edges:
                best_edges = len(result)
                best_result = result
            if best_edges == 190:  # complete graph
                break
    
    if best_result is not None:
        return best_result
    
    # If no valid complete graph found, try to find a valid subgraph.
    # Start with the 1-factorization and remove edges to fix violations.
    
    edge_color = build_1_factorization(n)
    
    # Iteratively remove edges that are in violating 4-cycles
    for iteration in range(100):
        if time.time() > deadline - 10:
            break
        
        violations = count_violations(edge_color, n)
        if violations == 0:
            break
        
        # Find an edge that's in many violations and remove it
        # For simplicity, just remove a random edge from a violating 4-cycle
        vertices = list(range(n))
        found = False
        for a, b, c, d in itertools.combinations(vertices, 4):
            cycles = [
                [(a,b), (b,c), (c,d), (d,a)],
                [(a,b), (b,d), (d,c), (c,a)],
                [(a,c), (c,b), (b,d), (d,a)]
            ]
            for edges in cycles:
                colors = [edge_color.get((min(u,v), max(u,v))) for u,v in edges]
                if len(set(colors)) < 4:
                    # This cycle is violating. Remove one of its edges.
                    for u,v in edges:
                        key = (min(u,v), max(u,v))
                        if key in edge_color:
                            del edge_color[key]
                            found = True
                            break
                    if found:
                        break
            if found:
                break
        
        if not found:
            break
    
    # Convert to result
    result = []
    for (u, v), c in sorted(edge_color.items()):
        result.append([u, v, c])
    
    # Verify the result is valid
    if len(result) > 0:
        # Rebuild edge_color from result for verification
        verify_edge_color = {}
        for u, v, c in result:
            verify_edge_color[(min(u,v), max(u,v))] = c
        
        if is_valid_4cycle_coloring(verify_edge_color, n):
            return result
    
    # Fallback: return a star (always valid, no 4-cycles)
    return [[0, v, 0] for v in range(1, 20)]
# EVOLVE-BLOCK-END
