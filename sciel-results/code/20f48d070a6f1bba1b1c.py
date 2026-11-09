# EVOLVE-BLOCK-START
import numpy as np
from functools import lru_cache


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a maximum square-difference-free set modulo 205 using a bitset-based
    branch-and-bound Maximum Independent Set solver (Tomita's algorithm).

    The forbidden differences are the nonzero quadratic residues mod 205.
    We model the problem as finding a maximum independent set in the Cayley
    graph Cay(Z/205Z, F) where F is the set of forbidden differences.
    """
    m = 205
    k = 2

    # Compute forbidden differences: nonzero squares mod 205
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)

    # Build adjacency bitsets using Python's arbitrary-precision integers as bitsets
    # adj[i] is a bitset where bit j is set if i and j are adjacent (forbidden difference)
    adj = [0] * m
    for i in range(m):
        for f in forbidden:
            j1 = (i + f) % m
            j2 = (i - f) % m
            adj[i] |= (1 << j1)
            adj[i] |= (1 << j2)
    # Remove self-loops
    for i in range(m):
        adj[i] &= ~(1 << i)

    # Precompute full mask
    full_mask = (1 << m) - 1

    # Tomita's Algorithm for Maximum Independent Set
    # We maintain a list of vertices `P` sorted by color.
    # The color of a vertex is an upper bound on the size of the MIS in the subgraph induced by vertices with color <= its color.
    
    # To speed up, we can precompute neighbors as lists? No, bitsets are faster for intersection.
    
    best_size = 0
    best_set = 0  # bitset representation of the best independent set found

    def expand(R, P):
        """
        R: current independent set (bitset)
        P: candidate vertices (list of vertices, sorted by color descending)
        """
        nonlocal best_size, best_set
        
        if not P:
            if bin(R).count('1') > best_size:
                best_size = bin(R).count('1')
                best_set = R
            return

        # Greedy coloring of the subgraph induced by P to get upper bound and ordering
        # We need to color the graph G[P].
        # Since we want to maximize, we process vertices in an order that allows tight bounds.
        # Tomita's algorithm typically colors the graph such that color(v) is the color assigned.
        # The upper bound for the current node is |R| + max_color.
        
        # Efficient greedy coloring for bitset-based MIS:
        # We partition P into color classes.
        # However, a simpler and very effective heuristic for MIS is to use the "degree" or "color" ordering.
        # Let's implement a standard greedy coloring on the candidate set P to determine the order and bound.
        
        # To avoid expensive re-coloring at every node, we can use a static ordering or a simple dynamic one.
        # But Tomita's method re-colors at each node. For N=205, this is feasible if optimized.
        
        # Let's try a simpler, robust approach:
        # 1. Compute a greedy coloring of G[P] to get an upper bound UB.
        # 2. Order vertices in P by decreasing color.
        # 3. Branch on the first vertex in the ordered list.
        
        # Implementation of greedy coloring on bitset P:
        # We want to find an upper bound on the independence number of G[P].
        # This is equivalent to the clique number of the complement, but we work with G.
        # Actually, for MIS in G, we can use a greedy coloring of G[P] itself?
        # No, the standard bound for MIS uses a coloring of the *candidate subgraph* such that vertices of the same color are *independent*?
        # Wait. If we color G[P] such that adjacent vertices have different colors, the number of colors is an upper bound on the Clique Number of G[P].
        # We want MIS of G.
        # Let H be the complement of G. MIS in G is Max Clique in H.
        # If we compute a greedy coloring of H[P], the number of colors is an upper bound on the Clique Number of H[P], which is the MIS size of G[P].
        # So we need to color the complement graph.
        
        # Alternatively, we can just use a simple heuristic bound:
        # UB = |P| is too loose.
        # A common bound for MIS is based on the maximum degree or similar.
        
        # Let's stick to the bitset branch and bound with a better vertex selection heuristic, 
        # which is often sufficient for N=205 if the graph is structured (Cayley graph).
        # The previous failure was likely due to the O(N^3) or worse coloring bound.
        
        # New Strategy:
        # Use a recursive search where we pick a vertex v from P.
        # Branch 1: Include v. New P = P \ (N(v) U {v}).
        # Branch 2: Exclude v. New P = P \ {v}.
        # To prune, we need a good upper bound.
        # A simple but effective bound for MIS is the "greedy coloring of the complement" bound.
        # Let's implement a fast greedy coloring of the complement graph induced by P.
        
        # Helper: Greedy coloring of complement graph induced by P
        # Input: bitset P
        # Output: max color used (upper bound on MIS size in P)
        
        def greedy_color_complement(P_bits):
            if P_bits == 0:
                return 0
            remaining = P_bits
            color = 0
            # We assign colors 1, 2, ...
            # A color class in the complement is an independent set in the complement,
            # which is a clique in the original graph G.
            # So we are finding a sequence of cliques in G that cover P.
            # The number of cliques is an upper bound on the independence number?
            # No. The number of colors in a proper coloring of the complement is an upper bound on the Clique Number of the complement,
            # which is the Independence Number of G.
            # So yes, we need to color the complement.
            
            # Greedy coloring:
            # Iterate through vertices. Assign the smallest possible color.
            # To make it fast, we iterate through vertices in a fixed order (e.g., index order) or heuristic order.
            # For N=205, even O(N^2) is fine.
            
            # However, doing this at EVERY node is expensive.
            # Let's try a different bound: 
            # The size of the maximum independent set in P is at most the size of the maximum independent set in any subset of P.
            # A simple bound: sum of 1/deg(v) + 1? No.
            
            # Let's use the "Tomita" ordering without full recoloring if it's too slow, 
            # or implement the coloring carefully.
            
            # Given N=205, a full greedy coloring of the complement at each node might be acceptable if optimized in C, 
            # but in Python, it might be slow.
            # Let's try a simpler bound first: 
            # UB = current_size + len(P)
            # This is weak.
            
            # Let's use the "Degree" heuristic for vertex selection and a simple bound.
            # Actually, for Cayley graphs, the structure is regular.
            # The independence number is roughly N / (d+1) where d is degree.
            # Here d = |F| * 2 (approx).
            
            # Let's implement the standard recursive MIS with a simple bound:
            # UB = current_size + popcount(P)
            # And use a good vertex ordering: Highest Degree in P first.
            
            # This is what the previous code did, but maybe the bound was the bottleneck?
            # No, the previous code had `greedy_color_bound` which was the bottleneck.
            # If we remove the expensive bound and just use `popcount(P)`, the search space is huge.
            
            # Compromise: Use a cached or static ordering? No, P changes.
            # Let's implement a fast, approximate bound:
            # Greedily find a maximal independent set in P. Let its size be L.
            # This gives a LOWER bound, not upper.
            
            # Let's try to implement the coloring bound efficiently.
            # We will iterate through vertices in P.
            # We maintain a list of color classes (bitsets).
            # For each vertex v in P, try to add it to the first color class where it is not adjacent to any vertex in that class (in the complement, i.e., NOT adjacent in G? No).
            # In the complement graph H, v is adjacent to u if v is NOT adjacent to u in G.
            # So v can be added to color class C if v is NOT adjacent to ANY vertex in C in H.
            # This means v IS adjacent to ALL vertices in C in G.
            # So v can be added to color class C if (adj[v] & C) == C.
            
            # This is exactly the clique cover of G.
            
            def get_bound(P_bits):
                if P_bits == 0:
                    return 0
                remaining = P_bits
                color_classes = []
                # We will process vertices in a specific order to maximize pruning.
                # Usually, vertices with high degree are processed first?
                # For coloring, the order matters.
                # Let's just iterate through bits.
                
                # To speed up, we can break if the number of colors already exceeds the best known?
                # We don't have the global best in this local scope easily without nonlocal.
                
                # Let's just compute the bound.
                # Optimization: If the number of colors reaches a value that doesn't improve the global bound, we can stop?
                # We need to pass best_size in.
                
                # Let's refine the main search loop to use this bound.
                
                return 0 # Placeholder

        # Let's rewrite the search using the "Tomita" style but with a simpler, faster bound calculation if possible.
        # Or, we can just use the fact that N=205 is small enough for a well-implemented branch and bound with a simple bound like `popcount(P)` IF the vertex ordering is very good.
        
        # Let's try the "Highest Degree" heuristic for vertex selection, which is often very effective for MIS.
        
        # 1. Find vertex v in P with highest degree in G[P].
        # 2. Branch Include v: P' = P \ (N(v) U {v})
        # 3. Branch Exclude v: P' = P \ {v}
        
        # To prune, we need an upper bound.
        # A very fast upper bound is the size of the maximum independent set of P, which we can approximate by a greedy heuristic? No, that's a lower bound.
        
        # Let's use the "Greedy Coloring" bound but optimize it.
        # We will implement `greedy_color_bound` to be as fast as possible.
        
        pass # This function is too complex to patch inline easily without restructuring.

    # Let's restart the solver implementation with a cleaner structure.
    
    best_size = 0
    best_set = 0
    
    # Precompute degrees for initial ordering?
    # We will use a recursive function.
    
    def solve(P_bits, R_bits):
        nonlocal best_size, best_set
        
        # Upper bound: current size + max possible from P
        # If we can't beat best_size, return.
        # Simple bound: popcount(P_bits)
        # If current_size + popcount(P_bits) <= best_size: return
        
        # Better bound: Greedy Coloring of Complement
        # We will implement a fast version.
        
        # To avoid deep recursion and Python overhead, we can use an iterative stack or just hope 205 is small enough.
        # 205 is small.
        
        # Let's implement the greedy coloring bound here.
        
        def calc_bound(P_bits):
            if P_bits == 0:
                return 0
            
            # We want to cover P_bits with cliques in G.
            # Each clique is a color class in the complement.
            # The number of cliques is an upper bound on the independence number.
            
            remaining = P_bits
            num_colors = 0
            
            # Heuristic: process vertices with high degree first?
            # For coloring, it's often better to process vertices with high degree in the subgraph.
            # Let's just iterate through vertices.
            
            # Optimization: If we already have enough colors to exceed the potential, we can stop?
            # We don't know the potential here without R_bits.
            
            while remaining:
                # Find a maximal clique in G induced by remaining
                clique = 0
                temp = remaining
                # To make it fast, we pick a vertex, then intersect neighbors.
                
                # Pick a vertex from temp.
                # Let's pick the one with highest degree in 'temp'?
                # Or just the first one?
                # Picking the first one is fast.
                
                if temp == 0:
                    break
                    
                # Get lowest set bit
                lb = temp & (-temp)
                v = lb.bit_length() - 1
                
                # Start clique with v
                clique = lb
                # Intersect remaining candidates with neighbors of v
                temp = temp & adj[v]
                
                # Remove v from temp? No, v is in clique.
                # The loop `while temp` will process other vertices.
                # We need to remove v from the pool of vertices we can add?
                # Actually, the standard algorithm for maximal clique:
                # Candidates = {v} U N(v)
                # While Candidates:
                #   Pick u
                #   Add u to clique
                #   Candidates = Candidates \ (N(u) U {u})  ... wait, this is for finding ONE maximal clique.
                
                # Let's use a simpler greedy clique finding:
                # Start with empty clique.
                # Candidates = remaining
                # While Candidates:
                #   Pick v from Candidates (e.g., highest degree)
                #   Add v to clique
                #   Candidates = Candidates \ N(v) (remove neighbors of v, because they can't be in the same clique)
                #   Remove v from Candidates
                
                clique = 0
                cand = remaining
                while cand:
                    # Pick vertex with max degree in cand?
                    # For speed, just pick the first one.
                    lb = cand & (-cand)
                    v = lb.bit_length() - 1
                    
                    clique |= lb
                    # Remove v and its neighbors from cand
                    cand &= ~(lb | adj[v])
                    
                num_colors += 1
                # Remove the clique from remaining
                remaining &= ~clique
                
                # Early termination?
                # If num_colors + current_size <= best_size, we can return num_colors?
                # No, we need the actual bound to prune.
                # But if num_colors is already large, we might want to stop calculating if it's not useful?
                # Actually, we just need the final count.
                
            return num_colors

        # Calculate bound
        ub = bin(R_bits).count('1') + calc_bound(P_bits)
        
        if ub <= best_size:
            return
        
        if P_bits == 0:
            if bin(R_bits).count('1') > best_size:
                best_size = bin(R_bits).count('1')
                best_set = R_bits
            return
        
        # Select vertex to branch on
        # Heuristic: Highest degree in P_bits
        temp = P_bits
        best_v = -1
        best_deg = -1
        
        # To speed up, we can limit the number of vertices we check?
        # With 205, checking all is fine.
        
        while temp:
            lb = temp & (-temp)
            v = lb.bit_length() - 1
            deg = bin(adj[v] & P_bits).count('1')
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp ^= lb
            
        if best_v == -1:
            return
            
        # Branch 1: Include best_v
        # New P = P \ (N(best_v) U {best_v})
        new_P = P_bits & ~(1 << best_v) & ~adj[best_v]
        new_R = R_bits | (1 << best_v)
        solve(new_P, new_R)
        
        # Branch 2: Exclude best_v
        new_P = P_bits & ~(1 << best_v)
        solve(new_P, R_bits)

    # Initialize with greedy solution
    greedy_set = 0
    for v in range(m):
        if (adj[v] & greedy_set) == 0:
            greedy_set |= (1 << v)
    best_size = bin(greedy_set).count('1')
    best_set = greedy_set

    # Start search
    solve((1 << m) - 1, 0)

    # Extract the best set
    result = []
    for v in range(m):
        if (best_set >> v) & 1:
            result.append(v)

    return np.array(result, dtype=np.int64)


# EVOLVE-BLOCK-END
