import time
import sys

def construct():
    """
    Constructs a C4-free subgraph of Q7 maximizing edge count.
    Strategy:
    1. Model the problem as Maximum Independent Set (MIS) on the conflict graph G_conflict.
       - Nodes of G_conflict: The 448 edges of Q7.
       - Edges of G_conflict: Two hypercube edges are connected if they share a 2-face (i.e., they are part of the same square).
       - An independent set in G_conflict corresponds to a set of hypercube edges with no 2-face containing all 4 edges.
       Wait, the constraint is "No 2-dimensional face may have all four edges selected".
       This is NOT an independent set constraint on pairs. It's a hypergraph constraint (clique of size 4 forbidden).
       
       Let's re-read the directive carefully.
       "Model the problem as a maximum independent set in the conflict graph where vertices are hypercube edges and edges connect pairs that share a 2-face"
       
       If I select a set of edges S such that no two edges in S share a face, that is MUCH stronger than required.
       The required condition is: No face has ALL 4 edges in S.
       It allows 3 edges in a face. It allows 2.
       
       So the "Conflict Graph" approach described in the directive is actually solving a DIFFERENT (harder) problem:
       Maximum set of edges such that no two edges share a face.
       This is equivalent to finding a maximum matching in the dual graph? Or just a packing of edges.
       
       Let's check the score of such a solution.
       If no two edges share a face, then for any face, at most 1 edge is selected?
       No, "share a face" means they are both in the same square.
       If e1 and e2 are in the same square, they are connected in G_conflict.
       An independent set in G_conflict means NO two selected edges are in the same square.
       So each square has at most 1 selected edge.
       Total squares = 672.
       Each edge is in 6 squares.
       Sum of edges in squares = 6 * |S|.
       Also Sum of edges in squares <= 672 * 1 = 672.
       So 6 * |S| <= 672 => |S| <= 112.
       
       The current best is ~304. A solution of size 112 would be terrible (score ~0.36).
       
       The directive says: "Model the problem as a maximum independent set in the conflict graph... which reframes the search... This approach exploits the graph's..."
       
       Perhaps the directive implies a different conflict graph?
       Or perhaps I should interpret "edges connect pairs that share a 2-face" loosely?
       
       Let's look at the "Idea target": "Does operator 'error_repair' change the chance that a child improves its island...?"
       This suggests I should maybe stick to a method that is robust and potentially better, but the directive is specific.
       
       However, if I follow the directive literally, I get a score of ~112/304 = 0.36, which is a massive drop.
       "A first try in a new direction may score worse; that is acceptable."
       
       BUT, is there a way to interpret the conflict graph such that it solves the ORIGINAL problem?
       The original problem is a Maximum 3-SAT-like problem or Maximum Independent Set in a 4-uniform hypergraph.
       
       Maybe the directive meant: "Conflict graph where vertices are hypercube edges and edges connect pairs that CANNOT be selected together"?
       Two edges cannot be selected together ONLY IF selecting them forces a C4? No, that's not a pairwise constraint.
       The constraint is global per face.
       
       If I MUST follow the directive to get a "structural change", and the literal interpretation yields a bad score, I should probably implement the literal interpretation but perhaps optimize it well?
       
       Wait, let's look at Inspiration 3: ILP.
       Let's look at Inspiration 1: Simulated Annealing.
       
       The directive is: "Model the problem as a maximum independent set in the conflict graph where vertices are hypercube edges and edges connect pairs that share a 2-face".
       
       If I implement this, I am solving: Maximize |S| such that for all faces f, |S intersect f| <= 1.
       This is a valid C4-free set (since |S intersect f| <= 1 implies not all 4 are present).
       So it IS a valid solution to the original problem, just not optimal.
       
       Is it possible the directive implies a DIFFERENT graph?
       "edges connect pairs that share a 2-face" -> This is the standard definition of the line graph of the hypercube's dual? No.
       
       Let's consider if there is a misunderstanding of the problem statement in the directive.
       "No 2-dimensional face may have all four edges selected."
       
       If I use the MIS on the "share a face" graph, I get <= 112 edges.
       
       Alternative interpretation:
       Maybe the "conflict graph" is built differently?
       What if the vertices are the FACES?
       No, "vertices are hypercube edges".
       
       Okay, I will implement the MIS on the "share a face" graph as requested. It is a structurally different approach (Global Optimization via Branch and Bound on a specific graph) vs Local Search. Even if the score is lower, it satisfies the "Structural Change" and "Different Design Family" requirements.
       
       To make it as good as possible within this constraint:
       1. Build the conflict graph G_conflict (448 nodes).
          Node e is connected to node f if e and f are in a common square.
          How many neighbors does each node have?
          Edge e is in 6 squares.
          Each square has 4 edges.
          So e shares a square with 3 other edges in each of the 6 squares.
          However, some neighbors might be shared between squares?
          In Q7, can two edges share MORE than one face?
          Two edges (u,v) and (x,y).
          If they share a face, they are coplanar.
          Can they share two faces?
          If they are disjoint? No, they must be in the same square.
          In a square, there are 4 edges.
          e1, e2, e3, e4.
          e1 shares a face with e2, e3, e4.
          e2 shares a face with e1, e3, e4.
          e3 shares a face with e1, e2, e4.
          e4 shares a face with e1, e2, e3.
          
          If edge e is in square S1 and S2.
          Neighbors in S1: n1, n2, n3.
          Neighbors in S2: m1, m2, m3.
          Can n1 == m1?
          If edge f shares a face with e in S1 and in S2, then e and f are in two different squares.
          This implies e and f are part of a 3-dimensional cube?
          In Q7, two edges can share at most 1 face?
          Let's check.
          Edge e: (0, 1) (diff bit 0).
          Edge f: (2, 3) (diff bit 0).
          Face 1: 0,1,3,2 (bits 0,1). e and f are in this face.
          Are they in another face?
          Face 2: 0,1,5,4 (bits 0,2). e is in it. f is not (2,3 differ in bit 0, but 2=10, 3=11. 4=100, 5=101. 2,3 are not in {0,1,4,5}).
          
          Actually, two edges in Q_n share a face if and only if they are parallel (same dimension) and their vertices are "close" (differ in one other bit), OR they are perpendicular and share a vertex?
          
          Case 1: Parallel edges (u, u^i) and (v, v^i).
          They share a face if v = u^j.
          Then the face is u, u^i, u^j, u^i^j.
          The edges are (u, u^i), (u^j, u^i^j), (u, u^j), (u^i, u^i^j).
          So e and f are opposite edges in the square.
          
          Case 2: Perpendicular edges (u, u^i) and (u, u^j).
          They share vertex u.
          They share a face defined by bits i, j.
          The face is u, u^i, u^j, u^i^j.
          
          So, for a given edge e=(u, u^i):
          1. It shares a face with (u, u^j) for all j != i. (6 edges, sharing vertex u)
          2. It shares a face with (u^i, u^i^j) for all j != i. (6 edges, sharing vertex u^i)
          3. It shares a face with (u^j, u^i^j) for all j != i. (6 edges, parallel, opposite in face)
          
          Total neighbors = 6 + 6 + 6 = 18?
          Are there duplicates?
          (u, u^j) is distinct from (u^i, u^i^j)? Yes, different vertices.
          (u, u^j) distinct from (u^j, u^i^j)? Yes.
          So degree is 18.
          
          Graph has 448 nodes, degree 18.
          We want Maximum Independent Set.
          
       2. Use Branch and Bound for MIS.
          - Recursion: Pick a node v.
          - Branch 1: Include v. Remove v and all its neighbors from the graph. Recurse.
          - Branch 2: Exclude v. Remove v from the graph. Recurse.
          - Bound: Use a greedy coloring of the remaining graph to estimate the max independent set size.
            The upper bound is the number of colors needed (chromatic number upper bound).
            Or simpler: Greedy set packing.
            
          Given the time limit (120s) and the small size (448 nodes), a well-implemented BB might find the optimal solution for THIS sub-problem (max set with no two edges sharing a face).
          
          Wait, is the optimal solution for "no two edges share a face" actually small?
          We estimated 112.
          Let's try to construct a large one.
          Color the vertices of Q7 with 2 colors (bipartite).
          Edges always go between color 0 and 1.
          If we select only edges from color 0 to color 1? That's all edges.
          
          The constraint is "no two edges share a face".
          This is equivalent to: The selected edges form a "linear hypergraph" in the dual sense?
          
          Actually, there is a known result for this.
          But let's just run the BB.
          
          To ensure we finish in time, I will add a timeout guard.
          
          However, a score of ~112 is very low.
          Is there ANY way the directive implies a different graph?
          "edges connect pairs that share a 2-face"
          
          If I interpret "share a 2-face" as "cannot be in the same independent set", then the independent set is the solution.
          
          I will proceed with this literal interpretation. It is a valid C4-free set.
          
          Optimization:
          Use Python's standard library. No networkx.
          Implement a fast MIS solver.
          
          Steps:
          1. Build adjacency list for G_conflict.
          2. Run Branch and Bound.
             - State: Set of available nodes (bitmask or list).
             - Since 448 nodes is too large for a simple bitmask in Python integers (actually 448 bits fits in a Python int, which is arbitrary precision), we can use bitmasks!
             - Python ints are perfect for sets up to thousands of bits.
             - `avail` is an integer where bit i is 1 if node i is available.
             - `neighbors[i]` is an integer bitmask of neighbors of i.
             
             Algorithm:
             def solve(avail, current_set):
                 if avail == 0: return current_set
                 # Bound
                 # Greedy coloring bound:
                 # We can approximate the bound by greedy set packing.
                 # Or just use a simple bound: popcount(avail) is a weak bound.
                 # Better: Greedy coloring.
                 # While avail != 0:
                 #   pick v
                 #   remove v and neighbors
                 #   count++
                 # This count is an upper bound on the size of the independent set?
                 # No, greedy coloring gives an upper bound on the clique cover, which is a lower bound on the chromatic number?
                 # For MIS, a common bound is based on coloring:
                 # If we can color the graph with k colors, the max independent set is at most... no, that's for max clique.
                 # For MIS, the complement graph is a Clique problem.
                 # Max Independent Set in G = Max Clique in G_complement.
                 # G_complement has 448 nodes.
                 # Degree in G is 18.
                 # Degree in G_comp is 448 - 1 - 18 = 429.
                 # Dense graph.
                 # Max Clique in dense graph is hard.
                 
                 # Let's stick to MIS on G.
                 # Bound:
                 # A simple bound is the greedy maximal independent set size.
                 # Greedy MIS:
                 #   remaining = avail
                 #   size = 0
                 #   while remaining:
                 #      pick v with min degree in remaining? or just first?
                 #      size += 1
                 #      remaining &= ~neighbors[v]
                 #      remaining &= ~(1<<v)
                 #   return size
                 
                 # This bound is computable fast with bitmasks.
                 
                 # Branching:
                 # Pick a vertex v.
                 # Heuristic: Pick v with highest degree in `avail`? Or lowest?
                 # Usually, for MIS, picking a high-degree vertex is good for pruning because it removes many neighbors.
                 
                 # Branch 1: Include v.
                 #   new_avail = avail & ~(neighbors[v] | (1<<v))
                 #   solve(new_avail, current_set | {v})
                 
                 # Branch 2: Exclude v.
                 #   new_avail = avail & ~(1<<v)
                 #   solve(new_avail, current_set)
                 
                 # To avoid exponential blowup, we need good bounds and ordering.
                 
          3. Since the graph is regular-ish and small, this might work.
          
          Let's implement this.

    """
    start_time = time.time()
    time_limit = 110.0  # Safety margin

    d = 7
    n = 2 ** d  # 128 vertices
    E = 448

    # 1. Precompute edges
    edge_list = []
    edge_to_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edge_list)
                edge_list.append((x, y))
                edge_to_idx[(x, y)] = idx

    # 2. Build Conflict Graph
    # Node i is connected to Node j if edge i and edge j share a face.
    # We determined that for each edge, there are 18 such neighbors.
    
    # To find neighbors efficiently:
    # For each edge e = (u, v) where v = u ^ (1 << i):
    # Neighbors are:
    # 1. (u, u^j) for j != i
    # 2. (v, v^j) for j != i
    # 3. (u^j, v^j) for j != i  (Note: v^j = (u^i)^j = u^i^j)
    
    adj_mask = [0] * E
    
    for idx, (u, v) in enumerate(edge_list):
        # Determine dimension i
        diff = u ^ v
        i = diff.bit_length() - 1
        
        mask = 0
        for j in range(d):
            if j == i:
                continue
            
            # Type 1: (u, u^j)
            u_j = u ^ (1 << j)
            e1 = (min(u, u_j), max(u, u_j))
            mask |= 1 << edge_to_idx[e1]
            
            # Type 2: (v, v^j)
            v_j = v ^ (1 << j)
            e2 = (min(v, v_j), max(v, v_j))
            mask |= 1 << edge_to_idx[e2]
            
            # Type 3: (u^j, v^j)
            # u^j and v^j = u^i^j
            # The edge is between u^j and u^i^j. They differ in bit i.
            a = u ^ (1 << j)
            b = v ^ (1 << j)
            e3 = (min(a, b), max(a, b))
            mask |= 1 << edge_to_idx[e3]
            
        adj_mask[idx] = mask

    # 3. Solve MIS using Branch and Bound with Bitmasks
    
    best_size = 0
    best_set_mask = 0
    
    # Precompute popcount for small masks? No, Python int.bit_count() is fast.
    
    # Greedy Bound Function
    def greedy_bound(avail):
        """
        Returns an upper bound on the size of the maximum independent set in the subgraph induced by 'avail'.
        Uses a greedy coloring-like approach (actually greedy set packing).
        """
        remaining = avail
        count = 0
        while remaining:
            # Pick a vertex. Heuristic: pick the one with smallest index? Or highest degree?
            # For a bound, any valid independent set found is a LOWER bound on the max.
            # We need an UPPER bound.
            # The size of a maximal independent set is NOT an upper bound.
            # Wait.
            # If I find an independent set of size K, the max is at least K.
            # I need an upper bound to prune.
            # A common upper bound for MIS is based on the clique cover of the complement?
            # Or simply: The sum of (1 / (degree + 1)) ? No.
            
            # Let's use a simple but effective bound:
            # The size of the maximum independent set is at most the number of vertices in the largest clique in the complement? No.
            
            # Actually, for regular graphs, a simple bound is just the number of vertices.
            # To prune effectively, we can use a greedy coloring of the GRAPH ITSELF?
            # No, coloring the graph gives a lower bound on chromatic number, which doesn't directly bound MIS.
            
            # However, we can use the following:
            # If we partition the vertices into cliques, the MIS can pick at most 1 from each clique.
            # So the number of cliques in a clique cover is an upper bound on MIS.
            # How to get a clique cover?
            # Greedily pick a clique, remove it, repeat.
            # In G, a clique is a set of edges that all share a face?
            # No, in G_conflict, an edge exists if they share a face.
            # So a clique in G_conflict is a set of hypercube edges where EVERY PAIR shares a face.
            # In Q7, what is the max clique size?
            # A set of edges all in the same square?
            # In a square, all 4 edges share a face with each other?
            # e1 shares with e2, e3, e4.
            # e2 shares with e1, e3, e4.
            # Yes, the 4 edges of a square form a K4 in G_conflict.
            # Can we have a K5?
            # 5 edges, every pair shares a face.
            # This would require them to be in a common structure.
            # Likely max clique is 4.
            
            # So a greedy clique cover will use cliques of size up to 4.
            # This gives a decent upper bound.
            
            cover_count = 0
            rem = remaining
            while rem:
                # Find a clique in rem
                # Pick a vertex v
                v_idx = (rem & -rem).bit_length() - 1
                v_mask = 1 << v_idx
                rem &= ~v_mask
                
                # Find neighbors of v in rem
                # Neighbors of v in G_conflict are adj_mask[v_idx]
                # We want to find a clique containing v.
                # A clique containing v is a subset of neighbors of v that are also mutually connected.
                # For a simple bound, we can just take v and its neighbors that are connected to each other?
                # Or just take v and as many neighbors as possible that form a clique?
                
                # Simple heuristic:
                # Start with clique = {v}
                # Add neighbors of v that are connected to all current clique members.
                
                clique_nodes = [v_idx]
                clique_mask = v_mask
                
                # Candidates: neighbors of v in rem
                candidates = adj_mask[v_idx] & rem
                
                # Iterate through candidates and add if connected to all in clique
                # To keep it fast, we can just greedily add.
                # Since max clique is small (4), this is fast.
                
                # Actually, for an UPPER bound, we want to pack LARGE cliques.
                # So we want to find a MAXIMAL clique in the remaining graph.
                
                # Let's just pick v and its neighbors that are mutually connected.
                # Since the graph is sparse (deg 18), we can check.
                
                # Simplified: Just take v and one neighbor?
                # No, let's take v and try to extend.
                
                # For speed, let's just use a very simple bound:
                # Upper bound = popcount(remaining)
                # This is too weak.
                
                # Let's use the "Coloring Bound" on the COMPLEMENT graph?
                # MIS in G = Clique in G_comp.
                # G_comp is dense.
                # Max Clique in G_comp.
                # Upper bound for Max Clique: Greedy coloring of G_comp.
                # If G_comp can be colored with k colors, Max Clique <= k.
                
                # Let's implement a fast greedy coloring of the subgraph induced by `remaining` in G_comp?
                # G_comp edge (i,j) exists if NOT (i==j or adj_mask[i] has bit j).
                # This is expensive to check on the fly.
                
                # Alternative:
                # Since we know the structure, maybe we can just use a loose bound and rely on the search?
                # With 448 nodes, if the bound is weak, it will explode.
                
                # Let's try a different bound:
                # The size of the maximum independent set is at most the minimum degree + 1? No.
                
                # Let's use a randomized greedy MIS to get a LOWER bound, and use that to prune?
                # No, we need an UPPER bound to prune branches where `current_size + upper_bound <= best_size`.
                
                # Let's use the number of vertices as the bound, but improve it by removing vertices that are "dominated"?
                
                # Actually, let's just use a simple recursive solver with a timeout.
                # If it times out, we return the best found so far.
                
                # To make it work, we need a decent bound.
                # Let's use the following:
                # Upper bound = size of a maximal independent set found greedily?
                # NO, that's a lower bound.
                
                # Wait, if I use `popcount(remaining)` as the bound, it's always >= actual.
                # It's just very loose.
                
                # Let's try to find a better bound using the structure.
                # Each edge is in 6 faces.
                # If we select an edge, we "block" 18 other edges.
                # So each selected edge "consumes" roughly 19 edges from the pool.
                # So |S| <= 448 / 19 ~ 23?
                # This is a very tight and valid upper bound!
                # Let's verify.
                # If I select edge e, I cannot select any of its 18 neighbors.
                # So the set of selected edges S is an independent set.
                # The closed neighborhood of any edge in S is disjoint?
                # Not necessarily disjoint, but they overlap.
                # However, the average degree is 18.
                # By Turan's theorem or similar, the size of an independent set in a d-regular graph is at most n / (d+1) * something?
                # Actually, for a d-regular graph, alpha(G) <= n / (d+1) is NOT always true.
                # But for our specific graph, let's check.
                
                # If the bound is ~23, and we found a solution of size 20, we can prune a lot.
                
                # Let's calculate the bound as: popcount(remaining) is too high.
                # Let's use: `popcount(remaining) // 2`? No.
                
                # Let's use a simple greedy coloring of G (not G_comp) to get a lower bound on chromatic number?
                # No.
                
                # Let's just use `popcount(remaining)` and hope the branching is efficient enough with good heuristics.
                # Or, better:
                # Bound = 0
                # rem = remaining
                # while rem:
                #   pick v
                #   Bound += 1
                #   rem &= ~adj_mask[v]
                #   rem &= ~(1<<v)
                # This Bound is the size of a MAXIMAL independent set.
                # This is a LOWER bound on the Max Independent Set.
                # We need an UPPER bound.
                
                # There is a theorem: alpha(G) <= n / (delta + 1) * (1 + delta/ (2 * ... ))?
                
                # Let's just use the trivial bound `popcount(remaining)` and add a time check.
                # If the search space is too big, we return the best found.
                
                # To improve pruning, we can sort vertices by degree.
                
                pass # Placeholder for bound logic
            
            return count # This function is currently broken/placeholder

    # Revised Plan:
    # Since the "share a face" graph is sparse and regular, and the target size is small (~20-30),
    # a simple DFS with pruning based on `current_size + popcount(remaining) <= best_size` might be sufficient
    # IF the branching factor is controlled.
    
    # However, 448 nodes is large for naive DFS.
    
    # Let's use a heuristic:
    # 1. Find a good initial solution using Greedy MIS.
    # 2. Use Branch and Bound.
    
    def greedy_mis(avail):
        """Returns a set (mask) of a greedy maximal independent set."""
        remaining = avail
        selected = 0
        while remaining:
            # Pick vertex with highest degree in remaining?
            # Or just first one.
            v_idx = (remaining & -remaining).bit_length() - 1
            v_mask = 1 << v_idx
            selected |= v_mask
            # Remove v and its neighbors
            remaining &= ~adj_mask[v_idx]
            remaining &= ~v_mask
        return selected

    # Initial Solution
    initial_set = greedy_mis((1 << E) - 1)
    best_size = initial_set.bit_count()
    best_set_mask = initial_set
    
    # Branch and Bound
    def solve(avail, current_mask, current_size):
        nonlocal best_size, best_set_mask
        nonlocal start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return
        
        if avail == 0:
            if current_size > best_size:
                best_size = current_size
                best_set_mask = current_mask
            return
        
        # Bound: current_size + popcount(avail) <= best_size
        if current_size + avail.bit_count() <= best_size:
            return
        
        # Heuristic: Pick a vertex to branch on.
        # Pick the vertex with the highest degree in the subgraph induced by `avail`.
        # This tends to prune more.
        
        # To find max degree quickly:
        # Iterate through bits in avail.
        # This is slow in Python.
        
        # Alternative: Just pick the first vertex.
        # Or, pick a vertex that is in the most "conflicts".
        
        # Let's pick the vertex with the smallest index for simplicity, 
        # but rely on the bound.
        
        # Actually, let's try to pick a vertex with high degree.
        # We can approximate degree by counting bits in (adj_mask[v] & avail).
        
        # To avoid iterating all 448 bits every time, we can just pick the first one.
        # Given the time limit, let's try to be smart.
        
        # Pick v = lowest set bit in avail.
        v_idx = (avail & -avail).bit_length() - 1
        v_mask = 1 << v_idx
        
        # Branch 1: Include v
        # New avail: remove v and its neighbors
        new_avail_1 = avail & ~adj_mask[v_idx] & ~v_mask
        solve(new_avail_1, current_mask | v_mask, current_size + 1)
        
        # Branch 2: Exclude v
        new_avail_2 = avail & ~v_mask
        solve(new_avail_2, current_mask, current_size)

    # Start B&B
    # To make it faster, we can order the vertices.
    # But the graph is symmetric.
    
    # Let's run it.
    solve((1 << E) - 1, 0, 0)
    
    # Extract result
    result = []
    for i in range(E):
        if (best_set_mask >> i) & 1:
            u, v = edge_list[i]
            result.append([u, v])
            
    return result

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Edges: {len(res)}, Time: {end-start:.2f}s")
