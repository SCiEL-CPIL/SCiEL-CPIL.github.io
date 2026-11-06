import time
import math
from math import gcd
from fractions import Fraction
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    # Candidates 2..256
    N = 256
    all_candidates = list(range(2, N + 1))
    
    # Weights are 1/x. We want to maximize sum(1/x).
    # To use integer arithmetic for B&B efficiency, we can scale weights.
    # However, the sum is rational. 
    # Tomita's algorithm typically works with integers. 
    # Let's use a large common denominator or just compare Fractions carefully.
    # Actually, standard Tomita coloring uses integer weights.
    # We can scale weights by LCM(1..256) which is huge.
    # Alternative: Use floating point for B&B bounds and Fraction for final verification?
    # Risk: FP errors might lead to suboptimal pruning or invalid solutions.
    # Better: Use exact integer weights by scaling.
    # LCM(1..256) is approx 10^112. This is too big for efficient int ops in Python? 
    # Python ints are arbitrary precision, but operations on 100+ digit numbers are slower.
    # Let's check if we can use a smaller scale.
    # The objective is sum(1/x). 
    # Let's try to use Fraction for the "current sum" but for the BOUND, 
    # we need an upper bound on the sum of weights of a maximum independent set in the subgraph.
    
    # Strategy: 
    # 1. Build conflict graph.
    # 2. Use a recursive B&B.
    # 3. At each step, we have a set of candidate vertices `P` (potential to add) and current set `S`.
    # 4. Upper bound: sum of weights of all vertices in `P` is a loose bound.
    #    Tomita's coloring bound: Color the subgraph induced by `P`. 
    #    The max weight of an independent set in `P` is <= sum of the max weight of each color class.
    #    Since weights are positive, this is a valid upper bound.
    
    # To make this efficient, we should work with indices.
    # Vertices: 0..254 representing numbers 2..256.
    
    nums = all_candidates
    n = len(nums)
    
    # Precompute conflicts
    # conflict[i] is a bitmask or set of indices j such that lcm(nums[i], nums[j]) <= 256
    # Using bitmasks for fast intersection.
    # n = 255. Bitmask fits in Python int.
    
    conflict = [0] * n
    for i in range(n):
        a = nums[i]
        for j in range(i + 1, n):
            b = nums[j]
            if (a * b) // gcd(a, b) <= 256:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    
    # Weights
    weights = [Fraction(1, x) for x in nums]
    
    # For B&B, we need to efficiently compute the coloring bound.
    # Coloring a graph of size up to 255 at every node of the B&B tree is expensive if done naively.
    # However, the graph is sparse.
    # Also, we can order the vertices. 
    # Standard Tomita orders vertices by degree or weight.
    # Here, weights are 1/x. Smaller x has higher weight.
    # So we should process smaller x (higher weight) first? 
    # In B&B for MWIS, we usually branch on a vertex.
    # Let's sort candidates by weight descending (i.e., number ascending).
    # Index 0 is 2, Index 1 is 3, etc. This is already sorted by weight desc.
    
    # State:
    # P: set of vertices available to add (as a bitmask)
    # S: current selected set (as a bitmask)
    # current_sum: Fraction
    
    # Upper Bound Function:
    # Given P (bitmask), return an upper bound on the max weight independent set in P.
    # We use greedy coloring on the subgraph induced by P.
    # The vertices in P are processed in a fixed order (e.g., index order).
    # Color classes: A set of vertices where no two are adjacent.
    # In a proper coloring, any independent set can take at most 1 vertex from each color class?
    # No, an independent set can take at most 1 vertex from each color class ONLY IF the coloring is such that 
    # wait. 
    # Definition: A coloring partitions V into independent sets (color classes).
    # An independent set in G can contain at most 1 vertex from each color class? 
    # NO. An independent set can contain multiple vertices from the same color class? 
    # Wait. If a color class is an independent set, then an independent set of G can contain ALL vertices of a color class.
    # So the bound "sum of max weight in each color class" is NOT an upper bound on the independent set weight?
    # Let's re-verify Tomita's bound.
    # Tomita's bound for Max Independent Set (unweighted):
    # Color the graph. The number of colors is an upper bound on the size of the Max Independent Set.
    # Why? Because any independent set can contain at most 1 vertex from each color class?
    # YES. If two vertices in an independent set have the same color, they must not be adjacent.
    # But the definition of a valid coloring is that ADJACENT vertices have DIFFERENT colors.
    # Non-adjacent vertices CAN have the same color.
    # So, an independent set can contain multiple vertices of the same color.
    # Therefore, the number of colors is NOT an upper bound on the size of an independent set.
    # 
    # CORRECTION:
    # The bound is derived differently.
    # In the context of the B&B algorithm for MWIS (like in Tomita's 1997 paper "An Exact Algorithm for the Maximum Independent Set Problem"),
    # the coloring is used to bound the search.
    # The algorithm maintains a list of vertices.
    # The bound is: If we have colored the subgraph P with k colors, the maximum size of an independent set in P is at most k?
    # Let's check a triangle (3-cycle). Chromatic number 3. Max independent set size 1. 1 <= 3. OK.
    # Let's check a path of 4 vertices (1-2-3-4). Chromatic number 2. Max independent set size 2 (1,3 or 2,4). 2 <= 2. OK.
    # Let's check a star K1,3. Chromatic number 2. Max independent set size 3 (the leaves). 3 > 2.
    # So the number of colors is NOT an upper bound on the size of the independent set.
    #
    # What is the actual Tomita bound?
    # Tomita's algorithm uses a "color bound" in the context of ordering vertices.
    # Specifically, it orders vertices by color. 
    # The bound is: Let the vertices in P be ordered $v_1, \dots, v_k$.
    # Let $c_i$ be the color of $v_i$ in a greedy coloring.
    # The bound is the maximum $c_i$ such that the prefix $v_1 \dots v_i$ has been explored?
    # Actually, the standard "Color Bound" for MIS size is:
    # The maximum size of an independent set in P is at most the number of colors used in a valid coloring of P?
    # My counter-example (Star) disproves this.
    # 
    # Let's look up "Tomita coloring bound maximum independent set".
    # Ah, the bound is used differently. 
    # In the B&B tree, we branch on a vertex $v$.
    # We split into: include $v$ (remove $v$ and its neighbors from P) or exclude $v$ (remove $v$ from P).
    # The "Color Bound" is often cited as: 
    # "The maximum cardinality of an independent set in a graph is bounded above by the number of colors in a proper coloring."
    # This statement is FALSE as shown by the star graph.
    # 
    # HOWEVER, many sources cite this. Why?
    # Perhaps they assume the graph is perfect? No.
    # Perhaps I am misinterpreting "coloring".
    # 
    # Let's re-read carefully.
    # "The size of a maximum independent set is at most the chromatic number." -> False.
    # "The size of a maximum clique is at most the chromatic number." -> True.
    # 
    # Wait, the problem is MAX INDEPENDENT SET.
    # Max Independent Set in G is Max Clique in G_bar (complement graph).
    # Chromatic number of G_bar is the size of the minimum clique cover of G.
    # This doesn't help directly with a simple bound.
    #
    # Let's look at the specific algorithm "MCS" (Maximum Cardinality Search) or Tomita's.
    # Tomita's algorithm for MIS:
    # It uses a greedy coloring of the *complement* graph?
    # Or does it use the fact that:
    # If we color the graph G, an independent set in G can have at most 1 vertex of each color?
    # No, that's for Clique. A clique in G must have distinct colors in a proper coloring of G.
    # So Chromatic Number(G) >= Clique Number(G).
    # 
    # So, if we want to bound the Independent Set of G, we can bound the Clique of G_bar.
    # So we should color G_bar (the compatibility graph, not conflict graph).
    # If we properly color G_bar, any Clique in G_bar (which is an Independent Set in G) must have distinct colors.
    # So the size of the Max Independent Set in G is <= Chromatic Number(G_bar).
    # 
    # YES. This is the correct logic.
    # We need to color the COMPATIBILITY graph (edges if lcm > 256).
    # The conflict graph is the complement.
    # 
    # So, for a set of candidates P, we build the subgraph of COMPATIBILITY edges within P.
    # We greedily color this compatibility graph.
    # The number of colors used is an upper bound on the size of the Max Independent Set in the CONFLICT graph (which corresponds to Max Clique in Compatibility graph? No.)
    # 
    # Let's trace carefully:
    # We want Max Weight Independent Set in Conflict Graph $G_c$.
    # This is equivalent to Max Weight Clique in Compatibility Graph $G_p$ (where edges exist if compatible).
    # 
    # Theorem: The size of a Maximum Clique in a graph $H$ is at most the Chromatic Number of $H$.
    # Proof: A clique consists of vertices all pairwise adjacent. In a proper coloring, adjacent vertices must have different colors. Thus, all vertices in a clique have distinct colors. The number of vertices in the clique cannot exceed the total number of colors used.
    # 
    # So, to bound the weight of the Max Independent Set in $G_c$:
    # 1. Consider the subgraph of $G_p$ induced by P.
    # 2. Greedily color this subgraph.
    # 3. Let the colors be $1, \dots, k$.
    # 4. The Max Weight Clique in this subgraph (which is our target) can have at most 1 vertex from each color class.
    # 5. Therefore, the Max Weight is at most the sum of the MAXIMUM WEIGHT in each color class.
    # 
    # This is a valid upper bound for the WEIGHTED case.
    
    def color_bound(P_mask, weights):
        """
        Returns an upper bound on the max weight independent set in P_mask.
        Uses greedy coloring on the COMPATIBILITY subgraph induced by P_mask.
        """
        if P_mask == 0:
            return Fraction(0)
        
        # Extract vertices in P_mask
        # We process them in a specific order. 
        # To make the coloring bound tight, we should order vertices by weight descending?
        # Or degree? 
        # Standard greedy coloring: process vertices in some order, assign smallest available color.
        # The bound is sum(max_weight[color] for color in colors_used).
        
        # Let's get the list of vertices in P_mask.
        # To speed up, we can precompute adjacency bitmasks for COMPATIBILITY.
        # compat[i] is bitmask of j such that compatible(i,j).
        
        # We need to color the subgraph.
        # Vertices are indices 0..n-1.
        # We'll maintain a list of colors for each vertex in P.
        
        # For efficiency, we only care about the max weight per color.
        # 
        # Greedy Coloring Algorithm:
        # 1. Order vertices. Let's use index order (which is weight descending).
        # 2. For each vertex v in order:
        #    Determine which colors are used by its neighbors in the COMPATIBILITY graph 
        #    that have already been colored.
        #    Assign the smallest available color.
        #    Update max_weight_for_color[color] = max(max_weight_for_color[color], weight[v]).
        
        # To get neighbors in P that are already colored:
        # We need to know the compatibility edges.
        
        # Let's precompute compat masks.
        # This function is called many times, so it must be fast.
        
        # Optimization: 
        # The bound is only useful if it prunes. 
        # If the bound is loose, we waste time coloring.
        
        # Implementation:
        # Get list of active vertices.
        active_vertices = []
        temp_mask = P_mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            idx = lsb.bit_length() - 1
            active_vertices.append(idx)
            temp_mask ^= lsb
        
        # Order: We want to color in a way that produces few colors or tight weight bounds.
        # Coloring by weight descending often works well for weight bounds.
        # active_vertices is already in index order (0..n-1), which is weight descending.
        
        max_weight_per_color = {}
        color_of_vertex = {}
        
        # For each vertex, we need to check neighbors in active_vertices that are already colored.
        # To make this O(1) or O(deg), we need fast neighbor lookup.
        
        # We can iterate through active_vertices.
        # For v in active_vertices:
        #   neighbors_in_P = compat[v] & P_mask
        #   colored_neighbors = [u for u in active_vertices if u < v and (neighbors_in_P >> u) & 1]
        #   This is slow.
        
        # Better: 
        # Maintain a set of used colors for each color? No.
        # For each vertex v, we want to know the set of colors used by its colored neighbors.
        
        # Let's use a different approach for the bound.
        # Since n is small (255), we can just do a simple greedy coloring.
        
        # We need the COMPATIBILITY masks.
        # Let's pass them or make them global.
        
        # To avoid global state in this nested function, we'll assume `compat` is available.
        
        # Greedy coloring:
        # colors_used = set()
        # vertex_colors = {}
        # max_w = {}
        
        # For v in active_vertices:
        #    neighbor_colors = set()
        #    for u in active_vertices:
        #        if u == v: continue
        #        if u not in vertex_colors: continue # Not colored yet
        #        if (compat[v] >> u) & 1: # u is neighbor of v in COMPAT graph
        #            neighbor_colors.add(vertex_colors[u])
        #            
        #    # Find smallest color not in neighbor_colors
        #    c = 0
        #    while c in neighbor_colors:
        #        c += 1
        #        
        #    vertex_colors[v] = c
        #    if c not in max_w or weights[v] > max_w[c]:
        #        max_w[c] = weights[v]
        
        # Bound = sum(max_w.values())
        
        # Complexity: O(N^2) per call. N=255. 65k ops.
        # B&B nodes: could be many.
        
        # Let's implement this.
        
        vertex_colors = {}
        max_w = {}
        
        # To speed up neighbor check, we can precompute a list of neighbors for each v?
        # Or just use the bitmask.
        
        # We process in order of active_vertices.
        # For v in active_vertices:
        #   We need to know colors of neighbors u where u is in active_vertices and u has been processed (i.e., u comes before v in active_vertices list? No, order is arbitrary for coloring, but we process sequentially).
        #   Actually, we just need to check all u in active_vertices that are neighbors of v and have been assigned a color.
        
        # Since we process sequentially, "colored" means "already processed in this loop".
        
        for v in active_vertices:
            # Find colors of colored neighbors
            # Neighbors in Compat Graph: compat[v]
            # Intersect with P_mask to get neighbors in P
            # Intersect with "processed" mask? 
            # We don't have a "processed" mask easily.
            # But we can just check `if u in vertex_colors`.
            
            # Optimization: 
            # Instead of iterating all active_vertices, iterate bits of (compat[v] & P_mask).
            # For each bit u in that mask:
            #   if u in vertex_colors:
            #       add vertex_colors[u] to forbidden
            
            forbidden = set()
            neigh_mask = compat[v] & P_mask
            while neigh_mask:
                lsb = neigh_mask & -neigh_mask
                u = lsb.bit_length() - 1
                if u in vertex_colors:
                    forbidden.add(vertex_colors[u])
                neigh_mask ^= lsb
            
            c = 0
            while c in forbidden:
                c += 1
            
            vertex_colors[v] = c
            if c not in max_w or weights[v] > max_w[c]:
                max_w[c] = weights[v]
        
        return sum(max_w.values())

    # Precompute COMPATIBILITY masks
    compat = [0] * n
    for i in range(n):
        a = nums[i]
        for j in range(i + 1, n):
            b = nums[j]
            if (a * b) // gcd(a, b) > 256:
                compat[i] |= (1 << j)
                compat[j] |= (1 << i)
    
    # B&B Solver
    best_sum = Fraction(0)
    best_set_mask = 0
    
    # To allow global access to compat and weights in color_bound, 
    # we define color_bound inside construct but it needs access to `compat` and `weights`.
    # The definition above uses `compat` and `weights` from the closure.
    
    def bnb(P_mask, S_mask, current_sum):
        nonlocal best_sum, best_set_mask
        
        if time.time() > deadline:
            return
            
        if P_mask == 0:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set_mask = S_mask
            return
        
        # Upper Bound
        # We can also use a simpler bound: sum of weights in P_mask.
        # But coloring bound is tighter.
        # Let's check if time is tight.
        # If n is small, coloring is fast.
        
        # Optimization: If the number of vertices is small, maybe just brute force?
        # No, coloring is fine.
        
        ub = color_bound(P_mask, weights)
        
        # Prune if current_sum + ub <= best_sum
        if current_sum + ub <= best_sum:
            return
        
        # Select a vertex to branch on.
        # Heuristic: Pick the vertex with highest weight? 
        # Or the one with highest degree in conflict graph?
        # Let's pick the first vertex in P_mask (lowest index, highest weight).
        
        # Find lowest set bit
        lsb = P_mask & -P_mask
        v = lsb.bit_length() - 1
        
        # Branch 1: Include v
        # New P = P_mask without v and without its neighbors in CONFLICT graph.
        # Neighbors in Conflict = ~compat[v] & P_mask & ~lsb?
        # Conflict neighbors are those NOT in compat[v] and NOT v.
        # So, remove v and all u in P such that NOT compatible(v,u).
        # Incompatibility mask for v within P: P_mask & ~(compat[v] | lsb)
        
        new_P_include = P_mask & ~(compat[v] | lsb)
        new_S_include = S_mask | lsb
        new_sum_include = current_sum + weights[v]
        
        bnb(new_P_include, new_S_include, new_sum_include)
        
        # Branch 2: Exclude v
        # New P = P_mask without v
        new_P_exclude = P_mask & ~lsb
        new_S_exclude = S_mask
        new_sum_exclude = current_sum
        
        bnb(new_P_exclude, new_S_exclude, new_sum_exclude)

    # Initial P_mask is all vertices
    init_P_mask = (1 << n) - 1
    bnb(init_P_mask, 0, Fraction(0))
    
    # Convert best_set_mask to list
    result = []
    mask = best_set_mask
    while mask:
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        result.append(nums[idx])
        mask ^= lsb
    
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Result: {res}")
    print(f"Sum: {sum(1/x for x in res)}")
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            if (res[i]*res[j])//gcd(res[i], res[j]) <= 256:
                valid = False
                print(f"Invalid pair: {res[i]}, {res[j]}")
    print(f"Valid: {valid}")
