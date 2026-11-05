import time
import sys
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s margin for safety
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # Build the conflict graph
    # Edge (u, v) exists if (u * v) % (u + v) == 0
    # We want to find the largest subset with NO such edges (Independent Set)
    
    # Precompute conflicts for each vertex
    # conflicts[i] is a set of vertices j such that (i * j) % (i + j) == 0
    conflicts = [set() for _ in range(N + 1)]
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                conflicts[i].add(j)
                conflicts[j].add(i)
                
    # Convert to lists for faster iteration if needed, but sets are good for membership
    # For backtracking, we need to quickly check if a candidate vertex conflicts with current solution
    # and to calculate upper bounds.
    
    # Heuristic ordering: Sort vertices by degree descending? 
    # Or just process 1..N? 
    # A good heuristic for MIS is to order vertices by degree (high degree first) 
    # or by some other structural property. 
    # Let's stick to a simple order but use a strong bound.
    
    # Upper bound heuristic: 
    # A simple bound is the number of remaining vertices.
    # A tighter bound: If we color the remaining graph with k colors, 
    # the independent set size is at most k.
    # Greedy coloring is fast.
    
    def greedy_color_bound(vertices_subset):
        """
        Returns an upper bound on the size of an independent set in the subgraph induced by vertices_subset.
        Uses greedy coloring. The chromatic number is a lower bound on clique cover, 
        but actually, for Independent Set, the upper bound is the number of colors needed to color the graph? 
        No. 
        Wait. If a graph can be colored with k colors, any independent set can contain at most one vertex from each color class? 
        No, an independent set can contain multiple vertices from the same color class? 
        No, vertices of the same color are NOT adjacent. So they CAN be in an independent set.
        So coloring gives a LOWER bound on Independent Set? No.
        
        Let's rethink.
        Clique Cover: If the graph can be covered by k cliques, then any independent set can have at most 1 vertex from each clique. So k is an UPPER bound.
        Greedy coloring of the COMPLEMENT graph gives a clique cover of the original graph.
        
        Alternatively, a simpler upper bound:
        Sum of (1 / (1 + degree_in_remaining)) is a bound? No.
        
        Let's use a simple recursive bound:
        If we pick a vertex v, we can either include it (remove v and its neighbors) or exclude it.
        Upper bound = max(1 + bound(G - N[v]), bound(G - v)).
        This is the exact algorithm. To make it fast, we need a cheap upper bound for pruning.
        
        Common cheap upper bound: 
        For each vertex v in remaining, calculate 1 + sum(1 for u in remaining if u != v and u not in N(v) and u not in N(u)?) 
        No, that's complex.
        
        Let's use the "Greedy Coloring of the Complement" for upper bound.
        Complement graph edges: (u,v) where (u*v) % (u+v) != 0.
        If we can color the complement graph with k colors, then the original graph can be covered by k cliques (each color class in complement is a clique in original).
        Thus, Independent Set size <= k.
        
        Implementing greedy coloring on the fly might be expensive.
        
        Alternative: 
        Since N=256 is small, we can use a bitset representation for the graph.
        """
        pass

    # Let's use a simpler, robust approach:
    # Branch and Bound with a simple upper bound: 
    # bound = len(remaining_vertices)
    # This is weak.
    
    # Better bound: 
    # For the remaining graph, calculate the maximum independent set size using a greedy heuristic 
    # (e.g., pick vertex with min degree, add to set, remove neighbors, repeat).
    # This gives a LOWER bound on the solution, not an upper bound.
    # We need an UPPER bound to prune.
    
    # Upper bound via coloring the complement:
    # Let's implement a fast greedy coloring for the complement graph.
    # Vertices: 1..N
    # Adjacency in complement: NOT (u in conflicts[v]) and u != v
    
    # To make it faster, we can precompute the complement adjacency matrix as bitsets.
    # Python ints can act as bitsets.
    
    comp_adj = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if i != j and j not in conflicts[i]:
                mask |= (1 << j)
        comp_adj[i] = mask
        
    def get_color_bound(remaining_mask):
        """
        Greedy coloring of the subgraph induced by remaining_mask in the COMPLEMENT graph.
        Returns the number of colors used. This is an upper bound on the Independent Set size of the original graph.
        """
        if remaining_mask == 0:
            return 0
            
        colors = 0
        uncolored = remaining_mask
        while uncolored:
            colors += 1
            # Pick a vertex from uncolored
            # Heuristic: pick vertex with highest degree in uncolored complement subgraph?
            # Or just lowest index for speed?
            v_idx = uncolored.bit_length() - 1 # Get highest set bit
            v = v_idx + 1 # If bit 0 is index 1? 
            # Let's map bit i to vertex i+1? 
            # Bit 0 -> vertex 1? 
            # Let's stick to bit i representing vertex i. 
            # But vertices are 1..256. 
            # Bit 1 -> vertex 1? 
            # Let's use bit (v-1) for vertex v.
            
            # Re-evaluate bit mapping.
            # Let's restart bitset logic carefully.
            # Vertex v (1..256) maps to bit (v-1).
            
            # Re-build comp_adj with correct bit mapping
            pass
        return colors

    # Let's simplify. N=256 is small enough that a well-optimized recursive search with 
    # "include/exclude" and a decent upper bound might work.
    # But 2^256 is too big.
    
    # Strategy: 
    # 1. Start with a good initial solution (e.g., all odds, or greedy max independent set).
    # 2. Use local search (k-opt swaps) to improve?
    # 3. Or use Branch and Bound with strong pruning.
    
    # Let's try a hybrid: 
    # 1. Generate a large candidate set using a greedy heuristic (pick vertex with min conflicts, add, remove neighbors).
    # 2. Try to improve via random restarts and local search.
    
    def greedy_mis(vertices):
        """Greedy heuristic to find a large independent set."""
        remaining = set(vertices)
        solution = []
        # Order by degree in the conflict graph (lower degree first is often better for MIS)
        # Or higher degree? 
        # In MIS, picking a vertex with few neighbors leaves more options.
        # So sort by degree ascending.
        
        # Calculate degrees
        degs = {v: len(conflicts[v]) for v in vertices}
        
        while remaining:
            # Pick vertex with min degree in remaining subgraph
            # To be fast, just pick min from current list
            # This is O(N^2) per step, total O(N^3) which is 16M ops, acceptable.
            min_v = None
            min_deg = float('inf')
            for v in remaining:
                # Count neighbors in remaining
                deg = 0
                for u in conflicts[v]:
                    if u in remaining:
                        deg += 1
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
            
            solution.append(min_v)
            remaining.remove(min_v)
            # Remove neighbors of min_v from remaining
            for u in conflicts[min_v]:
                if u in remaining:
                    remaining.remove(u)
                    
        return solution

    # Initial solution
    best_sol = greedy_mis(vertices)
    best_size = len(best_sol)
    
    # Local Search: 
    # Try to swap a vertex in the solution with a vertex outside to increase size? 
    # Or just try to add a vertex? 
    # Since it's an independent set, we can't just add. We must swap.
    # If we remove one vertex v from S, we can add up to 1 vertex u not in S 
    # such that u doesn't conflict with S \ {v}.
    # If we can add more than 1? No, we removed 1.
    # So size stays same or decreases unless we find a way to add 2? 
    # No, local search for MIS usually involves 1-for-1 swaps to maintain independence, 
    # or 2-for-3 etc.
    
    # Given the time limit, a simple random restart greedy might be best.
    # Randomize the order of tie-breaking in the greedy step.
    
    best_sol_set = set(best_sol)
    
    for _ in range(1000): # Number of restarts
        if time.time() > deadline:
            break
            
        # Randomized Greedy
        remaining = set(vertices)
        sol = []
        while remaining:
            # Pick a vertex with min degree in remaining, with some randomness
            # Or just random vertex? 
            # Min degree is a strong heuristic.
            min_v = None
            min_deg = float('inf')
            candidates = []
            for v in remaining:
                deg = 0
                for u in conflicts[v]:
                    if u in remaining:
                        deg += 1
                if deg < min_deg:
                    min_deg = deg
                    candidates = [v]
                elif deg == min_deg:
                    candidates.append(v)
            
            # Random choice among min-degree candidates
            min_v = random.choice(candidates)
            sol.append(min_v)
            remaining.remove(min_v)
            for u in conflicts[min_v]:
                if u in remaining:
                    remaining.remove(u)
            
        if len(sol) > best_size:
            best_size = len(sol)
            best_sol = sol
            best_sol_set = set(sol)
            
    return best_sol
# EVOLVE-BLOCK-END
