import time
import random

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # 1. Generate a 1-factorization of K_n.
    # A 1-factorization of K_n (n even) partitions the edges into n/1 perfect matchings (1-factors).
    # If we assign a unique color to each 1-factor, then any cycle of length 2k must use at least k distinct colors.
    # Specifically, a 4-cycle uses 4 edges. In a 1-factorization, no two edges in the same 1-factor share a vertex.
    # Thus, a 4-cycle cannot contain two edges from the same 1-factor (since they would be disjoint, 
    # but edges in a cycle are connected). 
    # Wait, let's verify: 
    # If a 4-cycle has edges e1, e2, e3, e4.
    # If e1 and e3 are in the same 1-factor, they are disjoint. In a 4-cycle u-v-w-x-u, e1=(u,v) and e3=(w,x) are disjoint. 
    # So a 4-cycle CAN have opposite edges in the same 1-factor.
    # The colors would be c1, c2, c1, c4. This is NOT rainbow.
    # So simple 1-factorization coloring does NOT guarantee rainbow 4-cycles.
    
    # However, the task is to MAXIMIZE edges. 
    # The current program uses SA to find a valid subset of edges and colors.
    # A better structural approach: 
    # Use the 1-factorization to define the graph structure (all edges) but we must color them such that 4-cycles are rainbow.
    # Actually, if we take the complete graph K_20, it has 190 edges. 
    # Is it possible to color K_20 with 20 colors such that every 4-cycle is rainbow?
    # This is equivalent to finding a proper edge coloring of the line graph of K_20 with respect to 4-cycles?
    # No, it's a specific constraint.
    
    # Let's rethink the "Different Strategy".
    # The current program is SA. 
    # Inspiration 3 mentioned MIS on conflict graph.
    # Inspiration 1 was Greedy.
    
    # Let's try a **Greedy with Conflict Graph Pruning** approach, but optimized.
    # Or better: **Backtracking with Forward Checking** but using a very specific ordering.
    
    # Actually, there is a known construction for "anti-Ramsey" or rainbow cycles.
    # For n=20, the maximum number of edges might be less than 190.
    
    # Let's implement a **Randomized Greedy with Backtracking (Limited Depth)** 
    # but with a crucial improvement: **Color Symmetry Breaking** and **Edge Ordering by Degree**.
    
    # Strategy:
    # 1. Sort edges by some heuristic (e.g., sum of endpoint degrees in a random spanning tree, or just random).
    # 2. Try to add edges one by one.
    # 3. For each edge, try colors in a random order.
    # 4. If no color works, skip the edge (or backtrack if we have budget).
    # 5. To improve density, we can use a "repair" phase or multiple restarts.
    
    # However, the current SA score is 3.26. 
    # A pure greedy usually scores ~2.2.
    # The jump to 3.26 in the current program comes from the SA allowing removals to unblock additions.
    
    # Alternative Strategy: **Simulated Annealing on the COLOR ASSIGNMENT ONLY** 
    # for a FIXED DENSE SUBGRAPH.
    # 1. Select a dense subgraph (e.g., all edges, or a random subset of 100-150 edges).
    # 2. The constraint is only on colors.
    # 3. Minimize the number of "bad" 4-cycles (non-rainbow).
    # 4. If we can find a coloring with 0 bad 4-cycles for a large subgraph, that's great.
    
    # Let's try this: 
    # 1. Start with all 190 edges.
    # 2. Assign random colors.
    # 3. Count violations (4-cycles with < 4 distinct colors).
    # 4. Use SA to change colors to reduce violations.
    # 5. If violations > 0, remove edges involved in violations (or the ones causing most conflicts).
    # 6. Repeat.
    
    # This "Color-First, then Edge-Removal" strategy is structurally different from "Edge-Addition SA".
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_index = {e: i for i, e in enumerate(all_edges)}
    
    # Precompute all 4-cycles for fast violation checking?
    # Number of 4-cycles in K_20 is C(20,4) * 3 = 4845 * 3 = 14535.
    # This is small enough to precompute and check quickly.
    
    cycles = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    # 3 ways to form a 4-cycle from 4 vertices: (i,j,k,l), (i,j,l,k), (i,k,j,l)
                    # Vertices: a,b,c,d
                    # Cycles: a-b-c-d-a, a-b-d-c-a, a-c-b-d-a
                    v1, v2, v3, v4 = i, j, k, l
                    c1 = [(v1, v2), (v2, v3), (v3, v4), (v4, v1)]
                    c2 = [(v1, v2), (v2, v4), (v4, v3), (v3, v1)]
                    c3 = [(v1, v3), (v3, v2), (v2, v4), (v4, v1)]
                    
                    # Normalize edges to (min, max)
                    def norm_edge(e):
                        return (min(e[0], e[1]), max(e[0], e[1]))
                    
                    cycles.append([norm_edge(e) for e in c1])
                    cycles.append([norm_edge(e) for e in c2])
                    cycles.append([norm_edge(e) for e in c3])
    
    num_cycles = len(cycles)
    
    def count_violations(active_edges, colors):
        """
        active_edges: set of edge tuples
        colors: dict edge -> color
        Returns number of non-rainbow 4-cycles.
        """
        violations = 0
        for cycle in cycles:
            # Check if all edges in this cycle are active
            if all(e in active_edges for e in cycle):
                cols = [colors[e] for e in cycle]
                if len(set(cols)) < 4:
                    violations += 1
        return violations

    def solve():
        best_edges = 0
        best_result = []
        
        # Multiple restarts
        for restart in range(10):
            if time.time() > deadline - 10:
                break
            
            # 1. Start with all edges active
            active_edges = set(all_edges)
            # 2. Random initial coloring
            colors = {e: random.randint(0, num_colors - 1) for e in all_edges}
            
            # 3. SA Loop
            T = 5.0
            cooling = 0.9995
            min_T = 0.01
            
            current_violations = count_violations(active_edges, colors)
            best_violations = current_violations
            
            # We want to minimize violations. 
            # If violations == 0, we have a valid coloring for the current active_edges.
            # Then we can try to add more edges? No, we started with all.
            # So if we reach 0 violations, we have a valid K_20? Unlikely.
            # So we must remove edges.
            
            # Objective: Maximize |active_edges| subject to violations == 0.
            # This is a hard constraint. 
            # SA usually minimizes a cost. 
            # Cost = violations * LARGE_PENALTY - |active_edges|
            
            LARGE_PENALTY = 1000
            
            def cost(active_edges, colors):
                v = count_violations(active_edges, colors)
                return v * LARGE_PENALTY - len(active_edges)
            
            current_cost = cost(active_edges, colors)
            best_cost = current_cost
            best_active = set(active_edges)
            best_colors = dict(colors)
            
            while T > min_T and time.time() < deadline - 5:
                # Move: Change color of a random edge, or toggle an edge.
                move_type = random.random()
                
                if move_type < 0.7:
                    # Change color
                    if not active_edges:
                        continue
                    e = random.choice(list(active_edges))
                    old_c = colors[e]
                    new_c = random.randint(0, num_colors - 1)
                    if new_c == old_c:
                        continue
                    
                    colors[e] = new_c
                    new_cost = cost(active_edges, colors)
                    
                    delta = new_cost - current_cost
                    if delta < 0 or random.random() < pow(2.718, -delta / T):
                        current_cost = new_cost
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_active = set(active_edges)
                            best_colors = dict(colors)
                    else:
                        colors[e] = old_c
                        
                else:
                    # Toggle edge
                    # If active, remove. If inactive, add.
                    # Bias towards adding if we have few edges? 
                    # Let's pick a random edge from all_edges.
                    e = random.choice(all_edges)
                    
                    if e in active_edges:
                        # Remove
                        active_edges.remove(e)
                        # Color doesn't matter if inactive, but keep it for potential re-add
                    else:
                        # Add
                        active_edges.add(e)
                        # Assign random color if not present? It is present in 'colors' dict for all edges.
                        
                    new_cost = cost(active_edges, colors)
                    delta = new_cost - current_cost
                    if delta < 0 or random.random() < pow(2.718, -delta / T):
                        current_cost = new_cost
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_active = set(active_edges)
                            best_colors = dict(colors)
                    else:
                        # Revert
                        if e in active_edges:
                            active_edges.remove(e)
                        else:
                            active_edges.add(e)
                            
                T *= cooling
            
            # Update global best
            if len(best_active) > best_edges:
                best_edges = len(best_active)
                # Construct result
                res = []
                for e in best_active:
                    res.append([e[0], e[1], best_colors[e]])
                best_result = res
                
        return best_result

    result = solve()
    return result

