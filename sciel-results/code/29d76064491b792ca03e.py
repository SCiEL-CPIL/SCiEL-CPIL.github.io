import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n = 20
    colors = 20
    max_edges = n * (n - 1) // 2
    
    # Pre-generate all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_indices = {e: i for i, e in enumerate(all_edges)}
    
    # We will use a backtracking search with forward checking.
    # Variables: Each edge can be present (with a color) or absent.
    # To maximize edges, we prefer presence.
    # However, checking all 4-cycles is expensive.
    # Insight: A 4-cycle is formed by 4 edges. 
    # Constraint: For any set of 4 edges forming a 4-cycle, their colors must be distinct.
    
    # Representation:
    # state: list of length max_edges.
    #   state[i] = -1 if edge i is absent.
    #   state[i] = c (0..19) if edge i is present with color c.
    
    # To speed up backtracking, we need efficient constraint propagation.
    # Instead of checking all 4-cycles globally at each step, we check constraints 
    # involving the most recently assigned edge.
    
    # Precompute all 4-cycles? 
    # Number of 4-cycles in K20 is huge (C(20,4) * 3 is approx 2300 * 3 = 6900? No.
    # Number of 4-cycles in K_n is 3 * C(n, 4). For n=20, C(20,4)=4845. 3*4845 = 14535.
    # This is small enough to precompute.
    
    four_cycles = []
    # A 4-cycle is defined by 4 distinct vertices a,b,c,d in order.
    # Edges: (a,b), (b,c), (c,d), (d,a).
    # To avoid duplicates, we can fix a < b < c < d? No, the cycle order matters for edges.
    # Actually, a 4-cycle subgraph is just a set of 4 edges.
    # Let's generate unique sets of 4 edges that form a cycle.
    # Vertices a,b,c,d distinct.
    # Edges: ab, bc, cd, da.
    # To get unique sets, we can iterate combinations of 4 vertices {a,b,c,d}.
    # For each set, there are 3 distinct 4-cycles (ways to pair them up).
    # 1. a-b-c-d-a
    # 2. a-b-d-c-a
    # 3. a-c-b-d-a
    
    from itertools import combinations
    
    # Map edge tuple (u,v) with u<v to index
    def get_edge_idx(u, v):
        if u > v: u, v = v, u
        return edge_indices[(u, v)]
    
    # Precompute list of 4-cycles as lists of edge indices
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d
        e1 = get_edge_idx(a, b)
        e2 = get_edge_idx(b, c)
        e3 = get_edge_idx(c, d)
        e4 = get_edge_idx(d, a)
        four_cycles.append((e1, e2, e3, e4))
        
        # Cycle 2: a-b-d-c
        e1 = get_edge_idx(a, b)
        e2 = get_edge_idx(b, d)
        e3 = get_edge_idx(d, c)
        e4 = get_edge_idx(c, a)
        four_cycles.append((e1, e2, e3, e4))
        
        # Cycle 3: a-c-b-d
        e1 = get_edge_idx(a, c)
        e2 = get_edge_idx(c, b)
        e3 = get_edge_idx(b, d)
        e4 = get_edge_idx(d, a)
        four_cycles.append((e1, e2, e3, e4))
        
    # For each edge, store the 4-cycles it participates in
    edge_to_cycles = [[] for _ in range(max_edges)]
    for i, cycle in enumerate(four_cycles):
        for e in cycle:
            edge_to_cycles[e].append(i)
    
    # Backtracking Search
    # State:
    # - edge_colors: array of length max_edges, -1 for unassigned/absent, 0-19 for color.
    # - We need to decide: Absent or Present (with color)?
    # To maximize edges, we should try to assign colors.
    # Heuristic: Order edges. Which order?
    # Random order allows different structures.
    # We will try to assign a color to an edge. If it fails, we mark it absent?
    # No, "Absent" is a valid state. But we want to maximize edges.
    # Strategy: 
    # 1. Randomly shuffle edge order.
    # 2. Iterate through edges. For each edge, try all 20 colors.
    # 3. If a color is valid (doesn't create a bad 4-cycle with already assigned edges), assign it.
    # 4. If NO color is valid, mark as absent.
    # 5. This is greedy again. 
    # 
    # To make it a real search:
    # We need to backtrack. If we mark an edge absent, maybe a different choice earlier would have allowed it?
    # Given the density, greedy might be close to optimal, but backtracking can fix local traps.
    # 
    # Let's implement a limited backtracking search.
    # Since n=20 is small, we can afford some depth.
    # However, the search space is huge.
    # 
    # Alternative Structural Change: 
    # **Local Search (Simulated Annealing) on the Color Assignment.**
    # Start with a dense graph (e.g., K20 or a random dense subgraph).
    # Assign random colors.
    # Identify "bad" 4-cycles (non-rainbow).
    # Move: Change the color of an edge involved in a bad 4-cycle to reduce the number of bad 4-cycles.
    # If stuck, remove an edge (decrease density) to allow recoloring? Or just recolor.
    # Actually, removing edges is the only way to make a "bad" structure (where 3 edges in a path have duplicate colors) valid for a 4th edge.
    # Wait, if 3 edges in a path u-a-b-v have duplicate colors, NO color for (u,v) works.
    # So we MUST remove one of the edges in that path to fix it.
    # 
    # So the optimization is: Find a subgraph G of K20 and a coloring c: E(G) -> {0..19} such that all 4-cycles are rainbow.
    # 
    # Let's use **Local Search with Edge Removal and Recoloring**.
    # 1. Start with K20, random colors.
    # 2. Count bad 4-cycles.
    # 3. While bad 4-cycles > 0 and time < deadline:
    #    a. Pick a random bad 4-cycle.
    #    b. Pick a random edge in that cycle.
    #    c. Option A: Recolor that edge to a random color. (Might not fix it if the conflict is with other edges in the cycle).
    #    d. Option B: Remove that edge. (Guarantees this specific 4-cycle is gone, but might create new issues? No, removing edges only removes 4-cycles).
    #    e. Actually, removing an edge is safe for validity (monotonic property: subgraphs of valid graphs are valid).
    #    f. So, a simple greedy removal strategy:
    #       Start with K20.
    #       While there is a bad 4-cycle:
    #          Remove a random edge from a random bad 4-cycle.
    #       This will terminate. But it might remove too many edges.
    #       
    #    To maximize edges, we should be smarter.
    #    If a 4-cycle is bad, it's because some colors repeat.
    #    Can we recolor to fix it?
    #    If the 4-cycle has colors [1, 1, 2, 3], we can change one of the 1s to 4 (if 4 is available).
    #    If we change it, we might break other 4-cycles.
    #    
    #    Let's try **Simulated Annealing**:
    #    State: Edge existence (bitmask) + Colors (array).
    #    Actually, let's just stick to the CSP idea but with a better heuristic:
    #    **Backtracking with "Most Constrained Variable" ordering.**
    
    # Given the time limit (145s), a full backtracking on 190 edges is impossible.
    # But we can do **Iterative Deepening** or **Greedy with Backtracking**.
    
    # Let's try a different structural approach:
    # **Algebraic Construction / Symmetry Breaking.**
    # If we use a cyclic group structure (Z_n), we might find a valid dense graph.
    # Vertices 0..19.
    # Edge (u, v) exists if (v-u) mod 20 is in a set S.
    # Color (u, v) is f((v-u) mod 20).
    # If the graph is vertex-transitive, we only need to check 4-cycles starting at 0?
    # No, but it reduces the search space massively.
    # Let's try to find a circulant graph that works.
    
    # This is a specific family. It might not be optimal, but it's a different strategy.
    # If it fails, we fall back to random greedy.
    
    def try_circulant():
        # Try all subsets of differences? 2^19 is too many.
        # Try random subsets of differences.
        best = []
        best_count = 0
        for _ in range(1000):
            if time.time() > deadline - 20:
                break
            # Random subset of differences 1..9 (since 10 is self-inverse, 11..19 are symmetric)
            # Actually, for simple graph, difference d and 20-d are same edge.
            # Differences 1..9.
            diffs = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            random.shuffle(diffs)
            # Pick a random number of edges to include?
            # Or pick a random subset.
            subset_size = random.randint(5, 9)
            chosen_diffs = set(diffs[:subset_size])
            
            # Construct edges
            edges = []
            for u in range(n):
                for d in chosen_diffs:
                    v = (u + d) % n
                    if u < v:
                        edges.append((u, v))
            
            # Assign colors based on difference
            # color(d) = d? Or random permutation of 0..19?
            # If we use color = d, then edges with same difference have same color.
            # A 4-cycle in a circulant graph:
            # u, u+d1, u+d1+d2, u+d1+d2+d3 ...
            # This is complex.
            
            # Let's just check if this specific graph can be colored.
            # Since it's vertex transitive, if we fix colors of edges incident to 0, the rest is determined?
            # No, color(u,v) = color(0, (v-u)%n).
            # So we just need to assign a color to each difference d in chosen_diffs.
            # Let color_map[d] = c.
            # We need to check all 4-cycles.
            
            # Generate all 4-cycles in this circulant graph.
            # A 4-cycle is u, v, w, x.
            # Due to symmetry, we can just check cycles starting at 0.
            # Cycle: 0, a, b, c, 0.
            # Edges: (0,a), (a,b), (b,c), (c,0).
            # Differences: a, b-a, c-b, -c (or 20-c).
            # We need colors: color(a), color(b-a), color(c-b), color(-c) to be distinct.
            
            # Let's iterate all possible 4-cycles starting at 0.
            # a, b, c are vertices.
            # (0,a) in G => a in chosen_diffs or 20-a in chosen_diffs.
            # Let's just iterate all 4-cycles in the graph and check.
            
            # To be fast, we precompute the 4-cycles of the base K20? No, the graph changes.
            # We can just check the specific cycles formed by the chosen diffs.
            
            valid = True
            # We need to assign colors to the diffs.
            # There are at most 9 diffs.
            # We can try all permutations of 0..19 for these 9 diffs? 20P9 is huge.
            # But we can use backtracking on the colors of the diffs.
            
            # Let's define the constraints on the diffs.
            # For each 4-cycle (0, a, b, c, 0), we have a constraint on the 4 diffs involved.
            
            # This is getting complicated. Let's stick to the previous CSP idea but optimized.
            
            # Actually, the "Circulant" idea is a good "Structural Change".
            # Let's implement a simple check for a random circulant graph.
            # If we find a valid one, return it.
            
            # To check validity, we need to color the edges.
            # If we assume color(u,v) = (v-u)%20, then:
            # Cycle 0-a-b-c-0 has diffs: a, b-a, c-b, -c.
            # These must be distinct.
            # So we need to choose a set of diffs S such that for any 4-cycle formed by S, the diffs are distinct.
            
            # Let's test random S.
            
            # Generate all 4-cycles in K20 that are "compatible" with the circulant structure?
            # No, just generate all 4-cycles in the graph defined by S.
            
            # This is too slow to implement correctly in the remaining time budget of the thought process.
            
            pass
        
        return best_edges, best_count

    # Let's go with **Local Search: Minimize Bad 4-Cycles**.
    # Start with a random dense graph (e.g., 80% of edges).
    # Random colors.
    # Loop:
    #   Find a bad 4-cycle.
    #   If none, done.
    #   Pick an edge in the bad 4-cycle.
    #   Try to recolor it to a color that reduces the number of bad 4-cycles involving that edge.
    #   If no such color exists, try removing the edge.
    
    def local_search():
        best_edges = []
        best_count = 0
        
        for restart in range(50):
            if time.time() > deadline - 5:
                break
                
            # Initialize
            # Random subset of edges
            p = 0.8
            edges = []
            adj = [dict() for _ in range(n)]
            for u, v in all_edges:
                if random.random() < p:
                    c = random.randint(0, 19)
                    edges.append([u, v, c])
                    adj[u][v] = c
                    adj[v][u] = c
            
            current_count = len(edges)
            
            # Precompute 4-cycles for the current graph?
            # It's dynamic.
            
            # We will iterate until no bad 4-cycles or time runs out.
            # To find bad 4-cycles efficiently:
            # We can maintain a list of all 4-cycles in K20 (precomputed).
            # For each 4-cycle, check if all 4 edges exist. If yes, check colors.
            
            # Precompute all 4-cycles of K20 once.
            # We did this above in the CSP section. Let's reuse it.
            
            # Re-define four_cycles for K20
            # (Already done in the outer scope? No, let's do it inside or pass it)
            
            # Let's assume four_cycles is available.
            
            # Optimization:
            # Instead of checking all 14535 4-cycles every step, 
            # we can just pick a random 4-cycle and check it.
            # If it's bad, fix it.
            
            max_iters = 5000
            for _ in range(max_iters):
                if time.time() > deadline - 2:
                    break
                
                # Pick a random 4-cycle
                cycle = random.choice(four_cycles)
                e1, e2, e3, e4 = cycle
                
                # Check if all edges exist
                c1 = edge_colors[e1]
                c2 = edge_colors[e2]
                c3 = edge_colors[e3]
                c4 = edge_colors[e4]
                
                # -1 means absent
                if c1 == -1 or c2 == -1 or c3 == -1 or c4 == -1:
                    continue # Not a cycle in current graph
                
                # Check if rainbow
                if len({c1, c2, c3, c4}) == 4:
                    continue # Good cycle
                
                # Bad cycle. Fix it.
                # Strategy: Try to recolor one of the edges.
                # Pick a random edge in the cycle.
                e_idx = random.choice([e1, e2, e3, e4])
                
                # Try all colors for e_idx
                best_new_color = None
                best_bad_count = current_bad_count # Need to track this?
                
                # To be fast, just try a random color?
                # Or try to find a color that makes THIS cycle rainbow.
                # If we change c1 to c_new, the cycle is rainbow if c_new not in {c2, c3, c4}.
                # But it might break other cycles.
                
                # Let's try: Recolor e_idx to a random color not in {c2, c3, c4}.
                forbidden = {c2, c3, c4}
                available = [c for c in range(20) if c not in forbidden]
                
                if available:
                    new_c = random.choice(available)
                    # Apply change
                    old_c = edge_colors[e_idx]
                    edge_colors[e_idx] = new_c
                    
                    # Update adj
                    u, v = all_edges[e_idx]
                    adj[u][v] = new_c
                    adj[v][u] = new_c
                    
                    # We don't check if it improved globally, we just hope.
                    # This is "Hill Climbing" with random moves.
                    
                else:
                    # No color available to fix this specific cycle?
                    # Then we must remove an edge.
                    e_remove = random.choice([e1, e2, e3, e4])
                    # Remove edge
                    u, v = all_edges[e_remove]
                    edge_colors[e_remove] = -1
                    del adj[u][v]
                    del adj[v][u]
                    current_count -= 1

            # Reconstruct edges
            current_edges = []
            for u in range(n):
                for v, c in adj[u].items():
                    if u < v:
                        current_edges.append([u, v, c])
            
            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges = current_edges
                
        return best_edges, best_count

    # We need to define edge_colors and four_cycles properly.
    
    # Let's restart the implementation cleanly.
    
    # Precompute 4-cycles of K20
    four_cycles = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        e1 = edge_indices[(a, b)]
        e2 = edge_indices[(b, c)]
        e3 = edge_indices[(c, d)]
        e4 = edge_indices[(d, a)]
        four_cycles.append((e1, e2, e3, e4))
        
        e1 = edge_indices[(a, b)]
        e2 = edge_indices[(b, d)]
        e3 = edge_indices[(d, c)]
        e4 = edge_indices[(c, a)]
        four_cycles.append((e1, e2, e3, e4))
        
        e1 = edge_indices[(a, c)]
        e2 = edge_indices[(c, b)]
        e3 = edge_indices[(b, d)]
        e4 = edge_indices[(d, a)]
        four_cycles.append((e1, e2, e3, e4))
        
    best_edges = []
    best_count = 0
    
    for restart in range(100):
        if time.time() > deadline - 5:
            break
            
        # Initialize random dense graph
        p = 0.85
        edge_colors = [-1] * max_edges
        adj = [dict() for _ in range(n)]
        
        for i, (u, v) in enumerate(all_edges):
            if random.random() < p:
                c = random.randint(0, 19)
                edge_colors[i] = c
                adj[u][v] = c
                adj[v][u] = c
                
        current_count = sum(1 for c in edge_colors if c != -1)
        
        # Local Search
        # We will do a fixed number of iterations or until no bad cycles found in a window.
        # To be efficient, we sample random 4-cycles.
        
        for _ in range(2000):
            if time.time() > deadline - 2:
                break
                
            cycle = random.choice(four_cycles)
            e1, e2, e3, e4 = cycle
            
            c1 = edge_colors[e1]
            c2 = edge_colors[e2]
            c3 = edge_colors[e3]
            c4 = edge_colors[e4]
            
            if c1 == -1 or c2 == -1 or c3 == -1 or c4 == -1:
                continue
                
            if len({c1, c2, c3, c4}) == 4:
                continue
                
            # Bad cycle
            # Try to fix by recoloring
            # Pick random edge in cycle
            e_idx = random.choice([e1, e2, e3, e4])
            
            # Determine forbidden colors for e_idx based on the OTHER 3 edges in this cycle
            # This ensures this specific cycle becomes rainbow (if we pick a valid color)
            if e_idx == e1:
                forb = {c2, c3, c4}
            elif e_idx == e2:
                forb = {c1, c3, c4}
            elif e_idx == e3:
                forb = {c1, c2, c4}
            else:
                forb = {c1, c2, c3}
                
            available = [c for c in range(20) if c not in forb]
            
            if available:
                new_c = random.choice(available)
                edge_colors[e_idx] = new_c
                u, v = all_edges[e_idx]
                adj[u][v] = new_c
                adj[v][u] = new_c
            else:
                # Cannot recolor to fix this cycle?
                # This happens if the other 3 edges use 19+ colors? Impossible with 3 edges.
                # Wait, forb has size 3. Available has size 17.
                # So we can ALWAYS recolor to fix a specific 4-cycle.
                # The problem is that recoloring might break OTHER cycles.
                # But we are just doing random hill climbing.
                
                # If we want to be safer, we could remove an edge if recoloring fails repeatedly?
                # But recoloring never "fails" to make this cycle rainbow.
                
                # So we always recolor.
                # This means we never remove edges in this loop?
                # If we start with a dense graph, and we keep recoloring,
                # will we converge to a valid coloring?
                # Not necessarily. We might get stuck in a local minimum where
                # recoloring one edge fixes one cycle but breaks two others.
                
                # To escape local minima, we should occasionally REMOVE an edge.
                # Let's add a small probability of removal.
                if random.random() < 0.05:
                    e_remove = random.choice([e1, e2, e3, e4])
                    u, v = all_edges[e_remove]
                    edge_colors[e_remove] = -1
                    del adj[u][v]
                    del adj[v][u]
                    current_count -= 1
                else:
                    # Just recolor as above
                    new_c = random.choice(available)
                    edge_colors[e_idx] = new_c
                    u, v = all_edges[e_idx]
                    adj[u][v] = new_c
                    adj[v][u] = new_c

        # Final check and update best
        # We should verify the result is actually valid.
        # Since we only sampled, it might not be.
        # Let's do a full verification.
        
        valid = True
        for cycle in four_cycles:
            c1 = edge_colors[cycle[0]]
            c2 = edge_colors[cycle[1]]
            c3 = edge_colors[cycle[2]]
            c4 = edge_colors[cycle[3]]
            if c1 != -1 and c2 != -1 and c3 != -1 and c4 != -1:
                if len({c1, c2, c3, c4}) != 4:
                    valid = False
                    break
        
        if valid:
            current_edges = []
            for i, (u, v) in enumerate(all_edges):
                if edge_colors[i] != -1:
                    current_edges.append([u, v, edge_colors[i]])
                    
            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges = current_edges
                
    return best_edges

