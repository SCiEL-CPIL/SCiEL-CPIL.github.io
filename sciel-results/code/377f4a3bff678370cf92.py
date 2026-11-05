import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def generate_instance(n, d, seed):
        """Generate the specific bounded degree graph."""
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        # Try to generate a graph with max degree d
        # The prompt implies a specific generation method: "bounded_degree_graph(n, d, seed)"
        # Based on the current code:
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            if u == v:
                continue
            e = (min(u, v), max(u, v))
            if e in edges:
                continue
            if degree[u] < d and degree[v] < d:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict_graph(edges):
        """
        Build the conflict graph for strong edge coloring.
        Vertices: 0 to m-1 corresponding to edges in 'edges'.
        Edge between i, j in conflict graph if edges[i] and edges[j] cannot share a color.
        Condition: edges[i] and edges[j] share a vertex OR are joined by an edge in G.
        """
        m = len(edges)
        if m == 0:
            return []
        
        # Map original vertex to list of incident edge indices
        vertex_to_edges = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            vertex_to_edges[u].append(i)
            vertex_to_edges[v].append(i)
            
        # Adjacency list for original graph for "joined by edge" check
        orig_adj = defaultdict(set)
        for u, v in edges:
            orig_adj[u].add(v)
            orig_adj[v].add(u)

        conflict = [set() for _ in range(m)]
        
        for i in range(m):
            u1, v1 = edges[i]
            # Check against all j > i
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                
                # 1. Share a vertex?
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflict[i].add(j)
                    conflict[j].add(i)
                    continue
                    
                # 2. Joined by an edge?
                # Check if (u1, u2), (u1, v2), (v1, u2), (v1, v2) are in original graph
                if (u2 in orig_adj[u1] or v2 in orig_adj[u1] or 
                    u2 in orig_adj[v1] or v2 in orig_adj[v1]):
                    conflict[i].add(j)
                    conflict[j].add(i)
        
        return conflict

    def dsatur_greedy(conflict, m):
        """
        DSATUR heuristic for initial coloring.
        Returns color array and num_colors.
        """
        if m == 0:
            return [], 0
            
        color = [-1] * m
        # sat_mask[i] is a bitmask of colors used by neighbors of i
        sat_mask = [0] * m
        num_colors = 0
        
        # Pre-compute degrees for tie-breaking
        degrees = [len(conflict[i]) for i in range(m)]
        
        for _ in range(m):
            # Find vertex with highest saturation, tie-break by degree
            best_v = -1
            best_sat = -1
            best_deg = -1
            
            for i in range(m):
                if color[i] == -1:
                    sat = bin(sat_mask[i]).count('1')
                    if sat > best_sat or (sat == best_sat and degrees[i] > best_deg):
                        best_sat = sat
                        best_deg = degrees[i]
                        best_v = i
            
            if best_v == -1:
                break
                
            # Find smallest available color
            used = sat_mask[best_v]
            c = 0
            while (used >> c) & 1:
                c += 1
            
            color[best_v] = c
            if c >= num_colors:
                num_colors = c + 1
                
            # Update neighbors
            for j in conflict[best_v]:
                if color[j] == -1:
                    sat_mask[j] |= (1 << c)
                    
        return color, num_colors

    def get_available_colors(i, color, conflict, num_colors):
        """Get list of colors available for vertex i."""
        used = set()
        for j in conflict[i]:
            if color[j] != -1:
                used.add(color[j])
        return [c for c in range(num_colors) if c not in used]

    def try_remove_color(color, conflict, target_color, m, num_colors):
        """
        Try to eliminate target_color by reassigning its edges to other colors.
        Returns True if successful, mutating color.
        """
        target_edges = [i for i in range(m) if color[i] == target_color]
        if not target_edges:
            return True
            
        # Sort by number of options (fewest conflicts with other colors)
        # Actually, just try to reassign one by one. 
        # To be more robust, we can use a simple backtracking or greedy with restarts for this subproblem.
        # Given the small size, a greedy pass with randomization works well.
        
        # We need to reassign all edges in target_edges to colors in [0, target_color-1]
        
        # Sort target edges by degree in the subgraph induced by non-target colors?
        # Simpler: Sort by number of available colors in current state (excluding target_color)
        
        def count_avail(i):
            cnt = 0
            for j in conflict[i]:
                if color[j] != -1 and color[j] != target_color:
                    # This neighbor uses a color we might need
                    pass
            # Actually, just check how many colors < target_color are available
            used = set(color[j] for j in conflict[i] if color[j] != -1 and color[j] != target_color)
            return len([c for c in range(target_color) if c not in used])
            
        target_edges.sort(key=count_avail)
        
        for i in target_edges:
            avail = get_available_colors(i, color, conflict, target_color) # Only look at colors < target_color
            if not avail:
                return False
            # Random choice to help with local minima
            color[i] = random.choice(avail)
            
        return True

    def solve_instance(n, d, seed, time_budget):
        edges = generate_instance(n, d, seed)
        m = len(edges)
        
        if m == 0:
            return []
            
        conflict = build_conflict_graph(edges)
        
        # Initial coloring
        color, num_colors = dsatur_greedy(conflict, m)
        
        best_color = color[:]
        best_nc = num_colors
        
        # Simulated Annealing
        # Temperature starts high, decays
        # Objective: Minimize number of colors
        
        T = 1.0
        cooling_rate = 0.995
        min_T = 0.01
        
        rng = random.Random(seed)
        
        # Precompute neighbor lists as lists for faster iteration
        conflict_lists = [list(c) for c in conflict]
        
        while time.time() < start_time + time_budget:
            if T < min_T:
                break
                
            # Pick a random move
            move_type = rng.random()
            
            if move_type < 0.7:
                # Recolor move
                i = rng.randrange(m)
                old_c = color[i]
                # Try to change to a different color
                # To potentially reduce colors, we should prefer colors that are already in use
                # Or if we want to explore, any valid color.
                
                # Find available colors
                used_by_neighbors = set(color[j] for j in conflict_lists[i])
                avail = [c for c in range(best_nc) if c not in used_by_neighbors]
                
                if not avail:
                    continue
                    
                if old_c in avail:
                    # We can stay, but we want to move
                    if len(avail) == 1:
                        continue
                    avail.remove(old_c)
                    
                new_c = rng.choice(avail)
                
                # Apply move
                color[i] = new_c
                
                # Calculate change in number of colors
                # Did we free up a color?
                # Check if old_c is still used by anyone
                # This is expensive O(m). 
                # Optimization: Keep a count of usage per color.
                
                # Let's maintain color_counts
                # We need to update color_counts outside this block or pass it in.
                # For simplicity and correctness, we calculate delta.
                
                # Actually, let's just recompute num_colors if the move looks promising?
                # No, that's O(m).
                
                # Better: Maintain `color_counts` array.
                
                # Let's restructure to maintain color_counts.
                
                # (Re-implementing inner loop with color_counts)
                
                # To avoid complex state management in the snippet, I will use a simpler 
                # local search that explicitly tries to remove the highest color, 
                # combined with random swaps to escape.
                
                color[i] = old_c # revert for now, will use different strategy below
                
            else:
                # Swap move
                i = rng.randrange(m)
                j = rng.randrange(m)
                if i == j:
                    continue
                if j in conflict[i]:
                    continue
                
                c_i = color[i]
                c_j = color[j]
                if c_i == c_j:
                    continue
                
                # Check if swap is valid
                # i takes c_j: c_j not in neighbors of i
                # j takes c_i: c_i not in neighbors of j
                valid_i = True
                for k in conflict_lists[i]:
                    if color[k] == c_j:
                        valid_i = False
                        break
                if not valid_i:
                    continue
                
                valid_j = True
                for k in conflict_lists[j]:
                    if color[k] == c_i:
                        valid_j = False
                        break
                if not valid_j:
                    continue
                
                # Apply swap
                color[i] = c_j
                color[j] = c_i
                
                # Check if this reduced color count?
                # It doesn't change the set of colors used, so num_colors stays same.
                # But it might allow future removals.
                
                # Accept swap with probability based on T (always accept if same cost)
                # Since cost is same, accept with prob 1.0 or high.
                # We keep the swap.
                
            T *= cooling_rate
            
            # Periodically try to remove the highest color
            if rng.random() < 0.1:
                trial = color[:]
                if try_remove_color(trial, conflict, best_nc - 1, m, best_nc):
                    color = trial
                    best_nc -= 1
                    best_color = color[:]
                    # Reset temperature to explore new landscape
                    T = max(T, 0.5)
                    
        # Final polish
        for _ in range(100):
            if time.time() > start_time + time_budget - 1:
                break
            trial = color[:]
            if try_remove_color(trial, conflict, best_nc - 1, m, best_nc):
                color = trial
                best_nc -= 1
                best_color = color[:]
            else:
                break
                
        return best_color

    # Allocation
    total_budget = deadline - start_time - 5
    # Weights based on difficulty (edges count approx n*d)
    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]
    weights = [40*4, 60*5, 80*6]
    total_w = sum(weights)
    
    out = []
    for i, (n, d, seed) in enumerate(specs):
        remaining = deadline - time.time() - 2
        if remaining < 5:
            out.append([])
            continue
            
        budget = min(total_budget * weights[i] / total_w, remaining)
        res = solve_instance(n, d, seed, budget)
        out.append(res)
        
    while len(out) < 3:
        out.append([])
        
    return out

# EVOLVE-BLOCK-END
