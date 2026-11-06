import time
import random

N = 16

# EVOLVE-BLOCK-START
def construct():
    # Base case for K4
    # Vertices 0,1,2,3
    # Edges (i,j) with i<j
    # (0,1): 1
    # (0,2): 4
    # (0,3): 1
    # (1,2): 2
    # (1,3): 4
    # (2,3): 3
    # Colors used: {1, 2, 3, 4}
    
    def get_base_coloring():
        # Returns a dictionary mapping (local_i, local_j) to color
        # and the max color value used
        base = {
            (0, 1): 1,
            (0, 2): 4,
            (0, 3): 1,
            (1, 2): 2,
            (1, 3): 4,
            (2, 3): 3
        }
        return base, 4

    def build_coloring(n):
        # Returns a dictionary mapping (local_i, local_j) to color
        # and the max color value used
        if n == 4:
            return get_base_coloring()
        
        n1 = n // 2
        n2 = n - n1
        
        # Recursively build for first half and second half
        # We need to handle n=1,2,3 as well?
        # The problem is for N=16, so we will hit 8, 4.
        # But what if n is not a power of 2?
        # For n=16, splits are 8, 8. Then 4, 4. So we only hit 4.
        # If we had n=10, 5, 5. 5 splits to 2, 3.
        # We need base cases for n=1,2,3.
        
        if n == 1:
            return {}, 0
        if n == 2:
            return {(0, 1): 1}, 1
        if n == 3:
            # K3. Any 3 edges. No C4. Can use 3 colors.
            return {(0, 1): 1, (0, 2): 2, (1, 2): 3}, 3
            
        col1, max1 = build_coloring(n1)
        col2, max2 = build_coloring(n2)
        
        # Shift colors of col2 to be above max1
        shifted_col2 = {}
        for (i, j), c in col2.items():
            shifted_col2[(i, j)] = c + max1
            
        bridge_color = max1 + max2 + 1
        
        new_col = {}
        
        # Edges in first half
        for (i, j), c in col1.items():
            new_col[(i, j)] = c
            
        # Edges in second half (shift indices)
        for (i, j), c in shifted_col2.items():
            new_col[(i + n1, j + n1)] = c
            
        # Edges between halves
        for i in range(n1):
            for j in range(n1, n):
                new_col[(i, j)] = bridge_color
                
        return new_col, bridge_color

    # Build for N=16
    coloring_dict, max_color = build_coloring(N)
    
    # Precompute all 4-cycles (all three cycle decompositions for each 4-vertex subset)
    # For vertices a<b<c<d, the three 4-cycles are:
    # Cycle 1: (a,b),(b,c),(c,d),(d,a) -> edges (a,b),(b,c),(c,d),(a,d)
    # Cycle 2: (a,b),(b,d),(d,c),(c,a) -> edges (a,b),(b,d),(c,d),(a,c)
    # Cycle 3: (a,c),(c,b),(b,d),(d,a) -> edges (a,c),(b,c),(b,d),(a,d)
    
    def get_four_cycles():
        cycles = []
        for a in range(N):
            for b in range(a+1, N):
                for c in range(b+1, N):
                    for d in range(c+1, N):
                        # Cycle 1: a-b-c-d-a
                        cycles.append([(a,b), (b,c), (c,d), (a,d)])
                        # Cycle 2: a-b-d-c-a
                        cycles.append([(a,b), (b,d), (c,d), (a,c)])
                        # Cycle 3: a-c-b-d-a
                        cycles.append([(a,c), (b,c), (b,d), (a,d)])
        return cycles
    
    four_cycles = get_four_cycles()
    
    # Helper to check if a coloring is valid (no rainbow C4)
    def is_valid(coloring):
        for cycle in four_cycles:
            colors = [coloring[e] for e in cycle]
            if len(set(colors)) == 4:
                return False
        return True
    
    # Helper to get number of distinct colors
    def count_distinct(coloring):
        return len(set(coloring.values()))
    
    # Local search: try to recolor edges to increase distinct colors
    def local_search(coloring, max_iter=5000, timeout=55.0):
        start_time = time.time()
        best_coloring = dict(coloring)
        best_count = count_distinct(best_coloring)
        
        # Precompute for each edge, which cycles it belongs to
        edge_to_cycles = {e: [] for e in coloring}
        for idx, cycle in enumerate(four_cycles):
            for e in cycle:
                edge_to_cycles[e].append(idx)
        
        # Try random edge recoloring
        edges = list(coloring.keys())
        
        for it in range(max_iter):
            if time.time() - start_time > timeout:
                break
            
            # Pick a random edge
            e = random.choice(edges)
            old_color = coloring[e]
            
            # Try new colors from 0 to 119
            # Prefer colors not currently used to increase diversity
            used_colors = set(coloring.values())
            
            # Try unused colors first
            candidate_colors = [c for c in range(120) if c not in used_colors]
            random.shuffle(candidate_colors)
            
            # Also try some random colors from used set
            for _ in range(3):
                candidate_colors.append(random.randint(0, 119))
            
            improved = False
            for new_color in candidate_colors:
                if new_color == old_color:
                    continue
                    
                coloring[e] = new_color
                
                # Check only affected cycles
                valid = True
                for cyc_idx in edge_to_cycles[e]:
                    cycle = four_cycles[cyc_idx]
                    colors = [coloring[ce] for ce in cycle]
                    if len(set(colors)) == 4:
                        valid = False
                        break
                
                if valid:
                    new_count = count_distinct(coloring)
                    if new_count >= best_count:
                        if new_count > best_count:
                            best_coloring = dict(coloring)
                            best_count = new_count
                            improved = True
                        break
                else:
                    coloring[e] = old_color
            
            if not improved and random.random() < 0.1:
                # Occasionally revert to best
                coloring = dict(best_coloring)
        
        return best_coloring, best_count
    
    # Run local search
    deadline = 55.0
    start = time.time()
    
    # First pass: try to improve
    improved_coloring, improved_count = local_search(coloring_dict, max_iter=8000, timeout=deadline - (time.time() - start) - 2.0)
    
    if improved_count > count_distinct(coloring_dict):
        coloring_dict = improved_coloring
    
    # Second pass if time remains
    remaining = deadline - (time.time() - start)
    if remaining > 5.0:
        improved_coloring2, improved_count2 = local_search(coloring_dict, max_iter=5000, timeout=remaining - 2.0)
        if improved_count2 > count_distinct(coloring_dict):
            coloring_dict = improved_coloring2
    
    # Create the final array
    # Lexicographically ordered pairs (i,j), 0<=i<j<N
    result = [0] * (N * (N - 1) // 2)
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            result[idx] = coloring_dict[(i, j)]
            idx += 1
            
    return result
# EVOLVE-BLOCK-END
