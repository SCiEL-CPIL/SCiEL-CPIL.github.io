"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    # Edge index mapping: (i,j) -> index in 0..119, lexicographic order
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        # Count edges before (i,j): for each k < i, there are (N-1-k) edges from k
        # Plus edges from i to vertices < j
        idx = 0
        for k in range(i):
            idx += (N - 1 - k)
        idx += (j - i - 1)
        return idx
    
    # Precompute edge indices
    edge_map = {}
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            edge_map[(i,j)] = idx
            idx += 1
    
    # All 4-vertex subsets and their 3 four-cycles
    # For 4 vertices a<b<c<d, the three 4-cycles are:
    # Cycle 1: a-b-c-d-a => edges (a,b), (b,c), (c,d), (d,a)
    # Cycle 2: a-b-d-c-a => edges (a,b), (b,d), (d,c), (c,a)
    # Cycle 3: a-c-b-d-a => edges (a,c), (c,b), (b,d), (d,a)
    
    four_cycles = []  # List of lists of 4 edge indices
    for a in range(N-3):
        for b in range(a+1, N-2):
            for c in range(b+1, N-1):
                for d in range(c+1, N):
                    # Cycle 1: a-b-c-d-a
                    four_cycles.append([
                        edge_map[(a,b)], edge_map[(b,c)], 
                        edge_map[(c,d)], edge_map[(d,a)]
                    ])
                    # Cycle 2: a-b-d-c-a
                    four_cycles.append([
                        edge_map[(a,b)], edge_map[(b,d)], 
                        edge_map[(c,d)], edge_map[(a,c)]
                    ])
                    # Cycle 3: a-c-b-d-a
                    four_cycles.append([
                        edge_map[(a,c)], edge_map[(b,c)], 
                        edge_map[(b,d)], edge_map[(d,a)]
                    ])
    
    # For each edge, precompute which four-cycles it participates in
    edge_to_cycles = [[] for _ in range(N*(N-1)//2)]
    for ci, cycle in enumerate(four_cycles):
        for ei in cycle:
            edge_to_cycles[ei].append(ci)
    
    # For each four-cycle, precompute the other 3 edges given one edge
    cycle_other_edges = {}
    for ci, cycle in enumerate(four_cycles):
        for pos in range(4):
            ei = cycle[pos]
            others = [cycle[k] for k in range(4) if k != pos]
            cycle_other_edges[(ci, ei)] = others
    
    # Greedy construction with backtracking
    best_colors = None
    best_count = 0
    
    # Try multiple random orderings
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Random edge ordering
        edge_order = list(range(N*(N-1)//2))
        random.shuffle(edge_order)
        
        colors = [-1] * (N*(N-1)//2)
        used_colors = set()
        
        # For tracking: for each color, which edges use it
        # For constraint checking: when assigning color to edge e, check all 4-cycles containing e
        # A 4-cycle is violated if all 4 edges have distinct colors
        
        # Efficient check: for each 4-cycle containing edge e with color c,
        # the other 3 edges must not all have distinct colors AND different from c
        
        success = True
        for ei in edge_order:
            if time.time() > deadline:
                success = False
                break
            
            # Try to assign a new color first
            assigned = False
            
            # First, try a new color (highest available)
            new_color = len(used_colors)
            if new_color < 120:
                # Check if assigning new_color to ei violates any constraint
                valid = True
                for ci in edge_to_cycles[ei]:
                    others = cycle_other_edges[(ci, ei)]
                    other_colors = []
                    for oe in others:
                        if colors[oe] == -1:
                            # Unassigned, so no conflict yet (we're building incrementally)
                            # Actually, we need to be careful: if some are unassigned,
                            # the constraint isn't fully determined yet.
                            # In greedy forward assignment, we only check fully assigned cycles
                            pass
                        else:
                            other_colors.append(colors[oe])
                    
                    # Only check if all 4 edges are assigned
                    if all(colors[oe] != -1 for oe in others):
                        all_colors = other_colors + [new_color]
                        if len(set(all_colors)) == 4:
                            valid = False
                            break
                
                if valid:
                    colors[ei] = new_color
                    used_colors.add(new_color)
                    assigned = True
            
            # If new color doesn't work, try existing colors
            if not assigned:
                for c in sorted(used_colors, reverse=True):  # Try higher colors first
                    valid = True
                    for ci in edge_to_cycles[ei]:
                        others = cycle_other_edges[(ci, ei)]
                        if all(colors[oe] != -1 for oe in others):
                            other_colors = [colors[oe] for oe in others]
                            all_colors = other_colors + [c]
                            if len(set(all_colors)) == 4:
                                valid = False
                                break
                    if valid:
                        colors[ei] = c
                        assigned = True
                        break
            
            if not assigned:
                success = False
                break
        
        if success:
            count = len(used_colors)
            if count > best_count:
                best_count = count
                best_colors = colors[:]
        
        # Early stop if we've used all 120 colors
        if best_count == 120:
            break
    
    if best_colors is None:
        # Fallback: all zeros
        return [0] * (N*(N-1)//2)
    
    return best_colors
# EVOLVE-BLOCK-END
