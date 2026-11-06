import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n = 16
    # Precompute all K4s and their edge indices
    k4s = []
    edge_index = {}
    edge_ij = []  # map edge index -> (i, j)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_index[(i,j)] = idx
            edge_ij.append((i, j))
            idx += 1
    
    for a,b,c,d in combinations(range(n), 4):
        edges = [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]
        edge_idxs = [edge_index[e] for e in edges]
        k4s.append(edge_idxs)
    
    # Precompute for each edge, which K4s it belongs to
    edge_to_k4s = [[] for _ in range(120)]
    for k4_idx, edge_idxs in enumerate(k4s):
        for ei in edge_idxs:
            edge_to_k4s[ei].append(k4_idx)
    
    # For each K4, track how many edges are assigned and their colors
    # We'll use a backtracking approach
    
    # Order edges by a heuristic: edges within the same vertex cluster first
    # This helps detect violations earlier
    edge_order = list(range(120))
    
    # Try to find a valid coloring with 13 colors
    palette_size = 13
    colors = [-1] * 120
    
    # For each K4, track the set of colors used on its assigned edges
    k4_colors = [set() for _ in range(len(k4s))]
    k4_assigned_count = [0] * len(k4s)
    
    # Count how many edges in each K4 are already assigned
    def assign_edge(ei, c):
        colors[ei] = c
        for k4_idx in edge_to_k4s[ei]:
            k4_colors[k4_idx].add(c)
            k4_assigned_count[k4_idx] += 1
    
    def unassign_edge(ei):
        c = colors[ei]
        colors[ei] = -1
        for k4_idx in edge_to_k4s[ei]:
            k4_assigned_count[k4_idx] -= 1
            # Remove color if no other edge in this K4 has this color
            edge_idxs = k4s[k4_idx]
            if not any(colors[ei2] == c and ei2 != ei for ei2 in edge_idxs):
                k4_colors[k4_idx].discard(c)
    
    def is_consistent(ei, c):
        """Check if assigning color c to edge ei is consistent with all K4s containing ei."""
        for k4_idx in edge_to_k4s[ei]:
            edge_idxs = k4s[k4_idx]
            assigned_count = k4_assigned_count[k4_idx]
            
            # If all 6 edges are assigned, we need at least 5 distinct colors
            if assigned_count + 1 == 6:
                # This would be the last edge
                # Current colors + new color must have >= 5 distinct
                current_colors = k4_colors[k4_idx].copy()
                current_colors.add(c)
                if len(current_colors) < 5:
                    return False
            else:
                # Still have unassigned edges
                # We need to be able to reach 5 distinct colors
                # Current distinct colors + potential new colors from unassigned edges
                current_distinct = len(k4_colors[k4_idx])
                # The new color c might add to distinct count
                if c not in k4_colors[k4_idx]:
                    potential_distinct = current_distinct + 1
                else:
                    potential_distinct = current_distinct
                
                # Remaining unassigned edges can add at most (6 - assigned_count - 1) new colors
                # But we need at least 5 total
                # If potential_distinct + remaining < 5, it's impossible
                remaining = 6 - assigned_count - 1
                if potential_distinct + remaining < 5:
                    return False
        
        return True
    
    def backtrack(ei_idx):
        """Backtracking search. ei_idx is the index in edge_order."""
        if time.time() > deadline:
            return False
        if ei_idx == 120:
            return True
        
        ei = edge_order[ei_idx]
        
        # Try each color in the palette
        for c in range(palette_size):
            if is_consistent(ei, c):
                assign_edge(ei, c)
                if backtrack(ei_idx + 1):
                    return True
                unassign_edge(ei)
        
        return False
    
    # Try backtracking with 13 colors
    if backtrack(0):
        return colors
    
    # If 13 colors didn't work, try 14
    palette_size = 14
    colors = [-1] * 120
    k4_colors = [set() for _ in range(len(k4s))]
    k4_assigned_count = [0] * len(k4s)
    
    def backtrack14(ei_idx):
        if time.time() > deadline:
            return False
        if ei_idx == 120:
            return True
        
        ei = edge_order[ei_idx]
        
        for c in range(palette_size):
            if is_consistent(ei, c):
                assign_edge(ei, c)
                if backtrack14(ei_idx + 1):
                    return True
                unassign_edge(ei)
        
        return False
    
    if backtrack14(0):
        return colors
    
    # Fallback: use the original modular construction
    return [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
