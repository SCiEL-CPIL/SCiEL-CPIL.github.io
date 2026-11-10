import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    # Pairs (i, j) in lexicographic order
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(pairs)
    # Map pair to index
    pair_index = {p: idx for idx, p in enumerate(pairs)}
    
    # Precompute all K4s with their edge indices
    k4s = list(combinations(range(n), 4))
    k4_edges = []
    # Also precompute for each edge, which K4s it belongs to
    edge_k4s = [[] for _ in range(num_edges)]
    
    for k4_idx, k4 in enumerate(k4s):
        v1, v2, v3, v4 = k4
        edges = [pair_index[(v1, v2)], pair_index[(v1, v3)], pair_index[(v1, v4)],
                 pair_index[(v2, v3)], pair_index[(v2, v4)], pair_index[(v3, v4)]]
        k4_edges.append(edges)
        for e in edges:
            edge_k4s[e].append(k4_idx)
            
    # Initialize with random colors from 0 to 14
    colors = [random.randint(0, 14) for _ in range(num_edges)]
    
    def get_k4_colors(k4_idx):
        """Get set of colors in a K4."""
        edges = k4_edges[k4_idx]
        return set(colors[e] for e in edges)
        
    def is_valid():
        """Check if all K4s have at least 5 distinct colors."""
        for k4_idx in range(len(k4s)):
            if len(get_k4_colors(k4_idx)) < 5:
                return False
        return True
        
    def repair():
        """Repair violations by changing random edges in violated K4s."""
        # Limit repair steps to avoid infinite loops, though unlikely
        for _ in range(1000):
            # Find a violated K4
            violated_k4 = -1
            for k4_idx in range(len(k4s)):
                if len(get_k4_colors(k4_idx)) < 5:
                    violated_k4 = k4_idx
                    break
            
            if violated_k4 == -1:
                return True # Valid
            
            # Pick a random edge in the violated K4
            edges = k4_edges[violated_k4]
            edge_idx = random.choice(edges)
            
            # Change to a random color, possibly introducing new colors
            # Using a wider range helps break symmetries
            colors[edge_idx] = random.randint(0, 119)
            
            # If we've been repairing for a long time, force a random restart of colors
            if _ > 500:
                 colors = [random.randint(0, 14) for _ in range(num_edges)]
        
        return is_valid()

    def try_merge():
        """Try to merge two colors to reduce count."""
        nonlocal colors
        
        # Count color usage
        color_counts = {}
        for c in colors:
            color_counts[c] = color_counts.get(c, 0) + 1
        
        active_colors = sorted(color_counts.keys(), key=lambda x: color_counts[x])
        if len(active_colors) < 2:
            return False
            
        # Try merging pairs of least used colors
        for i in range(len(active_colors)):
            for j in range(i + 1, len(active_colors)):
                c1 = active_colors[i]
                c2 = active_colors[j]
                
                # Check if merging c2 into c1 is valid
                # Only K4s containing at least one edge of color c2 need checking
                # because c2 disappears, potentially reducing distinct count.
                # If a K4 had {c1, c2, c3, c4, c5}, it becomes {c1, c3, c4, c5} (4 colors) -> Invalid.
                # So we check all K4s that contain c2.
                
                valid = True
                # Collect edges with color c2
                edges_with_c2 = [e for e, c in enumerate(colors) if c == c2]
                
                # Get unique K4s affected
                affected_k4s = set()
                for e in edges_with_c2:
                    affected_k4s.update(edge_k4s[e])
                    
                for k4_idx in affected_k4s:
                    # Simulate merge: replace c2 with c1 in this K4's color set
                    k4_col_set = get_k4_colors(k4_idx)
                    if c2 in k4_col_set:
                        # If c1 is also in the set, the size might decrease
                        # New set is k4_col_set - {c2} | {c1}
                        # Since c1 is in k4_col_set, it's just k4_col_set - {c2}
                        if len(k4_col_set) == 5:
                             # If it was 5, and we remove c2, it becomes 4 unless c1 was already there?
                             # Wait. If c1 is in k4_col_set, then {c1, c2, ...} becomes {c1, ...}.
                             # If the set was {c1, c2, c3, c4, c5}, removing c2 gives {c1, c3, c4, c5} (size 4).
                             # So any K4 with exactly 5 colors containing c2 will become invalid if c1 is also present?
                             # No, if c1 is NOT present, {c2, c3, c4, c5, c6} becomes {c1, c3, c4, c5, c6} (size 5). Valid.
                             # If c1 IS present, {c1, c2, c3, c4, c5} becomes {c1, c3, c4, c5} (size 4). Invalid.
                             
                             if c1 in k4_col_set:
                                 valid = False
                                 break
                             
                if valid:
                    # Perform merge
                    for k in range(num_edges):
                        if colors[k] == c2:
                            colors[k] = c1
                    return True
                    
        return False

    # Main Loop
    # 1. Get a valid solution first
    if not repair():
        # If repair fails, return whatever we have (should be rare)
        return colors

    # 2. Optimize: Merge colors
    while time.time() < deadline:
        # Try to merge
        if try_merge():
            continue
            
        # If no merge, do random recoloring to find new merge opportunities
        # Pick a random edge
        idx = random.randint(0, num_edges - 1)
        old_color = colors[idx]
        
        # Try changing to a random color
        new_color = random.randint(0, 119)
        if new_color == old_color:
            continue
            
        colors[idx] = new_color
        
        # Check validity for K4s containing this edge
        valid = True
        for k4_idx in edge_k4s[idx]:
            if len(get_k4_colors(k4_idx)) < 5:
                valid = False
                break
                
        if valid:
            # Check if old_color is now unused
            if old_color not in colors:
                pass # No explicit counter, try_merge handles it
        else:
            # Revert
            colors[idx] = old_color
            
    return colors
