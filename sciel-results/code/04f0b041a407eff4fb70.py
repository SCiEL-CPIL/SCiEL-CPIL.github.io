import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n = 16
    # Edge list in lexicographic order
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    num_edges = len(edges)
    
    # Initialize with (i+j) % 17
    colors = [(i + j) % 17 for i, j in edges]
    
    # Helper: check if coloring is valid
    def is_valid(col):
        # For each K4, check if it has >= 5 distinct colors
        for a in range(n-3):
            for b in range(a+1, n-2):
                for c in range(b+1, n-1):
                    for d in range(c+1, n):
                        # Find edge indices
                        # Edge (a,b), (a,c), (a,d), (b,c), (b,d), (c,d)
                        e1 = a*n - (a+1)*(a+2)//2 + (b - a - 1)  # This is getting complex
                        # Better to use a lookup
                        pass
        return True
    
    # Let me use a different approach with a lookup table
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
    
    def get_k4_colors(col, a, b, c, d):
        return [
            col[edge_idx[(a,b)]],
            col[edge_idx[(a,c)]],
            col[edge_idx[(a,d)]],
            col[edge_idx[(b,c)]],
            col[edge_idx[(b,d)]],
            col[edge_idx[(c,d)]]
        ]
    
    def count_bad_k4s(col):
        bad = 0
        for a in range(n-3):
            for b in range(a+1, n-2):
                for c in range(b+1, n-1):
                    for d in range(c+1, n):
                        k4c = get_k4_colors(col, a, b, c, d)
                        if len(set(k4c)) < 5:
                            bad += 1
        return bad
    
    # Local search to minimize max color
    current_max = max(colors)
    
    # Try to reduce colors by local search
    random.seed(42)
    
    while time.time() < deadline:
        improved = False
        # Try each edge
        for idx in range(num_edges):
            if time.time() > deadline:
                break
            old_color = colors[idx]
            # Try to change to a color < old_color, or try to reduce max
            # Try colors from 0 to current_max-1
            for new_color in range(0, current_max):
                if new_color == old_color:
                    continue
                colors[idx] = new_color
                bad = count_bad_k4s(colors)
                if bad == 0:
                    # Valid! Update max if we reduced it
                    new_max = max(colors)
                    if new_max < current_max:
                        current_max = new_max
                        improved = True
                    break
                else:
                    colors[idx] = old_color
            colors[idx] = old_color  # restore
            if improved:
                break
        if not improved:
            break
    
    return colors
