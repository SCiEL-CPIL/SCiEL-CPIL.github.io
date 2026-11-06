import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 16
    vertices = list(range(n))
    
    # Generate 1-factorization of K16
    # Vertices 0..14 and 15 (infinity)
    # M_k for k in 0..14
    matchings = []
    for k in range(15):
        m = []
        m.append((15, k)) # Edge with infinity
        for i in range(1, 8):
            u = (k + i) % 15
            v = (k - i) % 15
            if u > v:
                u, v = v, u
            m.append((u, v))
        matchings.append(m)
        
    # Map edge to matching index
    edge_to_match = {}
    for idx, m in enumerate(matchings):
        for u, v in m:
            edge_to_match[(u, v)] = idx
            
    # Precompute all K4s and their matching indices
    k4s = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
                    m_indices = set()
                    for u, v in edges:
                        m_indices.add(edge_to_match[(u, v)])
                    k4s.append(list(m_indices))
                    
    def check_assignment(color_of_match):
        # color_of_match is a list of 15 ints
        for m_indices in k4s:
            colors = set()
            for m_idx in m_indices:
                colors.add(color_of_match[m_idx])
            if len(colors) < 5:
                return False
        return True

    # Try 13 colours
    # We need to assign 15 matchings to 13 colours.
    # This means 2 pairs of matchings share a colour, and 11 singletons.
    # Or 1 triple and 12 singletons? No, 13 colours means sum of counts = 15.
    # Possible partitions: 2,2,1...1 or 3,1...1
    # 2,2 is likely safer?
    
    # Strategy: Randomly pick 2 pairs to merge.
    # There are C(15,2) ways to pick first pair, C(13,2) for second.
    # Too many.
    # Just try random assignments.
    
    best_colors = 15
    best_assignment = None
    
    # Guaranteed 14
    # Merge 0,1
    assign_14 = [0] + [0] + [i for i in range(1, 14)]
    # Wait, indices 0..14.
    # Merge 0 and 1 -> Colour 0.
    # 2 -> 1, 3 -> 2, ..., 14 -> 13.
    assign_14 = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    if check_assignment(assign_14):
        best_colors = 14
        best_assignment = assign_14
    else:
        # Should not happen
        best_colors = 15
        best_assignment = list(range(15))

    # Try to find 13
    if time.time() < deadline - 10.0:
        for _ in range(10000):
            if time.time() > deadline - 5.0:
                break
            # Randomly pick 2 indices to merge
            i, j = random.sample(range(15), 2)
            # Assign i, j to colour 0
            # Others to 1..12
            assign = [0] * 15
            # We need 13 distinct colours.
            # Let's just assign random colours from 0..12
            # But we must ensure exactly 13 are used? No, just <= 13.
            # Minimizing colours means we want to use as few as possible.
            # If we use 12, even better.
            
            # Let's try to map to 0..12
            # Random assignment
            assign = [random.randint(0, 12) for _ in range(15)]
            if check_assignment(assign):
                # Count distinct colours
                used = set(assign)
                if len(used) < best_colors:
                    best_colors = len(used)
                    best_assignment = assign
                    if best_colors <= 13:
                        break
                        
    # Construct final output
    result = []
    for i in range(n):
        for j in range(i+1, n):
            m_idx = edge_to_match[(i, j)]
            result.append(best_assignment[m_idx])
            
    return result
