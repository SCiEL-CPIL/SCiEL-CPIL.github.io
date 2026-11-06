"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def _check_k4(colors):
    """Check every K4 has at least 5 distinct colours."""
    n = 16
    # Build edge lookup
    edge_color = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_color[(i,j)] = colors[idx]
            idx += 1
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    cs = {
                        edge_color[(a,b)], edge_color[(a,c)], edge_color[(a,d)],
                        edge_color[(b,c)], edge_color[(b,d)], edge_color[(c,d)]
                    }
                    if len(cs) < 5:
                        return False
    return True

def _num_colours(colors):
    return len(set(colors))

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    best_colors = None
    best_count = 17  # initial solution uses 17 colours
    
    # Try (a*i + b*j) % p for various a, b, p
    # p should be prime to help with the constraint
    primes = [13, 17, 19, 23, 29, 31]
    
    for p in primes:
        if time.time() > deadline:
            break
        for a in range(1, p):
            for b in range(a, p):  # b >= a to avoid duplicates up to symmetry
                colors = []
                for i in range(16):
                    for j in range(i+1, 16):
                        colors.append((a*i + b*j) % p)
                cnt = _num_colours(colors)
                if cnt < best_count:
                    if _check_k4(colors):
                        best_colors = colors
                        best_count = cnt
                        if best_count <= 13:
                            return colors
    
    if best_colors is None:
        # Fallback to original
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    return best_colors
# EVOLVE-BLOCK-END
