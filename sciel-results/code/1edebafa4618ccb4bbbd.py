import time
import itertools

def _check_k4_fast(colors):
    """Check every K4 has at least 5 distinct colours. Returns True if valid."""
    n = 16
    edge = [[0]*n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            edge[i][j] = c
            edge[j][i] = c
            idx += 1
    
    for a in range(n):
        ea = edge[a]
        for b in range(a + 1, n):
            eb = edge[b]
            ab = ea[b]
            for c in range(b + 1, n):
                ac = ea[c]
                bc = eb[c]
                ec = edge[c]
                for d in range(c + 1, n):
                    mask = (1 << ab) | (1 << ac) | (1 << ea[d]) | (1 << bc) | (1 << eb[d]) | (1 << ec[d])
                    if mask.bit_count() < 5:
                        return False
    return True

def _num_colours(colors):
    return len(set(colors))

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 16
    # 1. Generate Additive Mod 17 coloring
    # c_ij = (i + j) % 17
    colors = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    
    # Verify it's valid (should be)
    if not _check_k4_fast(colors):
        # Fallback to random 17 colors if additive fails (unlikely)
        import random
        while time.time() < deadline - 10:
            colors = [random.randint(0, 16) for _ in range(n * (n - 1) // 2)]
            if _check_k4_fast(colors):
                break
    
    best_colors = colors
    best_count = _num_colours(colors)
    
    # 2. Try to reduce to 16 colors by merging
    # Get unique colors
    unique_colors = list(set(colors))
    
    # Try all pairs
    for c1, c2 in itertools.combinations(unique_colors, 2):
        if time.time() > deadline - 20:
            break
        
        # Create merged coloring
        # Map c1 and c2 to a new value, say min(c1, c2)
        # We just need to check validity.
        # We can create a list of new colors
        new_colors = [c1 if c == c2 else c for c in colors]
        
        if _check_k4_fast(new_colors):
            best_colors = new_colors
            best_count = _num_colours(new_colors)
            # Update unique_colors for next round? 
            # We can break and restart the loop with the new set
            # Or just continue checking other pairs in the original set?
            # Merging might make other merges invalid.
            # Let's restart the search with the new best.
            break
    
    # 3. Try to reduce to 15 colors
    if best_count == 16:
        unique_colors = list(set(best_colors))
        for c1, c2 in itertools.combinations(unique_colors, 2):
            if time.time() > deadline - 10:
                break
            
            new_colors = [c1 if c == c2 else c for c in best_colors]
            
            if _check_k4_fast(new_colors):
                best_colors = new_colors
                best_count = _num_colours(new_colors)
                break
                
    return best_colors
