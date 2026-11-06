"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _verify(colors, n=128):
    """Check if every non-constant 4-AP uses at least 3 distinct colors."""
    for a in range(n):
        for d in range(1, (n - a) // 3 + 1):
            i3 = a + 3 * d
            if i3 >= n:
                break
            c = {colors[a], colors[a + d], colors[a + 2 * d], colors[a + 3 * d]}
            if len(c) < 3:
                return False
    return True

def _try_modular(k, n=128):
    """Try coloring c(x) = x % k for positions 1..n. Return colors if valid, else None."""
    colors = [x % k for x in range(1, n + 1)]
    if _verify(colors, n):
        return colors
    return None

def construct():
    start = time.time()
    deadline = start + 55.0  # safety margin before 60s limit
    n = 128
    
    # Try modular colorings from smallest k upward
    best_colors = None
    best_k = None
    
    for k in range(2, 128):
        if time.time() > deadline:
            break
        colors = _try_modular(k, n)
        if colors is not None:
            best_colors = colors
            best_k = k
            break  # found smallest valid k, stop
    
    # If no modular coloring found (shouldn't happen, k=43 works), fall back
    if best_colors is None:
        # Use the greedy approach from the seed as fallback
        colors = []
        for x in range(1, 129):
            for color in range(128):
                if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 
                        for d in range(1, (x-1)//3 + 1))
                    and all(any(colors[x-1-j*d] != color for j in (1,2)) 
                            for d in range(1, (x-1)//2 + 1))):
                    colors.append(color)
                    break
        best_colors = colors
    
    # Optional: try to reduce colors via local search if time permits
    # This is a refinement pass - try to remap colors to use fewer distinct labels
    if time.time() < deadline - 5.0:
        # Count distinct colors used
        used = set(best_colors)
        if len(used) > 3:
            # Try to merge color pairs that never appear together in a way that violates constraints
            # This is complex; skip for now and return the modular solution
            pass
    
    return best_colors
# EVOLVE-BLOCK-END
