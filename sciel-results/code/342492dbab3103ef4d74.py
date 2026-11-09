"""Colorings with three colors in every four-term progression."""
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    N = 128
    
    # Precompute all 4-APs: (a, a+d, a+2d, a+3d) with 0-indexed positions
    # Nonconstant means d > 0
    aps = []
    for a in range(N):
        for d in range(1, (N - 1 - a) // 3 + 1):
            if a + 3*d < N:
                aps.append((a, a+d, a+2*d, a+3*d))
    
    n_aps = len(aps)
    
    # For each position, precompute which APs it participates in
    pos_aps = [[] for _ in range(N)]
    for i, ap in enumerate(aps):
        for p in ap:
            pos_aps[p].append(i)
    
    # For each AP, store the 4 positions
    ap_pos = np.array(aps, dtype=np.int32)  # shape (n_aps, 4)
    
    def is_valid(coloring):
        """Check if coloring is valid: every 4-AP has >= 3 distinct colors."""
        for ap in aps:
            colors = {coloring[ap[0]], coloring[ap[1]], coloring[ap[2]], coloring[ap[3]]}
            if len(colors) < 3:
                return False
        return True
    
    def try_solve(k, timeout=5.0):
        """Try to find a valid coloring using k colors (0..k-1).
        Uses backtracking with constraint propagation."""
        deadline = start_time + timeout
        coloring = [-1] * N
        
        # For each position, track which colors are still possible
        # We use a simple backtracking with forward checking
        
        # Precompute for each AP and each pair of colors, whether assigning
        # those colors to some subset forces issues
        # 
        # Key constraint: for a 4-AP (p0,p1,p2,p3), we need at least 3 distinct colors.
        # This is violated iff all 4 positions use colors from a set of size <= 2.
        # 
        # So the constraint is: NOT (all four positions have colors in {c1, c2} for some c1,c2)
        # Equivalently: at least 3 of the 4 positions have pairwise distinct colors (not quite)
        # 
        # More precisely: the set {coloring[p0], coloring[p1], coloring[p2], coloring[p3]} has size >= 3
        
        # Backtracking with MRV heuristic
        import sys
        sys.setrecursionlimit(10000)
        
        # Use iterative approach with explicit stack for speed
        # For each position, we'll try colors 0..k-1
        
        # Precompute AP membership for quick constraint checking
        # For each AP, track how many positions are assigned and what colors
        
        # State: for each AP, count of distinct colors used among assigned positions
        # and the set of colors used
        
        # Simpler approach: just check constraints when a position is assigned
        
        # Let's use a different approach: random restarts with local search
        # Start with random assignment of k colors, then fix violations
        
        best_coloring = None
        
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            # Random initial assignment
            coloring = list(np.random.randint(0, k, N))
            
            # Local search: find violated APs and recolor
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Find all violated APs
                violated = []
                for ap in aps:
                    colors = {coloring[ap[0]], coloring[ap[1]], coloring[ap[2]], coloring[ap[3]]}
                    if len(colors) < 3:
                        violated.append(ap)
                
                if not violated:
                    best_coloring = coloring[:]
                    break
                
                # Pick a random violated AP and try to fix it
                ap = violated[np.random.randint(len(violated))]
                p0, p1, p2, p3 = ap
                current_colors = [coloring[p0], coloring[p1], coloring[p2], coloring[p3]]
                distinct = set(current_colors)
                
                if len(distinct) >= 3:
                    continue  # Already fixed
                
                # Need to introduce a new color to one of the positions
                # Try changing one position to a color not in {current colors}
                # But we only have k colors total
                
                # Strategy: pick a position and try each color
                for pos_idx in [p0, p1, p2, p3]:
                    for new_color in range(k):
                        if new_color == coloring[pos_idx]:
                            continue
                        old_color = coloring[pos_idx]
                        coloring[pos_idx] = new_color
                        
                        # Check if this fixes the AP without creating too many new violations
                        # Quick check: does the AP now have >= 3 colors?
                        new_colors = {coloring[p0], coloring[p1], coloring[p2], coloring[p3]}
                        if len(new_colors) >= 3:
                            improved = True
                            break
                        else:
                            coloring[pos_idx] = old_color
                    
                    if improved:
                        break
            
            if best_coloring is not None:
                break
        
        return best_coloring
    
    # Binary search on k
    # Lower bound: k >= 3 (need at least 3 colors for any 4-AP)
    # Upper bound: k = 128 (trivially works with unique colors)
    
    # First, verify that k=3 is impossible or possible
    # Actually, let's just try increasing k from 3 upward
    
    for k in range(3, 129):
        if time.time() > DEADLINE:
            break
        
        result = try_solve(k, timeout=min(5.0, DEADLINE - time.time()))
        if result is not None:
            # Verify
            if is_valid(result):
                return result
    
    # Fallback: use more colors with a simpler construction
    # Pattern: use colors based on position mod something
    # If k is large enough, we can use a periodic pattern
    
    # Fallback: assign color = position % k for various k
    for k in range(3, 129):
        if time.time() > DEADLINE:
            break
        coloring = [i % k for i in range(N)]
        if is_valid(coloring):
            return coloring
    
    # Last resort: use 128 unique colors
    return list(range(N))

if __name__ == "__main__":
    result = construct()
    print(f"Number of colors used: {len(set(result))}")
    print(f"Result: {result[:20]}...")
