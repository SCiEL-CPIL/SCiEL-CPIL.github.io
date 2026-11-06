import time
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    N = 128
    
    # Precompute all 4-term progressions
    aps = []
    for a in range(1, N + 1):
        for d in range(1, N // 3 + 1):
            end = a + 3 * d
            if end <= N:
                aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    # Map position to list of AP indices it belongs to
    ap_indices = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            ap_indices[pos].append(idx)
    
    # Compute degree of each position (number of APs it belongs to)
    degree = [len(ap_indices[i]) for i in range(N)]
    
    # Order positions by degree (descending) for backtracking
    order = sorted(range(N), key=lambda i: -degree[i])
    order_inv = [0] * N
    for i, pos in enumerate(order):
        order_inv[pos] = i
    
    def check_ap_valid(ap_idx, colors):
        """Check if AP at ap_idx has at least 3 distinct colors among assigned positions."""
        a, b, c, d = aps[ap_idx]
        ca = colors[a]
        cb = colors[b]
        cc = colors[c]
        cd = colors[d]
        
        # Count distinct colors (only among assigned positions)
        assigned_colors = []
        if ca is not None:
            assigned_colors.append(ca)
        if cb is not None:
            assigned_colors.append(cb)
        if cc is not None:
            assigned_colors.append(cc)
        if cd is not None:
            assigned_colors.append(cd)
        
        distinct = len(set(assigned_colors))
        return distinct >= 3
    
    def forward_check(pos, color, colors):
        """Check if assigning color to pos creates any immediate violation."""
        for ap_idx in ap_indices[pos]:
            a, b, c, d = aps[ap_idx]
            # Check if this AP now has < 3 distinct colors among assigned positions
            ca = colors[a]
            cb = colors[b]
            cc = colors[c]
            cd = colors[d]
            
            # Temporarily assign color to pos
            if a == pos:
                ca = color
            elif b == pos:
                cb = color
            elif c == pos:
                cc = color
            elif d == pos:
                cd = color
            
            # Count distinct colors among assigned
            assigned = []
            if ca is not None:
                assigned.append(ca)
            if cb is not None:
                assigned.append(cb)
            if cc is not None:
                assigned.append(cc)
            if cd is not None:
                assigned.append(cd)
            
            if len(set(assigned)) < 3:
                return False
        
        return True
    
    def backtrack(k, colors, max_colors):
        """
        Backtracking search with forward checking.
        k: number of positions assigned (in order)
        colors: list of color assignments (None for unassigned)
        max_colors: maximum number of colors to use
        Returns: True if valid coloring found, False otherwise
        """
        if time.time() > deadline:
            return False
        
        if k == N:
            return True
        
        pos = order[k]
        
        # Try each color from 0 to max_colors-1
        for color in range(max_colors):
            # Forward check: does this assignment create an immediate violation?
            if not forward_check(pos, color, colors):
                continue
            
            colors[pos] = color
            
            # Recurse
            if backtrack(k + 1, colors, max_colors):
                return True
            
            colors[pos] = None
        
        return False
    
    def solve_with_k(max_colors, time_limit):
        """Try to find a valid coloring using at most max_colors colors."""
        end_time = time.time() + time_limit
        colors = [None] * N
        
        # Try multiple random orderings if initial fails
        for attempt in range(50):
            if time.time() > end_time:
                break
            
            # Shuffle the order slightly for diversity
            if attempt == 0:
                current_order = order[:]
            else:
                current_order = order[:]
                # Randomly swap some elements
                for _ in range(min(10, len(current_order) // 10)):
                    i, j = np.random.randint(0, N, 2)
                    current_order[i], current_order[j] = current_order[j], current_order[i]
            
            # Use a separate colors array for this attempt
            colors = [None] * N
            
            # Modified backtracking with custom order
            def bt(k, cols):
                if time.time() > end_time:
                    return False
                
                if k == N:
                    return True
                
                pos = current_order[k]
                
                for color in range(max_colors):
                    # Forward check
                    if not forward_check(pos, color, cols):
                        continue
                    
                    cols[pos] = color
                    
                    if bt(k + 1, cols):
                        return True
                    
                    cols[pos] = None
                
                return False
            
            if bt(0, colors):
                # Compress colors
                unique_colors = sorted(set(c for c in colors if c is not None))
                if len(unique_colors) < max_colors:
                    mapping = {c: i for i, c in enumerate(unique_colors)}
                    compressed = [mapping[c] for c in colors]
                    return compressed
                return colors
        
        return None
    
    # Try k from 3 upwards
    # Allocate time: 20s for k=3, 20s for k=4, 20s for k=5, 10s for k=6
    time_alloc = {3: 20, 4: 20, 5: 20, 6: 10}
    
    for k in [3, 4, 5, 6]:
        if time.time() > deadline - 5:
            break
        
        result = solve_with_k(k, time_alloc.get(k, 10))
        if result is not None:
            return result
    
    # Fallback: simple greedy
    def greedy_fallback():
        colors = []
        for x in range(1, N + 1):
            for color in range(128):
                valid = True
                
                # Check all APs where x is the last term
                for d in range(1, (x - 1) // 3 + 1):
                    c1 = colors[x - 1 - 3 * d]
                    c2 = colors[x - 1 - 2 * d]
                    c3 = colors[x - 1 - d]
                    if len({color, c1, c2, c3}) < 3:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Check all APs where x is the 3rd term
                for d in range(1, (x - 1) // 2 + 1):
                    c1 = colors[x - 1 - 2 * d]
                    c2 = colors[x - 1 - d]
                    if color == c1 and color == c2:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                colors.append(color)
                break
            else:
                colors.append(0)
        
        return colors
    
    return greedy_fallback()
