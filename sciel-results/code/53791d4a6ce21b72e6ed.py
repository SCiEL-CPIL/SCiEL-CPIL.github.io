def construct():
    n = 128
    colors = [-1] * (n + 1)
    
    # Precompute 4-AP endings? 
    # For each x, we need to check d such that x-3d >= 1.
    # Also check 3-APs ending at x-2d and x-d to prevent monochromatic 3-APs.
    
    for x in range(1, n + 1):
        forbidden = 0
        
        # 1. Constraints from 4-APs ending at x
        # AP: (x-3d, x-2d, x-d, x)
        # If |{c1, c2, c3}| == 2, c4 cannot be in {c1, c2, c3}
        # If |{c1, c2, c3}| == 1, invalid state (should not happen)
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            a = x - 3 * d
            c1 = colors[a]
            c2 = colors[a + d]
            c3 = colors[a + 2 * d]
            
            # Check if this 3-AP was monochromatic (error in previous steps)
            if c1 == c2 == c3:
                # This state is unrecoverable for this 4-AP.
                # But we should have prevented it.
                pass
            
            s = {c1, c2, c3}
            if len(s) == 2:
                for c in s:
                    forbidden |= (1 << c)
            elif len(s) == 1:
                # If we are here, it means we made a mistake previously.
                # However, if we prevented monochromatic 3-APs, this shouldn't happen.
                # If it does, we forbid that color to try to break it? No, it's already broken.
                # But wait, if c1=c2=c3=k, then c4 must be != k to get size 2.
                # But we need size >= 3. So it's impossible.
                # So we must ensure this case never occurs.
                pass

        # 2. Constraints to prevent creating monochromatic 3-APs
        # A 3-AP (u, v, w) is a prefix of a 4-AP (u, v, w, z) if w + (v-u) <= n.
        # If we set c_w, and c_u == c_v, then c_w cannot be c_u.
        # Here x plays the role of w.
        # So for any d such that x-2d >= 1, we have AP (x-2d, x-d, x).
        # If colors[x-2d] == colors[x-d], we must forbid colors[x-2d].
        # Also, we must ensure that this 3-AP is actually part of a 4-AP within [1, n].
        # The 4-AP would be (x-2d, x-d, x, x+d).
        # This requires x+d <= n.
        
        for d in range(1, x // 2 + 1):
            a = x - 2 * d
            b = x - d
            if a < 1:
                break
            # Check if this 3-AP can be extended to a 4-AP
            if x + d <= n:
                if colors[a] == colors[b]:
                    forbidden |= (1 << colors[a])
            
            # What if x is the middle element?
            # AP (x-d, x, x+d).
            # If we set c_x, and later c_{x+d} is forced to match c_{x-d}, we fail.
            # This is a future constraint. Greedy might fail here.
            # However, the problem asks to minimize colors.
            # Using the smallest available color is a standard heuristic.
            
            # What if x is the first element?
            # AP (x, x+d, x+2d).
            # If c_x == c_{x+d}, then c_{x+2d} cannot be c_x.
            # This is also a future constraint.

        # Select the smallest color not in forbidden
        chosen = -1
        for c in range(128):
            if not (forbidden & (1 << c)):
                chosen = c
                break
        
        if chosen == -1:
            # No valid color found.
            # This implies the greedy choice led to a dead end.
            # For a robust solution, we would need backtracking.
            # However, the directive suggests a specific greedy approach.
            # Let's assume the "prevent monochromatic 3-AP" heuristic is strong enough.
            raise Exception("No valid color found at position %d" % x)
        
        colors[x] = chosen
        
    return colors[1:]
