import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = 128
    
    # Precompute APs ending at each position for fast checking
    # aps_ending[i] = list of (a, d) such that a+3d = i (0-indexed)
    aps_ending = [[] for _ in range(n)]
    for i in range(n):
        for d in range(1, i // 3 + 1):
            a = i - 3 * d
            if a >= 0:
                aps_ending[i].append((a, d))
    
    # Precompute 3-term APs ending at each position for lookahead
    # threes_ending[i] = list of (a, d) such that a+2d = i (0-indexed)
    threes_ending = [[] for _ in range(n)]
    for i in range(n):
        for d in range(1, i // 2 + 1):
            a = i - 2 * d
            if a >= 0:
                threes_ending[i].append((a, d))
    
    def is_valid(colors):
        """Check if the coloring is valid for all 4-term APs."""
        for i in range(n):
            for a, d in aps_ending[i]:
                vals = {colors[a], colors[a+d], colors[a+2*d], colors[i]}
                if len(vals) < 3:
                    return False
        return True
    
    def greedy_with_lookahead(max_color, randomize=False):
        """
        Deterministic greedy: assign smallest valid color at each position.
        The lookahead constraint ensures that for any 3-term AP (a, a+d, a+2d) 
        ending at position i, if we assign color c to position i, then for the 
        4-term AP (a, a+d, a+2d, a+3d), the 4th position will have at least 
        3 distinct colors available.
        
        Actually, re-reading the seed: the second condition checks that for any 
        3-term AP ending at the previous position, the new color is different 
        from at least one of the two earlier terms. This prevents the 4th term 
        from being forced into a 2-color situation.
        """
        colors = [None] * n
        
        for i in range(n):
            if time.time() > deadline:
                return None
            
            # Find valid colors for position i
            valid_colors = []
            for c in range(max_color):
                # Check all 4-term APs ending at i
                ok_4 = True
                for a, d in aps_ending[i]:
                    vals = {colors[a], colors[a+d], colors[a+2*d], c}
                    if len(vals) < 3:
                        ok_4 = False
                        break
                if not ok_4:
                    continue
                
                # Lookahead: check 3-term APs where position i is the 3rd term
                # For AP (a, a+d, a+2d, a+3d) where a+2d = i, 
                # we need that c != colors[a] OR c != colors[a+d]
                # i.e., not (c == colors[a] AND c == colors[a+d])
                ok_3 = True
                for a, d in threes_ending[i]:
                    # The 3-term AP is (a, a+d, i) where i = a+2d
                    # For the 4-term AP (a, a+d, a+2d, a+3d), if c == colors[a] == colors[a+d],
                    # then the 4th position would need a 3rd distinct color, but might be forced
                    # Actually, the seed's condition is: any(colors[x-1-j*d] != color for j in (1,2))
                    # which means: colors[a] != c OR colors[a+d] != c
                    # i.e., NOT (colors[a] == c AND colors[a+d] == c)
                    if colors[a] == c and colors[a+d] == c:
                        ok_3 = False
                        break
                if not ok_3:
                    continue
                
                valid_colors.append(c)
            
            if not valid_colors:
                return None
            
            if randomize:
                colors[i] = random.choice(valid_colors)
            else:
                colors[i] = valid_colors[0]  # Smallest valid color
        
        # Final verification
        if is_valid(colors):
            return colors
        return None
    
    # Strategy 1: Deterministic greedy with lookahead (seed approach)
    result = greedy_with_lookahead(128, randomize=False)
    if result is not None:
        return result
    
    # Strategy 2: Try with smaller palettes using randomized greedy with lookahead
    for k in [3, 4, 5, 6]:
        if time.time() > deadline:
            break
        for attempt in range(5000):
            if time.time() > deadline:
                break
            result = greedy_with_lookahead(k, randomize=True)
            if result is not None:
                return result
    
    # Strategy 3: Pure randomized greedy without lookahead (fallback)
    for k in [3, 4, 5]:
        if time.time() > deadline:
            break
        for attempt in range(2000):
            if time.time() > deadline:
                break
            colors = [None] * n
            success = True
            for i in range(n):
                valid_colors = []
                for c in range(k):
                    ok = True
                    for a, d in aps_ending[i]:
                        vals = {colors[a], colors[a+d], colors[a+2*d], c}
                        if len(vals) < 3:
                            ok = False
                            break
                    if ok:
                        valid_colors.append(c)
                if not valid_colors:
                    success = False
                    break
                colors[i] = random.choice(valid_colors)
            if success and is_valid(colors):
                return colors
    
    # Final fallback: all distinct colors (always valid, score = 128)
    return list(range(n))
