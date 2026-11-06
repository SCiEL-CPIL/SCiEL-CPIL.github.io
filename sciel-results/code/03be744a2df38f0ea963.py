import numpy as np
import time

def construct():
    N = 128
    # Precompute 4-APs
    aps = []
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    aps = np.array(aps)
    
    def get_cost(colors):
        # colors is numpy array of shape (N,)
        ap_colors = colors[aps]
        # Sort each row to check for <= 2 distinct colors
        s = np.sort(ap_colors, axis=1)
        # Bad if:
        # 1 color: s[:,0] == s[:,3]
        # 2 colors: s[:,0] == s[:,2] (3+1 or 4) OR s[:,1] == s[:,3] (1+3 or 4)
        # Note: 2+2 case: a,a,b,b -> s[0]=a, s[1]=a, s[2]=b, s[3]=b.
        # s[0]!=s[2], s[1]!=s[3]. So 2+2 is NOT caught by the above?
        # Wait. If s is a,a,b,b:
        # s[0]=a, s[2]=b. a!=b.
        # s[1]=a, s[3]=b. a!=b.
        # So my condition misses 2+2!
        # Let's re-evaluate.
        # Distinct < 3 means distinct is 1 or 2.
        # If distinct is 2, the counts are (2,2), (3,1), or (1,3).
        # (3,1) sorted: a,a,a,b -> s[0]==s[2] (a==a). Caught.
        # (1,3) sorted: a,b,b,b -> s[1]==s[3] (b==b). Caught.
        # (2,2) sorted: a,a,b,b -> s[0]==s[1] and s[2]==s[3].
        # How to catch (2,2)?
        # If s[0]==s[1] and s[2]==s[3] and s[1]!=s[2].
        # So bad if:
        # (s[:,0] == s[:,3])  # 4
        # or (s[:,0] == s[:,2]) # 3+1
        # or (s[:,1] == s[:,3]) # 1+3
        # or ((s[:,0] == s[:,1]) & (s[:,2] == s[:,3]) & (s[:,1] != s[:,2])) # 2+2
        
        bad = (s[:, 0] == s[:, 3]) | \
              (s[:, 0] == s[:, 2]) | \
              (s[:, 1] == s[:, 3]) | \
              ((s[:, 0] == s[:, 1]) & (s[:, 2] == s[:, 3]) & (s[:, 1] != s[:, 2]))
        return np.sum(bad)

    # Search 3-coloring
    best_cost_3 = float('inf')
    best_colors_3 = None
    start_time = time.time()
    deadline_3 = start_time + 30
    
    # Initialize
    colors = np.random.randint(0, 3, N)
    cost = get_cost(colors)
    if cost < best_cost_3:
        best_cost_3 = cost
        best_colors_3 = colors.copy()
    
    while time.time() < deadline_3:
        if best_cost_3 == 0:
            return best_colors_3.tolist()
        
        i = np.random.randint(0, N)
        new_color = np.random.randint(0, 3)
        if colors[i] == new_color:
            continue
        
        old_color = colors[i]
        colors[i] = new_color
        new_cost = get_cost(colors)
        
        if new_cost < cost:
            cost = new_cost
            if cost < best_cost_3:
                best_cost_3 = cost
                best_colors_3 = colors.copy()
        else:
            # Revert
            colors[i] = old_color

    # Search 4-coloring if 3 failed
    best_cost_4 = float('inf')
    best_colors_4 = None
    start_time_4 = time.time()
    deadline_4 = start_time_4 + 25
    
    colors = np.random.randint(0, 4, N)
    cost = get_cost(colors)
    if cost < best_cost_4:
        best_cost_4 = cost
        best_colors_4 = colors.copy()
        
    while time.time() < deadline_4:
        if best_cost_4 == 0:
            return best_colors_4.tolist()
            
        i = np.random.randint(0, N)
        new_color = np.random.randint(0, 4)
        if colors[i] == new_color:
            continue
            
        old_color = colors[i]
        colors[i] = new_color
        new_cost = get_cost(colors)
        
        if new_cost < cost:
            cost = new_cost
            if cost < best_cost_4:
                best_cost_4 = cost
                best_colors_4 = colors.copy()
        else:
            colors[i] = old_color

    # Return best valid found
    if best_cost_3 == 0:
        return best_colors_3.tolist()
    if best_cost_4 == 0:
        return best_colors_4.tolist()
    
    # Fallback: Return best 4-coloring (likely invalid) or best 3-coloring
    # To maximize chance of non-zero score, return the one with lower cost?
    # No, invalid is 0.
    # If neither is valid, we are doomed.
    # But 4-coloring should be valid.
    return best_colors_4.tolist()
