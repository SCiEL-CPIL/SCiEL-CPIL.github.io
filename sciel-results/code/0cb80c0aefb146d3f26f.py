"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def _triple_index(a, b, c, n):
    """Map triple (a,b,c) with a<b<c to its index in lexicographic order."""
    # Number of triples with first vertex < a
    # For first vertex i < a: C(n-1-i, 2)
    idx = 0
    for i in range(a):
        idx += (n - 1 - i) * (n - 2 - i) // 2
    # For first vertex a, second vertex j < b: (n-1-a-j)
    for j in range(a + 1, b):
        idx += (n - 1 - a - j)
    # For first vertex a, second vertex b, third vertex k < c
    idx += (c - b - 1)
    return idx

def _triple_at(idx, n):
    """Map index to triple (a,b,c) in lexicographic order."""
    # Find a: largest a such that sum_{i=0}^{a-1} C(n-1-i, 2) <= idx
    a = 0
    cum = 0
    while a < n - 2:
        next_cum = cum + (n - 1 - a) * (n - 2 - a) // 2
        if next_cum > idx:
            break
        cum = next_cum
        a += 1
    remaining = idx - cum
    # Find b: largest b > a such that sum_{j=a+1}^{b-1} (n-1-a-j) <= remaining
    b = a + 1
    cum2 = 0
    while b < n - 1:
        next_cum2 = cum2 + (n - 1 - a - b)
        if next_cum2 > remaining:
            break
        cum2 = next_cum2
        b += 1
    c = b + 1 + (remaining - cum2)
    return (a, b, c)

def _init_violations(colors, n):
    """Count red K4 violations and blue K5 violations.
    Returns (red_viol_count, blue_viol_count, red_viol_set, blue_viol_set)"""
    # Precompute for each pair of vertices, the list of third vertices
    # For red K4: count for each 4-set how many of its 4 triples are red
    # For blue K5: count for each 5-set how many of its 10 triples are blue
    
    # Use numpy for speed
    colors = np.array(colors)
    
    # Build triple index mapping
    triple_list = list(combinations(range(n), 3))
    idx_map = {}
    for i, (a, b, c) in enumerate(triple_list):
        idx_map[(a, b, c)] = i
    
    # Count red triples per 4-set
    red_4_count = np.zeros((n, n, n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Triple (i,j,k) is red?
                if colors[idx_map[(i,j,k)]] == 0:
                    # For each l > k, this contributes to 4-set (i,j,k,l)
                    for l in range(k+1, n):
                        red_4_count[i,j,k,l] += 1
                # Also need to check other triples in the 4-set
                # Actually, let's do it differently: for each 4-set, count all 4 triples
    
    # Better approach: for each 4-set, count red triples directly
    red_4_viol = set()
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    count = 0
                    if colors[idx_map[(i,j,k)]] == 0: count += 1
                    if colors[idx_map[(i,j,l)]] == 0: count += 1
                    if colors[idx_map[(i,k,l)]] == 0: count += 1
                    if colors[idx_map[(j,k,l)]] == 0: count += 1
                    if count == 4:
                        red_4_viol.add((i,j,k,l))
    
    # Count blue K5 violations
    blue_5_viol = set()
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    for m in range(l+1, n):
                        count = 0
                        # 10 triples in {i,j,k,l,m}
                        triples_5 = [(i,j,k),(i,j,l),(i,j,m),(i,k,l),(i,k,m),(i,l,m),(j,k,l),(j,k,m),(j,l,m),(k,l,m)]
                        for t in triples_5:
                            if colors[idx_map[t]] == 1:
                                count += 1
                        if count == 10:
                            blue_5_viol.add((i,j,k,l,m))
    
    return len(red_4_viol), len(blue_5_viol), red_4_viol, blue_5_viol

def _count_violations(colors, n):
    """Count total violations (red K4 + blue K5)."""
    colors_arr = np.array(colors)
    triple_list = list(combinations(range(n), 3))
    idx_map = {}
    for i, (a, b, c) in enumerate(triple_list):
        idx_map[(a, b, c)] = i
    
    total = 0
    # Red K4
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    count = 0
                    if colors_arr[idx_map[(i,j,k)]] == 0: count += 1
                    if colors_arr[idx_map[(i,j,l)]] == 0: count += 1
                    if colors_arr[idx_map[(i,k,l)]] == 0: count += 1
                    if colors_arr[idx_map[(j,k,l)]] == 0: count += 1
                    if count == 4:
                        total += 1
    # Blue K5
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    for m in range(l+1, n):
                        count = 0
                        triples_5 = [(i,j,k),(i,j,l),(i,j,m),(i,k,l),(i,k,m),(i,l,m),(j,k,l),(j,k,m),(j,l,m),(k,l,m)]
                        for t in triples_5:
                            if colors_arr[idx_map[t]] == 1:
                                count += 1
                        if count == 10:
                            total += 1
    return total

def _is_valid(colors, n):
    """Check if coloring is valid (no red K4, no blue K5)."""
    return _count_violations(colors, n) == 0

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_n = 0
    best_colors = None
    
    # Try increasing n
    for n in range(5, 35):
        if time.time() > deadline - 10:
            break
        
        # Initialize with random coloring
        num_triples = n * (n-1) * (n-2) // 6
        colors = np.random.randint(0, 2, size=num_triples)
        
        triple_list = list(combinations(range(n), 3))
        idx_map = {}
        for i, (a, b, c) in enumerate(triple_list):
            idx_map[(a, b, c)] = i
        
        # Precompute for each triple, which 4-sets and 5-sets it belongs to
        # For incremental update: when flipping triple t, update violation counts
        
        # Initialize violation counts
        # red_4_count[s] = number of red triples in 4-set s
        # blue_5_count[s] = number of blue triples in 5-set s
        
        # For speed, use the direct counting approach with local search
        
        initial_viol = _count_violations(colors, n)
        
        if initial_viol == 0:
            best_n = n
            best_colors = colors.tolist()
            continue
        
        # Local search: flip random triples to reduce violations
        improved = True
        max_iterations = 50000
        iter_count = 0
        
        while improved and iter_count < max_iterations:
            if time.time() > deadline - 5:
                break
            improved = False
            iter_count += 1
            
            # Try flipping a random triple
            t_idx = random.randint(0, num_triples - 1)
            old_color = colors[t_idx]
            new_color = 1 - old_color
            
            # Compute change in violations
            # Flipping triple t from old_color to new_color
            # This affects all 4-sets containing t (for red) and all 5-sets containing t (for blue)
            
            # Get the triple
            a, b, c = triple_list[t_idx]
            
            # Find all 4-sets containing this triple
            # 4-sets: (a,b,c,d) for d > c, (a,b,d,c) not valid since d must be between... 
            # Actually 4-sets containing {a,b,c}: choose one more vertex d from {0..n-1}\{a,b,c}
            # But the 4-set is sorted, so we need to enumerate all d not in {a,b,c}
            
            # For red K4: the 4-set has all 4 triples red
            # When we flip triple t:
            # - If old_color was red (0) and new is blue (1): red count decreases by 1 for each 4-set containing t
            # - If old_color was blue (1) and new is red (0): red count increases by 1
            
            delta_red = 0
            delta_blue = 0
            
            # 4-sets containing triple (a,b,c): add one vertex d not in {a,b,c}
            for d in range(n):
                if d == a or d == b or d == c:
                    continue
                # The 4-set is {a,b,c,d} sorted
                s = tuple(sorted([a, b, c, d]))
                # Count current red triples in this 4-set
                # The 4 triples are: the 3 from the triple + the 4th
                # Actually, the 4 triples in 4-set {w,x,y,z} (w<x<y<z) are:
                # (w,x,y), (w,x,z), (w,y,z), (x,y,z)
                w, x, y, z = s
                # Check if t is one of these
                # t = (a,b,c)
                red_count = 0
                if colors[idx_map[(w,x,y)]] == 0: red_count += 1
                if colors[idx_map[(w,x,z)]] == 0: red_count += 1
                if colors[idx_map[(w,y,z)]] == 0: red_count += 1
                if colors[idx_map[(x,y,z)]] == 0: red_count += 1
                
                if old_color == 0:  # was red, becoming blue
                    # red_count decreases by 1
                    if red_count == 4:
                        # Was a violation, now not
                        delta_red -= 1
                    elif red_count == 3:
                        # Was not violation (3 reds), now has 2 reds, still not
                        pass
                else:  # was blue, becoming red
                    # red_count increases by 1
                    if red_count == 3:
                        # Was not violation, now becomes violation
                        delta_red += 1
            
            # 5-sets containing triple (a,b,c): add two vertices d,e not in {a,b,c}
            other_vertices = [v for v in range(n) if v != a and v != b and v != c]
            for i_d in range(len(other_vertices)):
                for i_e in range(i_d + 1, len(other_vertices)):
                    d = other_vertices[i_d]
                    e = other_vertices[i_e]
                    s = tuple(sorted([a, b, c, d, e]))
                    w, x, y, z, v = s
                    # 10 triples in 5-set
                    triples_5 = [(w,x,y),(w,x,z),(w,x,v),(w,y,z),(w,y,v),(w,z,v),(x,y,z),(x,y,v),(x,z,v),(y,z,v)]
                    blue_count = 0
                    for t5 in triples_5:
                        if colors[idx_map[t5]] == 1:
                            blue_count += 1
                    
                    if old_color == 1:  # was blue, becoming red
                        # blue_count decreases by 1
                        if blue_count == 10:
                            delta_blue -= 1
                    else:  # was red, becoming blue
                        # blue_count increases by 1
                        if blue_count == 9:
                            delta_blue += 1
            
            total_delta = delta_red + delta_blue
            
            if total_delta < 0:
                colors[t_idx] = new_color
                improved = True
            
            if not improved:
                # Try a few more random flips
                for _ in range(100):
                    if time.time() > deadline - 5:
                        break
                    t_idx2 = random.randint(0, num_triples - 1)
                    old_c = colors[t_idx2]
                    new_c = 1 - old_c
                    a2, b2, c2 = triple_list[t_idx2]
                    
                    dr = 0
                    db = 0
                    for d in range(n):
                        if d == a2 or d == b2 or d == c2:
                            continue
                        s = tuple(sorted([a2, b2, c2, d]))
                        w, x, y, z = s
                        rc = 0
                        if colors[idx_map[(w,x,y)]] == 0: rc += 1
                        if colors[idx_map[(w,x,z)]] == 0: rc += 1
                        if colors[idx_map[(w,y,z)]] == 0: rc += 1
                        if colors[idx_map[(x,y,z)]] == 0: rc += 1
                        if old_c == 0:
                            if rc == 4: dr -= 1
                        else:
                            if rc == 3: dr += 1
                    
                    other_v = [v for v in range(n) if v != a2 and v != b2 and v != c2]
                    for i_d in range(len(other_v)):
                        for i_e in range(i_d+1, len(other_v)):
                            d2 = other_v[i_d]
                            e2 = other_v[i_e]
                            s2 = tuple(sorted([a2, b2, c2, d2, e2]))
                            w2, x2, y2, z2, v2 = s2
                            triples_5_2 = [(w2,x2,y2),(w2,x2,z2),(w2,x2,v2),(w2,y2,z2),(w2,y2,v2),(w2,z2,v2),(x2,y2,z2),(x2,y2,v2),(x2,z2,v2),(y2,z2,v2)]
                            bc = 0
                            for t5 in triples_5_2:
                                if colors[idx_map[t5]] == 1:
                                    bc += 1
                            if old_c == 1:
                                if bc == 10: db -= 1
                            else:
                                if bc == 9: db += 1
                    
                    if dr + db < 0:
                        colors[t_idx2] = new_c
                        improved = True
                        break
        
        # Check if we found a valid coloring
        final_viol = _count_violations(colors, n)
        if final_viol == 0:
            best_n = n
            best_colors = colors.tolist()
            # Try to increase n further
            continue
        elif best_n == 0:
            # Keep the best we have
            best_n = n
            best_colors = colors.tolist()
    
    if best_colors is None:
        # Fallback to simple construction
        n = 9
        best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
        best_n = n
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
