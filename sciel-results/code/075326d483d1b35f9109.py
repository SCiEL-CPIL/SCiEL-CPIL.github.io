import time
import random

def construct():
    N = 128
    aps = []
    ap_indices = [[] for _ in range(N)]
    
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            idx_tuple = (a - 1, b - 1, c - 1, e - 1)
            aps.append(idx_tuple)
            for idx in idx_tuple:
                ap_indices[idx].append(len(aps) - 1)
                
    def is_bad_ap_colors(c0, c1, c2, c3):
        # Returns 1 if bad, 0 if good
        if c0 == c1 == c2 == c3:
            return 1
        # Check 2 colors
        # If not all same, count distinct
        # Optimization:
        # If c0 == c1 and c2 == c3 and c0 != c2 -> 2 colors
        # If c0 == c2 and c1 == c3 and c0 != c1 -> 2 colors
        # If c0 == c3 and c1 == c2 and c0 != c1 -> 2 colors
        # If 3 same, 1 diff -> 2 colors
        # Using set is O(1) effectively and fast enough in Python for 2700 items
        return 1 if len({c0, c1, c2, c3}) < 3 else 0

    def get_initial_cost(colors):
        cost = 0
        for ap in aps:
            cost += is_bad_ap_colors(colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]])
        return cost

    def solve(k, time_limit):
        start_time = time.time()
        best_cost = float('inf')
        best_colors = None
        
        while time.time() - start_time < time_limit:
            colors = [random.randint(0, k - 1) for _ in range(N)]
            current_cost = get_initial_cost(colors)
            
            if current_cost == 0:
                return colors
            
            stuck = 0
            while time.time() - start_time < time_limit:
                i = random.randint(0, N - 1)
                old_c = colors[i]
                
                best_delta = 0
                best_new_c = -1
                
                for new_c in range(k):
                    if new_c == old_c:
                        continue
                    delta = 0
                    for ap_idx in ap_indices[i]:
                        ap = aps[ap_idx]
                        c0 = colors[ap[0]]
                        c1 = colors[ap[1]]
                        c2 = colors[ap[2]]
                        c3 = colors[ap[3]]
                        
                        old_bad = is_bad_ap_colors(c0, c1, c2, c3)
                        
                        if ap[0] == i: c0 = new_c
                        elif ap[1] == i: c1 = new_c
                        elif ap[2] == i: c2 = new_c
                        elif ap[3] == i: c3 = new_c
                        
                        new_bad = is_bad_ap_colors(c0, c1, c2, c3)
                        delta += new_bad - old_bad
                        
                    if delta < best_delta:
                        best_delta = delta
                        best_new_c = new_c
                        
                if best_new_c != -1 and best_delta < 0:
                    colors[i] = best_new_c
                    current_cost += best_delta
                    stuck = 0
                    if current_cost == 0:
                        return colors
                else:
                    stuck += 1
                    if stuck > 500:
                        break
                        
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors[:]
                
        return best_colors

    # Try k=3
    res = solve(3, 45)
    if res:
        if get_initial_cost(res) == 0:
            return res
            
    # Try k=4
    res = solve(4, 12)
    if res:
        if get_initial_cost(res) == 0:
            return res
            
    # Try k=5
    res = solve(5, 3)
    if res:
        if get_initial_cost(res) == 0:
            return res
            
    # Fallback
    return res if res else [0]*N
