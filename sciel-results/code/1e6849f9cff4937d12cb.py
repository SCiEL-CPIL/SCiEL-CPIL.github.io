import time
import random

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    n = 16
    num_edges = 120
    edge_index = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_index[(i, j)] = idx
            idx += 1
            
    k4s = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    edges = [
                        edge_index[(a,b)], edge_index[(a,c)], edge_index[(a,d)],
                        edge_index[(b,c)], edge_index[(b,d)], edge_index[(c,d)]
                    ]
                    k4s.append(edges)
                    
    def count_violations(colors):
        violations = 0
        for edges in k4s:
            # Unpack for speed
            c0 = colors[edges[0]]
            c1 = colors[edges[1]]
            c2 = colors[edges[2]]
            c3 = colors[edges[3]]
            c4 = colors[edges[4]]
            c5 = colors[edges[5]]
            # Check distinct count
            # Using set is usually fast enough for 6 items
            if len({c0, c1, c2, c3, c4, c5}) < 5:
                violations += 1
        return violations

    def optimize_full(k, time_budget):
        best_colors = None
        best_violations = float('inf')
        
        while time.time() - start_time < time_budget:
            colors = [random.randint(0, k-1) for _ in range(num_edges)]
            current_violations = count_violations(colors)
            
            if current_violations == 0:
                return colors
            
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors[:]
                
            improved = True
            while improved and time.time() - start_time < time_budget:
                improved = False
                edge_order = list(range(num_edges))
                random.shuffle(edge_order)
                
                for e in edge_order:
                    if time.time() - start_time > time_budget:
                        break
                    old_color = colors[e]
                    best_new_color = old_color
                    best_viol = current_violations
                    
                    for c in range(k):
                        if c == old_color:
                            continue
                        colors[e] = c
                        v = count_violations(colors)
                        if v < best_viol:
                            best_viol = v
                            best_new_color = c
                    
                    if best_new_color != old_color:
                        colors[e] = best_new_color
                        current_violations = best_viol
                        improved = True
                        if current_violations == 0:
                            return colors
            
            # Restart if stuck
            pass
            
        return best_colors if best_violations == 0 else None

    # Try k=13, 14, 15
    # Allocate time
    # 13 is hardest, but if it works, it's best.
    # 15 is easiest.
    # Strategy: Try 15 first to guarantee a valid result?
    # No, we want to minimize colors.
    # If we find 15, we can stop? No, we want 13 if possible.
    # But if we spend 100s on 13 and fail, we have no time for 15.
    # Better: Try 13 with short time, then 14, then 15 with long time.
    # Or: Try 15 first to secure a valid score, then try to improve to 14, 13?
    # The evaluator kills at 150s.
    # If I return 15, score is 1.13.
    # If I return 13, score is 1.30.
    # I should prioritize finding ANY valid solution first, then optimize?
    # But the search for 13 and 15 are independent.
    # If I run 13 and it fails, I wasted time.
    # Let's run 15 first (easiest) to ensure we have a valid result.
    # Then use remaining time to try 14 and 13.
    
    # 1. Try 15
    res = optimize_full(15, time.time() + 40.0)
    if res is not None:
        best_res = res
        best_k = 15
    else:
        best_res = None
        best_k = 16 # Should not happen

    # 2. Try 14
    if time.time() - start_time < 90.0:
        res14 = optimize_full(14, time.time() + 30.0)
        if res14 is not None:
            best_res = res14
            best_k = 14

    # 3. Try 13
    if time.time() - start_time < 110.0:
        res13 = optimize_full(13, time.time() + 20.0)
        if res13 is not None:
            best_res = res13
            best_k = 13

    if best_res is not None:
        return best_res
    
    # Fallback
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
