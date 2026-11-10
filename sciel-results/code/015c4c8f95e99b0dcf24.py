import time
import math
import random

N = 64

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    # Precompute coordinates
    xs = [i // N for i in range(N * N)]
    ys = [i % N for i in range(N * N)]
    
    all_points = list(range(N * N))
    
    def dist2(i, j):
        dx = xs[i] - xs[j]
        dy = ys[i] - ys[j]
        return dx * dx + dy * dy
    
    def solve_once(seed, time_limit):
        start = time.time()
        limit = start + time_limit
        
        rng = random.Random(seed)
        
        # Try multiple greedy strategies
        best_greedy = set()
        best_greedy_size = 0
        
        # Strategy 1: Random
        def run_greedy(order):
            selected = set()
            dist_sets = {}
            
            def can_add(p):
                if p in selected:
                    return False
                seen_from_p = set()
                for q in selected:
                    d = dist2(p, q)
                    if d in seen_from_p:
                        return False
                    if d in dist_sets[q]:
                        return False
                    seen_from_p.add(d)
                return True

            def add(p):
                selected.add(p)
                dist_sets[p] = set()
                for q in selected:
                    if q == p:
                        continue
                    d = dist2(p, q)
                    dist_sets[p].add(d)
                    dist_sets[q].add(d)

            for p in order:
                if time.time() > limit - 10: # Reserve time for local search
                    break
                if can_add(p):
                    add(p)
            return selected, dist_sets

        # Generate orders
        order_random = all_points.copy()
        rng.shuffle(order_random)
        
        # Spiral order (center out)
        center = N // 2
        points_by_dist = sorted(all_points, key=lambda i: (abs(xs[i]-center)**2 + abs(ys[i]-center)**2))
        order_spiral = points_by_dist
        
        # Corner order
        points_by_corner = sorted(all_points, key=lambda i: (min(xs[i], N-1-xs[i]), min(ys[i], N-1-ys[i])))
        order_corner = points_by_corner
        
        candidates = [
            ("random", order_random),
            ("spiral", order_spiral),
            ("corner", order_corner)
        ]
        
        for name, order in candidates:
            if time.time() > limit - 20:
                break
            sel, ds = run_greedy(order)
            if len(sel) > best_greedy_size:
                best_greedy_size = len(sel)
                best_greedy = sel
                best_dist_sets = ds
        
        selected = best_greedy
        dist_sets = best_dist_sets
        unselected = [p for p in all_points if p not in selected]
        
        if not selected:
            return []
            
        def remove(p):
            selected.discard(p)
            if p not in dist_sets:
                return
            for q in selected:
                if q == p:
                    continue
                d = dist2(p, q)
                if d in dist_sets[q]:
                    dist_sets[q].discard(d)
            del dist_sets[p]
            
        def add(p):
            selected.add(p)
            dist_sets[p] = set()
            for q in selected:
                if q == p:
                    continue
                d = dist2(p, q)
                dist_sets[p].add(d)
                dist_sets[q].add(d)

        def try_swap(p_out, p_in):
            if p_in in selected or p_out not in selected:
                return False
                
            seen_from_pin = set()
            for r in selected:
                if r == p_out:
                    continue
                d = dist2(p_in, r)
                if d in seen_from_pin:
                    return False
                seen_from_pin.add(d)
                
            for r in selected:
                if r == p_out:
                    continue
                d_pr = dist2(p_in, r)
                d_pout_r = dist2(p_out, r)
                
                if d_pr in dist_sets[r]:
                    if d_pout_r != d_pr:
                        return False
                    
            remove(p_out)
            add(p_in)
            return True

        # Local Search
        # Use a mix of random swaps and systematic attempts
        no_improvement_count = 0
        max_no_improvement = 500
        
        while time.time() < limit:
            if no_improvement_count > max_no_improvement:
                break
                
            improved = False
            
            # Random swap
            if unselected and selected:
                p_out = rng.choice(list(selected))
                p_in = rng.choice(unselected)
                if try_swap(p_out, p_in):
                    unselected.remove(p_in)
                    unselected.append(p_out)
                    improved = True
            
            # Systematic swap: try to replace a random p_out with best p_in
            if not improved and selected and unselected:
                p_out = rng.choice(list(selected))
                # Try a few random p_in's
                for _ in range(10):
                    if not unselected:
                        break
                    p_in = rng.choice(unselected)
                    if try_swap(p_out, p_in):
                        unselected.remove(p_in)
                        unselected.append(p_out)
                        improved = True
                        break
            
            if improved:
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
        return [(i // N, i % N) for i in selected]

    best_result = []
    best_size = 0
    
    seeds = [42, 123, 456, 789, 101, 202, 303, 404]
    
    for seed in seeds:
        if time.time() > deadline - 10:
            break
        
        # Estimate time per run
        # We have ~280s total. 8 seeds. ~35s each.
        # But greedy is fast. Local search is slow.
        # Let's give each run 30s.
        time_per_run = 30.0
        if time.time() + time_per_run > deadline:
            time_per_run = max(5.0, deadline - time.time())
            
        result = solve_once(seed, time_per_run)
        
        if len(result) > best_size:
            best_size = len(result)
            best_result = result
            
    return best_result

# EVOLVE-BLOCK-END
