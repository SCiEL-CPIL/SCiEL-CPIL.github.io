import time
import random
import sys

def construct():
    start = time.time()
    deadline = start + 50  # safety margin before 90s kill
    
    n = 128
    positions = list(range(n))  # 0-indexed internally, map to 1..128
    
    # Precompute all nonconstant 4-APs in 1..128
    aps = []
    for a in range(1, n+1):
        for d in range(1, n):
            if a + 3*d > n:
                break
            aps.append((a-1, (a+d)-1, (a+2*d)-1, (a+3*d)-1))  # 0-indexed
    num_aps = len(aps)
    
    # For each position, list of AP indices
    pos_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for p in ap:
            pos_aps[p].append(idx)
    
    # For each AP, the 4 positions
    ap_positions = [list(ap) for ap in aps]
    
    def count_colors(colors, ap_idx):
        """Count distinct colors in an AP."""
        ap = ap_positions[ap_idx]
        return len(set(colors[p] for p in ap))
    
    def init_colors_structured(k):
        """
        Initialize with a modular pattern that tends to spread colors across APs.
        Try several patterns and pick the one with fewest violations.
        """
        best_colors = None
        best_violations = float('inf')
        
        for pattern_seed in range(30):
            rng = random.Random(pattern_seed * 12345 + 7)
            # Pattern: color[i] = (i * a + b) % k for random a,b, or hash-based
            a = rng.randint(1, max(1, k-1)) if k > 1 else 0
            b = rng.randint(0, k-1)
            colors = []
            for i in range(n):
                # Mix: use a hash-like function for better distribution
                h = (i * a + b * (i+1) * 7 + i*i * 3) 
                colors.append(h % k)
            
            # Count violations
            violations = 0
            for idx in range(num_aps):
                ap = ap_positions[idx]
                if len(set(colors[p] for p in ap)) < 3:
                    violations += 1
            
            if violations < best_violations:
                best_violations = violations
                best_colors = colors[:]
            if best_violations == 0:
                break
        
        return best_colors, best_violations
    
    def init_colors_greedy(k, seed=0):
        """Greedy: assign each position the color that minimizes new violations."""
        rng = random.Random(seed)
        colors = [-1] * n
        violations = [0] * num_aps  # whether AP is violated (given partial coloring)
        # For partial coloring, an AP is "potentially violated" if all its assigned 
        # positions use < 3 colors AND unassigned positions can't fix it.
        # Simplified: just count violations among fully-assigned APs at the end.
        
        for p in range(n):
            best_c = 0
            best_v = float('inf')
            for c in range(k):
                colors[p] = c
                # Count violations for APs containing p
                v = 0
                for ap_idx in pos_aps[p]:
                    ap = ap_positions[ap_idx]
                    all_assigned = all(colors[q] >= 0 for q in ap)
                    if all_assigned:
                        if len(set(colors[q] for q in ap)) < 3:
                            v += 1
                if v < best_v:
                    best_v = v
                    best_c = c
                if best_v == 0:
                    colors[p] = c
                    break
            colors[p] = best_c
        
        # Count total violations
        total_v = 0
        for idx in range(num_aps):
            ap = ap_positions[idx]
            if len(set(colors[q] for q in ap)) < 3:
                total_v += 1
        return colors, total_v
    
    def init_colors_random(k, seed=0):
        rng = random.Random(seed)
        colors = [rng.randint(0, k-1) for _ in range(n)]
        v = sum(1 for idx in range(num_aps) if len(set(colors[p] for p in ap_positions[idx])) < 3)
        return colors, v
    
    def anneal(colors, k, time_budget, seed=0):
        """Simulated annealing with pairwise recoloring moves."""
        rng = random.Random(seed)
        
        # Precompute current violation status
        violated = [False] * num_aps
        for idx in range(num_aps):
            violated[idx] = len(set(colors[p] for p in ap_positions[idx])) < 3
        energy = sum(violated)
        
        if energy == 0:
            return colors, energy
        
        t = max(time.time(), start + 0.1)
        end = start + time_budget
        temp = 1.0
        cool_rate = 0.9999
        
        best_colors = colors[:]
        best_energy = energy
        
        while t < end:
            # Pick a random violated AP
            violated_indices = [i for i in range(num_aps) if violated[i]]
            if not violated_indices:
                break
            
            ap_idx = rng.choice(violated_indices)
            ap = ap_positions[ap_idx]
            
            # Try: recolor one position in this AP to a random different color
            pos = rng.choice(ap)
            old_c = colors[pos]
            new_c = rng.randint(0, k-1)
            if new_c == old_c:
                new_c = (new_c + 1) % k
            
            colors[pos] = new_c
            
            # Compute delta energy for APs containing pos
            delta = 0
            for ap_i in pos_aps[pos]:
                was_v = violated[ap_i]
                now_v = len(set(colors[q] for q in ap_positions[ap_i])) < 3
                violated[ap_i] = now_v
                if now_v and not was_v:
                    delta += 1
                elif not now_v and was_v:
                    delta -= 1
            
            new_energy = energy + delta
            
            if new_energy <= energy:
                energy = new_energy
                if energy < best_energy:
                    best_energy = energy
                    best_colors = colors[:]
                if energy == 0:
                    break
            else:
                # Accept with probability exp(-delta/temp)
                import math
                if temp > 1e-6:
                    accept_prob = math.exp(-delta / temp)
                    if rng.random() < accept_prob:
                        energy = new_energy
                    else:
                        # Revert
                        colors[pos] = old_c
                        for ap_i in pos_aps[pos]:
                            violated[ap_i] = len(set(colors[q] for q in ap_positions[ap_i])) < 3
                        energy = best_energy  # revert energy too? No, we need to track properly
                        # Actually, let's just recompute from violated
                        energy = sum(violated)
                        best_energy = min(best_energy, energy)
                        if energy < best_energy:
                            best_energy = energy
                            best_colors = colors[:]
                else:
                    # Revert
                    colors[pos] = old_c
                    for ap_i in pos_aps[pos]:
                        violated[ap_i] = len(set(colors[q] for q in ap_positions[ap_i])) < 3
                    energy = sum(violated)
            
            temp *= cool_rate
            t = time.time()
        
        return best_colors, best_energy
    
    def try_merge_colors(colors, k):
        """
        Try to merge two color classes to reduce the number of distinct colors.
        Returns (merged_colors, new_k) or (None, k) if no merge is possible.
        """
        distinct = sorted(set(colors))
        if len(distinct) < 2:
            return colors, len(distinct)
        
        # Try all pairs of colors to merge
        for i in range(len(distinct)):
            for j in range(i+1, len(distinct)):
                c1, c2 = distinct[i], distinct[j]
                # Create merged coloring: c2 -> c1
                merged = [c1 if c == c2 else c for c in colors]
                # Check validity
                valid = True
                for idx in range(num_aps):
                    ap = ap_positions[idx]
                    if len(set(merged[p] for p in ap)) < 3:
                        valid = False
                        break
                if valid:
                    return merged, len(distinct) - 1
        
        return None, len(distinct)
    
    # Strategy:
    # 1. Start with k=3 (minimum possible)
    # 2. If fails, try k=4, k=5, etc.
    # 3. After finding valid, try to merge colors to reduce count
    
    best_result = None
    best_k_used = 999
    
    for k in [3, 4, 5, 6]:
        if time.time() > deadline:
            break
        
        # Try multiple initialization strategies
        candidates = []
        
        # Structured init
        cols, viol = init_colors_structured(k)
        candidates.append((cols, viol, 'structured'))
        
        # Greedy init
        cols, viol = init_colors_greedy(k, seed=42)
        candidates.append((cols, viol, 'greedy'))
        
        # A few random inits
        for s in range(5):
            cols, viol = init_colors_random(k, seed=s)
            candidates.append((cols, viol, 'random'))
        
        # Pick the best candidate (fewest violations)
        candidates.sort(key=lambda x: x[1])
        best_cols, best_viol, best_name = candidates[0]
        
        # Run annealing
        alloc = min(10, max(3, (deadline - (time.time() - start)) / k))
        cols, viol = anneal(best_cols, k, alloc, seed=123)
        
        if viol == 0:
            # Found valid coloring with k colors
            # Try to merge
            merged, new_k = try_merge_colors(cols, k)
            if merged is not None:
                best_result = merged
                best_k_used = new_k
            else:
                best_result = cols
                best_k_used = k
            # Continue trying higher k? No, we want minimum. But merging might reduce below k.
            # Actually, we should try to merge as much as possible.
            while True:
                merged, new_k = try_merge_colors(best_result, best_k_used)
                if merged is None:
                    break
                best_result = merged
                best_k_used = new_k
            break
        
        # If annealing didn't fix it, try a few more seeds
        for s in range(3):
            if time.time() > deadline:
                break
            cols_s, viol_s = anneal(best_cols, k, alloc, seed=100 + s)
            if viol_s == 0:
                merged, new_k = try_merge_colors(cols_s, k)
                if merged is not None:
                    best_result = merged
                    best_k_used = new_k
                else:
                    best_result = cols_s
                    best_k_used = k
                while True:
                    merged2, new_k2 = try_merge_colors(best_result, best_k_used)
                    if merged2 is None:
                        break
                    best_result = merged2
                    best_k_used = new_k2
                break
    
    if best_result is not None:
        # Final validation
        valid = True
        for idx in range(num_aps):
            ap = ap_positions[idx]
            if len(set(best_result[p] for p in ap)) < 3:
                valid = False
                break
        if valid:
            return best_result
    
    # Fallback: use a safe high-coloring
    # Color i = i % 10 (10 colors) should be safe
    colors = [i % 10 for i in range(n)]
    # Validate
    valid = True
    for idx in range(num_aps):
        ap = ap_positions[idx]
        if len(set(colors[p] for p in ap)) < 3:
            valid = False
            break
    if valid:
        return colors
    
    # Very safe fallback: all distinct colors
    return list(range(n))
