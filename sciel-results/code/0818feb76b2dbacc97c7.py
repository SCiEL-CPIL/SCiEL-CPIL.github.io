import time
import random
import sys

def construct():
    n = 128
    deadline = time.time() + 85.0
    
    # Precompute all 4-term APs
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for i in range(0, n - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
    
    # For each position, list indices of APs it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    def count_violations(colors):
        v = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                v += 1
        return v
    
    def is_ap_valid(colors, ap):
        c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
        return len({c0, c1, c2, c3}) >= 3
    
    def greedy_construct(k, seed=42):
        """Greedy construction with backtracking-lite."""
        rng = random.Random(seed)
        colors = [-1] * n
        
        # Try to assign colors greedily, preferring smallest label
        for pos in range(n):
            # Find which colors would create violations
            forbidden = set()
            for ap_idx in pos_to_aps[pos]:
                ap = aps[ap_idx]
                other_positions = [p for p in ap if p != pos]
                # Check if all other positions are colored
                if all(colors[p] != -1 for p in other_positions):
                    for c in range(k):
                        test_colors = colors[:]
                        test_colors[pos] = c
                        if not is_ap_valid(test_colors, ap):
                            forbidden.add(c)
            
            # Pick the smallest color not forbidden
            for c in range(k):
                if c not in forbidden:
                    colors[pos] = c
                    break
            else:
                # All colors forbidden - pick random (will need repair)
                colors[pos] = rng.randint(0, k - 1)
        
        return colors
    
    def repair_local_search(colors, k, time_limit):
        """Local search to fix violations."""
        current_violations = count_violations(colors)
        
        if current_violations == 0:
            return colors
        
        best_colors = colors[:]
        best_violations = current_violations
        
        temp = 1.0
        cooling = 0.9995
        min_temp = 0.005
        
        while time.time() < time_limit:
            # Find all violating APs
            violating_aps = [ap for ap in aps if not is_ap_valid(colors, ap)]
            
            if not violating_aps:
                return colors
            
            # Pick a random violating AP
            ap = random.choice(violating_aps)
            
            # Try changing one of the positions in this AP
            positions = list(ap)
            random.shuffle(positions)
            
            best_change = None
            best_delta = 0
            
            for pos in positions:
                old_color = colors[pos]
                
                for new_color in range(k):
                    if new_color == old_color:
                        continue
                    
                    colors[pos] = new_color
                    new_violations = count_violations(colors)
                    delta = new_violations - current_violations
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_change = (pos, new_color)
                    elif delta == best_delta and random.random() < 0.3:
                        best_change = (pos, new_color)
                    
                    colors[pos] = old_color
            
            if best_change is not None:
                pos, new_color = best_change
                colors[pos] = new_color
                current_violations += best_delta
                
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
                
                if current_violations == 0:
                    return colors
            
            temp *= cooling
            if temp < min_temp:
                break
        
        return best_colors if best_violations == 0 else None
    
    def hybrid_search(k, time_limit):
        """Combine greedy construction with local repair."""
        best_result = None
        best_violations = float('inf')
        
        num_trials = min(20, max(3, int((time_limit - time.time()) / 1.0)))
        
        for trial in range(num_trials):
            if time.time() > time_limit:
                break
            
            seed = trial * 7 + 13
            colors = greedy_construct(k, seed=seed)
            violations = count_violations(colors)
            
            if violations == 0:
                return colors
            
            if violations < best_violations:
                best_violations = violations
                best_result = colors[:]
            
            # Try to repair
            remaining_time = time_limit - time.time()
            if remaining_time > 0.5:
                repaired = repair_local_search(colors[:], k, time.time() + remaining_time * 0.8)
                if repaired is not None:
                    return repaired
                elif count_violations(repaired) < best_violations:
                    best_violations = count_violations(repaired)
                    best_result = repaired[:]
        
        return best_result if best_violations == 0 else None
    
    # Try increasing k
    k_values = [3, 4, 5, 6, 8]
    
    for k in k_values:
        if time.time() > deadline - 5.0:
            break
        
        time_slice = 15.0 if k <= 4 else 10.0
        if time.time() + time_slice > deadline - 2.0:
            time_slice = max(1.0, deadline - 2.0 - time.time())
        
        result = hybrid_search(k, time.time() + time_slice)
        if result is not None:
            return result
    
    # Fallback: identity mapping
    return list(range(n))

