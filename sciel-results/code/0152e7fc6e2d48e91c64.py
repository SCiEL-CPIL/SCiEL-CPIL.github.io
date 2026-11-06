import random
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for overhead/return

    def prepare_data_numpy(n):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Red K4s: all 4-subsets
        red_k4s = []
        for quad in combinations(range(n), 4):
            edges = [triple_to_idx[tri] for tri in combinations(quad, 3)]
            red_k4s.append(edges)
            
        # Blue K5s: all 5-subsets
        blue_k5s = []
        for quint in combinations(range(n), 5):
            edges = [triple_to_idx[tri] for tri in combinations(quint, 3)]
            blue_k5s.append(edges)
            
        num_k4s = len(red_k4s)
        num_k5s = len(blue_k5s)
        
        # Incidence matrices: M_red[k4_idx, edge_idx] = 1 if edge in K4
        M_red = np.zeros((num_k4s, num_triples), dtype=np.int8)
        for i, edges in enumerate(red_k4s):
            for e in edges:
                M_red[i, e] = 1
                
        M_blue = np.zeros((num_k5s, num_triples), dtype=np.int8)
        for i, edges in enumerate(blue_k5s):
            for e in edges:
                M_blue[i, e] = 1
        
        return triples, red_k4s, blue_k5s, M_red, M_blue

    def simulated_annealing(n, seed_val=0, time_limit=10.0):
        rng = random.Random(seed_val)
        triples, red_k4s, blue_k5s, M_red, M_blue = prepare_data_numpy(n)
        num_triples = len(triples)
        
        # Initial coloring: biased towards Red to avoid Blue K5s? 
        # P(Red K4) = p^4, P(Blue K5) = (1-p)^10.
        # For n=20: K4s=4845, K5s=15504.
        # If p=0.5: Exp Violations = 4845/16 + 15504/1024 ~ 300 + 15 = 315.
        # If p=0.6: P(R)=0.1296, P(B)=0.0107. Exp = 4845*0.13 + 15504*0.01 ~ 630 + 166 = 796. Worse.
        # If p=0.4: P(R)=0.0256, P(B)=0.0605. Exp = 4845*0.025 + 15504*0.06 ~ 123 + 930 = 1053. Worse.
        # p=0.5 seems best for n=20. For larger n, K5s grow faster?
        # C(n,4) ~ n^4/24, C(n,5) ~ n^5/120.
        # Ratio K5/K4 ~ n/5. For n=20, ratio is 4.
        # So Blue violations are more numerous in count, but probability is (1-p)^10 vs p^4.
        # (1-p)^10 drops off very fast.
        # Let's stick with p=0.5 or slightly biased towards Red (0.6) if Blue is easier to break?
        # Actually, breaking a Blue K5 requires changing 1 edge. Breaking Red K4 also 1 edge.
        # Let's use p=0.5.
        
        p_red = 0.5
        colors = np.array([1 if rng.random() > p_red else 0 for _ in range(num_triples)], dtype=np.int8)
        
        # Calculate initial violations
        # Red K4 is violation if all 4 edges are 0.
        # Sum M_red * (1-colors) along axis 1. If sum == 4, violation.
        # Actually, count of red edges in K4: sum(M_red * colors, axis=1). Violation if == 4.
        red_counts = M_red.dot(colors)
        blue_counts = M_blue.dot(1 - colors) # Count of blue edges (1-color)
        
        red_violations = np.sum(red_counts == 4)
        blue_violations = np.sum(blue_counts == 10)
        total_violations = red_violations + blue_violations
        
        if total_violations == 0:
            return colors.tolist()
            
        start = time.time()
        T = 1.0
        cooling = 0.995
        best_colors = colors.copy()
        best_violations = total_violations
        
        # Precompute neighbors for fast delta?
        # Delta for flipping edge e from c to 1-c:
        # For each K4 containing e:
        #   if c==0 (was red): red_count decreases by 1. If it was 4, violation -1. If it becomes 3, no change (unless we count "near").
        #   if c==1 (was blue): red_count increases by 1. If it was 3, violation +1.
        # Similarly for Blue K5.
        
        # To speed up, we can just update counts.
        while time.time() - start < time_limit:
            if total_violations == 0:
                return colors.tolist()
                
            if total_violations < best_violations:
                best_violations = total_violations
                best_colors = colors.copy()
                
            # Pick a random edge
            e = rng.randint(0, num_triples - 1)
            current_color = colors[e]
            new_color = 1 - current_color
            
            delta_viol = 0
            
            # Update Red K4s
            # Indices of K4s containing e
            k4_indices = np.where(M_red[:, e] == 1)[0]
            for idx in k4_indices:
                cnt = red_counts[idx]
                if current_color == 0:
                    # Removing a red edge
                    if cnt == 4:
                        delta_viol -= 1
                    # if cnt == 3, it becomes 2, no violation change
                else:
                    # Adding a red edge
                    if cnt == 3:
                        delta_viol += 1
                    # if cnt == 4, impossible since current_color was 1 (blue), so cnt < 4? 
                    # Wait, if current_color is 1, red_count does not include this edge.
                    # So max red_count is 3 (other 3 edges). If we add this, it becomes 4.
                    # So yes, if cnt == 3, it becomes a violation.
                    
            # Update Blue K5s
            k5_indices = np.where(M_blue[:, e] == 1)[0]
            for idx in k5_indices:
                cnt = blue_counts[idx]
                if current_color == 1:
                    # Removing a blue edge
                    if cnt == 10:
                        delta_viol -= 1
                else:
                    # Adding a blue edge
                    if cnt == 9:
                        delta_viol += 1
            
            # Acceptance
            if delta_viol < 0:
                accept = True
            elif delta_viol == 0:
                accept = True # Or random
            else:
                if T > 0:
                    accept = rng.random() < np.exp(-delta_viol / T)
                else:
                    accept = False
            
            if accept:
                colors[e] = new_color
                # Update counts
                for idx in k4_indices:
                    if current_color == 0:
                        red_counts[idx] -= 1
                    else:
                        red_counts[idx] += 1
                for idx in k5_indices:
                    if current_color == 1:
                        blue_counts[idx] -= 1
                    else:
                        blue_counts[idx] += 1
                total_violations += delta_viol
            else:
                # In SA, we usually don't update state on reject, but we might want to explore neighbors.
                # Sticking to standard SA: don't change state.
                pass
                
            T *= cooling
            
        # Return best found
        if best_violations == 0:
            return best_colors.tolist()
        return None

    best_n = 9
    # Fallback small valid coloring
    # For n=9, we can try to find one quickly or use a known one if available.
    # Let's just try SA for n=9 first to get a baseline.
    
    for n in range(9, 35):
        if time.time() > deadline - 5:
            break
            
        remaining = deadline - time.time() - 2
        if remaining < 1:
            break
            
        found = False
        # Allocate time based on remaining time and n
        # Larger n is harder, but we want to maximize n.
        # Try a few seeds for current n.
        
        # Estimate time per attempt
        # n=20 has ~1140 triples. SA steps can be fast.
        # Let's allow 2 attempts for small n, 1 for large.
        max_attempts = 3 if n < 15 else 2
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 2:
                break
                
            time_limit = min(remaining - 2, 10.0)
            colors = simulated_annealing(n, seed_val=attempt * 12345 + n, time_limit=time_limit)
            
            if colors is not None:
                best_n = n
                best_colors = colors
                found = True
                break
        
        if not found:
            # If we fail for n, we might fail for n+1 too, but let's try if time permits
            # Actually, if we fail for n, it's likely n is too big for the time budget.
            # But maybe a different seed works?
            # If we spent all time and failed, break.
            if time.time() > deadline - 10:
                break
            # Otherwise, try next n? No, if we can't solve n, n+1 is harder.
            # Break to save time? Or just return current best.
            # Let's break to ensure we return within time.
            break
            
    return {'n': best_n, 'colors': best_colors}
