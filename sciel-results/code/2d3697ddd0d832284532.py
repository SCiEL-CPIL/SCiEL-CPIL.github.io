import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    N = 128
    
    # Precompute all non-constant 4-term APs (0-indexed)
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    aps = np.array(aps, dtype=np.int32)
    
    def count_violations(colors):
        c = colors[aps[:, 0]]
        c2 = colors[aps[:, 1]]
        c3 = colors[aps[:, 2]]
        c4 = colors[aps[:, 3]]
        arr = np.stack([c, c2, c3, c4], axis=1)
        sorted_arr = np.sort(arr, axis=1)
        diffs = np.diff(sorted_arr, axis=1)
        distinct_counts = 1 + (diffs != 0).sum(axis=1)
        return (distinct_counts < 3).sum()

    def is_valid(colors):
        return count_violations(colors) == 0

    # Try deterministic patterns
    for pattern in [3, 4, 5]:
        colors = np.array([i % pattern for i in range(N)])
        if is_valid(colors):
            return colors.tolist()

    # Min-Conflicts
    best_violations = float('inf')
    best_colors = None
    
    for num_colors in [3, 4, 5]:
        if time.time() - start_time > DEADLINE:
            break
            
        for restart in range(10):
            if time.time() - start_time > DEADLINE:
                break
                
            colors = np.random.randint(0, num_colors, N)
            current_violations = count_violations(colors)
            
            if current_violations == 0:
                return colors.tolist()
                
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors.copy()
                
            for i in range(2000):
                if time.time() - start_time > DEADLINE:
                    break
                    
                # Find violated APs
                c = colors[aps[:, 0]]
                c2 = colors[aps[:, 1]]
                c3 = colors[aps[:, 2]]
                c4 = colors[aps[:, 3]]
                arr = np.stack([c, c2, c3, c4], axis=1)
                sorted_arr = np.sort(arr, axis=1)
                diffs = np.diff(sorted_arr, axis=1)
                distinct_counts = 1 + (diffs != 0).sum(axis=1)
                violated_mask = distinct_counts < 3
                violated_indices = np.where(violated_mask)[0]
                
                if len(violated_indices) == 0:
                    return colors.tolist()
                    
                current_violations = len(violated_indices)
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors.copy()
                    
                ap_idx = violated_indices[random.randint(0, len(violated_indices) - 1)]
                ap = aps[ap_idx]
                var_idx = random.choice(ap)
                
                old_color = colors[var_idx]
                best_new_color = old_color
                min_new_violations = float('inf')
                
                for new_color in range(num_colors):
                    if new_color == old_color:
                        continue
                    colors[var_idx] = new_color
                    v = count_violations(colors)
                    if v < min_new_violations:
                        min_new_violations = v
                        best_new_color = new_color
                    if v == 0:
                        return colors.tolist()
                        
                colors[var_idx] = best_new_color
                
        if best_violations == 0:
            return best_colors.tolist()

    # Fallback: Greedy
    colors = [0] * N
    for i in range(N):
        for color in range(128):
            colors[i] = color
            valid = True
            for d in range(1, i // 3 + 1):
                a = i - 3 * d
                if a < 0: break
                ap_colors = [colors[a], colors[a+d], colors[a+2*d], colors[i]]
                if len(set(ap_colors)) < 3:
                    valid = False
                    break
            if valid:
                break
    return colors

