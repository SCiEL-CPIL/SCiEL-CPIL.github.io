import numpy as np
import random
import time

def construct():
    n = 128
    
    # Precompute 4-APs
    aps = []
    for d in range(1, (n - 1) // 3 + 1):
        for i in range(1, n - 3 * d + 1):
            i0 = i - 1
            aps.append((i0, i0 + d, i0 + 2 * d, i0 + 3 * d))
    
    aps = np.array(aps, dtype=np.int32)
    M = len(aps)
    
    # Precompute index arrays for fast lookup
    idx0 = aps[:, 0]
    idx1 = aps[:, 1]
    idx2 = aps[:, 2]
    idx3 = aps[:, 3]
    
    def calculate_cost(colors):
        """
        Calculate number of invalid 4-APs.
        A 4-AP is invalid if it uses < 3 distinct colors.
        This happens when:
        - All 4 same (1 color)
        - 3 same, 1 different (2 colors)
        - 2 same, 2 same, different pairs (2 colors)
        
        We count valid APs as those with >= 3 distinct colors.
        """
        c1 = colors[idx0]
        c2 = colors[idx1]
        c3 = colors[idx2]
        c4 = colors[idx3]
        
        # For each AP, count distinct colors
        # An AP is invalid if it has 1 or 2 distinct colors
        
        # 1 distinct color: c1==c2==c3==c4
        all_same = (c1 == c2) & (c2 == c3) & (c3 == c4)
        
        # 2 distinct colors: exactly 3 of the same
        # Case 1: c1==c2==c3 != c4
        case1 = (c1 == c2) & (c2 == c3) & (c3 != c4)
        # Case 2: c1==c2==c4 != c3
        case2 = (c1 == c2) & (c2 == c4) & (c1 != c3)
        # Case 3: c1==c3==c4 != c2
        case3 = (c1 == c3) & (c3 == c4) & (c1 != c2)
        # Case 4: c2==c3==c4 != c1
        case4 = (c2 == c3) & (c3 == c4) & (c2 != c1)
        
        # 2 distinct colors: 2 and 2 split
        # Case 5: c1==c2, c3==c4, c1!=c3
        case5 = (c1 == c2) & (c3 == c4) & (c1 != c3)
        # Case 6: c1==c3, c2==c4, c1!=c2
        case6 = (c1 == c3) & (c2 == c4) & (c1 != c2)
        # Case 7: c1==c4, c2==c3, c1!=c2
        case7 = (c1 == c4) & (c2 == c3) & (c1 != c2)
        
        invalid = all_same | case1 | case2 | case3 | case4 | case5 | case6 | case7
        
        return np.sum(invalid)
    
    def fast_cost_update(colors, pos, old_color, new_color):
        """
        Estimate cost change by only checking APs that include position 'pos'.
        This is an approximation; for exact, we'd need to recompute all affected APs.
        For simplicity in SA, we'll use full cost but with early termination if possible.
        """
        return calculate_cost(colors)
    
    # Simulated Annealing with better initialization
    def sa(k, max_iters=3000, start_temp=5.0, end_temp=0.01):
        # Try multiple random initializations, keep the best
        best_initial = None
        best_initial_cost = float('inf')
        
        for init_try in range(10):
            current = np.random.randint(0, k, n)
            cost = calculate_cost(current)
            if cost < best_initial_cost:
                best_initial_cost = cost
                best_initial = current.copy()
            if cost == 0:
                break
        
        current = best_initial
        current_cost = best_initial_cost
        
        if current_cost == 0:
            return current
        
        best = current.copy()
        best_cost = current_cost
        
        T = start_temp
        cooling_rate = (end_temp / start_temp) ** (1.0 / max_iters)
        
        # Precompute for each position, which APs it belongs to
        # pos_to_aps[pos] = list of AP indices that include pos
        pos_to_aps = [[] for _ in range(n)]
        for ap_idx in range(M):
            for j in range(4):
                pos = aps[ap_idx, j]
                pos_to_aps[pos].append(ap_idx)
        
        for i in range(max_iters):
            # Pick a random position to change
            pos = np.random.randint(0, n)
            old_color = current[pos]
            
            # Pick a new color different from old
            if k > 1:
                new_color = np.random.randint(0, k - 1)
                if new_color >= old_color:
                    new_color += 1
            else:
                continue
            
            current[pos] = new_color
            new_cost = calculate_cost(current)
            
            delta = new_cost - current_cost
            
            if delta < 0 or (np.random.rand() < np.exp(-delta / T)):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best = current.copy()
                    if best_cost == 0:
                        break
            else:
                current[pos] = old_color
            
            T *= cooling_rate
        
        return best
    
    # Try increasing number of colors
    start_time = time.time()
    time_limit = 55.0  # Leave 5s buffer
    
    # Known: 2 colors is impossible for this task (can be proven)
    # So start from k=3
    for k in range(3, 129):
        if time.time() - start_time > time_limit:
            break
        
        # For small k, run more iterations
        if k <= 4:
            iters = 5000
        else:
            iters = 2000
        
        solution = sa(k, max_iters=iters)
        
        if calculate_cost(solution) == 0:
            # Verify and return
            unique_colors = np.unique(solution)
            return solution.tolist()
    
    # Fallback: if we didn't find a solution, try one more time with k=3
    if time.time() - start_time < time_limit:
        solution = sa(3, max_iters=10000)
        if calculate_cost(solution) == 0:
            return solution.tolist()
    
    # Last resort: return the last solution (might be invalid, but better than nothing)
    return solution.tolist()
