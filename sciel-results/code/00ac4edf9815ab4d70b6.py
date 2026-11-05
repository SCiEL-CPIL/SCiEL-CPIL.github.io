import time
import random

def construct():
    deadline = time.time() + 115
    
    def count_violations(colors, n):
        adj0 = [0] * n
        adj1 = [0] * n
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if colors[idx] == 0:
                    adj0[i] |= (1 << j)
                    adj0[j] |= (1 << i)
                else:
                    adj1[i] |= (1 << j)
                    adj1[j] |= (1 << i)
                idx += 1
        
        total_viol = 0
        
        def count_for_adj(adj):
            v = 0
            for u in range(n):
                nu = adj[u]
                for v_idx in range(u + 1, n):
                    nv = adj[v_idx]
                    s_mask = nu & nv
                    s_mask &= ~(1 << u)
                    s_mask &= ~(1 << v_idx)
                    
                    # Check K2,2 in s_mask
                    # We need to find x, y in s_mask such that (adj[x] & adj[y] & s_mask).bit_count() >= 2
                    # Iterate x
                    x_mask = s_mask
                    while x_mask:
                        x_bit = x_mask & -x_mask
                        x_idx = x_bit.bit_length() - 1
                        x_mask ^= x_bit
                        
                        nx = adj[x_idx]
                        # y must be in s_mask and y > x_idx to avoid double counting pairs?
                        # Actually, for counting, each pair (x,y) is a witness.
                        # If we just want to know if ANY exists, we can break.
                        # But we are counting.
                        # Let's count the number of pairs (x,y) that form a K2,2 base.
                        
                        y_mask = s_mask & (~((1 << (x_idx + 1)) - 1))
                        while y_mask:
                            y_bit = y_mask & -y_mask
                            y_idx = y_bit.bit_length() - 1
                            y_mask ^= y_bit
                            
                            ny = adj[y_idx]
                            common = nx & ny & s_mask
                            if common.bit_count() >= 2:
                                v += 1
            return v
            
        total_viol += count_for_adj(adj0)
        total_viol += count_for_adj(adj1)
        
        return total_viol

    # 1. Generate Paley(29)
    q = 29
    squares = {x * x % q for x in range(1, q)}
    colors_29 = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in squares:
                colors_29.append(0)
            else:
                colors_29.append(1)
    
    best_n = 29
    best_colors = colors_29
    if count_violations(colors_29, 29) != 0:
        best_n = 1
        best_colors = [0]

    # 2. Try n=30
    n = 30
    n_edges = n * (n - 1) // 2
    base_colors = colors_29.copy()
    
    found_30 = False
    best_viol_30 = float('inf')
    best_colors_30 = None
    
    # Use a more aggressive search
    # We will try to find a solution with 0 violations.
    
    # Strategy: 
    # 1. Random restarts
    # 2. Local search with random flips
    
    max_restarts = 100
    for r in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        current_colors = base_colors.copy()
        for _ in range(29):
            current_colors.append(random.randint(0, 1))
            
        current_viol = count_violations(current_colors, n)
        
        if current_viol == 0:
            return {'n': n, 'colors': current_colors}
        
        if current_viol < best_viol_30:
            best_viol_30 = current_viol
            best_colors_30 = current_colors.copy()
            
        # Local Search
        # Use a "plateau" strategy: if stuck, jump
        steps = 0
        max_steps = 500
        stuck_count = 0
        while steps < max_steps and time.time() < deadline - 10:
            steps += 1
            edge_idx = random.randint(0, n_edges - 1)
            old_val = current_colors[edge_idx]
            current_colors[edge_idx] = 1 - old_val
            
            new_viol = count_violations(current_colors, n)
            
            if new_viol < current_viol:
                current_viol = new_viol
                stuck_count = 0
                if new_viol == 0:
                    return {'n': n, 'colors': current_colors}
                if new_viol < best_viol_30:
                    best_viol_30 = new_viol
                    best_colors_30 = current_colors.copy()
            else:
                # Revert
                current_colors[edge_idx] = old_val
                stuck_count += 1
                
                # If stuck for a while, do a random jump (flip 3-5 edges)
                if stuck_count > 50:
                    for _ in range(3):
                        j = random.randint(0, n_edges - 1)
                        current_colors[j] = 1 - current_colors[j]
                    current_viol = count_violations(current_colors, n)
                    stuck_count = 0

    return {'n': best_n, 'colors': best_colors}
