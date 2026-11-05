import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 115
    
    def has_k222(adj, n):
        """Check if adj (list of int bitmasks) contains K2,2,2."""
        for u in range(n):
            nu = adj[u]
            for v_idx in range(u + 1, n):
                nv = adj[v_idx]
                s_mask = nu & nv
                s_mask &= ~(1 << u)
                s_mask &= ~(1 << v_idx)
                if s_mask.bit_count() < 4:
                    continue
                x_mask = s_mask
                while x_mask:
                    x_bit = x_mask & -x_mask
                    x_idx = x_bit.bit_length() - 1
                    x_mask ^= x_bit
                    nx = adj[x_idx]
                    y_mask = s_mask & (~((1 << (x_idx + 1)) - 1))
                    while y_mask:
                        y_bit = y_mask & -y_mask
                        y_idx = y_bit.bit_length() - 1
                        y_mask ^= y_bit
                        ny = adj[y_idx]
                        if (nx & ny & s_mask).bit_count() >= 2:
                            return True
        return False
    
    def build_adj(colors, n):
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
        return adj0, adj1
    
    def count_k222(adj, n):
        """Count number of K2,2,2 subgraphs."""
        v = 0
        for u in range(n):
            nu = adj[u]
            for v_idx in range(u + 1, n):
                nv = adj[v_idx]
                s_mask = nu & nv
                s_mask &= ~(1 << u)
                s_mask &= ~(1 << v_idx)
                if s_mask.bit_count() < 4:
                    continue
                x_mask = s_mask
                while x_mask:
                    x_bit = x_mask & -x_mask
                    x_idx = x_bit.bit_length() - 1
                    x_mask ^= x_bit
                    nx = adj[x_idx]
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
    
    def total_violations(colors, n):
        adj0, adj1 = build_adj(colors, n)
        return count_k222(adj0, n) + count_k222(adj1, n)
    
    # Generate Paley(29)
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
    if total_violations(colors_29, 29) != 0:
        best_n = 1
        best_colors = [0]
        return {'n': best_n, 'colors': best_colors}
    
    # Try n=30
    n = 30
    n_edges = n * (n - 1) // 2
    
    # Compute edge indices for edges involving vertex 29
    # Edge (i, j) with i < j has index: sum_{k=0}^{i-1} (n-1-k) + (j-i-1)
    # For j=29, i from 0 to 28:
    new_edge_indices = []
    for i in range(29):
        # Number of edges before row i: sum_{k=0}^{i-1} (29-k) = 29*i - i*(i-1)//2
        # Within row i, edge (i, 29) is at position (29-i-1)
        idx = 29 * i - i * (i - 1) // 2 + (29 - i - 1)
        new_edge_indices.append(idx)
    
    base_colors = colors_29.copy()
    
    best_viol_30 = float('inf')
    best_colors_30 = None
    
    max_restarts = 500
    for r in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        # Initialize with random coloring of new edges
        current_colors = base_colors.copy()
        for idx in new_edge_indices:
            current_colors.append(random.randint(0, 1))
        
        current_viol = total_violations(current_colors, n)
        
        if current_viol == 0:
            return {'n': n, 'colors': current_colors}
        
        if current_viol < best_viol_30:
            best_viol_30 = current_viol
            best_colors_30 = current_colors.copy()
        
        # Local search
        steps = 0
        max_steps = 500
        stuck_count = 0
        no_improve = 0
        
        while steps < max_steps and time.time() < deadline - 5:
            steps += 1
            
            # Choose which edge to flip
            if random.random() < 0.6:
                edge_idx = random.choice(new_edge_indices)
            else:
                edge_idx = random.randint(0, n_edges - 1)
            
            old_val = current_colors[edge_idx]
            current_colors[edge_idx] = 1 - old_val
            
            new_viol = total_violations(current_colors, n)
            
            if new_viol == 0:
                return {'n': n, 'colors': current_colors}
            
            if new_viol < current_viol:
                current_viol = new_viol
                stuck_count = 0
                no_improve = 0
                if new_viol < best_viol_30:
                    best_viol_30 = new_viol
                    best_colors_30 = current_colors.copy()
            else:
                current_colors[edge_idx] = old_val
                stuck_count += 1
                no_improve += 1
                
                if stuck_count > 20:
                    # Jump: flip 2-4 random new edges
                    num_flips = random.randint(2, 4)
                    for _ in range(num_flips):
                        j = random.choice(new_edge_indices)
                        current_colors[j] = 1 - current_colors[j]
                    current_viol = total_violations(current_colors, n)
                    if current_viol < best_viol_30:
                        best_viol_30 = current_viol
                        best_colors_30 = current_colors.copy()
                    stuck_count = 0
                    no_improve = 0
        
        # If we've been stuck for a while, try a more aggressive restart
        if no_improve > 100:
            break
    
    # If we found a better 30-vertex coloring, return it
    if best_viol_30 == 0 and best_colors_30 is not None:
        return {'n': 30, 'colors': best_colors_30}
    
    return {'n': best_n, 'colors': best_colors}
