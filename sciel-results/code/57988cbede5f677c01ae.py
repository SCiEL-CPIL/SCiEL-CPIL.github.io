import time
import random
import sys

def construct():
    deadline = time.time() + 145.0  # Safety margin for 150s limit
    
    # Known valid n=32 solution (Greenwood-Gleason doubled)
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    fallback_colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            fallback_colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    
    def verify(n, colors):
        """Fast verification using bitmasks"""
        m = n * (n - 1) // 2
        if len(colors) != m:
            return False
        # Build masks
        masks = [[0] * 4 for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                masks[i][c] |= (1 << j)
                masks[j][c] |= (1 << i)
                idx += 1
        # Check for monochromatic triangles
        for u in range(n):
            for c in range(4):
                mask_u = masks[u][c]
                if mask_u == 0:
                    continue
                # For each v in mask_u, check if masks[v][c] & mask_u has any bits
                # This counts triangles containing u
                temp = mask_u
                while temp:
                    lsb = temp & (-temp)
                    v = lsb.bit_length() - 1
                    if masks[v][c] & mask_u:
                        return False
                    temp ^= lsb
        return True
    
    def solve_for_n(n, time_budget):
        """Simulated annealing with bitmask delta computation"""
        start_time = time.time()
        m = n * (n - 1) // 2
        
        # Precompute edge index lookup
        idx = [[0] * n for _ in range(n)]
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                idx[i][j] = k
                idx[j][i] = k
                k += 1
        
        # Multiple restarts
        for restart in range(20):
            if time.time() > start_time + time_budget:
                break
            
            # Initialize random colors
            colors = [random.randint(0, 3) for _ in range(m)]
            
            # Build masks
            masks = [[0] * 4 for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[idx[i][j]]
                    masks[i][c] |= (1 << j)
                    masks[j][c] |= (1 << i)
            
            # Calculate initial triangle count using bitmasks
            tri_count = 0
            for u in range(n):
                for c in range(4):
                    mask_u = masks[u][c]
                    if mask_u == 0:
                        continue
                    temp = mask_u
                    while temp:
                        lsb = temp & (-temp)
                        v = lsb.bit_length() - 1
                        tri_count += (masks[v][c] & mask_u).bit_count()
                        temp ^= lsb
            # Each triangle counted 3 times (once per vertex)
            tri_count //= 3
            
            if tri_count == 0:
                return colors
            
            # Simulated annealing
            T = 1.0
            cooling = 0.9995
            min_T = 0.01
            no_improve_count = 0
            best_tri = tri_count
            
            while T > min_T and time.time() < start_time + time_budget - 2:
                # Pick random edge
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                
                i_idx = idx[u][v]
                old_c = colors[i_idx]
                new_c = random.randint(0, 3)
                if old_c == new_c:
                    continue
                
                # Calculate delta in triangle count
                lost = (masks[u][old_c] & masks[v][old_c]).bit_count()
                gained = (masks[u][new_c] & masks[v][new_c]).bit_count()
                delta = gained - lost
                
                if delta < 0 or random.random() < pow(2.718, -delta / T):
                    # Accept
                    colors[i_idx] = new_c
                    tri_count += delta
                    
                    # Update masks
                    masks[u][old_c] &= ~(1 << v)
                    masks[v][old_c] &= ~(1 << u)
                    masks[u][new_c] |= (1 << v)
                    masks[v][new_c] |= (1 << u)
                    
                    if tri_count < best_tri:
                        best_tri = tri_count
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    
                    if tri_count == 0:
                        return colors
                    
                    # Early termination if stuck too long
                    if no_improve_count > 50000:
                        break
                
                T *= cooling
            
            # If we got close, do a final polish
            if best_tri <= 5 and time.time() < start_time + time_budget - 2:
                # Try to eliminate remaining triangles
                for _ in range(100000):
                    if time.time() > start_time + time_budget - 2:
                        break
                    # Find a triangle
                    found = False
                    for u in range(n):
                        for c in range(4):
                            mask_u = masks[u][c]
                            if mask_u == 0:
                                continue
                            temp = mask_u
                            while temp:
                                lsb = temp & (-temp)
                                v = lsb.bit_length() - 1
                                common = masks[v][c] & mask_u
                                if common:
                                    # Found triangle (u, v, w) where w in common
                                    lsb2 = common & (-common)
                                    w = lsb2.bit_length() - 1
                                    # Try to break it by changing one edge
                                    for edge_u, edge_v in [(u, v), (u, w), (v, w)]:
                                        if edge_u > edge_v:
                                            edge_u, edge_v = edge_v, edge_u
                                        e_idx = idx[edge_u][edge_v]
                                        old_ec = colors[e_idx]
                                        for new_ec in range(4):
                                            if new_ec == old_ec:
                                                continue
                                            lost_e = (masks[edge_u][old_ec] & masks[edge_v][old_ec]).bit_count()
                                            gained_e = (masks[edge_u][new_ec] & masks[edge_v][new_ec]).bit_count()
                                            delta_e = gained_e - lost_e
                                            if delta_e < 0:
                                                # Accept this flip
                                                colors[e_idx] = new_ec
                                                tri_count += delta_e
                                                masks[edge_u][old_ec] &= ~(1 << edge_v)
                                                masks[edge_v][old_ec] &= ~(1 << edge_u)
                                                masks[edge_u][new_ec] |= (1 << edge_v)
                                                masks[edge_v][new_ec] |= (1 << edge_u)
                                                if tri_count == 0:
                                                    return colors
                                                found = True
                                                break
                                        if found:
                                            break
                                    if found:
                                        break
                                if found:
                                    break
                                temp ^= lsb
                            if found:
                                break
                        if found:
                            break
                    if not found:
                        break
        
        return None
    
    # Try n from 50 down to 33
    # Time budget allocation: give more time to higher n
    total_budget = deadline - time.time()
    n_values = list(range(50, 32, -1))
    # Allocate time proportional to difficulty (higher n gets more time)
    weights = [n * n for n in n_values]
    total_weight = sum(weights)
    time_per_n = {}
    for n in n_values:
        time_per_n[n] = total_budget * (n * n) / total_weight * 0.8  # 80% for search, 20% buffer
    
    for n in n_values:
        if time.time() > deadline - 5:
            break
        t_budget = min(time_per_n[n], deadline - time.time() - 5)
        if t_budget < 2:
            break
        result = solve_for_n(n, t_budget)
        if result is not None:
            # Verify
            if verify(n, result):
                return {'n': n, 'colors': result}
    
    # Fallback to n=32
    return {'n': 32, 'colors': fallback_colors}
