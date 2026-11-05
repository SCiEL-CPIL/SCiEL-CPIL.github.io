import numpy as np
import random
import math

def construct():
    N = 50
    E = N * (N - 1) // 2
    
    # Map (i, j) to edge index
    edge_idx = np.full((N, N), -1, dtype=int)
    idx = 0
    us = np.zeros(E, dtype=int)
    vs = np.zeros(E, dtype=int)
    for i in range(N):
        for j in range(i + 1, N):
            edge_idx[i, j] = idx
            edge_idx[j, i] = idx
            us[idx] = i
            vs[idx] = j
            idx += 1
            
    # Precompute neighbors for each edge?
    # For edge e=(u,v), the vertices w are all except u,v.
    # We need edge_idx[u,w] and edge_idx[v,w].
    
    # Initialize colors
    np.random.seed(42) # Fixed seed for reproducibility in dev, but maybe random is better for SA?
    # Let's use a fixed seed for the initial state to be deterministic, but SA randomness comes from the process.
    # Actually, let's use a different seed for SA to explore.
    colors = np.random.randint(0, 4, size=E)
    
    # Calculate initial energy (total number of monochromatic triangles)
    # Energy = sum of conflict_count[e] / 3 ? 
    # Each mono triangle contributes 3 to the sum of conflict_counts (1 for each edge).
    # So Energy = sum(conflict_count) / 3.
    # We can just minimize sum(conflict_count).
    
    conflict_count = np.zeros(E, dtype=int)
    # Efficient initial calculation
    for i in range(N):
        for j in range(i + 1, N):
            e1 = edge_idx[i, j]
            c1 = colors[e1]
            count = 0
            for k in range(N):
                if k == i or k == j: continue
                e2 = edge_idx[i, k]
                e3 = edge_idx[j, k]
                if colors[e2] == c1 and colors[e3] == c1:
                    count += 1
            conflict_count[e1] = count
            
    total_conflicts = np.sum(conflict_count)
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.999
    min_T = 0.001
    max_iters = 200000 # Adjust based on speed
    
    for iter in range(max_iters):
        if total_conflicts == 0:
            break
            
        # Pick random edge
        e = np.random.randint(E)
        u = us[e]
        v = vs[e]
        c_old = colors[e]
        
        # Pick random new color
        c_new = np.random.randint(0, 4)
        if c_new == c_old:
            continue
            
        # Calculate delta in total_conflicts
        # Only triangles containing e are affected.
        # Triangle (u, v, w) status changes.
        # delta = (new_mono_count - old_mono_count) * 3 ? 
        # Wait, conflict_count[e] counts w where c(u,v)=c(u,w)=c(v,w).
        # If we change c(u,v), it affects:
        # 1. conflict_count[e]
        # 2. conflict_count[e1] for all w where c(u,w) was c_old
        # 3. conflict_count[e2] for all w where c(v,w) was c_old
        
        # Let's just compute the change in total_conflicts directly.
        # old_total = sum(conflict_count)
        # new_total = sum(conflict_count) + delta
        
        delta = 0
        for w in range(N):
            if w == u or w == v:
                continue
            e1 = edge_idx[u, w]
            e2 = edge_idx[v, w]
            c1 = colors[e1]
            c2 = colors[e2]
            
            # Old status
            was_mono = (c1 == c_old and c2 == c_old)
            # New status
            is_mono = (c1 == c_new and c2 == c_new)
            
            if was_mono and not is_mono:
                # Triangle ceases to be mono.
                # conflict_count[e] decreases by 1
                # conflict_count[e1] decreases by 1
                # conflict_count[e2] decreases by 1
                delta -= 3
            elif not was_mono and is_mono:
                # Triangle becomes mono.
                # conflict_count[e] increases by 1
                # conflict_count[e1] increases by 1
                # conflict_count[e2] increases by 1
                delta += 3
        
        # Acceptance
        if delta < 0:
            accept = True
        else:
            # P = exp(-delta / T)
            # delta is multiple of 3.
            # If T is small, this is 0.
            accept = (random.random() < math.exp(-delta / T))
            
        if accept:
            colors[e] = c_new
            # Update conflict counts
            for w in range(N):
                if w == u or w == v:
                    continue
                e1 = edge_idx[u, w]
                e2 = edge_idx[v, w]
                c1 = colors[e1]
                c2 = colors[e2]
                
                was_mono = (c1 == c_old and c2 == c_old)
                is_mono = (c1 == c_new and c2 == c_new)
                
                if was_mono and not is_mono:
                    conflict_count[e] -= 1
                    conflict_count[e1] -= 1
                    conflict_count[e2] -= 1
                elif not was_mono and is_mono:
                    conflict_count[e] += 1
                    conflict_count[e1] += 1
                    conflict_count[e2] += 1
                    
            total_conflicts += delta
            
        T *= cooling
        if T < min_T:
            break
            
    if total_conflicts == 0:
        return {'n': N, 'colors': [int(c) for c in colors]}
    else:
        # Fallback to 32-vertex solution
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
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return {'n': 32, 'colors': colors}
