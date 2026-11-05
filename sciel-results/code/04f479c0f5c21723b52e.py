# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _make_adj(colors, n):
    adj = [np.zeros(n, dtype=np.uint64) for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    return adj

def _has_c6(adj, n):
    for c in range(4):
        a = adj[c]
        for v0 in range(n):
            row0 = a[v0]
            for v1 in range(v0 + 1, n):
                if not (row0 >> v1) & 1:
                    continue
                row1 = a[v1]
                for v2 in range(v1 + 1, n):
                    if not (row1 >> v2) & 1:
                        continue
                    row2 = a[v2]
                    for v3 in range(v2 + 1, n):
                        if not (row2 >> v3) & 1:
                            continue
                        row3 = a[v3]
                        for v4 in range(v3 + 1, n):
                            if not (row3 >> v4) & 1:
                                continue
                            row4 = a[v4]
                            common = row4 & row0
                            common &= ~(1 << v0)
                            common &= ~(1 << v1)
                            common &= ~(1 << v2)
                            common &= ~(1 << v3)
                            common &= ~(1 << v4)
                            if common:
                                v5 = int(np.log2(common & -common))
                                return True, c, (v0, v1, v2, v3, v4, v5)
    return False, -1, None

def _count_c6(adj, n):
    count = 0
    for c in range(4):
        a = adj[c]
        for v0 in range(n):
            row0 = a[v0]
            for v1 in range(v0 + 1, n):
                if not (row0 >> v1) & 1:
                    continue
                row1 = a[v1]
                for v2 in range(v1 + 1, n):
                    if not (row1 >> v2) & 1:
                        continue
                    row2 = a[v2]
                    for v3 in range(v2 + 1, n):
                        if not (row2 >> v3) & 1:
                            continue
                        row3 = a[v3]
                        for v4 in range(v3 + 1, n):
                            if not (row3 >> v4) & 1:
                                continue
                            row4 = a[v4]
                            common = row4 & row0
                            common &= ~(1 << v0)
                            common &= ~(1 << v1)
                            common &= ~(1 << v2)
                            common &= ~(1 << v3)
                            common &= ~(1 << v4)
                            count += np.bitwise_count(common)
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 112
    
    n = 17
    rng = random.Random(42)
    
    # Generate multiple initial colorings
    candidates = []
    
    # Pattern 1: distance mod 4
    colors1 = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            if d > n - d:
                d = n - d
            colors1.append(d % 4)
    candidates.append(colors1)
    
    # Pattern 2: quadratic residue split
    qr = {1, 2, 4, 8, 9, 13, 15, 16}
    colors2 = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            if d in qr:
                colors2.append(0 if d in {1, 2, 4, 8} else 1)
            else:
                colors2.append(2 if d in {3, 5, 6, 7} else 3)
    candidates.append(colors2)
    
    # Pattern 3: (3*d) % 4
    colors3 = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            if d > n - d:
                d = n - d
            colors3.append((3 * d) % 4)
    candidates.append(colors3)
    
    # Pattern 4: (5*d+1) % 4
    colors4 = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            if d > n - d:
                d = n - d
            colors4.append((5 * d + 1) % 4)
    candidates.append(colors4)
    
    # Pattern 5: (2*d+1) % 4
    colors5 = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            if d > n - d:
                d = n - d
            colors5.append((2 * d + 1) % 4)
    candidates.append(colors5)
    
    # Evaluate all candidates
    best_colors = None
    best_violations = float('inf')
    
    for colors in candidates:
        if time.time() > deadline - 40:
            break
        adj = _make_adj(colors, n)
        v = _count_c6(adj, n)
        if v < best_violations:
            best_violations = v
            best_colors = colors[:]
        if v == 0:
            break
    
    # Local search to eliminate C6s
    if best_violations > 0 and time.time() < deadline - 15:
        adj = _make_adj(best_colors, n)
        pair_list = []
        pair_idx = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                pair_list.append((i, j))
                pair_idx[(i, j)] = idx
                idx += 1
        
        no_improve = 0
        while time.time() < deadline - 10 and best_violations > 0:
            has, c, cycle = _has_c6(adj, n)
            if not has:
                best_violations = 0
                break
            
            v0, v1, v2, v3, v4, v5 = cycle
            edges = [(v0, v1), (v1, v2), (v2, v3), (v3, v4), (v4, v5), (v5, v0)]
            
            found_improvement = False
            for _ in range(30):
                e = rng.choice(edges)
                i, j = e
                if i > j:
                    i, j = j, i
                pi = pair_idx[(i, j)]
                old_c = best_colors[pi]
                new_c = rng.choice([x for x in range(4) if x != old_c])
                
                best_colors[pi] = new_c
                adj[old_c][i] &= ~(1 << j)
                adj[old_c][j] &= ~(1 << i)
                adj[new_c][i] |= (1 << j)
                adj[new_c][j] |= (1 << i)
                
                new_v = _count_c6(adj, n)
                
                if new_v < best_violations:
                    best_violations = new_v
                    found_improvement = True
                    if new_v == 0:
                        break
                else:
                    best_colors[pi] = old_c
                    adj[old_c][i] |= (1 << j)
                    adj[old_c][j] |= (1 << i)
                    adj[new_c][i] &= ~(1 << j)
                    adj[new_c][j] &= ~(1 << i)
            
            if not found_improvement:
                no_improve += 1
                # Random perturbation
                for _ in range(5):
                    pi = rng.randint(0, len(best_colors) - 1)
                    i, j = pair_list[pi]
                    old_c = best_colors[pi]
                    new_c = rng.choice([x for x in range(4) if x != old_c])
                    best_colors[pi] = new_c
                    adj[old_c][i] &= ~(1 << j)
                    adj[old_c][j] &= ~(1 << i)
                    adj[new_c][i] |= (1 << j)
                    adj[new_c][j] |= (1 << i)
                new_v = _count_c6(adj, n)
                if new_v < best_violations:
                    best_violations = new_v
                    no_improve = 0
                else:
                    if no_improve > 10:
                        break
            else:
                no_improve = 0
    
    # Try to extend to n=18
    if best_violations == 0 and time.time() < deadline - 20:
        n_new = 18
        best_ext = None
        rng2 = random.Random(777)
        
        for attempt in range(200):
            if time.time() > deadline - 5:
                break
            
            ext_colors = best_colors[:]
            for i in range(17):
                ext_colors.append(rng2.randint(0, 3))
            
            adj = _make_adj(ext_colors, n_new)
            v = _count_c6(adj, n_new)
            
            if v == 0:
                best_ext = ext_colors
                break
            
            if v <= 5:
                pair_list2 = []
                pair_idx2 = {}
                idx2 = 0
                for i in range(n_new):
                    for j in range(i + 1, n_new):
                        pair_list2.append((i, j))
                        pair_idx2[(i, j)] = idx2
                        idx2 += 1
                
                for _ in range(30):
                    has, c, cycle = _has_c6(adj, n_new)
                    if not has:
                        best_ext = ext_colors
                        break
                    v0, v1, v2, v3, v4, v5 = cycle
                    edges = [(v0, v1), (v1, v2), (v2, v3), (v3, v4), (v4, v5), (v5, v0)]
                    e = rng2.choice(edges)
                    i, j = e
                    if i > j:
                        i, j = j, i
                    pi = pair_idx2[(i, j)]
                    old_c = ext_colors[pi]
                    new_c = rng2.choice([x for x in range(4) if x != old_c])
                    ext_colors[pi] = new_c
                    adj[old_c][i] &= ~(1 << j)
                    adj[old_c][j] &= ~(1 << i)
                    adj[new_c][i] |= (1 << j)
                    adj[new_c][j] |= (1 << i)
                
                if best_ext:
                    break
        
        if best_ext:
            n = 18
            best_colors = best_ext
    
    return {'n': n, 'colors': best_colors}
# EVOLVE-BLOCK-END
