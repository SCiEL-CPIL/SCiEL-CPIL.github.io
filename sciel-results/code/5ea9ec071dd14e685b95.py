import time
import random
import numpy as np

def _build_bitsets(colors, n):
    bitsets = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx]:
                bitsets[i] |= (1 << j)
                bitsets[j] |= (1 << i)
            idx += 1
    return bitsets

def _is_valid(bitsets, n):
    for c in range(n - 1):
        bc = bitsets[c]
        for d in range(c + 1, n):
            if not (bc >> d) & 1:
                continue
            common = bc & bitsets[d]
            verts = []
            cb = common
            while cb:
                lsb = cb & (-cb)
                verts.append(lsb.bit_length() - 1)
                cb ^= lsb
            m = len(verts)
            if m < 4:
                continue
            for ii in range(m - 1):
                a = verts[ii]
                ba = bitsets[a]
                for jj in range(ii + 1, m):
                    b = verts[jj]
                    nb = ba & bitsets[b] & common
                    if nb & (nb - 1):
                        return False
    return True

def _count_k222_fast(bitsets, n):
    count = 0
    for c in range(n - 1):
        bc = bitsets[c]
        for d in range(c + 1, n):
            if not (bc >> d) & 1:
                continue
            common = bc & bitsets[d]
            verts = []
            cb = common
            while cb:
                lsb = cb & (-cb)
                verts.append(lsb.bit_length() - 1)
                cb ^= lsb
            m = len(verts)
            if m < 4:
                continue
            for ii in range(m - 1):
                a = verts[ii]
                ba = bitsets[a]
                for jj in range(ii + 1, m):
                    b = verts[jj]
                    nb = ba & bitsets[b] & common
                    cnt = nb.bit_count()
                    if cnt >= 2:
                        count += 1
    return count

def _edge_to_ij(idx, n):
    i = 0
    offset = 0
    while True:
        next_offset = offset + (n - 1 - i)
        if idx < next_offset:
            break
        i += 1
        offset = next_offset
    j = i + 1 + (idx - offset)
    return i, j

def _ij_to_edge(i, j, n):
    idx = 0
    for k in range(i):
        idx += (n - 1 - k)
    idx += (j - i - 1)
    return idx

def _paley(n, p):
    if p < n:
        return None
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(0 if (j - i) % p in squares else 1)
    return colors

def _targeted_repair(colors, n, time_limit=30):
    start_time = time.time()
    bitsets = _build_bitsets(colors, n)
    
    if _is_valid(bitsets, n):
        return colors, True
    
    for iteration in range(300):
        if time.time() - start_time > time_limit:
            break
        
        # Find violations
        violations = []
        for c in range(n - 1):
            bc = bitsets[c]
            for d in range(c + 1, n):
                if not (bc >> d) & 1:
                    continue
                common = bc & bitsets[d]
                verts = []
                cb = common
                while cb:
                    lsb = cb & (-cb)
                    verts.append(lsb.bit_length() - 1)
                    cb ^= lsb
                m = len(verts)
                if m < 4:
                    continue
                for ii in range(m - 1):
                    a = verts[ii]
                    ba = bitsets[a]
                    for jj in range(ii + 1, m):
                        b = verts[jj]
                        nb = ba & bitsets[b] & common
                        if nb & (nb - 1):
                            nb_copy = nb
                            lsb1 = nb_copy & (-nb_copy)
                            e = lsb1.bit_length() - 1
                            nb_copy ^= lsb1
                            lsb2 = nb_copy & (-nb_copy)
                            f = lsb2.bit_length() - 1
                            violations.append((c, d, a, b, e, f))
                            if len(violations) >= 200:
                                break
                    if len(violations) >= 200:
                        break
                if len(violations) >= 200:
                    break
            if len(violations) >= 200:
                break
        
        if not violations:
            return colors, True
        
        edge_freq = {}
        for (c, d, a, b, e, f) in violations:
            edge_pairs = [
                (c, a), (c, b), (c, e), (c, f),
                (d, a), (d, b), (d, e), (d, f),
                (a, e), (a, f), (b, e), (b, f)
            ]
            for (u, v) in edge_pairs:
                if u > v:
                    u, v = v, u
                key = (u, v)
                edge_freq[key] = edge_freq.get(key, 0) + 1
        
        if not edge_freq:
            break
        
        best_edge = max(edge_freq, key=edge_freq.get)
        u, v = best_edge
        idx = _ij_to_edge(u, v, n)
        
        old_val = (bitsets[u] >> v) & 1
        if old_val:
            bitsets[u] ^= (1 << v)
            bitsets[v] ^= (1 << u)
            colors[idx] = 0
        else:
            bitsets[u] |= (1 << v)
            bitsets[v] |= (1 << u)
            colors[idx] = 1
        
        if _is_valid(bitsets, n):
            return colors, True
    
    return colors, False

def _simulated_annealing(colors, n, max_iters=80000, time_limit=60):
    start_time = time.time()
    bitsets = _build_bitsets(colors, n)
    current_cost = _count_k222_fast(bitsets, n)
    
    if current_cost == 0:
        return colors, True
    
    num_edges = n * (n - 1) // 2
    T0 = float(max(1, current_cost))
    T_min = 0.05
    cooling = 0.999
    
    T = T0
    best_cost = current_cost
    best_colors = colors[:]
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        idx = random.randint(0, num_edges - 1)
        i, j = _edge_to_ij(idx, n)
        
        old_val = (bitsets[i] >> j) & 1
        if old_val:
            bitsets[i] ^= (1 << j)
            bitsets[j] ^= (1 << i)
            colors[idx] = 0
        else:
            bitsets[i] |= (1 << j)
            bitsets[j] |= (1 << i)
            colors[idx] = 1
        
        new_cost = _count_k222_fast(bitsets, n)
        delta = new_cost - current_cost
        
        if delta <= 0:
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors[:]
                T = max(T0 * 0.3, T_min)
        else:
            if T > T_min and random.random() < np.exp(-delta / T):
                current_cost = new_cost
            else:
                if old_val:
                    bitsets[i] |= (1 << j)
                    bitsets[j] |= (1 << i)
                    colors[idx] = 1
                else:
                    bitsets[i] ^= (1 << j)
                    bitsets[j] ^= (1 << i)
                    colors[idx] = 0
        
        T *= cooling
        
        if current_cost == 0:
            return colors, True
    
    return best_colors, best_cost == 0

def _local_search(colors, n, max_iters=30000, time_limit=40):
    start_time = time.time()
    bitsets = _build_bitsets(colors, n)
    current_cost = _count_k222_fast(bitsets, n)
    
    if current_cost == 0:
        return colors, True
    
    num_edges = n * (n - 1) // 2
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        if num_edges <= 406:
            candidates = list(range(num_edges))
            random.shuffle(candidates)
            candidates = candidates[:200]
        else:
            candidates = random.sample(range(num_edges), 200)
        
        best_idx = -1
        best_cost = current_cost
        
        for idx in candidates:
            i, j = _edge_to_ij(idx, n)
            
            old_val = (bitsets[i] >> j) & 1
            if old_val:
                bitsets[i] ^= (1 << j)
                bitsets[j] ^= (1 << i)
            else:
                bitsets[i] |= (1 << j)
                bitsets[j] |= (1 << i)
            
            new_cost = _count_k222_fast(bitsets, n)
            
            if old_val:
                bitsets[i] |= (1 << j)
                bitsets[j] |= (1 << i)
            else:
                bitsets[i] ^= (1 << j)
                bitsets[j] ^= (1 << i)
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_idx = idx
        
        if best_idx >= 0:
            i, j = _edge_to_ij(best_idx, n)
            old_val = (bitsets[i] >> j) & 1
            if old_val:
                bitsets[i] ^= (1 << j)
                bitsets[j] ^= (1 << i)
                colors[best_idx] = 0
            else:
                bitsets[i] |= (1 << j)
                bitsets[j] |= (1 << i)
                colors[best_idx] = 1
            
            current_cost = best_cost
            if current_cost == 0:
                return colors, True
        else:
            break
    
    return colors, current_cost == 0

def construct():
    deadline = time.time() + 140
    target_n = 29
    p = 29
    
    # 1. Try Paley(29) directly - this is the known solution
    colors = _paley(target_n, p)
    if colors:
        bitsets = _build_bitsets(colors, target_n)
        if _is_valid(bitsets, target_n):
            return {'n': target_n, 'colors': colors}
    
    # 2. Targeted repair from Paley
    if time.time() < deadline - 60:
        colors_copy = colors[:]
        _, success = _targeted_repair(colors_copy, target_n, time_limit=25)
        if success:
            return {'n': target_n, 'colors': colors_copy}
    
    # 3. Local search from Paley
    if time.time() < deadline - 50:
        colors_ls = colors[:]
        _, success = _local_search(colors_ls, target_n, max_iters=20000, time_limit=30)
        if success:
            return {'n': target_n, 'colors': colors_ls}
    
    # 4. Simulated annealing from Paley
    if time.time() < deadline - 40:
        colors_sa = colors[:]
        _, success = _simulated_annealing(colors_sa, target_n, max_iters=60000, time_limit=35)
        if success:
            return {'n': target_n, 'colors': colors_sa}
    
    # 5. Random restarts with targeted repair + SA
    for attempt in range(25):
        if time.time() > deadline - 15:
            break
        colors_rand = [random.randint(0, 1) for _ in range(target_n * (target_n - 1) // 2)]
        
        _, success = _targeted_repair(colors_rand, target_n, time_limit=8)
        if success:
            return {'n': target_n, 'colors': colors_rand}
        
        if time.time() < deadline - 15:
            _, success = _simulated_annealing(colors_rand, target_n, max_iters=20000, time_limit=10)
            if success:
                return {'n': target_n, 'colors': colors_rand}
    
    # 6. Fallback: try smaller n with Paley
    for n in range(28, 10, -1):
        if time.time() > deadline - 5:
            break
        for pp in [17, 19, 23, 29, 31]:
            if pp >= n:
                colors = _paley(n, pp)
                if colors:
                    bitsets = _build_bitsets(colors, n)
                    if _is_valid(bitsets, n):
                        return {'n': n, 'colors': colors}
    
    return {'n': 1, 'colors': [0]}
