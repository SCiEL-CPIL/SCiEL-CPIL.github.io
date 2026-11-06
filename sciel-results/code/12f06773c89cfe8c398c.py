import time
import random
import numpy as np

def _build_bitsets(colors, n):
    """Build adjacency bitsets from colors list. Returns list of Python ints."""
    bitsets = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx]:
                bitsets[i] |= (1 << j)
                bitsets[j] |= (1 << i)
            idx += 1
    return bitsets

def _find_k222_violations(bitsets, n, max_count=50):
    """Find up to max_count K2,2,2 subgraphs. Returns list of (c,d,a,b,e,f) tuples."""
    violations = []
    for c in range(n - 1):
        bc = bitsets[c]
        for d in range(c + 1, n):
            if not (bc >> d) & 1:
                continue
            common = bc & bitsets[d]
            # Get list of vertices in common
            verts = []
            cb = common
            while cb:
                lsb = cb & (-cb)
                verts.append(lsb.bit_length() - 1)
                cb ^= lsb
            m = len(verts)
            if m < 4:
                continue
            # Find K2,2 in induced subgraph on 'verts'
            for ii in range(m - 1):
                a = verts[ii]
                ba = bitsets[a]
                for jj in range(ii + 1, m):
                    b = verts[jj]
                    nb = ba & bitsets[b] & common
                    # Need at least 2 bits in nb
                    if nb & (nb - 1):
                        # Get two vertices e,f from nb
                        nb_copy = nb
                        lsb1 = nb_copy & (-nb_copy)
                        e = lsb1.bit_length() - 1
                        nb_copy ^= lsb1
                        lsb2 = nb_copy & (-nb_copy)
                        f = lsb2.bit_length() - 1
                        violations.append((c, d, a, b, e, f))
                        if len(violations) >= max_count:
                            return violations
    return violations

def _count_k222_fast(bitsets, n):
    """Count K2,2,2 subgraphs using bitset operations with early termination awareness."""
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

def _is_valid(bitsets, n):
    """Check if no K2,2,2 exists. Early exit."""
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

def _edge_to_ij(idx, n):
    """Convert linear edge index to (i, j) pair."""
    # Precomputed offset table for speed
    # idx for edge (i,j) where i<j: sum_{k=0}^{i-1}(n-1-k) + (j-i-1)
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

def _paley(n, p):
    """Paley coloring using prime p >= n."""
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
    """Repair a coloring by targeting specific K2,2,2 violations.
    
    Strategy: Find violations, identify edges that appear in the most violations,
    flip them, and re-check.
    """
    start_time = time.time()
    bitsets = _build_bitsets(colors, n)
    
    if _is_valid(bitsets, n):
        return colors, True
    
    for iteration in range(200):
        if time.time() - start_time > time_limit:
            break
        
        # Find violations
        violations = _find_k222_violations(bitsets, n, max_count=100)
        if not violations:
            return colors, True
        
        # Count edge participation in violations
        # Each K2,2,2 has 12 edges (all pairs among 6 vertices that are in the subgraph)
        # Actually K2,2,2 has 12 edges: between each pair of parts (2*2*3=12)
        edge_freq = {}
        for (c, d, a, b, e, f) in violations:
            # The 6 vertices
            verts = [c, d, a, b, e, f]
            # Edges in K2,2,2: all edges between different parts
            # Parts: {c,d}, {a,b}, {e,f}
            # Edges: c-a, c-b, c-e, c-f, d-a, d-b, d-e, d-f, a-e, a-f, b-e, b-f
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
        
        # Pick the edge with highest frequency
        best_edge = max(edge_freq, key=edge_freq.get)
        u, v = best_edge
        
        # Find the linear index for this edge
        # Compute idx for edge (u, v) where u < v
        idx = 0
        for k in range(u):
            idx += (n - 1 - k)
        idx += (v - u - 1)
        
        # Flip this edge
        old_val = (bitsets[u] >> v) & 1
        if old_val:
            bitsets[u] ^= (1 << v)
            bitsets[v] ^= (1 << u)
            colors[idx] = 0
        else:
            bitsets[u] |= (1 << v)
            bitsets[v] |= (1 << u)
            colors[idx] = 1
        
        # Check if valid now
        if _is_valid(bitsets, n):
            return colors, True
    
    return colors, False

def _simulated_annealing(colors, n, max_iters=50000, time_limit=60):
    """Simulated annealing with bitset maintenance and smart cooling."""
    start_time = time.time()
    bitsets = _build_bitsets(colors, n)
    current_cost = _count_k222_fast(bitsets, n)
    
    if current_cost == 0:
        return colors, True
    
    num_edges = n * (n - 1) // 2
    T0 = float(max(1, current_cost))
    T_min = 0.05
    cooling = 0.998
    
    T = T0
    best_cost = current_cost
    best_colors = colors[:]
    no_improve = 0
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Pick random edge
        idx = random.randint(0, num_edges - 1)
        i, j = _edge_to_ij(idx, n)
        
        # Flip edge
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
                no_improve = 0
                T = max(T0 * 0.3, T_min)
            else:
                no_improve += 1
        else:
            if T > T_min and random.random() < np.exp(-delta / T):
                current_cost = new_cost
                no_improve += 1
            else:
                # Revert
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

def _local_search(colors, n, max_iters=20000, time_limit=40):
    """Steepest-descent local search with bitset maintenance."""
    start_time = time.time()
    bitsets = _build_bitsets(colors, n)
    current_cost = _count_k222_fast(bitsets, n)
    
    if current_cost == 0:
        return colors, True
    
    num_edges = n * (n - 1) // 2
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Try a subset of edges
        if num_edges <= 406:
            candidates = list(range(num_edges))
            random.shuffle(candidates)
            candidates = candidates[:150]
        else:
            candidates = random.sample(range(num_edges), 150)
        
        best_idx = -1
        best_cost = current_cost
        
        for idx in candidates:
            i, j = _edge_to_ij(idx, n)
            
            # Flip
            old_val = (bitsets[i] >> j) & 1
            if old_val:
                bitsets[i] ^= (1 << j)
                bitsets[j] ^= (1 << i)
            else:
                bitsets[i] |= (1 << j)
                bitsets[j] |= (1 << i)
            
            new_cost = _count_k222_fast(bitsets, n)
            
            # Revert
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
    
    # 1. Try Paley(29) directly
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
        _, success = _local_search(colors_ls, target_n, max_iters=15000, time_limit=30)
        if success:
            return {'n': target_n, 'colors': colors_ls}
    
    # 4. Simulated annealing from Paley
    if time.time() < deadline - 40:
        colors_sa = colors[:]
        _, success = _simulated_annealing(colors_sa, target_n, max_iters=40000, time_limit=35)
        if success:
            return {'n': target_n, 'colors': colors_sa}
    
    # 5. Random restarts with targeted repair + SA
    for attempt in range(20):
        if time.time() > deadline - 15:
            break
        colors_rand = [random.randint(0, 1) for _ in range(target_n * (target_n - 1) // 2)]
        
        # Quick targeted repair
        _, success = _targeted_repair(colors_rand, target_n, time_limit=8)
        if success:
            return {'n': target_n, 'colors': colors_rand}
        
        # Then SA
        if time.time() < deadline - 15:
            _, success = _simulated_annealing(colors_rand, target_n, max_iters=15000, time_limit=10)
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
    
    # Absolute fallback
    return {'n': 1, 'colors': [0]}
