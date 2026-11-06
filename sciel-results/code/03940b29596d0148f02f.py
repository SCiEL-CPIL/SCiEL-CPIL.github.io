import time
import random
import itertools
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

def _count_k222_bitsets(bitsets, n):
    """Count K2,2,2 subgraphs using bitset operations. Fast O(n^4/word) per call."""
    count = 0
    for c in range(n - 1):
        bc = bitsets[c]
        for d in range(c + 1, n):
            if not (bc >> d) & 1:
                continue
            # Common neighbors of c and d (excluding c,d themselves via bitmasks)
            common = bc & bitsets[d]
            # Count K2,2 in subgraph induced by 'common'
            # For each pair (a,b) in common, count common neighbors of a and b within common
            # Vectorize: iterate over set bits in common
            common_bits = common
            verts = []
            while common_bits:
                lsb = common_bits & (-common_bits)
                v = lsb.bit_length() - 1
                verts.append(v)
                common_bits ^= lsb
            m = len(verts)
            if m < 4:
                continue
            for ii in range(m - 1):
                a = verts[ii]
                ba = bitsets[a]
                for jj in range(ii + 1, m):
                    b = verts[jj]
                    # Common neighbors of a and b that are also in common
                    nb = ba & bitsets[b] & common
                    # Count bits in nb
                    cnt = nb.bit_count()
                    if cnt >= 2:
                        count += 1
    return count

def _is_valid_bitsets(bitsets, n):
    return _count_k222_bitsets(bitsets, n) == 0

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

def _simulated_annealing_bitsets(colors, n, max_iters=20000, time_limit=100):
    """QUBO-inspired simulated annealing with dynamic temperature for K2,2,2 elimination."""
    start_time = time.time()
    bitsets = _build_bitsets(colors, n)
    current_cost = _count_k222_bitsets(bitsets, n)
    
    if current_cost == 0:
        return colors, True
    
    # Dynamic temperature schedule
    T0 = float(max(1, current_cost))
    T_min = 0.01
    cooling = 0.995  # geometric cooling
    
    T = T0
    best_cost = current_cost
    best_colors = colors[:]
    no_improve = 0
    
    num_edges = n * (n - 1) // 2
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Pick random edge to flip
        idx = random.randint(0, num_edges - 1)
        
        # Find i, j from idx
        i = 0
        base = 0
        while True:
            next_base = (i + 1) * (2 * n - i - 2) // 2
            if idx < next_base:
                break
            i += 1
            base = next_base
        j = i + 1 + (idx - base)
        
        # Flip edge (i, j)
        old_val = (bitsets[i] >> j) & 1
        if old_val:
            bitsets[i] ^= (1 << j)
            bitsets[j] ^= (1 << i)
            colors[idx] = 0
        else:
            bitsets[i] |= (1 << j)
            bitsets[j] |= (1 << i)
            colors[idx] = 1
        
        new_cost = _count_k222_bitsets(bitsets, n)
        delta = new_cost - current_cost
        
        if delta <= 0:
            # Accept improvement or equal
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors[:]
                no_improve = 0
                # Reset temperature on improvement
                T = max(T0 * 0.5, T_min)
            else:
                no_improve += 1
        else:
            # Metropolis criterion: accept with probability exp(-delta/T)
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
        
        # Cool down
        T *= cooling
        
        if current_cost == 0:
            return colors, True
    
    # Return best found
    return best_colors, best_cost == 0

def _local_search_bitsets(colors, n, max_iters=10000, time_limit=30):
    """Steepest-descent local search using bitset K2,2,2 counting."""
    start_time = time.time()
    bitsets = _build_bitsets(colors, n)
    current_cost = _count_k222_bitsets(bitsets, n)
    
    if current_cost == 0:
        return colors, True
    
    num_edges = n * (n - 1) // 2
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Try all single-edge flips, pick best improvement
        best_idx = -1
        best_cost = current_cost
        
        # Sample a subset of edges for speed (if n is large)
        if num_edges <= 500:
            candidates = range(num_edges)
        else:
            candidates = random.sample(range(num_edges), min(200, num_edges))
        
        for idx in candidates:
            i = 0
            base = 0
            while True:
                next_base = (i + 1) * (2 * n - i - 2) // 2
                if idx < next_base:
                    break
                i += 1
                base = next_base
            j = i + 1 + (idx - base)
            
            # Flip
            old_val = (bitsets[i] >> j) & 1
            if old_val:
                bitsets[i] ^= (1 << j)
                bitsets[j] ^= (1 << i)
            else:
                bitsets[i] |= (1 << j)
                bitsets[j] |= (1 << i)
            
            new_cost = _count_k222_bitsets(bitsets, n)
            
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
            # Apply best flip
            i = 0
            base = 0
            while True:
                next_base = (i + 1) * (2 * n - i - 2) // 2
                if best_idx < next_base:
                    break
                i += 1
                base = next_base
            j = i + 1 + (best_idx - base)
            
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
            # No improving move found
            break
    
    return colors, current_cost == 0

def construct():
    deadline = time.time() + 140
    target_n = 29
    p = 29  # 29 is prime
    
    # 1. Try Paley(29) directly
    colors = _paley(target_n, p)
    if colors:
        bitsets = _build_bitsets(colors, target_n)
        if _is_valid_bitsets(bitsets, target_n):
            return {'n': target_n, 'colors': colors}
    
    # 2. Local search from Paley
    colors_copy = colors[:]
    _, success = _local_search_bitsets(colors_copy, target_n, max_iters=5000, time_limit=40)
    if success:
        return {'n': target_n, 'colors': colors_copy}
    
    # 3. Simulated annealing from Paley
    if time.time() < deadline - 30:
        colors_sa = colors[:]
        _, success = _simulated_annealing_bitsets(colors_sa, target_n, max_iters=30000, time_limit=60)
        if success:
            return {'n': target_n, 'colors': colors_sa}
    
    # 4. Random restarts with simulated annealing
    for attempt in range(30):
        if time.time() > deadline - 10:
            break
        colors_rand = [random.randint(0, 1) for _ in range(target_n * (target_n - 1) // 2)]
        _, success = _simulated_annealing_bitsets(colors_rand, target_n, max_iters=10000, time_limit=15)
        if success:
            return {'n': target_n, 'colors': colors_rand}
    
    # 5. Fallback: try smaller n with Paley
    for n in range(28, 10, -1):
        if time.time() > deadline - 5:
            break
        for pp in [17, 19, 23, 29, 31]:
            if pp >= n:
                colors = _paley(n, pp)
                if colors:
                    bitsets = _build_bitsets(colors, n)
                    if _is_valid_bitsets(bitsets, n):
                        return {'n': n, 'colors': colors}
    
    # Absolute fallback
    return {'n': 1, 'colors': [0]}
