# EVOLVE-BLOCK-START
import numpy as np
import time

def _edge_index(i, j):
    """Compute the index of edge (i,j) in lexicographic order for given n.
    We'll pass n implicitly by precomputing a lookup table."""
    pass

def construct():
    """Find a 4-coloring of K_n with no monochromatic C6, maximizing n."""
    start_time = time.time()
    
    for n in range(18, 1, -1):
        if time.time() - start_time > 110:
            break
        
        m = n * (n - 1) // 2
        
        # Build edge index lookup
        edge_idx = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_idx[(i, j)] = idx
                idx += 1
        
        # Precompute all C6 patterns as tuples of edge indices
        patterns = []
        for v0 in range(n):
            for v1 in range(n):
                if v1 == v0:
                    continue
                for v2 in range(n):
                    if v2 in (v0, v1):
                        continue
                    for v3 in range(n):
                        if v3 in (v0, v1, v2):
                            continue
                        for v4 in range(n):
                            if v4 in (v0, v1, v2, v3):
                                continue
                            for v5 in range(n):
                                if v5 in (v0, v1, v2, v3, v4):
                                    continue
                                # Symmetry breaking: v0 is smallest, v1 is second smallest
                                verts = [v0, v1, v2, v3, v4, v5]
                                if min(verts) != v0:
                                    continue
                                remaining = sorted([v for v in verts if v != v0])
                                if v1 != remaining[0]:
                                    continue
                                # Build edge indices
                                edges = [(v0, v1), (v1, v2), (v2, v3), (v3, v4), (v4, v5), (v5, v0)]
                                edge_indices = []
                                valid = True
                                for a, b in edges:
                                    if a > b:
                                        a, b = b, a
                                    if (a, b) not in edge_idx:
                                        valid = False
                                        break
                                    edge_indices.append(edge_idx[(a, b)])
                                if valid:
                                    patterns.append(tuple(edge_indices))
        
        if len(patterns) == 0:
            return {'n': n, 'colors': [0] * m}
        
        # Convert patterns to numpy array for faster access
        patterns_np = np.array(patterns, dtype=np.int32)
        
        np.random.seed(42)
        colors = np.random.randint(0, 4, size=m, dtype=np.int8)
        
        def count_c6s(col):
            """Count monochromatic C6s efficiently."""
            if len(patterns_np) == 0:
                return 0
            # For each pattern, check if all 6 edges have the same color
            # col[pat] gives the 6 colors for that pattern
            pat_colors = col[patterns_np]  # shape (num_patterns, 6)
            # Check if all 6 are equal: min == max
            min_c = pat_colors.min(axis=1)
            max_c = pat_colors.max(axis=1)
            return np.sum(min_c == max_c)
        
        current_energy = count_c6s(colors)
        best_colors = colors.copy()
        best_energy = current_energy
        
        T = 1.0
        T_min = 0.01
        cooling = 0.997
        max_iter = 80000
        
        for it in range(max_iter):
            if time.time() - start_time > 115:
                break
            
            if current_energy == 0:
                return {'n': n, 'colors': colors.tolist()}
            
            ei = np.random.randint(m)
            old_color = colors[ei]
            new_color = np.random.randint(0, 4)
            if new_color == old_color:
                continue
            
            colors[ei] = new_color
            new_energy = count_c6s(colors)
            
            delta = new_energy - current_energy
            if delta < 0 or (T > 0 and np.random.rand() < np.exp(-delta / T)):
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_colors = colors.copy()
            else:
                colors[ei] = old_color
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        if best_energy == 0:
            return {'n': n, 'colors': best_colors.tolist()}
    
    # Fallback: n=17 with extended search
    n = 17
    m = n * (n - 1) // 2
    
    edge_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx[(i, j)] = idx
            idx += 1
    
    patterns = []
    for v0 in range(n):
        for v1 in range(n):
            if v1 == v0:
                continue
            for v2 in range(n):
                if v2 in (v0, v1):
                    continue
                for v3 in range(n):
                    if v3 in (v0, v1, v2):
                        continue
                    for v4 in range(n):
                        if v4 in (v0, v1, v2, v3):
                            continue
                        for v5 in range(n):
                            if v5 in (v0, v1, v2, v3, v4):
                                continue
                            verts = [v0, v1, v2, v3, v4, v5]
                            if min(verts) != v0:
                                continue
                            remaining = sorted([v for v in verts if v != v0])
                            if v1 != remaining[0]:
                                continue
                            edges = [(v0, v1), (v1, v2), (v2, v3), (v3, v4), (v4, v5), (v5, v0)]
                            edge_indices = []
                            valid = True
                            for a, b in edges:
                                if a > b:
                                    a, b = b, a
                                if (a, b) not in edge_idx:
                                    valid = False
                                    break
                                edge_indices.append(edge_idx[(a, b)])
                            if valid:
                                patterns.append(tuple(edge_indices))
    
    patterns_np = np.array(patterns, dtype=np.int32)
    
    np.random.seed(123)
    colors = np.random.randint(0, 4, size=m, dtype=np.int8)
    
    def count_c6s(col):
        if len(patterns_np) == 0:
            return 0
        pat_colors = col[patterns_np]
        min_c = pat_colors.min(axis=1)
        max_c = pat_colors.max(axis=1)
        return np.sum(min_c == max_c)
    
    current_energy = count_c6s(colors)
    best_colors = colors.copy()
    best_energy = current_energy
    
    T = 1.0
    T_min = 0.005
    cooling = 0.999
    max_iter = 150000
    
    for it in range(max_iter):
        if time.time() - start_time > 118:
            break
        
        if current_energy == 0:
            return {'n': n, 'colors': colors.tolist()}
        
        ei = np.random.randint(m)
        old_color = colors[ei]
        new_color = np.random.randint(0, 4)
        if new_color == old_color:
            continue
        
        colors[ei] = new_color
        new_energy = count_c6s(colors)
        
        delta = new_energy - current_energy
        if delta < 0 or (T > 0 and np.random.rand() < np.exp(-delta / T)):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_colors = colors.copy()
        else:
            colors[ei] = old_color
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    if best_energy == 0:
        return {'n': n, 'colors': best_colors.tolist()}
    
    # Ultimate fallback
    n = 9
    m = n * (n - 1) // 2
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append((j - i) % 4)
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
