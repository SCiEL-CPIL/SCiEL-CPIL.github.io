import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Try multiple strategies, keep best result
    best_result = None
    best_n = 0
    
    # Strategy 1: Try to construct n=26 with focused local search
    r = focused_local_search(26, time_limit=25.0)
    if r is not None:
        best_result = r
        best_n = 26
        # Try to extend to 27
        r27 = try_extend_focused(r, time_limit=40.0)
        if r27 is not None:
            return r27
    elif time.time() < deadline:
        # Strategy 2: Incremental construction
        r = incremental_construct(26, time_limit=20.0)
        if r is not None:
            best_result = r
            best_n = 26
            r27 = try_extend_focused(r, time_limit=40.0)
            if r27 is not None:
                return r27
    
    # Try more n=26 attempts if time permits
    if best_result is None and time.time() < deadline - 30:
        for attempt in range(3):
            if time.time() > deadline - 30:
                break
            r = focused_local_search(26, time_limit=8.0)
            if r is not None:
                best_result = r
                best_n = 26
                break
    
    if best_result is not None:
        return best_result
    
    # Fallback: smaller n
    for n in [25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]:
        if time.time() > deadline - 5:
            break
        r = focused_local_search(n, time_limit=5.0)
        if r is not None:
            return r
    
    # Absolute fallback
    return {'n': 11, 'colors': [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]}


def focused_local_search(n, time_limit=25.0):
    """Local search that focuses on edges involved in violations."""
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Initialize with random coloring
    colors = [random.randint(0, 4) for _ in range(num_edges)]
    
    # Build adjacency bitmasks
    adj_mask = [[0] * n for _ in range(5)]
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        adj_mask[c][i] |= (1 << j)
        adj_mask[c][j] |= (1 << i)
    
    # Compute violations and track which edges are in violations
    def compute_violations_and_edges(adj_mask):
        """Returns (total_violations, set_of_edge_indices_in_violations)."""
        total = 0
        violating_edges = set()
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj_mask[c][u] & adj_mask[c][v]
                    k = common.bit_count()
                    if k >= 2:
                        # C4 exists: u, v, w1, w2 where w1,w2 in common
                        total += k * (k - 1) // 2
                        # Mark edges (u,v) and edges from u,v to common neighbors
                        # Edge (u,v) is in the C4
                        # Also edges (u,w), (v,w) for w in common
                        common_list = []
                        temp = common
                        while temp:
                            w = (temp & -temp).bit_length() - 1
                            common_list.append(w)
                            temp &= temp - 1
                        # Edge (u,v)
                        if u < v:
                            ei = u * (n - 1) - u * (u - 1) // 2 + (v - u - 1)
                            violating_edges.add(ei)
                        # Edges (u,w) and (v,w) for each w in common
                        for w in common_list:
                            if u < w:
                                ei = u * (n - 1) - u * (u - 1) // 2 + (w - u - 1)
                                violating_edges.add(ei)
                            else:
                                ei = w * (n - 1) - w * (w - 1) // 2 + (u - w - 1)
                                violating_edges.add(ei)
                            if v < w:
                                ei = v * (n - 1) - v * (v - 1) // 2 + (w - v - 1)
                                violating_edges.add(ei)
                            else:
                                ei = w * (n - 1) - w * (w - 1) // 2 + (v - w - 1)
                                violating_edges.add(ei)
        return total, violating_edges
    
    current_violations, violating_edges = compute_violations_and_edges(adj_mask)
    
    if current_violations == 0:
        return {'n': n, 'colors': colors}
    
    best_violations = current_violations
    best_colors = colors[:]
    
    start_time = time.time()
    temperature = 1.5
    cooling_rate = 0.9997
    no_improve = 0
    max_no_improve = 3000
    
    while time.time() - start_time < time_limit:
        if no_improve > max_no_improve:
            # Restart
            colors = [random.randint(0, 4) for _ in range(num_edges)]
            adj_mask = [[0] * n for _ in range(5)]
            for idx, (i, j) in enumerate(edges):
                c = colors[idx]
                adj_mask[c][i] |= (1 << j)
                adj_mask[c][j] |= (1 << i)
            current_violations, violating_edges = compute_violations_and_edges(adj_mask)
            temperature = 1.5
            no_improve = 0
            continue
        
        # Select edge: 70% from violating edges, 30% random
        if violating_edges and random.random() < 0.7:
            e_idx = random.choice(list(violating_edges))
        else:
            e_idx = random.randint(0, num_edges - 1)
        
        i, j = edges[e_idx]
        old_color = colors[e_idx]
        
        # Try all 5 colors, pick best
        best_new_color = old_color
        best_new_violations = current_violations
        
        for new_color in range(5):
            if new_color == old_color:
                continue
            
            # Apply change
            adj_mask[old_color][i] &= ~(1 << j)
            adj_mask[old_color][j] &= ~(1 << i)
            adj_mask[new_color][i] |= (1 << j)
            adj_mask[new_color][j] |= (1 << i)
            
            new_violations, _ = compute_violations_and_edges(adj_mask)
            
            if new_violations < best_new_violations:
                best_new_violations = new_violations
                best_new_color = new_color
            
            # Revert
            adj_mask[new_color][i] &= ~(1 << j)
            adj_mask[new_color][j] &= ~(1 << i)
            adj_mask[old_color][i] |= (1 << j)
            adj_mask[old_color][j] |= (1 << i)
        
        if best_new_color != old_color:
            # Apply best change
            adj_mask[old_color][i] &= ~(1 << j)
            adj_mask[old_color][j] &= ~(1 << i)
            adj_mask[best_new_color][i] |= (1 << j)
            adj_mask[best_new_color][j] |= (1 << i)
            colors[e_idx] = best_new_color
            
            delta = best_new_violations - current_violations
            if delta < 0 or random.random() < math.exp(-delta / max(temperature, 0.01)):
                current_violations = best_new_violations
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
                    no_improve = 0
                    if best_violations == 0:
                        return {'n': n, 'colors': colors}
                else:
                    no_improve += 1
                # Recompute violating edges for next iteration
                current_violations, violating_edges = compute_violations_and_edges(adj_mask)
            else:
                # Revert
                adj_mask[best_new_color][i] &= ~(1 << j)
                adj_mask[best_new_color][j] &= ~(1 << i)
                adj_mask[old_color][i] |= (1 << j)
                adj_mask[old_color][j] |= (1 << i)
                colors[e_idx] = old_color
                no_improve += 1
        else:
            no_improve += 1
        
        temperature *= cooling_rate
        if temperature < 0.05:
            temperature = 0.05
    
    return None


def incremental_construct(n, time_limit=20.0):
    """Try to build a valid coloring incrementally vertex by vertex."""
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = n * (n - 1) // 2
    
    # Start with a small valid coloring and grow
    # For simplicity, start random and repair
    colors = [random.randint(0, 4) for _ in range(num_edges)]
    
    adj_mask = [[0] * n for _ in range(5)]
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        adj_mask[c][i] |= (1 << j)
        adj_mask[c][j] |= (1 << i)
    
    def count_violations():
        total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj_mask[c][u] & adj_mask[c][v]
                    k = common.bit_count()
                    if k >= 2:
                        total += k * (k - 1) // 2
        return total
    
    current_violations = count_violations()
    if current_violations == 0:
        return {'n': n, 'colors': colors}
    
    start_time = time.time()
    temperature = 1.0
    
    while time.time() - start_time < time_limit:
        if current_violations == 0:
            return {'n': n, 'colors': colors}
        
        # Pick a random edge
        e_idx = random.randint(0, num_edges - 1)
        i, j = edges[e_idx]
        old_color = colors[e_idx]
        new_color = random.randint(0, 4)
        if new_color == old_color:
            continue
        
        # Apply
        adj_mask[old_color][i] &= ~(1 << j)
        adj_mask[old_color][j] &= ~(1 << i)
        adj_mask[new_color][i] |= (1 << j)
        adj_mask[new_color][j] |= (1 << i)
        colors[e_idx] = new_color
        
        new_violations = count_violations()
        delta = new_violations - current_violations
        
        if delta < 0 or random.random() < math.exp(-delta / max(temperature, 0.01)):
            current_violations = new_violations
        else:
            # Revert
            adj_mask[new_color][i] &= ~(1 << j)
            adj_mask[new_color][j] &= ~(1 << i)
            adj_mask[old_color][i] |= (1 << j)
            adj_mask[old_color][j] |= (1 << i)
            colors[e_idx] = old_color
        
        temperature *= 0.999
        if temperature < 0.05:
            temperature = 0.05
    
    return None


def try_extend_focused(base_result, time_limit=40.0):
    """Extend a valid n-vertex coloring to n+1 vertices using focused search."""
    base_colors = base_result['colors']
    n_base = base_result['n']
    n_new = n_base + 1
    
    # Build adjacency masks for base
    adj_mask = [[0] * n_base for _ in range(5)]
    idx = 0
    for i in range(n_base):
        for j in range(i + 1, n_base):
            c = base_colors[idx]
            adj_mask[c][i] |= (1 << j)
            adj_mask[c][j] |= (1 << i)
            idx += 1
    
    def count_violations_new(new_colors):
        """Count C4 violations involving the new vertex."""
        violations = 0
        for c in range(5):
            neighbors = [i for i in range(n_base) if new_colors[i] == c]
            for a in range(len(neighbors)):
                u = neighbors[a]
                for b in range(a + 1, len(neighbors)):
                    v = neighbors[b]
                    common = adj_mask[c][u] & adj_mask[c][v]
                    k = common.bit_count()
                    if k >= 1:
                        violations += k
        return violations
    
    # Initialize
    new_colors = [random.randint(0, 4) for _ in range(n_base)]
    current_violations = count_violations_new(new_colors)
    
    if current_violations == 0:
        return build_full_coloring(base_colors, new_colors, n_base)
    
    best_violations = current_violations
    best_new_colors = new_colors[:]
    
    start_time = time.time()
    temperature = 1.0
    no_improve = 0
    
    while time.time() - start_time < time_limit:
        if best_violations == 0:
            break
        if no_improve > 2000:
            new_colors = [random.randint(0, 4) for _ in range(n_base)]
            current_violations = count_violations_new(new_colors)
            temperature = 1.0
            no_improve = 0
            continue
        
        # Pick a variable: prefer those in violating pairs
        e_idx = random.randint(0, n_base - 1)
        old_c = new_colors[e_idx]
        new_c = random.randint(0, 4)
        if new_c == old_c:
            continue
        
        new_colors[e_idx] = new_c
        new_violations = count_violations_new(new_colors)
        
        delta = new_violations - current_violations
        if delta < 0 or random.random() < math.exp(-delta / max(temperature, 0.01)):
            current_violations = new_violations
            if current_violations < best_violations:
                best_violations = current_violations
                best_new_colors = new_colors[:]
                no_improve = 0
                if best_violations == 0:
                    break
            else:
                no_improve += 1
        else:
            new_colors[e_idx] = old_c
            no_improve += 1
        
        temperature *= 0.999
        if temperature < 0.05:
            temperature = 0.05
    
    if best_violations == 0:
        return build_full_coloring(base_colors, best_new_colors, n_base)
    
    return None


def build_full_coloring(base_colors, new_colors, n_base):
    """Build the full (n_base+1)-vertex coloring from base and new vertex colors."""
    n_new = n_base + 1
    final_colors = [0] * (n_new * (n_new - 1) // 2)
    
    idx = 0
    for i in range(n_new):
        for j in range(i + 1, n_new):
            if j < n_base:
                # Edge in base graph
                base_idx = i * (n_base - 1) - i * (i - 1) // 2 + (j - i - 1)
                final_colors[idx] = base_colors[base_idx]
            else:
                # Edge to new vertex (j == n_base)
                final_colors[idx] = new_colors[i]
            idx += 1
    
    return {'n': n_new, 'colors': final_colors}
