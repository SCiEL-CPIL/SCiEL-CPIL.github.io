import time
import random
import math
import numpy as np

def construct():
    n = 128
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    # Precompute all nonconstant 4-term APs (0-indexed positions)
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for i in range(0, n - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
            
    num_aps = len(aps)
    
    # Map each position to the APs it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)

    def is_ap_valid(colors, ap):
        c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
        # Need at least 3 distinct colors
        if c0 == c1 == c2 == c3:
            return False
        if c0 == c1 == c2 or c0 == c1 == c3 or c0 == c2 == c3 or c1 == c2 == c3:
            return False
        if c0 == c1 and c2 == c3:
            return False
        if c0 == c2 and c1 == c3:
            return False
        if c0 == c3 and c1 == c2:
            return False
        return True

    def count_violations(colors):
        """Count number of APs with fewer than 3 distinct colors."""
        v = 0
        for ap in aps:
            if not is_ap_valid(colors, ap):
                v += 1
        return v

    def spectral_coloring(k, time_limit):
        """
        Spectral graph theory approach:
        1. Build a constraint graph where edges connect positions that appear together in 4-APs
        2. Weight edges by how many APs they share
        3. Compute eigenvectors of the Laplacian
        4. Assign colors based on clustering in eigenspace
        5. Refine with local search
        """
        # Build weighted adjacency matrix
        adj = np.zeros((n, n), dtype=np.float64)
        for ap in aps:
            a, b, c, d = ap
            # All pairs in this AP are constrained
            adj[a, b] += 1; adj[b, a] += 1
            adj[a, c] += 1; adj[c, a] += 1
            adj[a, d] += 1; adj[d, a] += 1
            adj[b, c] += 1; adj[c, b] += 1
            adj[b, d] += 1; adj[d, b] += 1
            adj[c, d] += 1; adj[d, c] += 1
        
        # Compute Laplacian L = D - A
        degree = adj.sum(axis=1)
        L = np.diag(degree) - adj
        
        # Compute eigenvectors
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
        except:
            # Fallback to random
            return [random.randint(0, k - 1) for _ in range(n)]
        
        # Use the first few non-trivial eigenvectors for embedding
        # Skip the first eigenvector (constant)
        num_components = min(k, n - 1)
        embedding = eigenvectors[:, 1:num_components + 1]
        
        # Cluster using simple thresholding on principal components
        # For k colors, use k-1 components and assign based on quadrant/region
        colors = [0] * n
        
        if k == 2:
            # Use first non-trivial eigenvector
            vals = embedding[:, 0]
            mean_val = np.mean(vals)
            for i in range(n):
                colors[i] = 0 if vals[i] < mean_val else 1
        elif k == 3:
            # Use first two non-trivial eigenvectors
            v1 = embedding[:, 0]
            v2 = embedding[:, 1]
            m1, m2 = np.mean(v1), np.mean(v2)
            for i in range(n):
                x, y = v1[i] - m1, v2[i] - m2
                if x >= 0 and y >= 0:
                    colors[i] = 0
                elif x < 0 and y >= 0:
                    colors[i] = 1
                elif x < 0 and y < 0:
                    colors[i] = 2
                else:
                    colors[i] = 0
        elif k == 4:
            # Use first three non-trivial eigenvectors
            v1, v2, v3 = embedding[:, 0], embedding[:, 1], embedding[:, 2]
            m1, m2, m3 = np.mean(v1), np.mean(v2), np.mean(v3)
            for i in range(n):
                x, y, z = v1[i] - m1, v2[i] - m2, v3[i] - m3
                if x >= 0 and y >= 0:
                    colors[i] = 0
                elif x >= 0 and y < 0:
                    colors[i] = 1
                elif x < 0 and y >= 0:
                    colors[i] = 2
                else:
                    colors[i] = 3
        else:
            # General case: use k-1 components and assign by rounding
            for i in range(n):
                # Use first k-1 components, normalize, and assign
                vec = embedding[i, :k-1]
                vec = vec / (np.linalg.norm(vec) + 1e-10)
                # Map to color by argmax of projected values
                proj = []
                for c in range(k):
                    # Simple projection
                    proj.append(np.dot(vec, np.cos(np.linspace(0, 2*np.pi, k)[c])))
                colors[i] = int(np.argmax(proj))
        
        # Local search refinement
        ap_status = [is_ap_valid(colors, ap) for ap in aps]
        violating_set = set(i for i, v in enumerate(ap_status) if not v)
        
        best_colors = colors[:]
        best_violations = len(violating_set)
        
        temp = 2.0
        cooling = 0.999
        min_temp = 0.001
        
        while time.time() < time_limit and temp > min_temp:
            if len(violating_set) < best_violations:
                best_violations = len(violating_set)
                best_colors = colors[:]
                if best_violations == 0:
                    break
            
            if len(violating_set) == 0:
                break
            
            pos = random.randint(0, n - 1)
            old_color = colors[pos]
            new_color = random.randint(0, k - 1)
            
            if new_color == old_color:
                temp *= cooling
                continue
            
            # Calculate delta
            delta = 0
            for ap_idx in pos_to_aps[pos]:
                ap = aps[ap_idx]
                old_valid = ap_status[ap_idx]
                
                # Simulate new color
                c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                if ap[0] == pos: c0 = new_color
                elif ap[1] == pos: c1 = new_color
                elif ap[2] == pos: c2 = new_color
                elif ap[3] == pos: c3 = new_color
                
                # Check validity
                new_valid = True
                if c0 == c1 == c2 == c3:
                    new_valid = False
                elif c0 == c1 == c2 or c0 == c1 == c3 or c0 == c2 == c3 or c1 == c2 == c3:
                    new_valid = False
                elif c0 == c1 and c2 == c3:
                    new_valid = False
                elif c0 == c2 and c1 == c3:
                    new_valid = False
                elif c0 == c3 and c1 == c2:
                    new_valid = False
                    
                if old_valid and not new_valid:
                    delta += 1
                elif not old_valid and new_valid:
                    delta -= 1
            
            # Acceptance
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = True
            else:
                prob = math.exp(-delta / temp)
                accept = random.random() < prob
            
            if accept:
                colors[pos] = new_color
                # Update state
                for ap_idx in pos_to_aps[pos]:
                    ap = aps[ap_idx]
                    c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                    new_valid = True
                    if c0 == c1 == c2 == c3:
                        new_valid = False
                    elif c0 == c1 == c2 or c0 == c1 == c3 or c0 == c2 == c3 or c1 == c2 == c3:
                        new_valid = False
                    elif c0 == c1 and c2 == c3:
                        new_valid = False
                    elif c0 == c2 and c1 == c3:
                        new_valid = False
                    elif c0 == c3 and c1 == c2:
                        new_valid = False
                        
                    old_valid = ap_status[ap_idx]
                    if old_valid != new_valid:
                        ap_status[ap_idx] = new_valid
                        if new_valid:
                            violating_set.discard(ap_idx)
                        else:
                            violating_set.add(ap_idx)
            
            temp *= cooling
        
        return best_colors

    def greedy_construct():
        """Greedy construction with many colors to ensure validity."""
        colors = [None] * n
        for x in range(n):
            for color in range(128):
                valid = True
                for ap_idx in pos_to_aps[x]:
                    ap = aps[ap_idx]
                    cset = set()
                    for pos in ap:
                        if pos == x:
                            cset.add(color)
                        else:
                            if colors[pos] is None:
                                valid = False
                                break
                            cset.add(colors[pos])
                    if not valid:
                        break
                    if len(cset) < 3:
                        valid = False
                        break
                if valid:
                    colors[x] = color
                    break
            if colors[x] is None:
                colors[x] = x
        return colors

    def try_reduce_colors(colors, time_limit):
        """Try to reduce the number of colors used."""
        current_colors = set(colors)
        if len(current_colors) <= 3:
            return colors
        
        # Map colors to 0..k-1
        color_map = {c: i for i, c in enumerate(sorted(current_colors))}
        mapped = [color_map[c] for c in colors]
        k = len(current_colors)
        
        # Try to merge colors
        for _ in range(20):
            if time.time() > time_limit:
                break
            if k <= 3:
                break
            
            # Try merging two random colors
            c1, c2 = random.sample(range(k), 2)
            new_colors = [c2 if c == c1 else c for c in mapped]
            
            # Check if still valid
            if count_violations(new_colors) == 0:
                # Relabel
                new_set = set(new_colors)
                new_map = {c: i for i, c in enumerate(sorted(new_set))}
                mapped = [new_map[c] for c in new_colors]
                k = len(new_set)
        
        return mapped

    # Main logic: try spectral approach with different k values
    best_result = None
    best_num_colors = 128
    
    for k in [3, 4, 5, 6]:
        if time.time() > deadline - 10:
            break
        
        time_limit = time.time() + 15.0
        result = spectral_coloring(k, time_limit)
        
        if result is not None:
            violations = count_violations(result)
            if violations == 0:
                num_used = len(set(result))
                if num_used < best_num_colors:
                    best_num_colors = num_used
                    best_result = result
                break  # Found valid solution with minimal k tried
    
    # If no valid solution found, use greedy
    if best_result is None:
        best_result = greedy_construct()
    
    # Try to reduce colors further
    if time.time() < deadline - 5:
        best_result = try_reduce_colors(best_result, time.time() + 5.0)
    
    # Final validation and relabeling
    final_colors = list(range(n))
    color_map = {}
    next_color = 0
    for c in best_result:
        if c not in color_map:
            color_map[c] = next_color
            next_color += 1
        final_colors[color_map[c]] = color_map[c]
    
    # Verify validity
    if not all(is_ap_valid(final_colors, ap) for ap in aps):
        # Fallback to greedy
        final_colors = greedy_construct()
        color_map = {}
        next_color = 0
        for c in final_colors:
            if c not in color_map:
                color_map[c] = next_color
                next_color += 1
        final_colors = [color_map[c] for c in final_colors]
    
    return final_colors
