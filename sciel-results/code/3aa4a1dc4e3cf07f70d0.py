import time
import itertools
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    """
    Return 120 integer colours in 0..119 for the pairs (i,j), 0<=i<j<16, in lexicographic order.
    Every set of four vertices must span edges of at least five distinct colours.
    Minimize the number of colours used.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = 16
    max_colors = 120
    
    # Precompute edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
            
    num_edges = len(edges)
    
    # Map (i, j) to edge index
    ij_to_edge_idx = {e: idx for idx, e in enumerate(edges)}
    
    # Precompute for each K4, the indices of its 6 edges
    k4s = list(itertools.combinations(range(n), 4))
    num_k4s = len(k4s)
    
    # k4_edges is an array of shape (num_k4s, 6) containing edge indices
    k4_edges = np.zeros((num_k4s, 6), dtype=np.int32)
    k4_idx = 0
    for verts in k4s:
        edge_indices_in_k4 = []
        for i in range(4):
            for j in range(i + 1, 4):
                u = verts[i]
                v = verts[j]
                edge_indices_in_k4.append(ij_to_edge_idx[(u, v)])
        k4_edges[k4_idx] = edge_indices_in_k4
        k4_idx += 1
        
    # Precompute edge_to_k4s for incremental updates
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx in range(num_k4s):
        for e_idx in k4_edges[k4_idx]:
            edge_to_k4s[int(e_idx)].append(k4_idx)
    
    # Convert edge_to_k4s to numpy arrays for faster access
    edge_to_k4s_np = [np.array(e2k, dtype=np.int32) if e2k else np.array([], dtype=np.int32) for e2k in edge_to_k4s]

    def count_distinct_vectorized(k4_indices, colors_np):
        """
        Vectorized count of distinct colors in each K4.
        Uses sorting and diff to count distinct values.
        """
        if len(k4_indices) == 0:
            return np.array([], dtype=np.int32)
        
        k4_idx_arr = np.array(k4_indices, dtype=np.int32)
        k4_colors = colors_np[k4_edges[k4_idx_arr]]  # shape (num_k4s, 6)
        
        # Sort each row
        sorted_colors = np.sort(k4_colors, axis=1)
        
        # Count distinct: 1 + number of positions where sorted[i] != sorted[i-1]
        # For 6 elements, diff has 5 elements
        diff = np.diff(sorted_colors, axis=1)  # shape (num_k4s, 5)
        counts = 1 + np.sum(diff != 0, axis=1)
        
        return counts.astype(np.int32)

    def get_violating_k4s(colors_np):
        """
        Returns list of K4 indices with < 5 distinct colors.
        """
        counts = count_distinct_vectorized(list(range(num_k4s)), colors_np)
        return np.where(counts < 5)[0].tolist()

    def is_valid(colors_np):
        """
        Check if coloring is valid (all K4s have >= 5 distinct colors).
        """
        counts = count_distinct_vectorized(list(range(num_k4s)), colors_np)
        return bool(np.all(counts >= 5))

    def try_merge_color_fast(colors_np, c_from, c_to):
        """
        Try to change all edges of color c_from to c_to.
        Returns (success, new_colors)
        """
        edges_with_from = np.where(colors_np == c_from)[0]
        if len(edges_with_from) == 0:
            return True, colors_np
        
        temp_colors = colors_np.copy()
        temp_colors[edges_with_from] = c_to
        
        # Only check K4s that contain at least one edge of color c_from
        k4s_to_check_set = set()
        for e_idx in edges_with_from:
            for k4_idx in edge_to_k4s_np[int(e_idx)]:
                k4s_to_check_set.add(int(k4_idx))
        
        if len(k4s_to_check_set) > 0:
            k4s_to_check_list = list(k4s_to_check_set)
            counts = count_distinct_vectorized(k4s_to_check_list, temp_colors)
            if np.any(counts < 5):
                return False, colors_np
        
        return True, temp_colors

    def get_initial_coloring(method=0, rng=None):
        """
        Generate different initial colorings.
        """
        if rng is None:
            rng = random.Random(42)
        
        if method == 0:
            # (i + j) % 17
            return np.array([(i + j) % 17 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
        elif method == 1:
            # (i + 2*j) % 17
            return np.array([(i + 2*j) % 17 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
        elif method == 2:
            # (2*i + j) % 17
            return np.array([(2*i + j) % 17 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
        elif method == 3:
            # (i + j) % 16
            return np.array([(i + j) % 16 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
        elif method == 4:
            # (i * j) % 17
            return np.array([(i * j) % 17 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
        elif method == 5:
            # (i + j) % 15
            return np.array([(i + j) % 15 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
        elif method == 6:
            # (i + 3*j) % 17
            return np.array([(i + 3*j) % 17 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
        elif method == 7:
            # (3*i + j) % 17
            return np.array([(3*i + j) % 17 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
        elif method == 8:
            # (i*j + i + j) % 17
            return np.array([(i*j + i + j) % 17 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
        elif method == 9:
            # (i^2 + j) % 17
            return np.array([(i*i + j) % 17 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
        elif method == 10:
            # (i + j^2) % 17
            return np.array([(i + j*j) % 17 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
        else:
            # Random with given seed
            return np.array([rng.randint(0, 16) for _ in range(num_edges)], dtype=np.int32)

    # Main search
    best_colors = None
    best_num_used = max_colors + 1
    
    rng = random.Random(42)
    
    # Try multiple initial colorings
    num_init_methods = 11
    for init_method in range(num_init_methods):
        if time.time() > deadline:
            break
        
        colors = get_initial_coloring(init_method, rng)
        
        # Check if valid
        violations = get_violating_k4s(colors)
        num_viol = len(violations)
        num_used = len(set(colors.tolist()))
        
        # If invalid, try to repair by adding colors
        if num_viol > 0:
            # For each violating K4, recolor one edge to a new color
            for v_k4 in violations:
                if time.time() > deadline:
                    break
                e_idxs = k4_edges[v_k4]
                e_idx = int(e_idxs[rng.randint(0, 5)])
                # Assign a new unique color
                new_c = int(max(colors)) + 1
                colors[e_idx] = new_c
            
            # Re-check
            violations = get_violating_k4s(colors)
            num_viol = len(violations)
            num_used = len(set(colors.tolist()))
        
        # Update best
        if num_viol == 0:
            if num_used < best_num_used:
                best_colors = colors.copy()
                best_num_used = num_used
        
        # Greedy merge phase - more aggressive
        current_colors = colors.copy()
        current_used = set(current_colors.tolist())
        
        max_merge_iters = 50
        for merge_iter in range(max_merge_iters):
            if time.time() > deadline:
                break
            
            progress = False
            colors_list = list(current_used)
            rng.shuffle(colors_list)
            
            for c_from in colors_list:
                if c_from not in current_used:
                    continue
                
                # Try to merge c_from into each other color
                for c_to in colors_list:
                    if c_from == c_to or c_to not in current_used:
                        continue
                    
                    success, new_colors = try_merge_color_fast(current_colors, c_from, c_to)
                    if success:
                        current_colors = new_colors
                        current_used.remove(c_from)
                        progress = True
                        
                        # Update best
                        num_used = len(current_used)
                        if num_used < best_num_used:
                            best_colors = current_colors.copy()
                            best_num_used = num_used
                        break  # Restart merge loop with new state
                
                if progress:
                    break
            
            if not progress:
                break
        
        # Local search phase to escape local minima
        if time.time() > deadline:
            break
        
        current_colors = current_colors.copy()
        current_used = set(current_colors.tolist())
        current_num_used = len(current_used)
        
        # Local search: try to reduce colors by recoloring individual edges
        ls_steps = 500
        for ls_step in range(ls_steps):
            if time.time() > deadline:
                break
            
            # Pick a random edge
            edge_idx = rng.randint(0, num_edges - 1)
            old_color = int(current_colors[edge_idx])
            
            # Try to change to another existing color
            other_colors = [c for c in current_used if c != old_color]
            if not other_colors:
                continue
            
            new_color = rng.choice(other_colors)
            
            # Check if this change is valid
            temp_colors = current_colors.copy()
            temp_colors[edge_idx] = new_color
            
            # Check affected K4s
            affected_k4s = edge_to_k4s_np[edge_idx]
            if len(affected_k4s) > 0:
                counts = count_distinct_vectorized(affected_k4s.tolist(), temp_colors)
                if np.any(counts < 5):
                    continue
            
            # Check if this reduces color count
            new_used = set(temp_colors.tolist())
            new_num_used = len(new_used)
            
            if new_num_used < current_num_used:
                current_colors = temp_colors
                current_used = new_used
                current_num_used = new_num_used
                
                # Update best
                if current_num_used < best_num_used:
                    best_colors = current_colors.copy()
                    best_num_used = current_num_used
        
        # Simulated annealing phase
        if time.time() > deadline:
            break
        
        current_colors = current_colors.copy()
        current_used = set(current_colors.tolist())
        current_num_used = len(current_used)
        
        # SA parameters
        current_temp = 30.0
        cooling_rate = 0.99
        sa_steps = 1000
        
        for sa_step in range(sa_steps):
            if time.time() > deadline:
                break
            
            # Generate a move: recolor a random edge
            edge_idx = rng.randint(0, num_edges - 1)
            old_color = int(current_colors[edge_idx])
            
            # Choose new color: 80% from existing, 20% try to merge
            if rng.random() < 0.8:
                existing = list(current_used)
                if existing:
                    new_color = rng.choice(existing)
                else:
                    new_color = old_color
            else:
                # Try to use a color that might reduce count
                max_c = max(current_used) if current_used else 0
                new_color = rng.randint(0, max_c)
            
            if old_color == new_color:
                continue
            
            current_colors[edge_idx] = new_color
            
            # Calculate energy change
            # Only check affected K4s
            affected_k4s = edge_to_k4s_np[edge_idx]
            if len(affected_k4s) > 0:
                counts = count_distinct_vectorized(affected_k4s.tolist(), current_colors)
                new_viol = int(np.sum(counts < 5))
            else:
                new_viol = 0
            
            # New color count
            new_used = set(current_colors.tolist())
            new_num_used = len(new_used)
            
            # Energy: prioritize reducing colors, then violations
            old_energy = current_num_used * 1000 + 0  # Assume current is valid
            new_energy = new_num_used * 1000 + new_viol
            
            delta_energy = new_energy - old_energy
            
            # Accept if better, or with probability exp(-delta/temp)
            if delta_energy < 0:
                current_num_used = new_num_used
                current_used = new_used
            else:
                prob = np.exp(-delta_energy / current_temp) if current_temp > 0 else 0
                if rng.random() < prob:
                    current_num_used = new_num_used
                    current_used = new_used
                else:
                    # Reject
                    current_colors[edge_idx] = old_color
            
            current_temp *= cooling_rate
            
            # Update best if valid and better
            if new_viol == 0 and new_num_used < best_num_used:
                best_colors = current_colors.copy()
                best_num_used = new_num_used

    # Final verification
    if best_colors is None:
        # Fallback: use the (i+j)%17 coloring
        best_colors = get_initial_coloring(0, rng)
        # Ensure it's valid
        violations = get_violating_k4s(best_colors)
        for v_k4 in violations:
            e_idxs = k4_edges[v_k4]
            e_idx = int(e_idxs[0])
            best_colors[e_idx] = int(max(best_colors)) + 1
    
    # Final check
    violations = get_violating_k4s(best_colors)
    if len(violations) > 0:
        # Emergency repair
        for v_k4 in violations:
            e_idxs = k4_edges[v_k4]
            e_idx = int(e_idxs[0])
            best_colors[e_idx] = int(max(best_colors)) + 1
    
    return best_colors.tolist()

# EVOLVE-BLOCK-END
