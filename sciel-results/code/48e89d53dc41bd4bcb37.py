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

    def count_distinct_in_k4s(k4_indices, colors_np):
        """
        Count distinct colors in each K4. Returns array of counts.
        """
        if len(k4_indices) == 0:
            return np.array([], dtype=np.int32)
        
        k4_idx_arr = np.array(k4_indices, dtype=np.int32)
        k4_colors = colors_np[k4_edges[k4_idx_arr]]  # shape (num_k4s, 6)
        
        # For each K4, count distinct colors
        # We can do this by sorting and counting, or using a trick with unique
        # Since each K4 has only 6 edges, we can use a simple approach
        counts = np.zeros(len(k4_indices), dtype=np.int32)
        for i in range(len(k4_indices)):
            c = k4_colors[i]
            # Count distinct: sort and count changes
            sorted_c = np.sort(c)
            # Number of distinct = 1 + number of positions where sorted_c[i] != sorted_c[i-1]
            if len(sorted_c) > 1:
                diff = np.diff(sorted_c)
                counts[i] = 1 + np.sum(diff != 0)
            else:
                counts[i] = 1
        return counts

    def count_violations_full(colors_np):
        """
        Returns number of K4s with < 5 distinct colors.
        """
        counts = count_distinct_in_k4s(list(range(num_k4s)), colors_np)
        return int(np.sum(counts < 5))

    def get_violating_k4s(colors_np):
        """
        Returns list of K4 indices with < 5 distinct colors.
        """
        counts = count_distinct_in_k4s(list(range(num_k4s)), colors_np)
        return [i for i in range(num_k4s) if counts[i] < 5]

    def try_merge_color(colors_np, c_from, c_to):
        """
        Try to change all edges of color c_from to c_to.
        Returns (success, new_colors, changed_edges)
        """
        edges_with_from = np.where(colors_np == c_from)[0]
        if len(edges_with_from) == 0:
            return True, colors_np, []
        
        temp_colors = colors_np.copy()
        temp_colors[edges_with_from] = c_to
        
        # Only check K4s that contain at least one edge of color c_from
        k4s_to_check = set()
        for e_idx in edges_with_from:
            k4s_to_check.update(edge_to_k4s[int(e_idx)])
        
        if len(k4s_to_check) > 0:
            k4s_to_check_list = list(k4s_to_check)
            counts = count_distinct_in_k4s(k4s_to_check_list, temp_colors)
            if np.any(counts < 5):
                return False, colors_np, []
        
        return True, temp_colors, list(edges_with_from)

    def get_initial_coloring(method=0):
        """
        Generate different initial colorings.
        """
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
        else:
            # Random
            return np.random.randint(0, 17, size=num_edges).astype(np.int32)

    # Main search
    best_colors = None
    best_num_used = max_colors + 1
    best_violations = num_k4s + 1
    
    rng = random.Random(42)
    
    # Try multiple initial colorings
    for init_method in range(6):
        if time.time() > deadline:
            break
        
        colors = get_initial_coloring(init_method)
        
        # Check if valid
        violations = get_violating_k4s(colors)
        num_viol = len(violations)
        num_used = len(set(colors))
        
        # If invalid, try to repair by adding colors
        if num_viol > 0:
            # For each violating K4, recolor one edge to a new color
            for v_k4 in violations:
                if time.time() > deadline:
                    break
                e_idxs = k4_edges[v_k4]
                e_idx = e_idxs[rng.randint(0, 5)]
                # Assign a new unique color
                new_c = max(colors) + 1
                colors[e_idx] = new_c
            
            # Re-check
            violations = get_violating_k4s(colors)
            num_viol = len(violations)
            num_used = len(set(colors))
        
        # Update best
        if num_viol == 0:
            if num_used < best_num_used:
                best_colors = colors.copy()
                best_num_used = num_used
                best_violations = 0
        
        # Greedy merge phase
        current_colors = colors.copy()
        current_used = set(current_colors)
        
        progress = True
        while progress and time.time() < deadline:
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
                    
                    success, new_colors, _ = try_merge_color(current_colors, c_from, c_to)
                    if success:
                        current_colors = new_colors
                        current_used.remove(c_from)
                        progress = True
                        
                        # Update best
                        num_used = len(current_used)
                        if num_used < best_num_used:
                            best_colors = current_colors.copy()
                            best_num_used = num_used
                            best_violations = 0
                        break  # Restart merge loop with new state
                
                if progress:
                    break
        
        # Simulated annealing phase to escape local minima
        if time.time() > deadline:
            break
        
        current_colors = current_colors.copy()
        current_used = set(current_colors)
        current_num_used = len(current_used)
        
        # SA parameters
        current_temp = 50.0
        cooling_rate = 0.995
        sa_steps = 2000
        
        for sa_step in range(sa_steps):
            if time.time() > deadline:
                break
            
            # Generate a move: recolor a random edge
            edge_idx = rng.randint(0, num_edges - 1)
            old_color = current_colors[edge_idx]
            
            # Choose new color: 70% from existing, 30% new (if < max_colors)
            if rng.random() < 0.7:
                existing = list(current_used)
                if existing:
                    new_color = rng.choice(existing)
                else:
                    new_color = old_color
            else:
                # Try to use a color that might reduce count
                # Pick from 0 to max(current_used)
                max_c = max(current_used) if current_used else 0
                new_color = rng.randint(0, max_c + 1)
            
            if old_color == new_color:
                continue
            
            current_colors[edge_idx] = new_color
            
            # Calculate energy change
            # Only check affected K4s
            affected_k4s = set(edge_to_k4s[edge_idx])
            
            old_counts = count_distinct_in_k4s(list(affected_k4s), current_colors)
            new_viol = int(np.sum(old_counts < 5))
            
            # New color count
            new_used = set(current_colors)
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
                best_violations = 0

    # Final verification
    if best_colors is None:
        # Fallback: use the (i+j)%17 coloring
        best_colors = get_initial_coloring(0)
        # Ensure it's valid
        violations = get_violating_k4s(best_colors)
        for v_k4 in violations:
            e_idxs = k4_edges[v_k4]
            e_idx = e_idxs[0]
            best_colors[e_idx] = max(best_colors) + 1
    
    # Final check
    violations = get_violating_k4s(best_colors)
    if len(violations) > 0:
        # Emergency repair
        for v_k4 in violations:
            e_idxs = k4_edges[v_k4]
            e_idx = e_idxs[0]
            best_colors[e_idx] = max(best_colors) + 1
    
    return best_colors.tolist()

# EVOLVE-BLOCK-END
