import time
import random
import math
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Fallback: n=11 (Known valid: K11 decomposed into 5 Hamiltonian cycles)
    n_fallback = 11
    colors_fallback = [min(j - i, n_fallback - j + i) - 1 for i in range(n_fallback) for j in range(i + 1, n_fallback)]
    best_n = n_fallback
    best_colors = colors_fallback

    def get_idx(i, j, n):
        # Lexicographic order: (0,1), (0,2)...(0,n-1), (1,2)...
        # Index = i*(n-1) - i*(i+1)//2 + j - i - 1
        return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

    def check_validity(colors, n):
        # Fast check using bitmasks
        adj_mask = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj_mask[c][i] |= (1 << j)
                adj_mask[c][j] |= (1 << i)
                idx += 1
        
        for c in range(5):
            for u in range(n):
                mask_u = adj_mask[c][u]
                if mask_u == 0:
                    continue
                for v in range(u + 1, n):
                    mask_v = adj_mask[c][v]
                    if mask_v == 0:
                        continue
                    common = mask_u & mask_v
                    if common:
                        if common.bit_count() > 1:
                            return False
        return True

    def count_violations(colors, n):
        adj_mask = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj_mask[c][i] |= (1 << j)
                adj_mask[c][j] |= (1 << i)
                idx += 1
        
        violations = 0
        for c in range(5):
            for u in range(n):
                mask_u = adj_mask[c][u]
                if mask_u == 0:
                    continue
                for v in range(u + 1, n):
                    mask_v = adj_mask[c][v]
                    if mask_v == 0:
                        continue
                    common = mask_u & mask_v
                    if common:
                        cnt = common.bit_count()
                        if cnt > 1:
                            violations += cnt - 1
        return violations

    def targeted_local_search(colors, n, max_time):
        """
        Local search that focuses on edges incident to vertices involved in conflicts.
        """
        current_violations = count_violations(colors, n)
        if current_violations == 0:
            return colors, True
        
        # Precompute edge map
        inv_map = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                inv_map[idx] = (i, j)
                idx += 1
        
        num_edges = n * (n - 1) // 2
        temp = 2.0
        cooling = 0.95
        min_temp = 0.01
        best_colors = colors[:]
        best_violations = current_violations
        no_improve_count = 0
        
        while temp > min_temp and time.time() < max_time:
            # Identify vertices involved in violations
            # We can approximate by just picking random edges, but let's try to be smart.
            # For simplicity and speed, we pick a random edge, but with higher probability
            # for edges that are "critical".
            # Actually, calculating critical edges is expensive. 
            # Let's stick to random edge selection but with a faster violation update?
            # No, count_violations is O(n^2 * 5). For n=20, that's 190*5 = 950 ops. Very fast.
            # So random edge selection is fine.
            
            e_idx = random.randint(0, num_edges - 1)
            old_color = colors[e_idx]
            new_color = random.randint(0, 4)
            if new_color == old_color:
                continue
            
            colors[e_idx] = new_color
            new_violations = count_violations(colors, n)
            
            delta = new_violations - current_violations
            
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_violations = new_violations
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                colors[e_idx] = old_color
                no_improve_count += 1
            
            if no_improve_count > 100:
                # Restart temperature or break if stuck
                if current_violations == best_violations:
                     # Try a few random jumps?
                     pass
                no_improve_count = 0
                temp = max(min_temp, temp * 0.5) # Cool down faster if stuck
            
            temp *= cooling
            
        if best_violations == 0:
            return best_colors, True
        return best_colors, False

    # Strategy: Incremental Construction with Backtracking/Repair
    # We try to build n=12, 13, ... up to a limit.
    # For each n, we initialize with a structured pattern or random, then repair.
    
    # The key insight: The constraint is local (C4). 
    # Random initialization for n=12 has many conflicts.
    # A better initialization: Try to mimic the n=11 structure?
    # Or use a greedy approach:
    # 1. Start with n=11 valid coloring.
    # 2. Add vertex 11. Assign colors to (11, 0..10) greedily to avoid C4s.
    # 3. If stuck, backtrack or use local search on the new edges.
    
    # Let's implement a "Greedy + Local Repair" for n > 11.
    
    def greedy_extend(n_target):
        """
        Try to extend the n=11 solution to n_target.
        """
        if n_target < 11:
            return None
            
        # Start with n=11 solution
        current_n = 11
        # colors list for current_n
        curr_colors = colors_fallback[:]
        
        # Map from (i,j) to index in curr_colors
        # We need to add vertices one by one.
        
        for new_v in range(11, n_target):
            if time.time() > deadline - 10:
                return None
                
            # We need to assign colors for edges (new_v, j) for j in 0..new_v-1
            # Let's use local search to find valid colors for these new edges,
            # keeping the existing edges fixed.
            
            # Create a temporary list of new edges
            new_edges = [(new_v, j) for j in range(new_v)]
            num_new = len(new_edges)
            
            # We will run a local search that only modifies these new edges.
            # The objective is to minimize violations in the whole graph.
            
            # Initialize new edges randomly
            initial_new_colors = [random.randint(0, 4) for _ in range(num_new)]
            
            def get_full_colors(new_edge_colors):
                full = curr_colors[:]
                # Insert new colors at the correct positions
                # Lexicographic order: (0,1)..(0,n-1), (1,2)..
                # New edges are (new_v, j) with j < new_v.
                # In lexicographic order, (j, new_v) comes after (j, new_v-1) and before (j+1, j+2)?
                # No, lexicographic is sorted by first vertex, then second.
                # (0, 1), (0, 2), ..., (0, new_v-1), (0, new_v)
                # (1, 2), ..., (1, new_v-1), (1, new_v)
                # ...
                # (new_v-1, new_v)
                
                # We need to build the full list for n_target (or current_n+1)
                # Actually, it's easier to just build the full list from scratch if n is small.
                # But for efficiency, let's just construct the full list.
                
                full_n = new_v + 1
                full_colors = [0] * (full_n * (full_n - 1) // 2)
                
                # Copy old colors
                idx_old = 0
                for i in range(11):
                    for j in range(i + 1, 11):
                        full_colors[get_idx(i, j, full_n)] = curr_colors[idx_old]
                        idx_old += 1
                
                # Set new colors
                for k, c in enumerate(new_edge_colors):
                    j = new_edges[k][1]
                    full_colors[get_idx(j, new_v, full_n)] = c
                    
                return full_colors, full_n
            
            # Run local search on the new edges only
            best_new_colors = initial_new_colors
            best_violations = float('inf')
            
            # To speed up, we can use a simplified local search that only checks violations
            # involving the new vertex.
            # Violations involving new vertex:
            # 1. C4s where new_v is a vertex: (new_v, a, b, c) -> a,b,c are neighbors of new_v in same color?
            #    No, C4 is u-v-w-x-u.
            #    If new_v is in C4, it must be adjacent to two vertices a,b in the C4.
            #    So a,b must be neighbors of new_v in the same color c.
            #    And a,b must have a common neighbor x in color c (other than new_v).
            # 2. C4s not involving new_v: These are already fixed in curr_colors (valid).
            
            # So we only need to check C4s involving new_v.
            # This is much faster!
            
            def check_violations_new_vertex(new_edge_colors, full_n):
                # new_edge_colors[k] is color of (new_v, new_edges[k][1])
                # new_edges[k] = (new_v, j)
                
                # Build adjacency for new_v
                # adj_new_v[c] = set of neighbors j such that color(new_v, j) == c
                adj_new_v = [set() for _ in range(5)]
                for k, c in enumerate(new_edge_colors):
                    j = new_edges[k][1]
                    adj_new_v[c].add(j)
                
                # For each color c:
                # For each pair of neighbors (a, b) of new_v in color c:
                #   Check if a and b have a common neighbor x in color c (x != new_v).
                #   If yes, violation.
                
                # We need the adjacency of the old graph (n=11) for each color.
                # Let's precompute it once outside the loop?
                # For now, let's just use the full graph check but it's expensive.
                # Let's optimize: Precompute common neighbors in old graph.
                
                # Actually, let's just use the full count_violations for simplicity in this step,
                # but only on the full graph. It's O(n^2). For n=12, it's tiny.
                
                full_colors, _ = get_full_colors(new_edge_colors)
                return count_violations(full_colors, full_n)
            
            # Local search on new_edge_colors
            temp = 5.0
            cooling = 0.9
            for _ in range(200): # Fixed number of steps for this sub-problem
                if time.time() > deadline - 10:
                    break
                    
                # Pick a random new edge to change
                k = random.randint(0, num_new - 1)
                old_c = new_edge_colors[k]
                new_c = random.randint(0, 4)
                if new_c == old_c:
                    continue
                
                new_edge_colors[k] = new_c
                v_new = check_violations_new_vertex(new_edge_colors, new_v + 1)
                
                # We don't track current_violations easily here without storing state.
                # Let's just keep the best found so far.
                if v_new < best_violations:
                    best_violations = v_new
                    best_new_colors = new_edge_colors[:]
                else:
                    # Revert if worse? Or accept with prob?
                    # Since we don't have current, let's just revert if worse.
                    new_edge_colors[k] = old_c
                    
            # After local search, if best_violations == 0, we have a valid extension.
            if best_violations == 0:
                # Update curr_colors to include the new vertex
                curr_colors, current_n = get_full_colors(best_new_colors)
                # curr_colors is now for n = new_v + 1
                # We need to update the mapping for the next iteration?
                # No, we just rebuild curr_colors each time.
            else:
                # Failed to extend
                return None
                
        return curr_colors

    # Try to extend
    for n_target in range(12, 16):
        if time.time() > deadline - 20:
            break
            
        result = greedy_extend(n_target)
        if result is not None:
            # Verify
            if check_validity(result, n_target):
                best_n = n_target
                best_colors = result
            else:
                # Something went wrong
                pass
        else:
            # If greedy fails, try full random + local search for this n
            # (Fallback to previous strategy)
            colors = [random.randint(0, 4) for _ in range(n_target * (n_target - 1) // 2)]
            t_limit = time.time() + max(1.0, (deadline - 10.0 - time.time()) / 2.0)
            result_colors, is_valid = targeted_local_search(colors, n_target, t_limit)
            if is_valid:
                best_n = n_target
                best_colors = result_colors
            else:
                break # If we can't do n, we likely can't do n+1

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
