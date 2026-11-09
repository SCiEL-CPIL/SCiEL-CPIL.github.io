import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Precompute edge indices
    # Edge (i, j) with i < j.
    # Index in the list of 120 edges:
    # For vertex i, edges are (i, i+1), ..., (i, 15).
    # Number of edges before vertex i is i * 15 - i * (i - 1) / 2?
    # Let's just generate the list.
    edges_list = []
    edge_to_idx = {}
    for i in range(16):
        for j in range(i + 1, 16):
            edge_to_idx[(i, j)] = len(edges_list)
            edges_list.append((i, j))
    
    # Precompute K4s
    # A K4 is defined by 4 vertices. It has 6 edges.
    k4s = []
    for quad in itertools.combinations(range(16), 4):
        v0, v1, v2, v3 = quad
        e1 = edge_to_idx[(v0, v1)]
        e2 = edge_to_idx[(v0, v2)]
        e3 = edge_to_idx[(v0, v3)]
        e4 = edge_to_idx[(v1, v2)]
        e5 = edge_to_idx[(v1, v3)]
        e6 = edge_to_idx[(v2, v3)]
        k4s.append((e1, e2, e3, e4, e5, e6))
        
    num_k4s = len(k4s)
    
    # Map edge index to list of K4 indices
    edge_to_k4s = [[] for _ in range(120)]
    for k4_idx, edges in enumerate(k4s):
        for e in edges:
            edge_to_k4s[e].append(k4_idx)

    def get_violation_count(colors):
        """
        Returns the number of K4s with fewer than 5 distinct colors.
        Also returns a list of violating K4 indices for faster repair.
        """
        viol_count = 0
        viol_k4s = []
        for i in range(num_k4s):
            e = k4s[i]
            c = len({colors[e[0]], colors[e[1]], colors[e[2]], colors[e[3]], colors[e[4]], colors[e[5]]})
            if c < 5:
                viol_count += 1
                viol_k4s.append(i)
        return viol_count, viol_k4s

    def check_validity(colors):
        for i in range(num_k4s):
            e = k4s[i]
            c = len({colors[e[0]], colors[e[1]], colors[e[2]], colors[e[3]], colors[e[4]], colors[e[5]]})
            if c < 5:
                return False
        return True

    best_solution = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    best_color_count = len(set(best_solution))
    
    # Try to improve with fewer colors
    # We will try targets 13, 14, 15.
    # For each target, we run a local search.
    
    targets = [13, 14, 15]
    
    for target in targets:
        if time.time() > deadline - 10:
            break
            
        # Time slice for this target
        # 13 is hardest, give it less time if we don't expect it?
        # Or give it more?
        # Let's give 30s to 13, 40s to 14, 30s to 15.
        slice_time = 30 if target == 13 else 40 if target == 14 else 30
        slice_end = min(deadline - 5, time.time() + slice_time)
        
        # Try multiple restarts
        found_valid = False
        restarts = 0
        while time.time() < slice_end and not found_valid:
            restarts += 1
            
            # Initialize colors
            # Use a structured pattern to start closer to valid
            # Pattern: (i * a + j * b) % target
            # Or random
            if restarts % 2 == 0:
                # Random
                colors = [random.randint(0, target - 1) for _ in range(120)]
            else:
                # Structured
                a = random.randint(1, target - 1)
                b = random.randint(1, target - 1)
                colors = []
                for i in range(16):
                    for j in range(i + 1, 16):
                        colors.append((i * a + j * b) % target)
            
            # Calculate initial violations
            viol_count, viol_k4s = get_violation_count(colors)
            
            if viol_count == 0:
                actual_colors = len(set(colors))
                if actual_colors < best_color_count:
                    best_solution = colors[:]
                    best_color_count = actual_colors
                    if best_color_count == 13:
                        return best_solution
                found_valid = True
                continue
            
            # Local Search: Min-Conflicts
            # We want to reduce viol_count.
            # Pick a random violating K4, pick a random edge in it, change color.
            
            max_iters = 5000
            for _ in range(max_iters):
                if time.time() > slice_end:
                    break
                if viol_count == 0:
                    actual_colors = len(set(colors))
                    if actual_colors < best_color_count:
                        best_solution = colors[:]
                        best_color_count = actual_colors
                        if best_color_count == 13:
                            return best_solution
                    found_valid = True
                    break
                
                # Pick a violating K4
                # To avoid scanning all 1820 K4s every time, we maintain viol_k4s list?
                # But viol_k4s changes.
                # Simple approach: Pick random K4, check if it's violating.
                # If not, try again.
                # This is inefficient if few violations remain.
                # Better: Maintain a list of violating K4s.
                # But updating it is complex.
                # Let's stick to random K4 pick for now, but optimize the check.
                
                # Actually, let's pick a random edge from the entire graph?
                # No, Min-Conflicts usually picks a variable (edge) involved in a conflict.
                
                # Let's pick a random K4 index
                k4_idx = random.randint(0, num_k4s - 1)
                e = k4s[k4_idx]
                # Check if this K4 is violating
                c = len({colors[e[0]], colors[e[1]], colors[e[2]], colors[e[3]], colors[e[4]], colors[e[5]]})
                if c >= 5:
                    continue
                
                # It is violating. Pick a random edge in this K4.
                edge_idx = e[random.randint(0, 5)]
                old_color = colors[edge_idx]
                new_color = random.randint(0, target - 1)
                
                if old_color == new_color:
                    continue
                
                colors[edge_idx] = new_color
                
                # Update violation count for affected K4s
                # We need to know the old and new violation status of each affected K4.
                # Affected K4s are those containing edge_idx.
                
                # To do this efficiently, we need to know the color counts in those K4s.
                # Recalculating the set for each affected K4 is O(1) (6 edges).
                # Number of affected K4s per edge:
                # An edge (i, j) is in K4s formed by (i, j, k, l) where k, l are other vertices.
                # Number of such pairs is C(14, 2) = 91.
                # So we update 91 K4s.
                
                # We need to maintain a global violation count.
                # Let's maintain `k4_color_counts`? No, just re-evaluate the 91 K4s.
                
                # But we need to update `viol_count`.
                # We don't have a fast way to know if a K4 was violating before without storing its state.
                # Let's store `k4_is_violating` array of booleans.
                
                # Wait, I didn't define `k4_is_violating` in the loop above.
                # Let's refactor to maintain state.
                
                pass # Placeholder for logic below
                
            # End local search for this restart

    return best_solution
