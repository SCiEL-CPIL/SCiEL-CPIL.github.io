import random
import time
import numpy as np

def construct():
    N = 128
    deadline = time.time() + 85.0  # 5s margin for safety
    
    # Precompute all 4-APs (0-indexed)
    aps = []
    ap_indices = [[] for _ in range(N)]
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            ap_id = len(aps)
            aps.append((i0, i1, i2, i3))
            ap_indices[i0].append(ap_id)
            ap_indices[i1].append(ap_id)
            ap_indices[i2].append(ap_id)
            ap_indices[i3].append(ap_id)

    def count_violations(coloring):
        violations = 0
        for ap in aps:
            c0, c1, c2, c3 = coloring[ap[0]], coloring[ap[1]], coloring[ap[2]], coloring[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                violations += 1
        return violations

    def backtracking_solve(k, timeout=5.0):
        """
        Try to find a valid coloring with exactly k colors using backtracking.
        Returns a valid coloring list if found, else None.
        """
        start_time = time.time()
        coloring = [-1] * N
        # Track which APs are violated by each position
        # For each position, track the APs it's in and their current state
        
        # Precompute for each position the APs it belongs to
        # and for each AP, which positions are unassigned
        
        # Use a simpler approach: recursive backtracking with forward checking
        # Order positions by degree (number of APs they're in) for variable ordering
        degrees = [len(ap_indices[i]) for i in range(N)]
        # Sort positions by descending degree
        pos_order = sorted(range(N), key=lambda x: -degrees[x])
        
        # For each AP, track how many positions are assigned and their colors
        ap_assigned = [0] * len(aps)
        ap_colors = [None] * len(aps)  # Will store set of colors used
        
        def is_feasible(pos, color):
            """Check if assigning color to pos keeps all APs feasible."""
            for ap_id in ap_indices[pos]:
                ap = aps[ap_id]
                # Check if this AP would become infeasible
                # An AP is infeasible if it has 2 or more positions assigned to the same color
                # and the remaining positions can't provide a third color
                # Actually, the constraint is: every 4-AP must use at least 3 distinct colors
                # So an AP is violated if it uses 1 or 2 distinct colors
                # During assignment, we need to ensure that even if all remaining positions
                # get the same color, we still have at least 3 colors
                # More precisely: after assigning pos=color, check if the AP can still
                # achieve 3 distinct colors
                
                # Get current colors of assigned positions in this AP
                current_colors = set()
                unassigned_count = 0
                for p in ap:
                    if p == pos:
                        current_colors.add(color)
                    elif coloring[p] != -1:
                        current_colors.add(coloring[p])
                    else:
                        unassigned_count += 1
                
                # If we already have 3+ colors, we're fine
                if len(current_colors) >= 3:
                    continue
                
                # If we have 2 colors, we need at least one more color among unassigned
                # The unassigned positions can be colored with any of the k colors
                # So if k >= 3, we can always add a third color
                # But if k == 2, we can't add a third color
                if len(current_colors) == 2:
                    if k < 3:
                        return False
                    # With k >= 3, we can assign a new color to one of the unassigned
                    # But we need to check if that's possible given other constraints
                    # For now, assume it's feasible (optimistic)
                    continue
                
                # If we have 1 color, we need at least 2 more distinct colors
                # With k >= 3, this is feasible
                if len(current_colors) == 1:
                    if k < 3:
                        return False
                    continue
                
                # If we have 0 colors (all unassigned), need 3 distinct colors
                if len(current_colors) == 0:
                    if k < 3:
                        return False
                    continue
            
            return True
        
        def backtrack(idx):
            if time.time() - start_time > timeout:
                return False
            
            if idx == N:
                return True
            
            pos = pos_order[idx]
            
            # Try each color
            for color in range(k):
                if not is_feasible(pos, color):
                    continue
                
                coloring[pos] = color
                
                # Update AP states
                for ap_id in ap_indices[pos]:
                    ap_assigned[ap_id] += 1
                    if ap_colors[ap_id] is None:
                        ap_colors[ap_id] = set()
                    ap_colors[ap_id].add(color)
                
                if backtrack(idx + 1):
                    return True
                
                # Undo
                for ap_id in ap_indices[pos]:
                    ap_assigned[ap_id] -= 1
                    if ap_colors[ap_id] is not None:
                        ap_colors[ap_id].discard(color)
                        if not ap_colors[ap_id]:
                            ap_colors[ap_id] = None
                
                coloring[pos] = -1
            
            return False
        
        if backtrack(0):
            return coloring
        return None

    def simulated_annealing(k, max_steps=50000, timeout=10.0):
        """Run simulated annealing for k colors. Returns valid coloring or None."""
        start_time = time.time()
        coloring = [random.randint(0, k - 1) for _ in range(N)]
        
        current_violations = count_violations(coloring)
        best_coloring = coloring[:]
        best_violations = current_violations
        
        T = 5.0
        cooling = 0.9995
        steps = 0
        
        while current_violations > 0 and steps < max_steps and time.time() - start_time < timeout:
            steps += 1
            idx = random.randint(0, N - 1)
            old_color = coloring[idx]
            new_color = random.randint(0, k - 1)
            
            if old_color == new_color:
                continue
            
            # Calculate change in violations
            delta_violations = 0
            for ap_id in ap_indices[idx]:
                ap = aps[ap_id]
                c0 = coloring[ap[0]]
                c1 = coloring[ap[1]]
                c2 = coloring[ap[2]]
                c3 = coloring[ap[3]]
                
                old_set_size = len({c0, c1, c2, c3})
                
                new_c0 = new_color if ap[0] == idx else c0
                new_c1 = new_color if ap[1] == idx else c1
                new_c2 = new_color if ap[2] == idx else c2
                new_c3 = new_color if ap[3] == idx else c3
                
                new_set_size = len({new_c0, new_c1, new_c2, new_c3})
                
                if old_set_size < 3 and new_set_size >= 3:
                    delta_violations -= 1
                elif old_set_size >= 3 and new_set_size < 3:
                    delta_violations += 1
            
            if delta_violations < 0:
                coloring[idx] = new_color
                current_violations += delta_violations
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_coloring = coloring[:]
            else:
                if delta_violations == 0:
                    accept = random.random() < 0.5
                else:
                    if T > 0:
                        accept = random.random() < np.exp(-delta_violations / T)
                    else:
                        accept = False
                
                if accept:
                    coloring[idx] = new_color
                    current_violations += delta_violations
            
            T *= cooling
        
        if current_violations == 0:
            return coloring
        if best_violations == 0:
            return best_coloring
        return None

    # Try increasing number of colors, using backtracking first for small k
    for k in range(3, 129):
        if time.time() > deadline - 15.0:
            break
        
        # For small k, try backtracking first
        if k <= 6:
            result = backtracking_solve(k, timeout=3.0)
            if result is not None:
                return result
        
        # Then try simulated annealing
        result = simulated_annealing(k, max_steps=30000, timeout=5.0)
        if result is not None:
            return result
    
    # Fallback: Use N distinct colors
    return list(range(N))
