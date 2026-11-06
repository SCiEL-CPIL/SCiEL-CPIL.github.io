import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit to be safe
    
    n = 128
    # Generate all 4-APs
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for start in range(1, n - 3 * d + 1):
            aps.append((start, start + d, start + 2 * d, start + 3 * d))
    
    # Map position to list of AP indices
    pos_to_aps = [[] for _ in range(n + 1)]
    for i, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(i)
            
    # Order positions by degree (number of APs they are in)
    # Higher degree first
    pos_order = sorted(range(1, n + 1), key=lambda x: len(pos_to_aps[x]), reverse=True)
    
    best_solution = None
    best_k = 128 # Start with worst case
    
    # Try increasing number of colors
    for k in range(3, 10): # Try 3 to 9 colors
        if time.time() > deadline - 5:
            break
            
        assignment = [-1] * (n + 1)
        
        # State for forward checking
        # For each AP, we track the set of colors used so far
        # And count of unassigned
        ap_colors = [set() for _ in range(len(aps))]
        ap_unassigned = [4] * len(aps)
        
        def is_consistent(ap_idx):
            colors = ap_colors[ap_idx]
            unassigned = ap_unassigned[ap_idx]
            # Pruning rules
            if unassigned == 0:
                return len(colors) >= 3
            if unassigned == 1:
                return len(colors) >= 2
            if unassigned == 2:
                return len(colors) >= 1
            return True
            
        def backtrack(idx):
            nonlocal best_solution
            if time.time() > deadline:
                return False
                
            if idx == len(pos_order):
                # All assigned
                # Final check just in case
                for i in range(len(aps)):
                    if len(ap_colors[i]) < 3:
                        return False
                best_solution = assignment[1:].copy()
                return True
                
            pos = pos_order[idx]
            
            # Try each color
            for color in range(k):
                # Check if assigning this color to pos causes any inconsistency
                valid = True
                affected_aps = []
                for ap_idx in pos_to_aps[pos]:
                    ap = aps[ap_idx]
                    # Check if this AP is affected
                    # It is, because pos is in it
                    # Simulate assignment
                    old_color = assignment[pos]
                    assignment[pos] = color
                    
                    # Update AP state
                    # We need to know if the color was already in the set
                    # This is tricky with sets. We need to revert.
                    # Better to just check consistency after hypothetical update?
                    # Or maintain counts?
                    
                    # Let's just do a quick check without full revert if possible
                    # Actually, let's just update and revert.
                    
                    # Check if color is already in ap_colors[ap_idx]
                    color_in_set = color in ap_colors[ap_idx]
                    
                    if not color_in_set:
                        ap_colors[ap_idx].add(color)
                        
                    ap_unassigned[ap_idx] -= 1
                    
                    if not is_consistent(ap_idx):
                        valid = False
                        # Revert
                        ap_unassigned[ap_idx] += 1
                        if not color_in_set:
                            ap_colors[ap_idx].remove(color)
                        assignment[pos] = old_color
                        break
                        
                    affected_aps.append(ap_idx)
                    
                if not valid:
                    continue
                    
                if backtrack(idx + 1):
                    return True
                    
                # Revert
                for ap_idx in affected_aps:
                    ap_unassigned[ap_idx] += 1
                    # Note: removing color from set is hard if it appeared multiple times?
                    # No, ap_colors is a SET of colors used in the AP.
                    # If I remove the assignment, I should remove the color ONLY IF it is not used by any other vertex in the AP.
                    # This requires knowing the counts per color in the AP.
                    # My simple set update is flawed for reverting.
                    
                # This revert logic is complex. Let's use a simpler, slower but correct approach:
                # Just check consistency of the whole AP after assignment, without maintaining incremental state that is hard to revert.
                # Or, maintain counts: ap_color_counts[ap_idx][color]
                
                # Let's restart the logic with counts.
                
            return False
            
        # Reset state for each k
        # The above backtrack has a flaw in revert. Let's implement a cleaner version.
        
        # Clean implementation:
        ap_color_counts = [[0] * k for _ in range(len(aps))]
        ap_unassigned = [4] * len(aps)
        
        def is_consistent_clean(ap_idx):
            counts = ap_color_counts[ap_idx]
            unassigned = ap_unassigned[ap_idx]
            distinct_colors = sum(1 for c in counts if c > 0)
            
            if unassigned == 0:
                return distinct_colors >= 3
            if unassigned == 1:
                return distinct_colors >= 2
            if unassigned == 2:
                return distinct_colors >= 1
            return True
            
        def backtrack_clean(idx):
            nonlocal best_solution
            if time.time() > deadline:
                return False
                
            if idx == len(pos_order):
                for i in range(len(aps)):
                    if sum(1 for c in ap_color_counts[i] if c > 0) < 3:
                        return False
                best_solution = assignment[1:].copy()
                return True
                
            pos = pos_order[idx]
            
            # Heuristic: try colors that are less frequent in neighbors?
            # For now, try 0..k-1
            
            for color in range(k):
                assignment[pos] = color
                valid = True
                affected_aps = []
                
                for ap_idx in pos_to_aps[pos]:
                    ap_color_counts[ap_idx][color] += 1
                    ap_unassigned[ap_idx] -= 1
                    affected_aps.append(ap_idx)
                    
                    if not is_consistent_clean(ap_idx):
                        valid = False
                        break
                        
                if not valid:
                    # Revert
                    for ap_idx in affected_aps:
                        ap_color_counts[ap_idx][color] -= 1
                        ap_unassigned[ap_idx] += 1
                    assignment[pos] = -1
                    continue
                    
                if backtrack_clean(idx + 1):
                    return True
                    
                # Revert
                for ap_idx in affected_aps:
                    ap_color_counts[ap_idx][color] -= 1
                    ap_unassigned[ap_idx] += 1
                assignment[pos] = -1
                
            return False
            
        # Start search
        backtrack_clean(0)
        
        if best_solution is not None:
            return best_solution
            
    # If no solution found with < 10 colors, return a greedy solution or seed-like solution
    # Fallback to seed logic or random
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 for d in range(1, (x-1)//3+1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) for d in range(1, (x-1)//2+1))):
                colors.append(color)
                break
    return colors
