import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 128
    
    # Pre-compute all 4-term APs and index them by position for fast updates
    aps = []
    ap_indices = [[] for _ in range(N)]
    for d in range(1, N // 3 + 1):
        for i in range(N - 3 * d):
            ap = (i, i + d, i + 2 * d, i + 3 * d)
            ap_idx = len(aps)
            aps.append(ap)
            for idx in ap:
                ap_indices[idx].append(ap_idx)
                
    num_aps = len(aps)
    
    def count_violations(colors):
        count = 0
        for ap in aps:
            if len({colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}) < 3:
                count += 1
        return count

    def run_seed():
        colors=[]
        for x in range(1,129):
            for color in range(128):
                if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                    and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                    colors.append(color);break
        return colors

    # Get initial seed solution
    seed_colors = run_seed()
    best_colors = seed_colors[:]
    best_k = len(set(seed_colors))
    
    # CSP backtracking solver with forward checking and MRV
    def csp_solve(target_k, time_limit):
        """
        Solve the CSP: assign each position a color from {0..target_k-1}
        such that every 4-term AP uses at least 3 distinct colors.
        Uses backtracking with forward checking and MRV heuristic.
        """
        colors = [-1] * N  # -1 means unassigned
        
        # For each position, track which APs it's involved in
        # and for each AP, track the colors used at assigned positions
        
        # Pre-compute: for each AP, which positions are in it
        # We'll track for each AP: set of colors used, count of assigned positions
        
        # State: for each AP, (assigned_count, color_counts)
        # color_counts: dict color -> count, but we only care about distinct colors >= 3
        
        # For forward checking: when we assign position p to color c,
        # for each AP containing p, we update the state and check if still feasible
        
        # Feasibility check for an AP:
        # - If all 4 positions assigned: must have >= 3 distinct colors
        # - If 3 positions assigned: the 4th must be assigned a color that makes >= 3 distinct
        #   If the 3 assigned use only 1 or 2 colors, the 4th must add a new color
        # - If 2 positions assigned: always feasible (can always add 2 more colors)
        # - If 1 or 0 positions assigned: always feasible
        
        def is_ap_feasible(ap, colors):
            assigned_colors = []
            unassigned_count = 0
            for pos in ap:
                if colors[pos] != -1:
                    assigned_colors.append(colors[pos])
                else:
                    unassigned_count += 1
            
            if unassigned_count == 0:
                return len(set(assigned_colors)) >= 3
            
            distinct_assigned = set(assigned_colors)
            
            if unassigned_count == 1:
                # The one unassigned position must make it >= 3 distinct
                # If we already have >= 3 distinct, any color works
                if len(distinct_assigned) >= 3:
                    return True
                # If we have 2 distinct, the unassigned must be a 3rd color
                if len(distinct_assigned) == 2:
                    return True  # As long as target_k >= 3, we can pick a 3rd color
                # If we have 1 distinct, the unassigned must be a different color
                if len(distinct_assigned) == 1:
                    return target_k >= 2
                # If we have 0 distinct (shouldn't happen if unassigned_count==1)
                return True
            
            if unassigned_count == 2:
                # With 2 unassigned, we can always make it work if target_k >= 2
                # Worst case: both assigned are same color, we need 2 different colors for the rest
                # Or both assigned are different, we need 1 more different
                if len(distinct_assigned) <= 1:
                    return target_k >= 2
                return True
            
            # unassigned_count >= 3: always feasible
            return True
        
        # Backtracking solver
        def backtrack(assigned_count):
            if time.time() > time_limit:
                return None
            
            if assigned_count == N:
                return colors[:]
            
            # MRV: pick the unassigned position with fewest valid colors
            # For each unassigned position, count how many colors are feasible
            min_domain = target_k + 1
            min_pos = -1
            
            for pos in range(N):
                if colors[pos] != -1:
                    continue
                
                # Count feasible colors for this position
                feasible_count = 0
                for c in range(target_k):
                    colors[pos] = c
                    # Check all APs containing pos
                    feasible = True
                    for ai in ap_indices[pos]:
                        if not is_ap_feasible(aps[ai], colors):
                            feasible = False
                            break
                    colors[pos] = -1
                    if feasible:
                        feasible_count += 1
                
                if feasible_count < min_domain:
                    min_domain = feasible_count
                    min_pos = pos
                    if feasible_count <= 1:
                        break
            
            if min_pos == -1:
                return None  # No unassigned positions
            
            if min_domain == 0:
                return None  # Dead end
            
            # Try each color for min_pos
            for c in range(target_k):
                colors[min_pos] = c
                
                # Forward check: verify all APs containing min_pos are still feasible
                feasible = True
                for ai in ap_indices[min_pos]:
                    if not is_ap_feasible(aps[ai], colors):
                        feasible = False
                        break
                
                if not feasible:
                    colors[min_pos] = -1
                    continue
                
                result = backtrack(assigned_count + 1)
                if result is not None:
                    return result
                
                colors[min_pos] = -1
            
            return None
        
        return backtrack(0)
    
    # Try to find a valid coloring with fewer colors using CSP
    for target_k in range(3, best_k):
        if time.time() > deadline - 5:
            break
        
        time_limit = min(deadline - 2, time.time() + 10.0)
        result = csp_solve(target_k, time_limit)
        
        if result is not None and count_violations(result) == 0:
            actual_k = len(set(result))
            if actual_k < best_k:
                best_k = actual_k
                best_colors = result[:]
            else:
                best_colors = result[:]
            break
    
    # If no improvement found, return the seed or best found
    return best_colors
# EVOLVE-BLOCK-END
