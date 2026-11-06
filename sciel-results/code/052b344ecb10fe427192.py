import time
import random
import numpy as np
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
    
    # Convert to numpy arrays for vectorized operations
    aps_np = np.array(aps, dtype=np.int32)  # shape (num_aps, 4)
    
    def count_violations(colors):
        colors = np.array(colors, dtype=np.int32)
        # Get colors at all positions of all APs
        ap_colors = colors[aps_np]  # shape (num_aps, 4)
        # Count distinct colors in each AP
        # An AP violates if it has < 3 distinct colors, i.e., <= 2 distinct
        # Check: if all 4 same, or 3 same + 1 different, or 2+2
        c0, c1, c2, c3 = ap_colors[:, 0], ap_colors[:, 1], ap_colors[:, 2], ap_colors[:, 3]
        # Violation if: all same, or exactly 2 distinct
        all_same = (c0 == c1) & (c1 == c2) & (c2 == c3)
        # Exactly 2 distinct: pairs of equal
        pair_01_23 = (c0 == c1) & (c2 == c3) & (c0 != c2)
        pair_01_32 = (c0 == c1) & (c3 == c2) & (c0 != c3)
        pair_02_13 = (c0 == c2) & (c1 == c3) & (c0 != c1)
        pair_03_12 = (c0 == c3) & (c1 == c2) & (c0 != c1)
        # Also: one different from three same
        one_diff_0 = (c1 == c2) & (c2 == c3) & (c0 != c1)
        one_diff_1 = (c0 == c2) & (c2 == c3) & (c1 != c0)
        one_diff_2 = (c0 == c1) & (c1 == c3) & (c2 != c0)
        one_diff_3 = (c0 == c1) & (c1 == c2) & (c3 != c0)
        violation = all_same | pair_01_23 | pair_01_32 | pair_02_13 | pair_03_12 | \
                    one_diff_0 | one_diff_1 | one_diff_2 | one_diff_3
        return int(np.sum(violation))
    
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
    
    # Improved CSP backtracking solver with forward checking and MRV
    def csp_solve(target_k, time_limit):
        """
        Solve the CSP: assign each position a color from {0..target_k-1}
        such that every 4-term AP uses at least 3 distinct colors.
        Uses backtracking with forward checking and MRV heuristic.
        """
        colors = np.full(N, -1, dtype=np.int32)  # -1 means unassigned
        assigned_count = [0]
        
        # Pre-compute for each AP, which positions it contains
        # For forward checking: when we assign position p to color c,
        # for each AP containing p, we update the state and check if still feasible
        
        def is_ap_feasible(ap_idx, colors):
            """Check if an AP can still be satisfied given current assignments"""
            ap = aps_np[ap_idx]
            assigned_colors = []
            unassigned_count = 0
            for pos in ap:
                c = colors[pos]
                if c != -1:
                    assigned_colors.append(c)
                else:
                    unassigned_count += 1
            
            if unassigned_count == 0:
                return len(set(assigned_colors)) >= 3
            
            distinct_assigned = set(assigned_colors)
            
            if unassigned_count == 1:
                # The one unassigned position must make it >= 3 distinct
                if len(distinct_assigned) >= 3:
                    return True
                if len(distinct_assigned) == 2:
                    return target_k >= 3
                if len(distinct_assigned) == 1:
                    return target_k >= 2
                return True
            
            if unassigned_count == 2:
                if len(distinct_assigned) <= 1:
                    return target_k >= 2
                return True
            
            return True
        
        # Count feasible colors for a position (for MRV)
        def count_feasible_colors(pos, colors):
            feasible_count = 0
            for c in range(target_k):
                colors[pos] = c
                feasible = True
                for ai in ap_indices[pos]:
                    if not is_ap_feasible(ai, colors):
                        feasible = False
                        break
                colors[pos] = -1
                if feasible:
                    feasible_count += 1
            return feasible_count
        
        # Backtracking solver
        def backtrack(assigned_count):
            if time.time() > time_limit:
                return None
            
            if assigned_count == N:
                return colors.copy()
            
            # MRV: pick the unassigned position with fewest valid colors
            min_domain = target_k + 1
            min_pos = -1
            
            for pos in range(N):
                if colors[pos] != -1:
                    continue
                
                feasible_count = count_feasible_colors(pos, colors)
                
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
                    if not is_ap_feasible(ai, colors):
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
    
    # Local search to reduce number of colors used
    def local_search_reduce(colors, max_iterations=100000, time_limit=None):
        """
        Try to reduce the number of distinct colors by remapping colors.
        """
        if time_limit is None:
            time_limit = deadline - 5
        
        colors = list(colors)
        current_k = len(set(colors))
        
        # Build color frequency
        from collections import Counter
        color_freq = Counter(colors)
        
        # Try to merge colors: pick two colors, try to remap one to the other
        # if it doesn't create violations
        for iteration in range(max_iterations):
            if time.time() > time_limit:
                break
            
            if current_k <= 3:
                break
            
            # Get colors sorted by frequency (least frequent first)
            sorted_colors = sorted(color_freq.keys(), key=lambda c: color_freq[c])
            
            # Try to eliminate the least frequent color
            target_color = sorted_colors[0]
            target_freq = color_freq[target_color]
            
            # Find positions with this color
            positions = [i for i, c in enumerate(colors) if c == target_color]
            
            # Try remapping to each other color
            other_colors = [c for c in sorted_colors if c != target_color]
            
            success = False
            for new_color in other_colors:
                # Test remapping
                test_colors = colors[:]
                for pos in positions:
                    test_colors[pos] = new_color
                
                if count_violations(test_colors) == 0:
                    # Accept
                    colors = test_colors
                    color_freq[new_color] += target_freq
                    del color_freq[target_color]
                    current_k -= 1
                    success = True
                    break
            
            if not success:
                # Try pairwise swaps between two colors
                # This is more expensive, so only do it occasionally
                if iteration % 10 == 0 and len(sorted_colors) >= 2:
                    c1 = sorted_colors[0]
                    c2 = sorted_colors[1]
                    test_colors = colors[:]
                    for i in range(N):
                        if test_colors[i] == c1:
                            test_colors[i] = c2
                        elif test_colors[i] == c2:
                            test_colors[i] = c1
                    
                    if count_violations(test_colors) == 0:
                        # This swap doesn't change k, but might enable future reductions
                        # Only accept if it helps (we'll skip for now)
                        pass
        
        return colors, len(set(colors))
    
    # Try to find a valid coloring with fewer colors using CSP
    for target_k in range(3, best_k):
        if time.time() > deadline - 10:
            break
        
        time_limit = min(deadline - 5, time.time() + 8.0)
        result = csp_solve(target_k, time_limit)
        
        if result is not None:
            result = list(result)
            if count_violations(result) == 0:
                # Try local search to further reduce
                improved, improved_k = local_search_reduce(result, time_limit=deadline - 3)
                if improved_k < best_k:
                    best_k = improved_k
                    best_colors = improved
                else:
                    best_colors = result
                break
    
    # Final local search on best solution
    if time.time() < deadline - 3:
        improved, improved_k = local_search_reduce(best_colors, time_limit=deadline - 1)
        if improved_k < best_k:
            best_k = improved_k
            best_colors = improved
    
    return best_colors
# EVOLVE-BLOCK-END
