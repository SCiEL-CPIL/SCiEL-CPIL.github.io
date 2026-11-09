"""Rewritten program: greedy construction with backtracking and time guard."""
import time
import random

def construct():
    """
    Returns n=128 integer color labels in 0..127.
    Every nonconstant four-term progression of positions must use AT LEAST three distinct colors.
    Minimize the number of labels actually used.
    """
    deadline = time.time() + 55.0  # 55s safety margin
    N = 128
    
    # Precompute all nonconstant 4-term progressions (a, a+d, a+2d, a+3d) with 0<=a, a+3d<N, d>=1
    # Positions are 0-indexed internally
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # For each position, precompute the list of APs that include it and the index within the AP
    pos_aps = [[] for _ in range(N)]
    for i, ap in enumerate(aps):
        for j, p in enumerate(ap):
            pos_aps[p].append((i, j))
    
    def count_distinct_colors(ap, colors):
        """Count distinct colors in an AP given current colors (None means uncolored)."""
        cs = set()
        for p in ap:
            if colors[p] is not None:
                cs.add(colors[p])
        return len(cs)
    
    def is_valid_partial(colors, pos, color, max_colors):
        """Check if assigning 'color' to 'pos' is valid given current partial coloring."""
        # Check all APs containing pos
        for ap_idx, j in pos_aps[pos]:
            ap = aps[ap_idx]
            # Temporarily assign
            old = colors[pos]
            colors[pos] = color
            
            # Check this AP: must have at least 3 distinct colors among colored positions
            # But we also need to ensure it CAN be extended to have >=3 distinct colors
            # For now, just check that we don't violate the constraint:
            # If all 4 positions are colored, must have >=3 distinct
            # If some are uncolored, we need to ensure it's still possible
            
            distinct = set()
            uncolored_count = 0
            for p in ap:
                if colors[p] is None:
                    uncolored_count += 1
                else:
                    distinct.add(colors[p])
            
            colors[pos] = old
            
            # If all colored, check constraint
            if uncolored_count == 0:
                if len(distinct) < 3:
                    return False
            # If some uncolored, we need at least 3 distinct in the end
            # The uncolored positions can be assigned any color from 0..max_colors-1
            # So we need: len(distinct) + uncolored_count >= 3 is necessary but not sufficient
            # Actually, we need to be able to assign colors to uncolored positions such that total distinct >= 3
            # This is possible iff len(distinct) + uncolored_count >= 3 AND we have enough color choices
            # But a tighter check: if len(distinct) == 0, we need uncolored_count >= 3 (which is always true since we just colored one)
            # If len(distinct) == 1, we need at least one uncolored position to get a different color
            # If len(distinct) == 2, we're already fine as long as we don't force all uncolored to be one of those two
            # Actually the constraint is "at least 3 distinct", so if we have 2 distinct and some uncolored,
            # we can always assign a third color to one of the uncolored positions
            # The only problematic case is if we have 0 or 1 distinct and not enough uncolored positions
            # But since we're building left to right, let's just check the hard constraint when all are colored
            
            # For a stronger check during construction: if the AP has some uncolored positions,
            # verify that it's still possible to achieve >=3 distinct colors
            # Max distinct we can achieve = len(distinct) + uncolored_count (if we use new colors for all uncolored)
            # But we're limited to max_colors colors total
            # So max achievable distinct = min(max_colors, len(distinct) + uncolored_count)
            # We need this to be >= 3
            if len(distinct) + uncolored_count < 3:
                return False
            # Also, if we've used all max_colors and len(distinct) < 3 and uncolored_count > 0,
            # we can still use existing colors, but we need at least 3 distinct among all
            # This is getting complex; for now, just check the hard constraint when fully colored
            # and a simple feasibility check
    
        return True
    
    def greedy_with_backtrack(max_colors, seed):
        """Greedy with limited backtracking."""
        rng = random.Random(seed)
        colors = [None] * N
        colors[0] = 0  # First position always color 0
        
        def backtrack(pos):
            if pos == N:
                return True
            
            if time.time() > deadline:
                return False
            
            # Try colors in order 0..max_colors-1
            for color in range(max_colors):
                if is_valid_partial(colors, pos, color, max_colors):
                    colors[pos] = color
                    
                    if backtrack(pos + 1):
                        return True
                    
                    colors[pos] = None
            
            return False
        
        # Simple greedy first, then backtrack if needed
        # For efficiency, do greedy first
        for pos in range(1, N):
            if time.time() > deadline:
                return None
            found = False
            for color in range(max_colors):
                if is_valid_partial(colors, pos, color, max_colors):
                    colors[pos] = color
                    found = True
                    break
            if not found:
                # Backtrack
                break
        
        if all(c is not None for c in colors):
            return colors
        
        # Try backtracking from the failure point
        # Find the first uncolored position
        fail_pos = 0
        while fail_pos < N and colors[fail_pos] is not None:
            fail_pos += 1
        
        if fail_pos < N:
            if backtrack(fail_pos):
                return colors
        
        return None
    
    # Try with increasing number of colors
    # Start with 3 colors (minimum possible)
    for num_colors in range(3, 129):
        if time.time() > deadline:
            break
        
        # Try a few random seeds for backtracking
        for seed in range(10):
            if time.time() > deadline:
                break
            
            result = greedy_with_backtrack(num_colors, seed)
            if result is not None:
                # Verify the solution
                valid = True
                for ap in aps:
                    distinct = set(result[p] for p in ap)
                    if len(distinct) < 3:
                        valid = False
                        break
                if valid:
                    return result
    
    # Fallback: return the seed program's result if we ran out of time
    # This shouldn't happen, but just in case
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 for d in range(1, (x-1)//3+1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) for d in range(1, (x-1)//2+1))):
                colors.append(color)
                break
    return colors
