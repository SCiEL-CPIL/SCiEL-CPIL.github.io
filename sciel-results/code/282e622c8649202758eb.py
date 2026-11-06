"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = 128
    
    # Precompute 4-term progressions ending at each index
    # prog_end[x] contains tuples (a, b, c) such that a, b, c, x form a 4-AP
    # a + 3d = x => a = x - 3d
    # b = a + d = x - 2d
    # c = a + 2d = x - d
    prog_end = [[] for _ in range(n)]
    for x in range(n):
        # d can range such that a >= 0
        # x - 3d >= 0 => 3d <= x => d <= x // 3
        for d in range(1, x // 3 + 1):
            a = x - 3 * d
            b = x - 2 * d
            c = x - d
            prog_end[x].append((a, b, c))
            
    def solve(k, time_limit):
        """
        Backtracking solver to find a valid coloring with k colors.
        Returns list of colors if found, else None.
        """
        colors = [-1] * n
        end_time = time.time() + time_limit
        
        def is_valid_assignment(x, c):
            """Check if assigning color c to x is valid given previous assignments."""
            # Check all progressions ending at x
            for a, b, c_prev in prog_end[x]:
                # The colors at a, b, c_prev are already assigned
                # We need len({colors[a], colors[b], colors[c_prev], c}) >= 3
                # Optimization: 
                # Let S = {colors[a], colors[b], colors[c_prev]}
                # If len(S) == 3, adding c keeps it >= 3.
                # If len(S) == 2, c must be different from both colors in S.
                # If len(S) == 1, c must be different from the color in S? 
                #   If S={v}, set is {v, c}. Size 2. We need size >= 3. 
                #   So if len(S)==1, it's impossible to reach 3 with just one more element?
                #   Wait. The progression has 4 elements.
                #   If a,b,c are all v, and x is c_new.
                #   Set is {v, c_new}. Size is 2 (if c_new != v) or 1 (if c_new == v).
                #   Both are < 3.
                #   So if len(S) == 1, it's ALWAYS INVALID.
                #   If len(S) == 2, say {u, v}. Set is {u, v, c_new}.
                #   Size is 3 if c_new not in {u,v}. Size is 2 if c_new in {u,v}.
                #   So c_new must NOT be in S.
                #   If len(S) == 3, set is S union {c_new}. Size is 3 or 4. Always >= 3.
                
                ca = colors[a]
                cb = colors[b]
                cc = colors[c_prev]
                
                # Count distinct in first 3
                if ca == cb:
                    if cb == cc:
                        return False # All 3 same, adding 4th can't make 3 distinct
                    else:
                        # 2 distinct: {ca, cc}
                        if c == ca or c == cc:
                            return False
                else:
                    if ca == cc:
                        # 2 distinct: {ca, cb}
                        if c == ca or c == cb:
                            return False
                    elif cb == cc:
                        # 2 distinct: {ca, cb}
                        if c == ca or c == cb:
                            return False
                    else:
                        # 3 distinct: {ca, cb, cc}
                        pass # Always valid regardless of c
            
            return True

        def backtrack(x):
            if time.time() > end_time:
                return False
            if x == n:
                return True
            
            # Heuristic: try colors in order, maybe randomize for symmetry breaking?
            # For now, standard order.
            for c in range(k):
                if is_valid_assignment(x, c):
                    colors[x] = c
                    if backtrack(x + 1):
                        return True
                    colors[x] = -1
                    
            return False

        if backtrack(0):
            return colors
        return None

    # Try k=3 first, then 4, etc.
    # k=2 is impossible as established.
    for k in range(3, 128):
        # Allocate more time for smaller k if it's likely to be hard?
        # Actually, if k=3 is impossible, it might run until timeout.
        # Let's give it 20s.
        time_allowed = 20 if k == 3 else 40
        if time.time() + time_allowed > deadline:
            break
            
        sol = solve(k, time_allowed)
        if sol is not None:
            return sol
            
    # Fallback: return a valid coloring with more colors if we ran out of time
    # Greedy construction with 128 colors (identity) is always valid?
    # If colors are 0..127, any 4-AP has 4 distinct indices, thus 4 distinct colors.
    # So identity is always valid.
    return list(range(128))
# EVOLVE-BLOCK-END
