import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    def is_valid(idx, color, colors):
        # Check all 4-APs ending at idx
        # d=1 to (idx-1)//3
        # Actually idx is 0-indexed.
        # AP: idx-3d, idx-2d, idx-d, idx
        # Need idx-3d >= 0 => 3d <= idx => d <= idx//3
        max_d = idx // 3
        for d in range(1, max_d + 1):
            i3 = idx - 3 * d
            i2 = idx - 2 * d
            i1 = idx - d
            # Get previous colors
            c3 = colors[i3]
            c2 = colors[i2]
            c1 = colors[i1]
            # Check if adding 'color' makes size >= 3
            # If {c3, c2, c1} already has size 3, any color is valid for this AP
            # If size 2, 'color' must be different from both
            # If size 1, 'color' can be anything? No, {A, A, A, B} -> size 2. Invalid.
            # So if prev size 1, 'color' must be different?
            # If prev is {A, A, A}, set is {A, color}. Size 2. Invalid.
            # So 'color' must be different from A? No, even if different, size is 2.
            # Wait. If prev is {A, A, A}, and color=B, set is {A, B}. Size 2.
            # So if prev size 1, it's ALWAYS invalid?
            # Yes, because 4 elements from 2 values max have size 2.
            # So if any of c3,c2,c1 are all same, it's invalid regardless of color?
            # No, if c3=c2=c1=A, and color=B, set is {A,B}. Size 2.
            # So yes, if the first 3 are same, it's impossible to satisfy with the 4th.
            # So we can short-circuit: if c3==c2==c1, return False.
            
            if c3 == c2 and c2 == c1:
                return False
                
            # If prev set size is 2, say {A, B}, color must be not in {A, B}?
            # If color is A, set is {A, B}. Size 2.
            # If color is C, set is {A, B, C}. Size 3.
            # So if prev size 2, color must be new.
            
            # If prev set size is 3, any color is ok.
            
            prev_set = {c3, c2, c1}
            if len(prev_set) == 1:
                return False
            elif len(prev_set) == 2:
                if color in prev_set:
                    return False
            # else len(prev_set) == 3, ok
            
        return True

    def greedy_construct(color_set):
        colors = []
        for i in range(128):
            found = False
            perm = list(color_set)
            random.shuffle(perm)
            for c in perm:
                if is_valid(i, c, colors):
                    colors.append(c)
                    found = True
                    break
            if not found:
                return None
        return colors

    # Try 3 colors
    # Deterministic first? No, random is better for finding a solution if it exists.
    # But deterministic is faster to check "impossible".
    # Let's just do random restarts.
    
    # Optimization: Use a fixed seed for reproducibility? No, random is fine.
    
    while time.time() < deadline - 5:
        colors = greedy_construct([0, 1, 2])
        if colors:
            # Final verification
            valid = True
            for i in range(128):
                if not is_valid(i, colors[i], colors[:i]):
                    valid = False
                    break
            if valid:
                return colors

    # If 3 colors failed, try 4
    while time.time() < deadline:
        colors = greedy_construct([0, 1, 2, 3])
        if colors:
            valid = True
            for i in range(128):
                if not is_valid(i, colors[i], colors[:i]):
                    valid = False
                    break
            if valid:
                return colors

    # Fallback
    colors = []
    for i in range(128):
        for c in range(128):
            if is_valid(i, c, colors):
                colors.append(c)
                break
    return colors
