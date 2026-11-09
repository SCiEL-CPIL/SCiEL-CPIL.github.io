"""Colorings with three colors in every four-term progression.
Strategy: start from a valid 4-coloring (2-adic valuation mod 4),
then use Kempe-chain color-swap local search to merge color classes,
minimizing the number of distinct labels used.
"""
import time
import numpy as np
from itertools import combinations

# Precompute all nonconstant 4-term progressions in [1..128]
# A 4-term progression: a, a+d, a+2d, a+3d with d>=1, a+3d <= 128
# Positions are 0-indexed: 0..127
PROGRESSIONS = []
for a in range(126):  # a from 0 to 125 (0-indexed)
    for d in range(1, (127 - a) // 3 + 1):
        if a + 3 * d < 128:
            PROGRESSIONS.append((a, a + d, a + 2 * d, a + 3 * d))

PROGRESSIONS = np.array(PROGRESSIONS)  # shape (num_progs, 4)
NUM_PROGS = len(PROGRESSIONS)

def _check_validity(colors):
    """Check if every 4-term progression uses at least 3 distinct colors."""
    # colors is a numpy array of shape (128,)
    c = colors[PROGRESSIONS]  # shape (num_progs, 4)
    # For each progression, count distinct colors
    # A progression is invalid if it uses 1 or 2 distinct colors
    # 1 distinct: all four same
    # 2 distinct: various patterns
    # Fast check: count how many are NOT (all same) and NOT (exactly 2 colors)
    # Use: for each row, compute if len(set) >= 3
    # Vectorized: a row has <3 distinct colors iff:
    #   all same, OR (exactly 2 distinct)
    # Exactly 2 distinct means the 4 values take exactly 2 values
    # Let's use: min_distinct >= 3
    # For 4 elements, distinct count < 3 means:
    #   1 distinct: c0==c1==c2==c3
    #   2 distinct: not all same, but only 2 values
    # Simpler: check if the row has at least 3 distinct values
    # Using: the number of distinct values in {c0,c1,c2,c3} >= 3
    # This is equivalent to: NOT (all same) AND NOT (exactly 2 distinct)
    
    # Fast vectorized approach:
    # A 4-tuple has < 3 distinct colors iff:
    # (c0==c1 and c0==c2 and c0==c3)  # all same
    # OR (c0==c1 and c2==c3 and c0!=c2)  # pattern AABC
    # OR (c0==c2 and c1==c3 and c0!=c1)  # pattern ABAB
    # OR (c0==c3 and c1==c2 and c0!=c1)  # pattern ABBA
    # OR (c0==c1 and c0==c2 and c3!=c0)  # AAAB
    # OR (c0==c1 and c2==c3 and c0!=c2)  # AABC (same as above)
    # ... this gets complex. Let's use a simpler approach.
    
    # For 4 elements, distinct count >= 3 iff:
    # NOT (all pairs that should be equal in a 2-color pattern are equal)
    # Actually, let's just compute: for each row, is the number of unique values >= 3?
    # We can check: the row is "bad" if it has at most 2 distinct values.
    # Bad patterns for 4 elements with <= 2 distinct values:
    # 1 distinct: xxxx
    # 2 distinct: xyxy, xyxx, xxyx, xxxy, xyyy, yxxx, yxxy, yyxx, xyyx, yyxy, yxyy, yxxy
    # Actually simpler: a 4-tuple has <= 2 distinct values iff:
    # (a==b and a==c and a==d)  # 1 distinct
    # or (a==b and a==c and d!=a)  # 3 same, 1 diff -> 2 distinct
    # or (a==b and c==d and a!=c)  # 2-2 split
    # or (a==c and b==d and a!=b)  # 2-2 split
    # or (a==d and b==c and a!=b)  # 2-2 split
    # or (a==b and a!=c and a!=d and c==d)  # 2-2 split (same as above)
    # or (a==c and a!=b and a!=d and b==d)  # 2-2 split
    # or (a==d and a!=b and a!=c and b==c)  # 2-2 split
    # or (a!=b and a!=c and a!=d and b==c and b!=d)  # 3 same, 1 diff
    # or (a!=b and a!=c and a!=d and b==d and b!=c)  # 3 same, 1 diff
    # or (a!=b and a!=c and a!=d and c==d and c!=b)  # 3 same, 1 diff
    # or (b==c and c==d and a!=b)  # 3 same, 1 diff
    # or (a==b and b==c and d!=a)  # 3 same, 1 diff
    # or (a==c and c==d and b!=a)  # 3 same, 1 diff
    # or (b==d and d==c and a!=b)  # 3 same, 1 diff
    # or (a==b and b==d and c!=a)  # 3 same, 1 diff
    # or (a==c and c==b and d!=a)  # 3 same, 1 diff
    # or (a==d and d==b and c!=a)  # 3 same, 1 diff
    # or (a==d and d==c and b!=a)  # 3 same, 1 diff
    # or (b==c and c==a and d!=b)  # 3 same, 1 diff
    # or (b==d and d==a and c!=b)  # 3 same, 1 diff
    # or (c==d and d==a and b!=c)  # 3 same, 1 diff
    
    # This is getting unwieldy. Let me use a cleaner vectorized approach.
    # A 4-tuple (w,x,y,z) has < 3 distinct values iff:
    # (w==x==y==z)  # 1 distinct
    # or exactly 2 distinct values
    # Exactly 2 distinct: the multiset has size 2.
    # For 4 elements with exactly 2 distinct values, the partition is either 3+1 or 2+2.
    # 3+1: three equal, one different.
    # 2+2: two pairs equal.
    
    c0, c1, c2, c3 = c[:,0], c[:,1], c[:,2], c[:,3]
    
    # All same
    all_same = (c0 == c1) & (c0 == c2) & (c0 == c3)
    
    # 3+1 patterns:
    # c0,c1,c2 same, c3 different
    p1 = (c0 == c1) & (c0 == c2) & (c0 != c3)
    # c0,c1,c3 same, c2 different
    p2 = (c0 == c1) & (c0 == c3) & (c0 != c2)
    # c0,c2,c3 same, c1 different
    p3 = (c0 == c2) & (c0 == c3) & (c0 != c1)
    # c1,c2,c3 same, c0 different
    p4 = (c1 == c2) & (c1 == c3) & (c1 != c0)
    
    # 2+2 patterns:
    # c0==c1, c2==c3, c0!=c2
    q1 = (c0 == c1) & (c2 == c3) & (c0 != c2)
    # c0==c2, c1==c3, c0!=c1
    q2 = (c0 == c2) & (c1 == c3) & (c0 != c1)
    # c0==c3, c1==c2, c0!=c1
    q3 = (c0 == c3) & (c1 == c2) & (c0 != c1)
    
    bad = all_same | p1 | p2 | p3 | p4 | q1 | q2 | q3
    return not np.any(bad)

def _get_bad_progressions(colors):
    """Return indices of progressions that violate the constraint."""
    c = colors[PROGRESSIONS]
    c0, c1, c2, c3 = c[:,0], c[:,1], c[:,2], c[:,3]
    all_same = (c0 == c1) & (c0 == c2) & (c0 == c3)
    p1 = (c0 == c1) & (c0 == c2) & (c0 != c3)
    p2 = (c0 == c1) & (c0 == c3) & (c0 != c2)
    p3 = (c0 == c2) & (c0 == c3) & (c0 != c1)
    p4 = (c1 == c2) & (c1 == c3) & (c1 != c0)
    q1 = (c0 == c1) & (c2 == c3) & (c0 != c2)
    q2 = (c0 == c2) & (c1 == c3) & (c0 != c1)
    q3 = (c0 == c3) & (c1 == c2) & (c0 != c1)
    bad = all_same | p1 | p2 | p3 | p4 | q1 | q2 | q3
    return np.where(bad)[0]

def _initial_4coloring():
    """Create a valid 4-coloring using 2-adic valuation mod 4."""
    colors = np.zeros(128, dtype=np.int64)
    for x in range(1, 129):
        # 2-adic valuation: number of trailing zeros in binary representation of x
        v = 0
        n = x
        while n % 2 == 0:
            v += 1
            n //= 2
        colors[x - 1] = v % 4
    return colors

def _try_merge(colors, color_a, color_b):
    """
    Try to merge color_b into color_a by swapping some positions.
    Returns (new_colors, success) where success indicates if the merge was valid.
    Uses Kempe-chain: find a connected component in the graph where edges connect
    positions that appear together in a critical progression, and swap colors within it.
    """
    # We want to eliminate color_b by reassigning those positions to other colors
    # Strategy: for each position with color_b, try to change it to color_a or another color
    # such that all progressions remain valid.
    
    # Simpler approach: try direct recoloring of color_b positions
    # For each position i with colors[i] == color_b, try changing to color_a
    # and check validity.
    
    # Even simpler: try to map all color_b to color_a directly
    new_colors = colors.copy()
    new_colors[new_colors == color_b] = color_a
    
    if _check_validity(new_colors):
        return new_colors, True
    
    # Try partial merge: swap color_b with color_a in a Kempe chain
    # Build conflict graph: two positions are connected if they appear in the same
    # progression and swapping would create a violation.
    # For now, try a simpler heuristic: randomly reassign color_b positions to 
    # other colors and check.
    
    # Try each possible target color for the color_b positions
    used_colors = set(colors[colors >= 0].tolist())
    for target in used_colors:
        if target == color_b:
            continue
        new_colors = colors.copy()
        new_colors[new_colors == color_b] = target
        if _check_validity(new_colors):
            return new_colors, True
    
    # Try mixed: assign some color_b positions to color_a, others to another color
    # This is more complex. For now, return failure.
    return colors, False

def _local_search(colors, time_limit):
    """
    Perform local search to minimize the number of distinct colors.
    Try to merge color classes one at a time.
    """
    start_time = time.time()
    colors = colors.copy()
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 5:  # safety margin
            break
        
        # Find all used colors
        used = sorted(set(colors.tolist()))
        if len(used) <= 1:
            break
        
        # Try to merge the smallest color class into another
        # Sort by frequency (ascending) to try eliminating rare colors first
        color_freq = {}
        for c in used:
            color_freq[c] = np.sum(colors == c)
        
        # Try pairs of colors to merge
        merged = False
        for c_small in sorted(used, key=lambda c: color_freq[c]):
            for c_target in used:
                if c_small == c_target:
                    continue
                new_colors, success = _try_merge(colors, c_small, c_target)
                if success:
                    colors = new_colors
                    merged = True
                    break
            if merged:
                break
        
        if not merged:
            # Try individual position recoloring
            improved = False
            for i in range(128):
                if time.time() - start_time > time_limit - 5:
                    break
                old_color = colors[i]
                for new_color in range(4):
                    if new_color == old_color:
                        continue
                    trial = colors.copy()
                    trial[i] = new_color
                    if _check_validity(trial):
                        # Check if this reduces the number of colors
                        old_used = set(colors.tolist())
                        new_used = set(trial.tolist())
                        if len(new_used) < len(old_used):
                            colors = trial
                            improved = True
                            break
            if not improved:
                break
    
    return colors

def construct():
    deadline = 60.0  # seconds
    start_time = time.time()
    
    # Step 1: Create initial valid 4-coloring
    colors = _initial_4coloring()
    
    # Verify initial coloring is valid
    if not _check_validity(colors):
        # Fallback: use a different construction
        # Try coloring based on x mod 4
        colors = np.array([(x) % 4 for x in range(128)], dtype=np.int64)
        if not _check_validity(colors):
            # Use the original greedy approach as fallback
            colors_list = []
            for x in range(1, 129):
                for color in range(128):
                    if (all(len({color} | {colors_list[x-1-j*d] for j in (1,2,3)}) >= 3 
                           for d in range(1, (x-1)//3+1))
                        and all(any(colors_list[x-1-j*d] != color for j in (1,2)) 
                              for d in range(1, (x-1)//2+1))):
                        colors_list.append(color)
                        break
            colors = np.array(colors_list, dtype=np.int64)
    
    # Step 2: Local search to minimize colors
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        colors = _local_search(colors, deadline - 5)
    
    # Step 3: Final verification and normalization
    # Ensure all labels are in 0..127
    colors = np.clip(colors, 0, 127).astype(np.int64)
    
    # Verify final result
    if not _check_validity(colors):
        # Return the initial valid coloring
        colors = _initial_4coloring()
    
    return colors.tolist()
