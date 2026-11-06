"""Randomized local search (simulated annealing) for 3-colors-per-4AP coloring."""
import time
import random
import numpy as np

N = 128
DEADLINE = 85.0  # safety margin before 90s kill

# Precompute all non-constant 4-term progressions: (a, a+d, a+2d, a+3d)
PROGS = []
for a in range(N - 3):
    max_d = (N - 1 - a) // 3
    for d in range(1, max_d + 1):
        PROGS.append((a, a + d, a + 2 * d, a + 3 * d))
PROGS = np.array(PROGS, dtype=np.int32)
P = len(PROGS)

def _is_valid(c):
    """True if every 4AP uses >= 3 distinct colors."""
    cols = c[PROGS]
    # A 4AP is bad iff all 4 equal, or exactly 2 distinct colors with a 2-2 split
    # Simplest: count distinct via min/max and equality patterns
    # Fast: bad iff (c0==c1==c2==c3) or (c0==c1==c2!=c3 with c3!=c0) etc.
    # Use: number of distinct < 3.
    # Compute via: sort-free check
    a, b, cc, dd = cols[:, 0], cols[:, 1], cols[:, 2], cols[:, 3]
    # all four equal
    bad = (a == b) & (b == cc) & (cc == dd)
    # exactly 2 colors: need at least one pair equal and the rest from 2 values
    # distinct count < 3 means: the set {a,b,cc,dd} has size 1 or 2
    # size 2: at least two equal, and all values in {v1,v2}
    # Equivalent: (a==b) | (a==cc) | (a==dd) | (b==cc) | (b==dd) | (cc==dd)  AND not all equal
    # and the two values that appear are exactly 2 (i.e., not 3 or 4)
    # Simpler: size < 3 iff  (a==b or a==cc or a==dd) and (b==cc or b==dd or cc==dd) ... messy
    # Use numpy unique per row is slow. Instead:
    # size < 3 iff (a,b,cc,dd) take values in at most 2 symbols.
    # That is: (a==b) & (cc==dd) & (a!=cc)  -> 2 colors (2-2)  OR
    #          (a==cc) & (b==dd) & (a!=b)  -> 2 colors (1-2-1) OR
    #          (a==dd) & (b==cc) & (a!=b)  -> 2 colors (1-1-2) OR
    #          any three equal and fourth different -> 2 colors (3-1)
    # Let's just compute: m = max, M = min; if m==M -> 1 color. Else if the only distinct values are 2.
    # Fast vectorized: 
    #   eq1 = (a==b)|(a==cc)|(a==dd)
    #   eq2 = (b==cc)|(b==dd)|(cc==dd)
    #   if eq1 & eq2: at most 2 colors? Not exactly; could be 3.
    # Safest fast method: 
    #   distinct = 4 - (number of equal pairs among the 6 pairs) ... no, that's not right either.
    # Use: (a==b) & (b==cc) & (cc==dd) -> 1
    #       else: check if all in {a, b} OR all in {a, cc} etc.
    #   in_ab = (a==a) & (b==a|b==b) ... 
    # Let's do: 
    #   v1 = a; 
    #   rest_in_v1_v2 = (b==v1)|(b==b) ... 
    # I'll use a clean method: 
    #   two_color = ((a==b) & ((cc==a)|(cc==b)) & ((dd==a)|(dd==b))) 
    #               | ((a==cc) & ((b==a)|(b==cc)) & ((dd==a)|(dd==cc))) 
    #               | ((a==dd) & ((b==a)|(b==dd)) & ((cc==a)|(cc==dd))) 
    #               | ((b==cc) & ((a==b)|(a==cc)) & ((dd==b)|(dd==cc))) 
    #               | ((b==dd) & ((a==b)|(a==dd)) & ((cc==b)|(cc==dd))) 
    #               | ((cc==dd) & ((a==cc)|(a==dd)) & ((b==cc)|(b==dd)))
    #   But this overcounts 1-color case; that's fine since we OR with bad.
    # Actually simpler: 
    #   size<3 iff NOT (size==3 or size==4). 
    #   size==4: all distinct. size==3: exactly one pair equal, others distinct from it and each other.
    #   Let's just compute size via: 
    #     s = 4
    #     s -= (a==b)
    #     s -= (a==cc) & ~(a==b)  # careful with double counting
    # This is getting complex. Use a different fast trick:
    #   Pack into a single integer and use a lookup? No.
    #   Use: bad = ( (a==b)&(b==cc)&(cc==dd) ) |  # 1 color
    #         ( ((a==b)&(cc==dd)&(a!=cc)) |  # 2-2
    #           ((a==cc)&(b==dd)&(a!=b)) |  # 1-2-1
    #           ((a==dd)&(b==cc)&(a!=b)) |  # 1-1-2
    #           ((a==b)&(b==cc)&(a!=dd)) |  # 3-1
    #           ((a==cc)&(cc==dd)&(a!=b)) |  # 1-3
    #           ((a==dd)&(dd==b)&(a!=cc)) )  # 1-1-2 variant
    #   Wait, 3-1 cases: three equal, one different. There are 4 such patterns.
    #   Let me enumerate all 2-color patterns:
    #   Positions (0,1,2,3). Two colors X,Y. Patterns with both X and Y present:
    #   XXXY, XXYY, XYYX, XYYY, XXYX, XYXY, XYYX, YXXX, YXXY, YXYX, YYYX, YXYY
    #   That's a lot. 
    #   Simpler: size < 3 iff the 4 values span at most 2 symbols.
    #   Equivalent: (a,b,cc,dd) all in {min(a,b,cc,dd), max(a,b,cc,dd)} AND not all equal.
    #   Let mn = min(a,b,cc,dd), mx = max(a,b,cc,dd). 
    #   all_in_two = (a>=mn)&(a<=mx)&(b>=mn)&(b<=mx)&(cc>=mn)&(cc<=mx)&(dd>=mn)&(dd<=mx)  # always true by definition
    #   So that doesn't help. 
    #   The correct fast check: 
    #     bad = (a==b) & (b==cc) & (cc==dd)  # all equal
    #     # or exactly 2 distinct:
    #     #   (a==b) & (cc==dd) & (a!=cc)
    #     #   (a==cc) & (b==dd) & (a!=b)
    #     #   (a==dd) & (b==cc) & (a!=b)
    #     #   (a==b) & (b==cc) & (a!=dd)
    #     #   (a==cc) & (cc==dd) & (a!=b)
    #     #   (a==dd) & (dd==b) & (a!=cc)
    #     #   (b==cc) & (cc==dd) & (b!=a)
    #     #   (b==dd) & (dd==cc) & (b!=a)
    #   Let me just do: 
    #     p1 = (a==b) & (cc==dd) & (a!=cc)
    #     p2 = (a==cc) & (b==dd) & (a!=b)
    #     p3 = (a==dd) & (b==cc) & (a!=b)
    #     p4 = (a==b) & (b==cc) & (a!=dd)
    #     p5 = (a==cc) & (cc==dd) & (a!=b)
    #     p6 = (a==dd) & (dd==b) & (a!=cc)
    #     p7 = (b==cc) & (cc==dd) & (b!=a)
    #     p8 = (b==dd) & (dd==cc) & (b!=a)
    #     bad = bad | p1 | p2 | p3 | p4 | p5 | p6 | p7 | p8
    #   This covers all 2-color cases.
    p1 = (a == b) & (cc == dd) & (a != cc)
    p2 = (a == cc) & (b == dd) & (a != b)
    p3 = (a == dd) & (b == cc) & (a != b)
    p4 = (a == b) & (b == cc) & (a != dd)
    p5 = (a == cc) & (cc == dd) & (a != b)
    p6 = (a == dd) & (dd == b) & (a != cc)
    p7 = (b == cc) & (cc == dd) & (b != a)
    p8 = (b == dd) & (dd == cc) & (b != a)
    bad = bad | p1 | p2 | p3 | p4 | p5 | p6 | p7 | p8
    return not bad.any()

def construct():
    start = time.time()
    rng = np.random.default_rng(42)
    random.seed(42)
    
    best_c = None
    best_count = N + 1
    
    # Try multiple restarts with different color pool sizes
    for trial in range(30):
        if time.time() - start > DEADLINE:
            break
        # Start with a random coloring using a small pool
        # Try pool sizes: 3, 4, 5, ... but start small
        pool_size = 3 + (trial % 5)  # 3,4,5,6,7,3,4,...
        c = rng.integers(0, pool_size, size=N).astype(np.int32)
        
        # Simulated annealing
        T = 1.0
        cool = 0.9995
        steps = 0
        max_steps = 50000
        
        # Track used colors
        used = set(c.tolist())
        cur_count = len(used)
        
        # First, try to fix validity by hill-climbing
        # Count violations
        def count_violations(c):
            cols = c[PROGS]
            a, b, cc, dd = cols[:, 0], cols[:, 1], cols[:, 2], cols[:, 3]
            bad = (a == b) & (b == cc) & (cc == dd)
            p1 = (a == b) & (cc == dd) & (a != cc)
            p2 = (a == cc) & (b == dd) & (a != b)
            p3 = (a == dd) & (b == cc) & (a != b)
            p4 = (a == b) & (b == cc) & (a != dd)
            p5 = (a == cc) & (cc == dd) & (a != b)
            p6 = (a == dd) & (dd == b) & (a != cc)
            p7 = (b == cc) & (cc == dd) & (b != a)
            p8 = (b == dd) & (dd == cc) & (b != a)
            bad = bad | p1 | p2 | p3 | p4 | p5 | p6 | p7 | p8
            return bad.sum()
        
        cur_viol = count_violations(c)
        
        # Phase 1: reduce violations to 0
        while cur_viol > 0 and steps < max_steps and time.time() - start < DEADLINE:
            if time.time() - start > DEADLINE:
                break
            pos = int(rng.integers(0, N))
            old = c[pos]
            new = int(rng.integers(0, pool_size))
            if new == old:
                continue
            c[pos] = new
            new_viol = count_violations(c)
            if new_viol <= cur_viol:
                cur_viol = new_viol
            else:
                # Revert
                c[pos] = old
            steps += 1
            if steps % 100 == 0:
                T *= cool
        
        if cur_viol == 0:
            # Valid! Now minimize color count
            used = set(c.tolist())
            cur_count = len(used)
            if cur_count < best_count:
                best_count = cur_count
                best_c = c.copy()
            
            # Phase 2: try to reduce color count by recoloring
            # Try to eliminate rare colors
            for _ in range(200):
                if time.time() - start > DEADLINE:
                    break
                used = list(set(c.tolist()))
                if len(used) <= 3:
                    break
                # Pick a color with fewest occurrences
                counts = {col: int(np.sum(c == col)) for col in used}
                rarest = min(counts, key=counts.get)
                # Try to recolor all positions with rarest to another color
                positions = np.where(c == rarest)[0]
                success = True
                for pos in positions:
                    # Try each other color
                    found = False
                    for new_col in used:
                        if new_col == rarest:
                            continue
                        c[pos] = new_col
                        if count_violations(c) == 0:
                            found = True
                            break
                        c[pos] = rarest
                    if not found:
                        success = False
                        break
                if success:
                    c[positions] = 0  # placeholder, will be set
                    # Recalculate
                    used = set(c.tolist())
                    cur_count = len(used)
                    if cur_count < best_count:
                        best_count = cur_count
                        best_c = c.copy()
                else:
                    # Random perturbation
                    pos = int(rng.integers(0, N))
                    old = c[pos]
                    new = int(rng.integers(0, pool_size))
                    c[pos] = new
                    if count_violations(c) > 0:
                        c[pos] = old
        
        # Record best
        if _is_valid(c):
            used = set(c.tolist())
            cur_count = len(used)
            if cur_count < best_count:
                best_count = cur_count
                best_c = c.copy()
    
    if best_c is None:
        # Fallback: use the greedy-like construction
        colors = []
        for x in range(1, N + 1):
            for color in range(128):
                ok = True
                for d in range(1, (x - 1) // 3 + 1):
                    prog_colors = {color} | {colors[x - 1 - j * d] for j in (1, 2, 3)}
                    if len(prog_colors) < 3:
                        ok = False
                        break
                if not ok:
                    continue
                for d in range(1, (x - 1) // 2 + 1):
                    if all(colors[x - 1 - j * d] == color for j in (1, 2)):
                        ok = False
                        break
                if ok:
                    colors.append(color)
                    break
        return colors
    
    # Map colors to 0..127 range (they already are)
    return best_c.tolist()
