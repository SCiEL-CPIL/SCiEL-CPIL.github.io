import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = 128
    
    # Precompute all nonconstant 4-term progressions
    progressions = []
    for a in range(1, n + 1):
        for d in range(1, n):
            a3 = a + 3 * d
            if a3 <= n:
                progressions.append((a - 1, a + d - 1, a + 2 * d - 1, a3 - 1))
    
    num_progs = len(progressions)
    
    # For each position, which progressions does it appear in?
    pos_progs = [[] for _ in range(n)]
    for i, prog in enumerate(progressions):
        for pos in prog:
            pos_progs[pos].append(i)
    
    def is_violation(c0, c1, c2, c3):
        """Check if a 4-tuple violates (has < 3 distinct colors)"""
        # Count distinct colors
        if c0 == c1 == c2 == c3:
            return True
        # Check if only 2 distinct colors
        if c0 == c1 and c0 == c2 and c0 == c3:
            return True
        if c0 == c1 and c0 == c2 and c1 == c3:
            return True
        # More efficiently: count how many are different from c0
        if c0 == c1:
            if c0 == c2:
                # c0=c1=c2, check c3
                if c0 == c3:
                    return True
                # c0=c1=c2 != c3: 2 distinct, violation
                return True
            else:
                # c0=c1 != c2
                if c2 == c3:
                    # c0=c1, c2=c3, c0!=c2: 2 distinct, violation
                    return True
                # c0=c1, c2!=c3, c0!=c2: 3 distinct, ok
                return False
        else:
            # c0 != c1
            if c0 == c2:
                # c0=c2 != c1
                if c1 == c3:
                    # c0=c2, c1=c3, c0!=c1: 2 distinct, violation
                    return True
                # c0=c2, c1!=c3, c0!=c1: 3 distinct, ok
                return False
            else:
                # c0 != c1, c0 != c2
                if c1 == c2:
                    # c1=c2 != c0
                    if c0 == c3:
                        # c0=c3, c1=c2, c0!=c1: 2 distinct, violation
                        return True
                    # c0!=c3, c1=c2!=c0: 3 distinct, ok
                    return False
                else:
                    # c0, c1, c2 all different: at least 3 distinct, ok
                    return False
    
    def count_violations_full(colors):
        viol = 0
        for prog in progressions:
            if is_violation(colors[prog[0]], colors[prog[1]], colors[prog[2]], colors[prog[3]]):
                viol += 1
        return viol
    
    # Incremental: when we change position p from old_c to new_c,
    # only progressions containing p can change their violation status
    def delta_violation(colors, pos, old_c, new_c):
        """Return change in violation count when changing pos from old_c to new_c"""
        delta = 0
        for pi in pos_progs[pos]:
            prog = progressions[pi]
            c0 = colors[prog[0]]
            c1 = colors[prog[1]]
            c2 = colors[prog[2]]
            c3 = colors[prog[3]]
            
            # Current violation status
            was_viol = is_violation(c0, c1, c2, c3)
            
            # New violation status
            new_colors = [c0, c1, c2, c3]
            for j, p in enumerate(prog):
                if p == pos:
                    new_colors[j] = new_c
            
            now_viol = is_violation(new_colors[0], new_colors[1], new_colors[2], new_colors[3])
            
            if was_viol and not now_viol:
                delta -= 1
            elif not was_viol and now_viol:
                delta += 1
        
        return delta
    
    best_colors = None
    best_viol = float('inf')
    
    # Try deterministic patterns first
    patterns = [
        [i % 3 for i in range(n)],
        [(i + 1) % 3 for i in range(n)],
        [(i + 2) % 3 for i in range(n)],
        [i % 3 if i < 64 else (i + 1) % 3 for i in range(n)],
        [i % 3 if i < 64 else (i + 2) % 3 for i in range(n)],
        [(i * 2) % 3 for i in range(n)],
        [(i * 2 + 1) % 3 for i in range(n)],
    ]
    
    for pat in patterns:
        v = count_violations_full(pat)
        if v == 0:
            return pat
        if v < best_viol:
            best_viol = v
            best_colors = pat[:]
    
    # Local search from best pattern
    if best_colors is not None:
        colors = best_colors[:]
        viol = best_viol
        
        # Hill climbing
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for pos in range(n):
                if time.time() > deadline:
                    break
                old_c = colors[pos]
                for new_c in [0, 1, 2]:
                    if new_c == old_c:
                        continue
                    delta = delta_violation(colors, pos, old_c, new_c)
                    if delta < 0:
                        colors[pos] = new_c
                        viol += delta
                        improved = True
                        if viol == 0:
                            return colors
                    elif delta == 0 and random.random() < 0.1:
                        # Accept neutral move with small probability to escape local minima
                        colors[pos] = new_c
                        improved = True
                if viol == 0:
                    break
        
        if viol < best_viol:
            best_viol = viol
            best_colors = colors[:]
    
    # More random restarts with local search
    for attempt in range(50):
        if time.time() > deadline:
            break
        random.seed(attempt * 12345 + 67890)
        colors_r = [random.choice([0, 1, 2]) for _ in range(n)]
        viol_r = count_violations_full(colors_r)
        
        if viol_r < best_viol:
            best_viol = viol_r
            best_colors = colors_r[:]
        
        # Local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for pos in range(n):
                if time.time() > deadline:
                    break
                old_c = colors_r[pos]
                for new_c in [0, 1, 2]:
                    if new_c == old_c:
                        continue
                    delta = delta_violation(colors_r, pos, old_c, new_c)
                    if delta < 0:
                        colors_r[pos] = new_c
                        viol_r += delta
                        improved = True
                        if viol_r == 0:
                            return colors_r
                if viol_r == 0:
                    break
        
        if viol_r < best_viol:
            best_viol = viol_r
            best_colors = colors_r[:]
    
    if best_colors is not None:
        return best_colors
    
    return [i % 3 for i in range(n)]
