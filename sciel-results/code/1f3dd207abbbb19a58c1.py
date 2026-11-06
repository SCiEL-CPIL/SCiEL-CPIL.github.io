import time
import random
import numpy as np

def _seed_fallback():
    """Deterministic seed algorithm - guaranteed valid."""
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x - 1 - j * d] for j in (1, 2, 3)}) >= 3 
                    for d in range(1, (x - 1) // 3 + 1))
                and all(any(colors[x - 1 - j * d] != color for j in (1, 2)) 
                        for d in range(1, (x - 1) // 2 + 1))):
                colors.append(color)
                break
    return colors

def _validate(colors):
    """Check if every non-constant 4-AP uses at least 3 distinct colors."""
    n = len(colors)
    for a in range(n):
        for d in range(1, n):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e >= n:
                break
            cols = {colors[a], colors[b], colors[c], colors[e]}
            if len(cols) < 3:
                return False
    return True

def _precompute_aps(n=128):
    """Precompute all 4-APs as (a, b, c, e) tuples and index by position."""
    all_aps = []
    pos_to_aps = [[] for _ in range(n)]
    for a in range(n):
        for d in range(1, (n - a - 1) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e < n:
                ap = (a, b, c, e)
                all_aps.append(ap)
                for p in ap:
                    pos_to_aps[p].append(ap)
    return all_aps, pos_to_aps

def _greedy_with_lookahead(n=128, seed=0, max_lookahead=3):
    """Greedy with limited lookahead to reduce color count."""
    rng = random.Random(seed)
    all_aps, pos_to_aps = _precompute_aps(n)
    colors = [-1] * n
    
    for i in range(n):
        # Get valid colors for position i based on already-assigned positions
        valid_colors = {0, 1, 2}
        for ap in pos_to_aps[i]:
            a, b, c, e = ap
            # Check if other positions in this AP are assigned
            others = [p for p in ap if p != i]
            if all(colors[p] != -1 for p in others):
                other_colors = {colors[p] for p in others}
                for col in list(valid_colors):
                    if len(other_colors | {col}) < 3:
                        valid_colors.discard(col)
        
        if not valid_colors:
            return None
        
        # Try each valid color with a quick lookahead
        best_color = None
        best_score = float('inf')
        
        for col in valid_colors:
            colors[i] = col
            # Quick check: count how many future positions become constrained
            # Just check immediate feasibility
            feasible = True
            for j in range(i + 1, min(i + max_lookahead + 1, n)):
                if colors[j] == -1:
                    # Check if position j still has valid colors
                    j_valid = {0, 1, 2}
                    for ap in pos_to_aps[j]:
                        a, b, c, e = ap
                        others = [p for p in ap if p != j]
                        if all(colors[p] != -1 for p in others):
                            other_colors = {colors[p] for p in others}
                            for c2 in list(j_valid):
                                if len(other_colors | {c2}) < 3:
                                    j_valid.discard(c2)
                    if not j_valid:
                        feasible = False
                        break
            if feasible:
                # Score: prefer colors already used
                used = set(colors[:i])
                score = 0 if col in used else 1
                if score < best_score:
                    best_score = score
                    best_color = col
            colors[i] = -1
        
        if best_color is None:
            # Fallback: just pick any valid color
            best_color = rng.choice(list(valid_colors))
        
        colors[i] = best_color
    
    return colors

def _local_search_reduce_colors(colors, n=128, max_iters=5000, seed=0):
    """Try to reduce color count by merging colors."""
    rng = random.Random(seed)
    colors = list(colors)
    current_count = len(set(colors))
    
    if current_count <= 3:
        return colors
    
    # Try to merge two colors into one
    color_list = sorted(set(colors))
    
    for _ in range(max_iters):
        if len(color_list) <= 3:
            break
        
        # Pick two colors to try merging
        c1, c2 = rng.sample(color_list, 2)
        
        # Try merging c2 into c1
        new_colors = list(colors)
        for i in range(n):
            if new_colors[i] == c2:
                new_colors[i] = c1
        
        # Validate
        if _validate(new_colors):
            colors = new_colors
            color_list = sorted(set(colors))
            current_count = len(color_list)
            if current_count <= 3:
                break
    
    return colors

def _construct_3color_direct(n=128, seed=0, time_limit=3.0):
    """Direct construction of 3-coloring using pattern-based approach."""
    start_time = time.time()
    rng = random.Random(seed)
    
    # Try various patterns
    # Pattern 1: Based on 2-adic valuation
    for attempt in range(50):
        if time.time() - start_time > time_limit:
            break
        
        if attempt == 0:
            # 2-adic valuation mod 3
            colors = []
            for x in range(1, n + 1):
                v2 = 0
                tmp = x
                while tmp % 2 == 0:
                    tmp //= 2
                    v2 += 1
                colors.append(v2 % 3)
        elif attempt == 1:
            # 3-adic valuation mod 3
            colors = []
            for x in range(1, n + 1):
                v3 = 0
                tmp = x
                while tmp % 3 == 0:
                    tmp //= 3
                    v3 += 1
                colors.append(v3 % 3)
        elif attempt == 2:
            # Sum of digits in base 2 mod 3
            colors = []
            for x in range(1, n + 1):
                s = bin(x - 1).count('1')
                colors.append(s % 3)
        elif attempt == 3:
            # (x-1) mod 3 with perturbation
            colors = []
            for x in range(1, n + 1):
                base = (x - 1) % 3
                # Add some randomness
                if rng.random() < 0.1:
                    base = (base + 1) % 3
                colors.append(base)
        elif attempt < 10:
            # Random with constraint checking
            colors = [-1] * n
            valid = True
            for i in range(n):
                valid_colors = {0, 1, 2}
                for d in range(1, (i) // 3 + 1):
                    a = i - 3 * d
                    b = i - 2 * d
                    c = i - d
                    if a >= 0:
                        c1, c2, c3 = colors[a], colors[b], colors[c]
                        for col in list(valid_colors):
                            if len({c1, c2, c3, col}) < 3:
                                valid_colors.discard(col)
                if not valid_colors:
                    valid = False
                    break
                colors[i] = rng.choice(list(valid_colors))
            if not valid:
                continue
        else:
            # Greedy with preference for used colors
            colors = [-1] * n
            valid = True
            for i in range(n):
                valid_colors = {0, 1, 2}
                for d in range(1, (i) // 3 + 1):
                    a = i - 3 * d
                    b = i - 2 * d
                    c = i - d
                    if a >= 0:
                        c1, c2, c3 = colors[a], colors[b], colors[c]
                        for col in list(valid_colors):
                            if len({c1, c2, c3, col}) < 3:
                                valid_colors.discard(col)
                if not valid_colors:
                    valid = False
                    break
                used = set(colors[:i])
                preferred = valid_colors & used
                if preferred:
                    colors[i] = rng.choice(list(preferred))
                else:
                    colors[i] = rng.choice(list(valid_colors))
            if not valid:
                continue
        
        if _validate(colors):
            if len(set(colors)) == 3:
                return colors
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = 128
    
    # First, get the seed fallback (guaranteed valid)
    seed_colors = _seed_fallback()
    best = seed_colors
    best_count = len(set(seed_colors))
    
    # Try direct 3-coloring construction
    remaining = deadline - time.time()
    if remaining > 5:
        result = _construct_3color_direct(n, seed=0, time_limit=min(4.0, remaining - 2))
        if result is not None:
            best = result
            best_count = 3
            return best
    
    # Try greedy with lookahead
    remaining = deadline - time.time()
    if remaining > 3:
        for seed in range(100):
            if time.time() > deadline - 5:
                break
            result = _greedy_with_lookahead(n, seed, max_lookahead=2)
            if result is not None:
                if _validate(result):
                    count = len(set(result))
                    if count < best_count:
                        best = result
                        best_count = count
                        if best_count == 3:
                            return best
    
    # Try local search to reduce colors
    remaining = deadline - time.time()
    if remaining > 2 and best_count > 3:
        for seed in range(50):
            if time.time() > deadline - 1:
                break
            result = _local_search_reduce_colors(best, n, max_iters=1000, seed=seed)
            count = len(set(result))
            if count < best_count:
                best = result
                best_count = count
                if best_count == 3:
                    return best
    
    return best
