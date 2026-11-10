"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys
import numpy as np

N = 16
NUM_PAIRS = N * (N - 1) // 2  # 120
MAX_COLOR = 119  # colors 0..119

def pair_index(i, j):
    """Return the index in the flat array for pair (i,j) where i<j."""
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

# Precompute the three cycles for each 4-subset
def build_cycles():
    from itertools import combinations
    cycles = []
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        e1 = [pair_index(a,b), pair_index(b,c), pair_index(c,d), pair_index(d,a)]
        e2 = [pair_index(a,b), pair_index(b,d), pair_index(d,c), pair_index(c,a)]
        e3 = [pair_index(a,c), pair_index(c,b), pair_index(b,d), pair_index(d,a)]
        cycles.append(tuple(e1))
        cycles.append(tuple(e2))
        cycles.append(tuple(e3))
    return cycles

ALL_CYCLES = build_cycles()
NUM_CYCLES = len(ALL_CYCLES)

# Precompute for each edge index, which cycles it belongs to
EDGE_CYCLES = [[] for _ in range(NUM_PAIRS)]
for ci, cyc in enumerate(ALL_CYCLES):
    for idx in cyc:
        EDGE_CYCLES[idx].append(ci)

def count_colors(coloring):
    return len(set(coloring))

def is_valid(coloring):
    for cyc in ALL_CYCLES:
        c0, c1, c2, c3 = coloring[cyc[0]], coloring[cyc[1]], coloring[cyc[2]], coloring[cyc[3]]
        if len({c0, c1, c2, c3}) == 4:
            return False
    return True

def check_violation(coloring, cyc):
    c0, c1, c2, c3 = coloring[cyc[0]], coloring[cyc[1]], coloring[cyc[2]], coloring[cyc[3]]
    return len({c0, c1, c2, c3}) == 4

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin
    
    best_coloring = None
    best_score = 0
    
    rng = random.Random(42)
    
    def parametric_coloring(a, b, c, M):
        """Generate coloring: color(i,j) = (a*i + b*j + c) % M, mapped to 0..119"""
        coloring = []
        for i in range(N):
            for j in range(i+1, N):
                coloring.append((a*i + b*j + c) % M)
        return coloring
    
    def try_parametric_and_refine(a, b, c, M, max_iters=20000):
        """Start with parametric coloring, then refine via local search."""
        coloring = parametric_coloring(a, b, c, M)
        
        # First, ensure validity by fixing violations
        for _ in range(5000):
            violated = False
            for cyc in ALL_CYCLES:
                if check_violation(coloring, cyc):
                    violated = True
                    idx = rng.choice(cyc)
                    coloring[idx] = rng.randint(0, MAX_COLOR)
                    break
            if not violated:
                break
        
        if not is_valid(coloring):
            return None, 0
        
        current_score = count_colors(coloring)
        
        # Local search: try to increase color count
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            idx = rng.randint(0, NUM_PAIRS - 1)
            old_color = coloring[idx]
            
            # Try to assign a new color not currently used, or a random one
            used_colors = set(coloring)
            if len(used_colors) < MAX_COLOR + 1:
                # Prefer unused colors to increase diversity
                unused = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
                if unused:
                    new_color = rng.choice(unused)
                else:
                    new_color = rng.randint(0, MAX_COLOR)
            else:
                new_color = rng.randint(0, MAX_COLOR)
            
            if new_color == old_color:
                continue
            
            coloring[idx] = new_color
            
            # Check validity only for affected cycles
            valid = True
            for ci in EDGE_CYCLES[idx]:
                cyc = ALL_CYCLES[ci]
                c0, c1, c2, c3 = coloring[cyc[0]], coloring[cyc[1]], coloring[cyc[2]], coloring[cyc[3]]
                if len({c0, c1, c2, c3}) == 4:
                    valid = False
                    break
            
            if valid:
                new_score = count_colors(coloring)
                if new_score >= current_score:
                    current_score = new_score
                else:
                    coloring[idx] = old_color
            else:
                coloring[idx] = old_color
        
        if is_valid(coloring):
            return list(coloring), current_score
        return None, 0
    
    # Try many parametric initializations with different parameters
    param_configs = []
    for M in range(2, 120):
        for a in range(0, min(M, 10)):
            for b in range(0, min(M, 10)):
                for c in range(0, min(M, 5)):
                    if gcd_gcd(a, b, M) != 0:  # ensure some variation
                        param_configs.append((a, b, c, M))
    
    # Shuffle and try a subset
    rng.shuffle(param_configs)
    
    # Also add some structured patterns
    structured_patterns = [
        (1, 2, 0, 17),
        (1, 3, 0, 19),
        (2, 3, 1, 23),
        (1, 4, 0, 29),
        (3, 5, 2, 31),
        (1, 1, 0, 11),
        (2, 1, 0, 13),
        (1, 5, 0, 37),
        (4, 1, 0, 41),
        (1, 7, 0, 43),
    ]
    param_configs.extend(structured_patterns)
    
    # Try parametric colorings
    for a, b, c, M in param_configs[:200]:
        if time.time() > deadline - 10.0:
            break
        
        coloring, score = try_parametric_and_refine(a, b, c, M, max_iters=10000)
        
        if coloring is not None:
            if score > best_score:
                best_score = score
                best_coloring = coloring
    
    # If we have a good result, do more aggressive refinement
    if best_coloring is not None and best_score > 0:
        # Further improvement: try to increase color count
        for _ in range(3):
            if time.time() > deadline - 3.0:
                break
            improved = False
            for idx in range(NUM_PAIRS):
                if time.time() > deadline - 2.0:
                    break
                old_color = best_coloring[idx]
                used_colors = set(best_coloring)
                for new_color in range(MAX_COLOR + 1):
                    if new_color in used_colors:
                        continue
                    best_coloring[idx] = new_color
                    # Check affected cycles
                    valid = True
                    for ci in EDGE_CYCLES[idx]:
                        cyc = ALL_CYCLES[ci]
                        c0, c1, c2, c3 = best_coloring[cyc[0]], best_coloring[cyc[1]], best_coloring[cyc[2]], best_coloring[cyc[3]]
                        if len({c0, c1, c2, c3}) == 4:
                            valid = False
                            break
                    if valid:
                        improved = True
                        break
                    else:
                        best_coloring[idx] = old_color
            if not improved:
                break
    
    if best_coloring is None:
        return [0] * NUM_PAIRS
    
    return best_coloring

def gcd_gcd(a, b, M):
    """Return 0 if a and b are both 0 mod M, else 1."""
    if a % M == 0 and b % M == 0:
        return 0
    return 1
# EVOLVE-BLOCK-END
