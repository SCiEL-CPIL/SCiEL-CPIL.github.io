import time
import random
import sys
import numpy as np

N = 16
NUM_PAIRS = N * (N - 1) // 2
MAX_COLOR = 119

def pair_index(i, j):
    if i > j: i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

# Precompute cycles
ALL_CYCLES = []
EDGE_CYCLES = [[] for _ in range(NUM_PAIRS)]
CYCLE_OTHER_EDGES = []  # For each cycle, store the other 3 edges

def build_cycles():
    from itertools import combinations
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        e1 = [pair_index(a,b), pair_index(b,c), pair_index(c,d), pair_index(d,a)]
        # Cycle 2: a-b-d-c-a
        e2 = [pair_index(a,b), pair_index(b,d), pair_index(d,c), pair_index(c,a)]
        # Cycle 3: a-c-b-d-a
        e3 = [pair_index(a,c), pair_index(c,b), pair_index(b,d), pair_index(d,a)]
        
        for cyc in [e1, e2, e3]:
            idx = len(ALL_CYCLES)
            ALL_CYCLES.append(tuple(cyc))
            CYCLE_OTHER_EDGES.append(tuple(x for x in cyc))
            for e in cyc:
                EDGE_CYCLES[e].append(idx)

build_cycles()

def is_valid(coloring):
    for cyc in ALL_CYCLES:
        c0, c1, c2, c3 = coloring[cyc[0]], coloring[cyc[1]], coloring[cyc[2]], coloring[cyc[3]]
        if len({c0, c1, c2, c3}) == 4:
            return False
    return True

def check_edge_change_fast(coloring, edge_idx, new_color):
    """Fast check: for each cycle containing edge_idx, verify no rainbow."""
    old_color = coloring[edge_idx]
    if old_color == new_color:
        return True
    
    for cyc_idx in EDGE_CYCLES[edge_idx]:
        cyc = ALL_CYCLES[cyc_idx]
        # Get the 3 other edge colors
        others = []
        for e in cyc:
            if e != edge_idx:
                others.append(coloring[e])
        
        # Check if adding new_color to others creates 4 distinct colors
        # This happens iff new_color is not in others AND all others are distinct
        if new_color not in others:
            if len(set(others)) == 3:
                return False
        
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    rng = random.Random(42)
    
    best_score = 0
    best_coloring = None
    
    def parametric_coloring(a, b, c, M):
        coloring = []
        for i in range(N):
            for j in range(i+1, N):
                coloring.append((a*i + b*j + c) % M)
        return coloring

    def refine(coloring, max_iters=500000):
        """Local search to maximize distinct colors."""
        current_coloring = list(coloring)
        used_colors = set(current_coloring)
        current_score = len(used_colors)
        
        # If invalid, fix it
        if not is_valid(current_coloring):
             return None, 0

        stuck_counter = 0
        max_stuck = 500
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            if current_score == MAX_COLOR + 1:
                break
                
            # Pick random edge
            idx = rng.randint(0, NUM_PAIRS - 1)
            
            # Strategy: Try to add a new color
            if len(used_colors) < MAX_COLOR + 1:
                # Pick a random unused color
                unused = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
                new_color = rng.choice(unused)
                
                if check_edge_change_fast(current_coloring, idx, new_color):
                    current_coloring[idx] = new_color
                    used_colors.add(new_color)
                    current_score += 1
                    stuck_counter = 0
                    continue
            
            # If we didn't add a color, try perturbation with existing colors
            stuck_counter += 1
            if stuck_counter > max_stuck:
                # Perform a random valid perturbation
                for _ in range(200):
                    p_idx = rng.randint(0, NUM_PAIRS - 1)
                    p_old = current_coloring[p_idx]
                    p_new = rng.choice(list(used_colors))
                    if p_new == p_old:
                        continue
                    if check_edge_change_fast(current_coloring, p_idx, p_new):
                        current_coloring[p_idx] = p_new
                        break
                stuck_counter = 0
                max_stuck = int(max_stuck * 1.3)
                
        if is_valid(current_coloring):
            return current_coloring, current_score
        return None, 0

    # Try various parametric starts
    moduli = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 119]
    
    for M in moduli:
        if time.time() > deadline - 10.0:
            break
            
        # Try a few coefficients for each M
        for a in range(0, min(M, 5)):
            for b in range(0, min(M, 5)):
                if a == 0 and b == 0: continue
                for c in range(0, min(M, 3)):
                    init_col = parametric_coloring(a, b, c, M)
                    
                    # Check validity of initial coloring
                    if not is_valid(init_col):
                        continue
                        
                    coloring, score = refine(init_col)
                    
                    if coloring is not None:
                        if score > best_score:
                            best_score = score
                            best_coloring = coloring
                            
                            # If we hit max, we can stop early
                            if best_score == 120:
                                return best_coloring
                                
    if best_coloring is None:
        return [0] * NUM_PAIRS
        
    return best_coloring
# EVOLVE-BLOCK-END
