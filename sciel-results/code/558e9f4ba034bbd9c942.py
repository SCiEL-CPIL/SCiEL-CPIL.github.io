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
            for e in cyc:
                EDGE_CYCLES[e].append(idx)

build_cycles()

def is_valid(coloring):
    for cyc in ALL_CYCLES:
        c0, c1, c2, c3 = coloring[cyc[0]], coloring[cyc[1]], coloring[cyc[2]], coloring[cyc[3]]
        if len({c0, c1, c2, c3}) == 4:
            return False
    return True

def check_edge_change(coloring, edge_idx, new_color):
    # Check if changing coloring[edge_idx] to new_color creates a rainbow cycle
    old_color = coloring[edge_idx]
    if old_color == new_color:
        return True
    
    for cyc_idx in EDGE_CYCLES[edge_idx]:
        cyc = ALL_CYCLES[cyc_idx]
        # Get colors, replacing the one at edge_idx
        c0 = coloring[cyc[0]] if cyc[0] != edge_idx else new_color
        c1 = coloring[cyc[1]] if cyc[1] != edge_idx else new_color
        c2 = coloring[cyc[2]] if cyc[2] != edge_idx else new_color
        c3 = coloring[cyc[3]] if cyc[3] != edge_idx else new_color
        
        # Check if 4 distinct
        # Optimization: if any two are equal, it's not rainbow
        if c0 == c1 or c0 == c2 or c0 == c3 or c1 == c2 or c1 == c3 or c2 == c3:
            continue
        # If we get here, they are all distinct
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

    def refine(coloring, max_iters=100000):
        """Local search to maximize distinct colors."""
        current_coloring = list(coloring)
        used_colors = set(current_coloring)
        current_score = len(used_colors)
        
        # If invalid, fix it (shouldn't happen if init is good, but safe)
        if not is_valid(current_coloring):
             return None, 0

        stuck_counter = 0
        max_stuck = 2000 # Reset or shuffle if stuck for too long
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            if current_score == MAX_COLOR + 1:
                break
                
            # Pick random edge
            idx = rng.randint(0, NUM_PAIRS - 1)
            
            # Strategy 1: Try to add a new color
            if len(used_colors) < MAX_COLOR + 1:
                # Pick a random unused color
                unused = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
                # To speed up, don't build list every time?
                # 120 colors, set lookup is O(1). List comp is O(120). Fine.
                new_color = rng.choice(unused)
                
                if check_edge_change(current_coloring, idx, new_color):
                    current_coloring[idx] = new_color
                    used_colors.add(new_color)
                    current_score += 1
                    stuck_counter = 0
                    continue
            else:
                # All colors used, we are done (score 120)
                break
                
            # If we didn't add a color, try to shuffle (move to different used color)
            # This helps escape local optima where no *new* color fits, but a rearrangement might.
            # Actually, if no new color fits at idx, maybe another edge allows a new color?
            # The random selection of idx handles this.
            
            # If we are stuck for a while, try a random valid recoloring with existing colors
            stuck_counter += 1
            if stuck_counter > max_stuck:
                # Perform a random valid perturbation
                perturbed = False
                for _ in range(100):
                    p_idx = rng.randint(0, NUM_PAIRS - 1)
                    p_old = current_coloring[p_idx]
                    # Pick a random used color
                    p_new = rng.choice(list(used_colors))
                    if p_new == p_old:
                        continue
                    if check_edge_change(current_coloring, p_idx, p_new):
                        current_coloring[p_idx] = p_new
                        perturbed = True
                        break
                stuck_counter = 0
                max_stuck = int(max_stuck * 1.5) # Increase patience
                
        if is_valid(current_coloring):
            return current_coloring, current_score
        return None, 0

    # Try various parametric starts
    # Small moduli are safer for validity
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
                            
                            # If we hit max, we can stop early?
                            # But maybe we can do better? No, 120 is max.
                            if best_score == 120:
                                return best_coloring
                                
    if best_coloring is None:
        return [0] * NUM_PAIRS
        
    return best_coloring
# EVOLVE-BLOCK-END
