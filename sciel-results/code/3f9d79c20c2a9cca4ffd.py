import time
import random
import sys
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    # Sum of edges from vertex 0 to i-1: 0+1+...+(i-1) = i*(i-1)/2
    # Then within vertex i's edges, j is at position (j-i-1)
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

# Precompute for each edge, which 4-sets it participates in
EDGE_4SETS = [[] for _ in range(NUM_EDGES)]
# Precompute the 3 cycle edge-triples for each 4-set
FOUR_SETS = []
for quad in combinations(range(N), 4):
    a, b, c, d = quad
    e_ab = edge_idx(a, b)
    e_ac = edge_idx(a, c)
    e_ad = edge_idx(a, d)
    e_bc = edge_idx(b, c)
    e_bd = edge_idx(b, d)
    e_cd = edge_idx(c, d)
    # Three 4-cycles: (ab,bc,cd,ad), (ab,bd,cd,ac), (ac,bc,bd,ad)
    cycles = [(e_ab, e_bc, e_cd, e_ad),
              (e_ab, e_bd, e_cd, e_ac),
              (e_ac, e_bc, e_bd, e_ad)]
    FOUR_SETS.append((quad, cycles))
    for e in [e_ab, e_ac, e_ad, e_bc, e_bd, e_cd]:
        EDGE_4SETS[e].append(len(FOUR_SETS) - 1)

def is_rainbow(c1, c2, c3, c4):
    return c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4

def check_edge_valid(colors, e_idx, new_color):
    """Check if setting edge e_idx to new_color violates any rainbow 4-cycle."""
    for fs_idx in EDGE_4SETS[e_idx]:
        quad, cycles = FOUR_SETS[fs_idx]
        for cyc in cycles:
            if e_idx in cyc:
                pos = cyc.index(e_idx)
                c = [new_color if p == pos else colors[cyc[p]] for p in range(4)]
                if is_rainbow(c[0], c[1], c[2], c[3]):
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Start with a valid Gallai coloring: color(i,j) = min(i,j)
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            colors[edge_idx(i, j)] = i
    
    best_count = len(set(colors))
    best_result = colors[:]
    
    edge_list = list(range(NUM_EDGES))
    
    # Strategy: Try to introduce new colors by swapping existing colors
    # to free up "room" for new colors on other edges
    for restart in range(200):
        if time.time() > deadline - 3:
            break
        
        # Reset to base
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                colors[edge_idx(i, j)] = i
        
        next_color = 15
        random.shuffle(edge_list)
        
        # Phase 1: Greedily add new colors
        for e in edge_list:
            if time.time() > deadline - 2:
                break
            if check_edge_valid(colors, e, next_color):
                colors[e] = next_color
                next_color += 1
        
        cur_count = len(set(colors))
        if cur_count > best_count:
            best_count = cur_count
            best_result = colors[:]
        
        # Phase 2: Try to improve by recoloring edges with existing colors
        # to enable more new colors
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            random.shuffle(edge_list)
            
            for e in edge_list:
                if time.time() > deadline - 2:
                    break
                
                old_color = colors[e]
                # Try assigning a new color
                if check_edge_valid(colors, e, next_color):
                    colors[e] = next_color
                    next_color += 1
                    improved = True
                    cur_count = len(set(colors))
                    if cur_count > best_count:
                        best_count = cur_count
                        best_result = colors[:]
                    continue
                
                # Try swapping with another edge's color to enable new color
                # Pick a random other edge and see if swapping helps
                for _ in range(3):
                    if time.time() > deadline - 2:
                        break
                    e2 = random.choice(edge_list)
                    if e2 == e:
                        continue
                    c1, c2 = colors[e], colors[e2]
                    if c1 == c2:
                        continue
                    
                    # Try: swap colors, then try new color on e
                    colors[e], colors[e2] = c2, c1
                    if check_edge_valid(colors, e, next_color):
                        colors[e] = next_color
                        next_color += 1
                        improved = True
                        cur_count = len(set(colors))
                        if cur_count > best_count:
                            best_count = cur_count
                            best_result = colors[:]
                    else:
                        # Revert swap
                        colors[e], colors[e2] = c1, c2
    
    return best_result

if __name__ == "__main__":
    c = construct()
    print(len(set(c)))
