import time
import random
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
NUM_COLORS = 120  # 0..119

# Precompute edge index mapping
def edge_idx(i, j):
    if i > j:
        i, j = j, i
    # Lexicographic ordering of pairs (i,j) with 0<=i<j<16
    # (0,1), (0,2), ..., (0,15), (1,2), (1,3), ..., (1,15), ..., (14,15)
    # Index = i*(N-1) - i*(i-1)//2 + (j - i - 1)
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

# Precompute all 4-cycles and their edge indices
def precompute_cycles():
    cycles = []
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        # Three 4-cycles on these 4 vertices:
        # 1: a-b-c-d-a => edges ab, bc, cd, da
        # 2: a-b-d-c-a => edges ab, bd, dc, ca
        # 3: a-c-b-d-a => edges ac, cb, bd, da
        cycles.append((edge_idx(a, b), edge_idx(b, c), edge_idx(c, d), edge_idx(d, a)))
        cycles.append((edge_idx(a, b), edge_idx(b, d), edge_idx(d, c), edge_idx(c, a)))
        cycles.append((edge_idx(a, c), edge_idx(c, b), edge_idx(b, d), edge_idx(d, a)))
    return cycles

CYCLES = precompute_cycles()

# For each edge, precompute which cycles it participates in
def precompute_edge_cycles():
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for idx, cycle in enumerate(CYCLES):
        for e in cycle:
            edge_to_cycles[e].append(idx)
    return edge_to_cycles

EDGE_CYCLES = precompute_edge_cycles()

def count_distinct_colors(colors):
    return len(set(colors))

def is_valid(colors):
    """Check if no 4-cycle has 4 distinct colors."""
    for e1, e2, e3, e4 in CYCLES:
        c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
        if len({c1, c2, c3, c4}) == 4:
            return False
    return True

def check_cycle_valid(colors, cycle):
    e1, e2, e3, e4 = cycle
    c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
    return len({c1, c2, c3, c4}) != 4

def construct():
    deadline = time.time() + 85  # Leave 5s safety margin
    
    random.seed(42)
    
    # Initialize with all zeros (valid, 1 color)
    colors = [0] * NUM_EDGES
    
    # Greedy initialization: try to assign distinct colors while maintaining validity
    # Start by trying to use as many colors as possible
    best_colors = colors[:]
    best_count = 1
    
    # Greedy: for each edge, try to assign a new color if possible
    for e in range(NUM_EDGES):
        # Try colors from 0 to NUM_COLORS-1, prefer unused ones
        assigned = False
        for c in range(NUM_COLORS):
            colors[e] = c
            # Check only cycles involving edge e
            valid = True
            for cyc_idx in EDGE_CYCLES[e]:
                if not check_cycle_valid(colors, CYCLES[cyc_idx]):
                    valid = False
                    break
            if valid:
                assigned = True
                break
        if not assigned:
            colors[e] = 0  # Fallback
        
        cur_count = count_distinct_colors(colors)
        if cur_count > best_count:
            best_count = cur_count
            best_colors = colors[:]
        
        if time.time() > deadline:
            break
    
    # Now use simulated annealing to improve
    if time.time() < deadline - 2:
        cur_colors = best_colors[:]
        cur_count = count_distinct_colors(cur_colors)
        best_sa = cur_colors[:]
        best_sa_count = cur_count
        
        T = 1.0
        T_min = 0.01
        cooling = 0.9995
        
        # Track color usage
        color_counts = [0] * NUM_COLORS
        for c in cur_colors:
            color_counts[c] += 1
        
        steps = 0
        max_steps = 500000
        
        while time.time() < deadline - 2 and steps < max_steps:
            steps += 1
            
            # Choose a move
            if random.random() < 0.5:
                # Recolor a single edge
                e = random.randint(0, NUM_EDGES - 1)
                old_c = cur_colors[e]
                new_c = random.randint(0, NUM_COLORS - 1)
                if new_c == old_c:
                    continue
                
                # Check if recoloring edge e maintains validity
                valid = True
                for cyc_idx in EDGE_CYCLES[e]:
                    cyc = CYCLES[cyc_idx]
                    # Temporarily set new color
                    cur_colors[e] = new_c
                    if not check_cycle_valid(cur_colors, cyc):
                        cur_colors[e] = old_c
                        valid = False
                        break
                if not valid:
                    continue
                
                # Calculate delta in distinct colors
                old_count = color_counts[old_c]
                new_count = color_counts[new_c]
                delta = 0
                if old_count == 1:
                    delta -= 1
                if new_count == 0:
                    delta += 1
                
                # Accept or reject
                if delta >= 0:
                    accept = True
                else:
                    import math
                    accept = random.random() < math.exp(delta / T)
                
                if accept:
                    color_counts[old_c] -= 1
                    color_counts[new_c] += 1
                    cur_count += delta
                    if cur_count > best_sa_count:
                        best_sa_count = cur_count
                        best_sa = cur_colors[:]
                else:
                    cur_colors[e] = old_c
            else:
                # Swap colors between two edges
                e1 = random.randint(0, NUM_EDGES - 1)
                e2 = random.randint(0, NUM_EDGES - 1)
                if e1 == e2:
                    continue
                c1 = cur_colors[e1]
                c2 = cur_colors[e2]
                if c1 == c2:
                    continue
                
                # Check validity after swap
                cur_colors[e1] = c2
                cur_colors[e2] = c1
                
                valid = True
                for e in (e1, e2):
                    for cyc_idx in EDGE_CYCLES[e]:
                        if not check_cycle_valid(cur_colors, CYCLES[cyc_idx]):
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    cur_colors[e1] = c1
                    cur_colors[e2] = c2
                    continue
                
                # Delta for swap: check if colors become unused or newly used
                delta = 0
                if color_counts[c1] == 1:
                    delta -= 1
                if color_counts[c2] == 1:
                    delta -= 1
                if color_counts[c1] == 1 and color_counts[c2] == 1:
                    # Both were unique, after swap both still exist (swapped)
                    delta = 0
                elif color_counts[c1] == 1:
                    delta = 0  # c1 still used at e2
                elif color_counts[c2] == 1:
                    delta = 0  # c2 still used at e1
                
                # Actually for swap, distinct count doesn't change unless...
                # Wait, swap doesn't change the set of colors used at all!
                # So delta is always 0 for swaps. They're for exploration.
                delta = 0
                
                # Always accept swaps for exploration (or use SA)
                accept = True
                if accept:
                    # Update color counts
                    # c1 was at e1, now at e2; c2 was at e2, now at e1
                    # Counts don't change
                    if cur_count > best_sa_count:
                        best_sa_count = cur_count
                        best_sa = cur_colors[:]
                else:
                    cur_colors[e1] = c1
                    cur_colors[e2] = c2
            
            T *= cooling
            if T < T_min:
                T = T_min
    
    # Return the best valid coloring found
    result = best_sa if best_sa_count >= best_count else best_colors
    # Ensure validity
    if not is_valid(result):
        # Fallback: try to fix
        for e in range(NUM_EDGES):
            for c in range(NUM_COLORS):
                result[e] = c
                if is_valid(result):
                    break
            else:
                result[e] = 0
        # Final validity check - if still invalid, return all zeros
        if not is_valid(result):
            result = [0] * NUM_EDGES
    
    return result

# Verify at module level that we have a valid baseline
if __name__ == "__main__":
    start = time.time()
    result = construct()
    elapsed = time.time() - start
    distinct = len(set(result))
    valid = is_valid(result)
    print(f"Time: {elapsed:.2f}s, Distinct colors: {distinct}, Valid: {valid}")
