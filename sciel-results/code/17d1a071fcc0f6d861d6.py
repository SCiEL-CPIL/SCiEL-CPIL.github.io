import time
import random
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def build_four_cycles():
    """Precompute all 4-cycles as tuples of edge indices."""
    cycles = []
    for a in range(N):
        for b in range(a+1, N):
            for c in range(b+1, N):
                for d in range(c+1, N):
                    # Cycle 1: a-b-c-d-a
                    e1 = (edge_index(a,b), edge_index(b,c), edge_index(c,d), edge_index(d,a))
                    cycles.append(e1)
                    # Cycle 2: a-b-d-c-a
                    e2 = (edge_index(a,b), edge_index(b,d), edge_index(d,c), edge_index(c,a))
                    cycles.append(e2)
                    # Cycle 3: a-c-b-d-a
                    e3 = (edge_index(a,c), edge_index(c,b), edge_index(b,d), edge_index(d,a))
                    cycles.append(e3)
    return cycles

def construct():
    deadline = time.time() + 55  # safety margin before 60s limit
    
    # Precompute cycles and index by edge
    cycles = build_four_cycles()
    cycles_by_edge = [[] for _ in range(NUM_EDGES)]
    for cycle in cycles:
        for e in cycle:
            cycles_by_edge[e].append(cycle)
    
    # Greedy construction: assign colors in order, preferring new colors
    # Order edges by some heuristic (e.g., degree of involvement in cycles)
    edge_order = list(range(NUM_EDGES))
    
    # Try multiple strategies
    best_colors = None
    best_count = 0
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Strategy 1: Greedy with random edge ordering
        colors = [None] * NUM_EDGES
        used_colors = set()
        
        # Randomize edge order for diversity
        edge_order = list(range(NUM_EDGES))
        rng.shuffle(edge_order)
        
        success = True
        for e in edge_order:
            # Try colors from high to low (prefer new colors)
            # First try a completely new color
            found = False
            # Try new colors first (from high range)
            for c in range(NUM_EDGES - 1, -1, -1):
                if c in used_colors:
                    continue
                # Check if assigning color c to edge e is valid
                valid = True
                for cycle in cycles_by_edge[e]:
                    # Get colors of other 3 edges in this cycle
                    other_colors = []
                    for oe in cycle:
                        if oe == e:
                            continue
                        if colors[oe] is None:
                            valid = False
                            break
                        other_colors.append(colors[oe])
                    if not valid:
                        break
                    # Check if {c} ∪ other_colors has 4 distinct values
                    all_colors = set(other_colors)
                    all_colors.add(c)
                    if len(all_colors) == 4:
                        valid = False
                        break
                if valid:
                    colors[e] = c
                    used_colors.add(c)
                    found = True
                    break
            
            if not found:
                # Try existing colors
                for c in sorted(used_colors, key=lambda x: -x):
                    valid = True
                    for cycle in cycles_by_edge[e]:
                        other_colors = []
                        for oe in cycle:
                            if oe == e:
                                continue
                            if colors[oe] is None:
                                valid = False
                                break
                            other_colors.append(colors[oe])
                        if not valid:
                            break
                        all_colors = set(other_colors)
                        all_colors.add(c)
                        if len(all_colors) == 4:
                            valid = False
                            break
                    if valid:
                        colors[e] = c
                        found = True
                        break
            
            if not found:
                success = False
                break
        
        if success:
            cur_count = len(set(colors))
            if cur_count > best_count:
                best_count = cur_count
                best_colors = colors[:]
        
        # Strategy 2: Local search improvement
        if best_colors is not None and time.time() < deadline:
            colors = best_colors[:]
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for _ in range(2000):
                    if time.time() > deadline:
                        break
                    e = rng.randint(0, NUM_EDGES - 1)
                    old_c = colors[e]
                    # Try to change to a new color not currently used
                    used = set(colors)
                    for c in range(NUM_EDGES - 1, -1, -1):
                        if c in used:
                            continue
                        valid = True
                        for cycle in cycles_by_edge[e]:
                            other_colors = [colors[oe] for oe in cycle if oe != e]
                            all_colors = set(other_colors)
                            all_colors.add(c)
                            if len(all_colors) == 4:
                                valid = False
                                break
                        if valid:
                            colors[e] = c
                            improved = True
                            break
            
            cur_count = len(set(colors))
            if cur_count > best_count:
                best_count = cur_count
                best_colors = colors[:]
    
    if best_colors is None:
        # Fallback: all zeros (always valid)
        best_colors = [0] * NUM_EDGES
    
    return best_colors
