import time
import random
import math

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119

def edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def build_four_cycles():
    """Precompute all 4-cycles as tuples of 4 edge indices."""
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    e1 = (edge_index(a, b), edge_index(b, c), edge_index(c, d), edge_index(d, a))
                    cycles.append(e1)
                    e2 = (edge_index(a, b), edge_index(b, d), edge_index(d, c), edge_index(c, a))
                    cycles.append(e2)
                    e3 = (edge_index(a, c), edge_index(c, b), edge_index(b, d), edge_index(d, a))
                    cycles.append(e3)
    return cycles

_CYCLES = build_four_cycles()
_CYCLES_BY_EDGE = [[] for _ in range(NUM_EDGES)]
for cyc in _CYCLES:
    for e in cyc:
        _CYCLES_BY_EDGE[e].append(cyc)

# Precompute: for each edge, list of (cycle, position) so we can quickly check validity
# Also precompute the other 3 edges in each cycle for each edge
_CYCLE_OTHERS = [[] for _ in range(NUM_EDGES)]
for e in range(NUM_EDGES):
    for cyc in _CYCLES_BY_EDGE[e]:
        others = tuple(oe for oe in cyc if oe != e)
        _CYCLE_OTHERS[e].append(others)

def is_valid_assign(colors, e, nc):
    """Check if assigning color nc to edge e is valid (all cycles containing e have <=3 distinct colors)."""
    for others in _CYCLE_OTHERS[e]:
        c0, c1, c2 = colors[others[0]], colors[others[1]], colors[others[2]]
        # Check if {c0, c1, c2, nc} has 4 distinct values
        # Fast check: if any of c0,c1,c2 equals nc, we're fine (at most 3 distinct)
        if nc == c0 or nc == c1 or nc == c2:
            continue
        # Otherwise need c0,c1,c2 to have at most 2 distinct values
        if c0 == c1 or c0 == c2 or c1 == c2:
            continue
        # All four are distinct -> invalid
        return False
    return True

def count_distinct(colors):
    return len(set(colors))

def greedy_init(rng, edge_order):
    """Greedy initialization assigning colors to maximize distinct colors."""
    colors = [0] * NUM_EDGES
    used_colors = set([0])
    
    for e in edge_order:
        best_c = -1
        # Try unused colors first (shuffled)
        unused = [c for c in range(1, MAX_COLOR + 1) if c not in used_colors]
        rng.shuffle(unused)
        for c in unused:
            if is_valid_assign(colors, e, c):
                best_c = c
                break
        if best_c == -1:
            # Try used colors
            for c in list(used_colors):
                if c != colors[e] and is_valid_assign(colors, e, c):
                    best_c = c
                    break
        if best_c == -1:
            best_c = colors[e]  # keep original
        colors[e] = best_c
        used_colors.add(best_c)
    
    return colors

def simulated_annealing(colors, rng, deadline):
    """
    Simulated annealing with:
    - Single edge recoloring (to new or existing color)
    - Color swaps between two edges that share a cycle
    Objective: maximize number of distinct colors
    """
    colors = colors[:]
    best_colors = colors[:]
    best_count = count_distinct(colors)
    cur_count = best_count
    
    used_colors = set(colors)
    
    T = 1.0  # temperature
    T_min = 0.01
    cooling = 0.995
    
    edge_order = sorted(range(NUM_EDGES), key=lambda e: len(_CYCLES_BY_EDGE[e]), reverse=True)
    
    # Precompute shared edges for swap moves
    shared_edges = [set() for _ in range(NUM_EDGES)]
    for e in range(NUM_EDGES):
        for cyc in _CYCLES_BY_EDGE[e]:
            for oe in cyc:
                if oe != e:
                    shared_edges[e].add(oe)
    
    steps_since_improve = 0
    max_steps_no_improve = 200
    
    while time.time() < deadline and T > T_min:
        improved = False
        
        # Phase 1: Try to add new colors via single edge recoloring
        for _ in range(50):
            if time.time() >= deadline:
                break
            e = rng.randint(0, NUM_EDGES - 1)
            old_c = colors[e]
            
            # Try unused colors
            unused = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
            if unused:
                rng.shuffle(unused)
                for nc in unused[:10]:
                    if nc == old_c:
                        continue
                    if is_valid_assign(colors, e, nc):
                        colors[e] = nc
                        used_colors.add(nc)
                        cur_count += 1
                        improved = True
                        if cur_count > best_count:
                            best_count = cur_count
                            best_colors = colors[:]
                        break
            if improved:
                continue
            
            # Try existing colors (might enable future improvements)
            existing = list(used_colors)
            rng.shuffle(existing)
            for nc in existing[:5]:
                if nc == old_c:
                    continue
                if is_valid_assign(colors, e, nc):
                    colors[e] = nc
                    # cur_count unchanged
                    improved = True
                    break
        
        # Phase 2: Try color swaps between edges that share a cycle
        for _ in range(30):
            if time.time() >= deadline:
                break
            e1 = rng.randint(0, NUM_EDGES - 1)
            if not shared_edges[e1]:
                continue
            e2 = rng.choice(list(shared_edges[e1]))
            
            c1, c2 = colors[e1], colors[e2]
            if c1 == c2:
                continue
            
            # Try swap: e1 gets c2, e2 gets c1
            # Check validity
            valid = True
            # Check cycles for e1 with new color c2
            for others in _CYCLE_OTHERS[e1]:
                oc = [colors[oe] for oe in others]
                # e2 might be in others; if so, its color would be c1
                oc_adj = []
                for x in oc:
                    # We need to check if any other edge in this cycle is e2
                    # Actually we just check the actual colors after swap
                    pass
                # Simpler: temporarily assign and check
                old_e1, old_e2 = colors[e1], colors[e2]
                colors[e1] = c2
                colors[e2] = c1
                for cyc in _CYCLES_BY_EDGE[e1]:
                    vals = [colors[ce] for ce in cyc]
                    if len(set(vals)) == 4:
                        valid = False
                        break
                if valid:
                    for cyc in _CYCLES_BY_EDGE[e2]:
                        vals = [colors[ce] for ce in cyc]
                        if len(set(vals)) == 4:
                            valid = False
                            break
                colors[e1] = old_e1
                colors[e2] = old_e2
                if not valid:
                    break
            
            if valid:
                # The swap doesn't change the number of distinct colors,
                # but it might enable future improvements
                # Accept with some probability based on temperature
                if rng.random() < math.exp(-0.1 / T):  # small penalty for neutral moves
                    colors[e1], colors[e2] = c2, c1
                    improved = True
        
        # Phase 3: Aggressive recoloring of singleton-color edges
        if not improved:
            color_counts = {}
            for c in colors:
                color_counts[c] = color_counts.get(c, 0) + 1
            
            singleton_edges = [e for e in range(NUM_EDGES) if color_counts[colors[e]] == 1]
            rng.shuffle(singleton_edges)
            
            for e in singleton_edges[:10]:
                if time.time() >= deadline:
                    break
                old_c = colors[e]
                unused = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
                if unused:
                    rng.shuffle(unused)
                    for nc in unused[:15]:
                        if is_valid_assign(colors, e, nc):
                            colors[e] = nc
                            used_colors.add(nc)
                            cur_count += 1
                            improved = True
                            if cur_count > best_count:
                                best_count = cur_count
                                best_colors = colors[:]
                            break
        
        if improved:
            steps_since_improve = 0
        else:
            steps_since_improve += 1
        
        # Perturbation if stuck
        if steps_since_improve >= max_steps_no_improve:
            num_perturb = rng.randint(5, 15)
            for _ in range(num_perturb):
                if time.time() >= deadline:
                    break
                e = rng.randint(0, NUM_EDGES - 1)
                nc = rng.randint(0, MAX_COLOR)
                if nc == colors[e]:
                    continue
                if is_valid_assign(colors, e, nc):
                    colors[e] = nc
                    # Update used_colors
                    if colors.count(nc) == 1:
                        used_colors.add(nc)
                    # Remove old color if no longer used
                    old_c = nc  # this is wrong, let me fix
            # Rebuild used_colors
            used_colors = set(colors)
            cur_count = len(used_colors)
            if cur_count > best_count:
                best_count = cur_count
                best_colors = colors[:]
            steps_since_improve = 0
        
        T *= cooling
    
    return best_colors, best_count

def construct():
    deadline = time.time() + 55
    
    rng = random.Random(42)
    
    # Edge ordering by constraint tightness
    edge_order = sorted(range(NUM_EDGES), key=lambda e: len(_CYCLES_BY_EDGE[e]), reverse=True)
    
    best_colors = None
    best_count = 0
    
    # Multiple restarts with different strategies
    for restart in range(8):
        if time.time() >= deadline - 5:
            break
        
        if restart == 0:
            # Greedy init
            colors = greedy_init(rng, edge_order)
        elif restart == 1:
            # Random init with constraint checking
            colors = [0] * NUM_EDGES
            used = set([0])
            for e in edge_order:
                c = 0
                for _ in range(20):
                    c = rng.randint(0, MAX_COLOR)
                    if is_valid_assign(colors, e, c):
                        break
                colors[e] = c
                used.add(c)
        else:
            # Start from best found so far with perturbation
            if best_colors is None:
                colors = greedy_init(rng, edge_order)
            else:
                colors = best_colors[:]
                # Perturb
                for _ in range(rng.randint(5, 15)):
                    e = rng.randint(0, NUM_EDGES - 1)
                    nc = rng.randint(0, MAX_COLOR)
                    if is_valid_assign(colors, e, nc):
                        colors[e] = nc
        
        # Run simulated annealing
        anneal_deadline = min(deadline - 2, time.time() + 12)
        result, count = simulated_annealing(colors, rng, anneal_deadline)
        
        if count > best_count:
            best_count = count
            best_colors = result
    
    if best_colors is None:
        best_colors = greedy_init(rng, edge_order)
    
    return best_colors
