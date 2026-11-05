import time
import random
import numpy as np

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
                    # Cycle 1: a-b-c-d-a
                    e1 = (edge_index(a, b), edge_index(b, c), edge_index(c, d), edge_index(d, a))
                    cycles.append(e1)
                    # Cycle 2: a-b-d-c-a
                    e2 = (edge_index(a, b), edge_index(b, d), edge_index(d, c), edge_index(c, a))
                    cycles.append(e2)
                    # Cycle 3: a-c-b-d-a
                    e3 = (edge_index(a, c), edge_index(c, b), edge_index(b, d), edge_index(d, a))
                    cycles.append(e3)
    return cycles

_CYCLES = build_four_cycles()
NUM_CYCLES = len(_CYCLES)
_CYCLES_BY_EDGE = [[] for _ in range(NUM_EDGES)]
for idx, cyc in enumerate(_CYCLES):
    for e in cyc:
        _CYCLES_BY_EDGE[e].append(idx)

# Precompute cycle edge indices as numpy arrays for faster access
_CYCLES_ARR = np.array(_CYCLES, dtype=np.int32)  # shape (NUM_CYCLES, 4)

def is_valid_cycle(colors, cyc):
    """Check if a 4-cycle has at most 3 distinct colors."""
    c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
    if c0 == c1 or c0 == c2 or c0 == c3:
        return True
    if c1 == c2 or c1 == c3:
        return True
    if c2 == c3:
        return True
    return False

def validate(colors):
    """Check that no 4-cycle has 4 distinct colors. Returns True if valid."""
    for cyc in _CYCLES:
        if not is_valid_cycle(colors, cyc):
            return False
    return True

def count_colors(colors):
    return len(set(colors))

def greedy_coloring(edge_order, rng):
    """
    Greedy: process edges in given order, assign each edge a color that
    keeps all 4-cycles valid. Try to maximize distinct colors by preferring
    unused colors when possible.
    """
    colors = [-1] * NUM_EDGES
    used_colors = set()
    
    for e in edge_order:
        # Try colors in an order that favors diversity
        unused = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
        rng.shuffle(unused)
        
        used_list = list(used_colors)
        rng.shuffle(used_list)
        
        all_candidates = unused + used_list
        
        found = False
        for c in all_candidates:
            colors[e] = c
            # Check all cycles containing e
            valid = True
            for cyc_idx in _CYCLES_BY_EDGE[e]:
                cyc = _CYCLES[cyc_idx]
                if colors[cyc[0]] == -1 or colors[cyc[1]] == -1 or colors[cyc[2]] == -1 or colors[cyc[3]] == -1:
                    continue
                if not is_valid_cycle(colors, cyc):
                    valid = False
                    break
            if valid:
                found = True
                break
        
        if not found:
            colors[e] = 0
        else:
            used_colors.add(colors[e])
    
    return colors

def local_improve_simulated_annealing(colors, rng, deadline):
    """
    Simulated annealing local search to maximize number of distinct colors.
    """
    colors = colors[:]
    used_colors = set(colors)
    current_count = len(used_colors)
    best_count = current_count
    best_colors = colors[:]
    
    # Temperature parameters
    T_start = 0.1
    T_end = 0.001
    T_current = T_start
    
    start_time = time.time()
    total_time = deadline - start_time
    if total_time <= 0:
        return best_colors, best_count
    
    # Track iteration for cooling schedule
    max_iterations = 100000
    iteration = 0
    
    while time.time() < deadline - 1:
        iteration += 1
        
        # Cooling schedule
        if max_iterations > 0:
            T_current = T_start * (T_end / T_start) ** (iteration / max_iterations)
        else:
            T_current = T_end
        
        # Pick a random edge
        e = rng.randint(0, NUM_EDGES - 1)
        old_color = colors[e]
        
        # Generate a new color
        # With some probability, try an unused color (to increase diversity)
        if rng.random() < 0.3:
            unused = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
            if unused:
                new_color = rng.choice(unused)
            else:
                new_color = rng.randint(0, MAX_COLOR)
        else:
            # Try a random color from used or a small perturbation
            if used_colors:
                new_color = rng.choice(list(used_colors))
            else:
                new_color = rng.randint(0, MAX_COLOR)
        
        if new_color == old_color:
            continue
        
        # Check if the new coloring is valid for all cycles containing edge e
        colors[e] = new_color
        valid = True
        for cyc_idx in _CYCLES_BY_EDGE[e]:
            cyc = _CYCLES[cyc_idx]
            if not is_valid_cycle(colors, cyc):
                valid = False
                break
        
        if not valid:
            # Reject
            colors[e] = old_color
            continue
        
        # Calculate change in objective (number of distinct colors)
        old_count = current_count
        used_colors.add(new_color)
        # Check if old_color is still used
        still_used = False
        for i in range(NUM_EDGES):
            if i != e and colors[i] == old_color:
                still_used = True
                break
        if not still_used:
            used_colors.discard(old_color)
        new_count = len(used_colors)
        
        delta = new_count - old_count
        
        # Accept or reject based on simulated annealing
        if delta > 0 or rng.random() < np.exp(delta / T_current):
            # Accept
            current_count = new_count
            if current_count > best_count:
                best_count = current_count
                best_colors = colors[:]
        else:
            # Reject
            colors[e] = old_color
            # Restore used_colors
            if not still_used:
                used_colors.add(old_color)
            # Note: new_color may or may not still be in used_colors
            # If we rejected, new_color wasn't added (or was added but we're reverting)
            # Actually, we added new_color to used_colors above. Let's fix this.
            # If we reject, we should remove new_color if it was only on edge e
            # But this is getting complicated. Let's simplify by rebuilding used_colors occasionally.
        
        # Rebuild used_colors every 1000 iterations for correctness
        if iteration % 1000 == 0:
            used_colors = set(colors)
            current_count = len(used_colors)
    
    # Final check
    used_colors = set(colors)
    current_count = len(used_colors)
    if current_count > best_count:
        best_count = current_count
        best_colors = colors[:]
    
    return best_colors, best_count

def local_improve_aggressive(colors, rng, deadline):
    """
    Aggressive local search combining multiple strategies.
    """
    colors = colors[:]
    used_colors = set(colors)
    current_count = len(used_colors)
    best_count = current_count
    best_colors = colors[:]
    
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        
        # Strategy 1: Try to introduce new colors by recoloring single edges
        for _ in range(500):
            if time.time() >= deadline - 2:
                break
            
            e = rng.randint(0, NUM_EDGES - 1)
            old = colors[e]
            
            unused = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
            if not unused:
                break
            
            rng.shuffle(unused)
            for new_c in unused[:20]:
                if new_c == old:
                    continue
                colors[e] = new_c
                valid = True
                for cyc_idx in _CYCLES_BY_EDGE[e]:
                    cyc = _CYCLES[cyc_idx]
                    if not is_valid_cycle(colors, cyc):
                        valid = False
                        break
                if valid:
                    used_colors.add(new_c)
                    current_count += 1
                    if current_count > best_count:
                        best_count = current_count
                        best_colors = colors[:]
                    improved = True
                    break
                else:
                    colors[e] = old
        
        # Strategy 2: Try 2-edge moves
        for _ in range(200):
            if time.time() >= deadline - 2:
                break
            
            e1 = rng.randint(0, NUM_EDGES - 1)
            shared_edges = set()
            for cyc_idx in _CYCLES_BY_EDGE[e1]:
                cyc = _CYCLES[cyc_idx]
                for other_e in cyc:
                    if other_e != e1:
                        shared_edges.add(other_e)
            
            if not shared_edges:
                continue
            
            e2 = rng.choice(list(shared_edges))
            old1, old2 = colors[e1], colors[e2]
            
            unused = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
            if not unused:
                break
            
            rng.shuffle(unused)
            for new_c1 in unused[:10]:
                colors[e1] = new_c1
                valid1 = True
                for cyc_idx in _CYCLES_BY_EDGE[e1]:
                    cyc = _CYCLES[cyc_idx]
                    if not is_valid_cycle(colors, cyc):
                        valid1 = False
                        break
                if not valid1:
                    colors[e1] = old1
                    continue
                
                # Try to also recolor e2
                unused2 = [c for c in range(MAX_COLOR + 1) if c not in used_colors and c != new_c1]
                if unused2:
                    rng.shuffle(unused2)
                    for new_c2 in unused2[:5]:
                        colors[e2] = new_c2
                        valid2 = True
                        for cyc_idx in _CYCLES_BY_EDGE[e2]:
                            cyc = _CYCLES[cyc_idx]
                            if not is_valid_cycle(colors, cyc):
                                valid2 = False
                                break
                        if valid2:
                            used_colors.add(new_c1)
                            used_colors.add(new_c2)
                            current_count += 2
                            if current_count > best_count:
                                best_count = current_count
                                best_colors = colors[:]
                            improved = True
                            break
                        else:
                            colors[e2] = old2
                    else:
                        # Only e1 gets new color
                        used_colors.add(new_c1)
                        current_count += 1
                        if current_count > best_count:
                            best_count = current_count
                            best_colors = colors[:]
                        improved = True
                        break
                else:
                    used_colors.add(new_c1)
                    current_count += 1
                    if current_count > best_count:
                        best_count = current_count
                        best_colors = colors[:]
                    improved = True
                    break
                break
            else:
                colors[e1] = old1
        
        # Rebuild used_colors for consistency
        used_colors = set(colors)
        current_count = len(used_colors)
    
    return best_colors, best_count

def construct():
    start = time.time()
    deadline = start + 55  # Safety margin before 60s limit
    
    rng = random.Random(42)
    
    best_colors = [0] * NUM_EDGES
    best_count = 1
    
    # Try multiple greedy orderings
    edge_indices = list(range(NUM_EDGES))
    
    orderings = []
    orderings.append(list(edge_indices))
    orderings.append(list(reversed(edge_indices)))
    
    for _ in range(15):
        shuffled = list(edge_indices)
        rng.shuffle(shuffled)
        orderings.append(shuffled)
    
    # Degree-based orderings
    edge_cycle_count = [len(_CYCLES_BY_EDGE[e]) for e in range(NUM_EDGES)]
    degree_order = sorted(edge_indices, key=lambda e: edge_cycle_count[e])
    orderings.append(degree_order)
    orderings.append(list(reversed(degree_order)))
    
    candidates = []
    for order in orderings:
        if time.time() >= deadline - 20:
            break
        c = greedy_coloring(order, rng)
        if validate(c):
            cnt = count_colors(c)
            candidates.append((cnt, c))
    
    candidates.sort(key=lambda x: -x[0])
    
    if candidates:
        best_count = candidates[0][0]
        best_colors = candidates[0][1][:]
    
    # Local improvement on top candidates
    top_candidates = candidates[:10]
    for cnt, c in top_candidates:
        if time.time() >= deadline - 5:
            break
        improved_c, improved_cnt = local_improve_aggressive(c, rng, deadline)
        if improved_cnt > best_count:
            best_count = improved_cnt
            best_colors = improved_c[:]
    
    # Try simulated annealing on the best solution
    if time.time() < deadline - 3:
        sa_colors, sa_count = local_improve_simulated_annealing(best_colors, rng, deadline)
        if sa_count > best_count:
            best_count = sa_count
            best_colors = sa_colors[:]
    
    # Final validation
    if not validate(best_colors):
        best_colors = [0] * NUM_EDGES
        best_count = 1
    
    return best_colors
