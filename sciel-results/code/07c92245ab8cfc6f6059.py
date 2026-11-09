"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import sys

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

# EVOLVE-BLOCK-START

def _edge_idx(i, j):
    """Return the index of edge (i,j) in the lexicographic ordering."""
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)

def _build_edge_list():
    """Return list of (i, j) for all edges in lexicographic order."""
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
    return edges

def _build_cycles():
    """
    For every 4-vertex subset, there are 3 distinct 4-cycles.
    A 4-cycle on vertices {a,b,c,d} (a<b<c<d) can be:
      (a,b,c,d): edges (a,b),(b,c),(c,d),(d,a)
      (a,b,d,c): edges (a,b),(b,d),(d,c),(c,a)
      (a,c,b,d): edges (a,c),(c,b),(b,d),(d,a)
    Return list of tuples of 4 edge indices for each cycle.
    """
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Cycle 1: a-b-c-d-a
                    e1 = (_edge_idx(a, b), _edge_idx(b, c), _edge_idx(c, d), _edge_idx(d, a))
                    cycles.append(e1)
                    # Cycle 2: a-b-d-c-a
                    e2 = (_edge_idx(a, b), _edge_idx(b, d), _edge_idx(d, c), _edge_idx(c, a))
                    cycles.append(e2)
                    # Cycle 3: a-c-b-d-a
                    e3 = (_edge_idx(a, c), _edge_idx(c, b), _edge_idx(b, d), _edge_idx(d, a))
                    cycles.append(e3)
    return cycles

def _count_violations(colors, cycles):
    """Count number of 4-cycles with 4 distinct colors."""
    count = 0
    for (e1, e2, e3, e4) in cycles:
        c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
        if len({c1, c2, c3, c4}) == 4:
            count += 1
    return count

def _can_assign(colors, edge_i, color, cycles, edge_to_cycles):
    """Check if assigning 'color' to edge edge_i creates any new violation."""
    colors[edge_i] = color
    # Only need to check cycles containing edge_i
    for cyc in edge_to_cycles[edge_i]:
        (e1, e2, e3, e4) = cyc
        c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
        if len({c1, c2, c3, c4}) == 4:
            colors[edge_i] = -1  # restore
            return False
    colors[edge_i] = -1  # restore
    return True

def _greedy_randomized(deadline):
    """Greedy randomized construction."""
    edges = _build_edge_list()
    cycles = _build_cycles()
    
    # Build edge_to_cycles mapping
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for cyc in cycles:
        for e in cyc:
            edge_to_cycles[e].append(cyc)
    
    colors = [-1] * NUM_EDGES
    used_colors = set()
    
    # Shuffle edges for randomization
    edge_order = list(range(NUM_EDGES))
    random.shuffle(edge_order)
    
    for idx in edge_order:
        if time.time() > deadline:
            break
        # Try to assign a new color first (to maximize distinct colors)
        # First try unused colors
        available_new = [c for c in range(120) if c not in used_colors and _can_assign(colors, idx, c, cycles, edge_to_cycles)]
        if available_new:
            colors[idx] = available_new[0]
            used_colors.add(available_new[0])
        else:
            # Try used colors
            available_old = [c for c in sorted(used_colors) if _can_assign(colors, idx, c, cycles, edge_to_cycles)]
            if available_old:
                # Prefer colors that are most constrained (appear in most cycles with this edge)
                colors[idx] = available_old[0]
            else:
                # Should not happen; assign 0 as fallback
                colors[idx] = 0
                used_colors.add(0)
    
    # Fill any remaining -1
    for i in range(NUM_EDGES):
        if colors[i] == -1:
            colors[i] = 0
            used_colors.add(0)
    
    return colors, len(used_colors)

def _local_search(colors, cycles, edge_to_cycles, deadline, max_iters=50000):
    """Local search: try to change colors to reduce violations, then increase distinct colors."""
    colors = list(colors)
    violations = _count_violations(colors, cycles)
    used_colors = set(colors)
    best_violations = violations
    best_distinct = len(used_colors)
    
    iters = 0
    while time.time() < deadline and iters < max_iters:
        iters += 1
        if iters % 1000 == 0:
            violations = _count_violations(colors, cycles)
            used_colors = set(colors)
            if violations < best_violations or (violations == best_violations and len(used_colors) > best_distinct):
                best_violations = violations
                best_distinct = len(used_colors)
        
        if violations == 0:
            # Try to increase distinct colors
            # Pick a random edge and try to change it to a new color
            for _ in range(100):
                if time.time() > deadline:
                    break
                idx = random.randint(0, NUM_EDGES - 1)
                # Try a random new color
                new_color = random.randint(0, 119)
                if new_color == colors[idx]:
                    continue
                old_color = colors[idx]
                colors[idx] = new_color
                new_viol = _count_violations(colors, cycles)
                if new_viol <= violations:
                    violations = new_viol
                    used_colors = set(colors)
                    if len(used_colors) > best_distinct:
                        best_distinct = len(used_colors)
                else:
                    colors[idx] = old_color
            continue
        
        # Try to fix violations
        # Find a violated cycle
        violated = None
        for cyc in cycles:
            (e1, e2, e3, e4) = cyc
            c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
            if len({c1, c2, c3, c4}) == 4:
                violated = cyc
                break
        
        if violated is None:
            violations = 0
            continue
        
        # Try to change one edge in the violated cycle to match another
        (e1, e2, e3, e4) = violated
        c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
        
        # Try changing each edge to match one of the others
        attempts = [
            (e1, c2), (e1, c3), (e1, c4),
            (e2, c1), (e2, c3), (e2, c4),
            (e3, c1), (e3, c2), (e3, c4),
            (e4, c1), (e4, c2), (e4, c3),
        ]
        random.shuffle(attempts)
        
        found_improvement = False
        for edge_idx, target_color in attempts:
            if colors[edge_idx] == target_color:
                continue
            old_color = colors[edge_idx]
            colors[edge_idx] = target_color
            new_viol = _count_violations(colors, cycles)
            if new_viol < violations:
                violations = new_viol
                found_improvement = True
                break
            else:
                colors[edge_idx] = old_color
        
        if not found_improvement:
            # Try random perturbation
            idx = random.randint(0, NUM_EDGES - 1)
            old_color = colors[idx]
            new_color = random.randint(0, 119)
            colors[idx] = new_color
            new_viol = _count_violations(colors, cycles)
            if new_viol > violations + 5:
                colors[idx] = old_color
    
    # Final count
    used_colors = set(colors)
    return colors, len(used_colors)

def construct():
    deadline = time.time() + 75  # Leave 15s safety margin
    
    best_colors = None
    best_distinct = 0
    
    edges = _build_edge_list()
    cycles = _build_cycles()
    
    # Build edge_to_cycles mapping
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for cyc in cycles:
        for e in cyc:
            edge_to_cycles[e].append(cyc)
    
    # Multiple restarts with different random seeds
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        random.seed(attempts * 12345 + 678)
        
        colors, distinct = _greedy_randomized(deadline - 5)
        
        if distinct > best_distinct:
            best_distinct = distinct
            best_colors = list(colors)
        
        # Local search to improve
        if time.time() < deadline - 3:
            colors2, distinct2 = _local_search(colors, cycles, edge_to_cycles, deadline - 2, max_iters=30000)
            if distinct2 > best_distinct:
                best_distinct = distinct2
                best_colors = list(colors2)
        
        # If we've used many colors, try more targeted improvements
        if best_distinct >= 30 and time.time() < deadline - 5:
            # Try to add more colors via local search with focus on new colors
            colors3 = list(best_colors)
            used = set(colors3)
            unused = [c for c in range(120) if c not in used]
            random.shuffle(unused)
            
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                for new_c in unused[:]:
                    if time.time() > deadline - 2:
                        break
                    for idx in range(NUM_EDGES):
                        if colors3[idx] == new_c:
                            continue
                        old_c = colors3[idx]
                        colors3[idx] = new_c
                        if _count_violations(colors3, cycles) == 0:
                            used.add(new_c)
                            unused.remove(new_c)
                            improved = True
                            break
                        else:
                            colors3[idx] = old_c
                    if improved:
                        break
            
            final_distinct = len(set(colors3))
            if final_distinct > best_distinct:
                best_distinct = final_distinct
                best_colors = list(colors3)
        
        if best_distinct >= 50:
            # Diminishing returns, reduce restarts
            if attempts > 10:
                break
        else:
            if attempts > 20:
                break
    
    if best_colors is None:
        best_colors = [0] * NUM_EDGES
    
    return best_colors

# EVOLVE-BLOCK-END
