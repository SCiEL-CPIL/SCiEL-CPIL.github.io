import time
import random
import sys

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLOR = 119
DEADLINE = 85.0  # Safety margin below 90s

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    # Precompute all 4-cycles
    # For vertices a<b<c<d, the three 4-cycles are:
    # Cycle 1: (a,b),(b,c),(c,d),(d,a)
    # Cycle 2: (a,b),(b,d),(c,d),(a,c)
    # Cycle 3: (a,c),(b,c),(b,d),(a,d)
    
    four_cycles = []
    for a in range(N):
        for b in range(a+1, N):
            for c in range(b+1, N):
                for d in range(c+1, N):
                    four_cycles.append([(a,b), (b,c), (c,d), (a,d)])
                    four_cycles.append([(a,b), (b,d), (c,d), (a,c)])
                    four_cycles.append([(a,c), (b,c), (b,d), (a,d)])
    
    num_cycles = len(four_cycles)
    
    # Edge index mapping
    all_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
    edge_to_idx = {e: idx for idx, e in enumerate(all_edges)}
    
    # For each cycle, store edge indices
    cycle_edges = []
    for cycle in four_cycles:
        cycle_edges.append([edge_to_idx[e] for e in cycle])
    
    # For each edge, store which cycles contain it and the positions
    edge_cycle_info = [[] for _ in range(NUM_EDGES)]
    for cidx, ce in enumerate(cycle_edges):
        for pos, eidx in enumerate(ce):
            edge_cycle_info[eidx].append((cidx, pos))
    
    # For fast checking: for each edge, precompute the other 3 edge indices in each cycle
    edge_other_edges = [[] for _ in range(NUM_EDGES)]
    for eidx in range(NUM_EDGES):
        for cidx, pos in edge_cycle_info[eidx]:
            ce = cycle_edges[cidx]
            others = [ce[(pos + k) % 4] for k in range(1, 4)]
            edge_other_edges[eidx].append(others)
    
    def count_distinct(coloring):
        return len(set(coloring))
    
    def is_valid_full(coloring):
        for ce in cycle_edges:
            colors = [coloring[e] for e in ce]
            if len(set(colors)) == 4:
                return False
        return True
    
    def check_edge_valid(eidx, new_color, coloring):
        """Check if setting edge eidx to new_color keeps all affected cycles valid."""
        for others in edge_other_edges[eidx]:
            c0 = new_color
            c1 = coloring[others[0]]
            c2 = coloring[others[1]]
            c3 = coloring[others[2]]
            if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                return False
        return True
    
    def local_search(coloring, timeout=30.0, aggressive=False):
        """Local search to maximize distinct colors while maintaining validity."""
        best = list(coloring)
        best_count = count_distinct(best)
        current = list(coloring)
        current_count = best_count
        
        for _ in range(1000000):
            if elapsed() > timeout:
                break
            
            # Pick a random edge
            eidx = random.randint(0, NUM_EDGES - 1)
            old_color = current[eidx]
            
            # Determine candidate colors
            used = set(current)
            
            if aggressive:
                # Try unused colors first (to increase diversity)
                candidates = [c for c in range(MAX_COLOR + 1) if c not in used]
                random.shuffle(candidates)
                # Also try some random used colors for flexibility
                for _ in range(10):
                    candidates.append(random.randint(0, MAX_COLOR))
            else:
                # Mix of unused and random
                unused = [c for c in range(MAX_COLOR + 1) if c not in used]
                random.shuffle(unused)
                candidates = unused[:20]
                for _ in range(10):
                    candidates.append(random.randint(0, MAX_COLOR))
            
            found_improvement = False
            for new_color in candidates:
                if new_color == old_color:
                    continue
                
                if not check_edge_valid(eidx, new_color, current):
                    continue
                
                # Apply and check if it improves
                current[eidx] = new_color
                
                # Compute new distinct count
                # If old_color was unique and we're changing it, count might decrease
                # If new_color is new, count increases
                # Quick check:
                if new_color not in used and old_color in used:
                    # Potentially +1 if old_color appears elsewhere, or same if it was unique
                    # Just recompute for accuracy in this small case
                    new_count = count_distinct(current)
                elif new_color in used:
                    # Color was already used, count stays same or decreases if old was unique
                    new_count = count_distinct(current)
                else:
                    new_count = count_distinct(current)
                
                if new_count > current_count:
                    current_count = new_count
                    if new_count > best_count:
                        best = list(current)
                        best_count = new_count
                    found_improvement = True
                    break
                elif new_count == current_count and random.random() < 0.3:
                    # Accept neutral moves for exploration
                    found_improvement = True
                    break
                else:
                    # Revert
                    current[eidx] = old_color
            
            if not found_improvement and random.random() < 0.1:
                # Revert to best
                current = list(best)
                current_count = best_count
        
        return best, best_count
    
    def two_opt_search(coloring, timeout=20.0):
        """Try pairs of edge recolorings to escape local optima."""
        best = list(coloring)
        best_count = count_distinct(best)
        current = list(coloring)
        current_count = best_count
        
        for _ in range(500000):
            if elapsed() > timeout:
                break
            
            # Pick two random edges
            e1 = random.randint(0, NUM_EDGES - 1)
            e2 = random.randint(0, NUM_EDGES - 1)
            if e1 == e2:
                continue
            
            old1 = current[e1]
            old2 = current[e2]
            
            # Try random new colors for both
            c1 = random.randint(0, MAX_COLOR)
            c2 = random.randint(0, MAX_COLOR)
            
            if c1 == old1 and c2 == old2:
                continue
            
            # Check validity after both changes
            current[e1] = c1
            current[e2] = c2
            
            valid = True
            # Check all cycles containing e1 or e2
            affected_cycles = set()
            for cidx, pos in edge_cycle_info[e1]:
                affected_cycles.add(cidx)
            for cidx, pos in edge_cycle_info[e2]:
                affected_cycles.add(cidx)
            
            for cidx in affected_cycles:
                ce = cycle_edges[cidx]
                colors = [current[e] for e in ce]
                if len(set(colors)) == 4:
                    valid = False
                    break
            
            if valid:
                new_count = count_distinct(current)
                if new_count > current_count:
                    current_count = new_count
                    if new_count > best_count:
                        best = list(current)
                        best_count = new_count
                elif new_count == current_count and random.random() < 0.2:
                    pass  # Accept neutral
                else:
                    current[e1] = old1
                    current[e2] = old2
            else:
                current[e1] = old1
                current[e2] = old2
        
        return best, best_count
    
    # Generate initial coloring using modular arithmetic
    def gen_modular_coloring(a, b, mod):
        """c(i,j) = (a*i + b*j) % mod, mapped to 0..MAX_COLOR"""
        coloring = [0] * NUM_EDGES
        for idx, (i, j) in enumerate(all_edges):
            coloring[idx] = (a * i + b * j) % (mod + 1)  # +1 to get 0..mod
        return coloring
    
    def gen_random_coloring():
        coloring = [random.randint(0, MAX_COLOR) for _ in range(NUM_EDGES)]
        return coloring
    
    def repair_coloring(coloring, timeout=10.0):
        """Fix any rainbow 4-cycles by recoloring edges."""
        current = list(coloring)
        
        for _ in range(100000):
            if elapsed() > timeout:
                break
            
            # Find a violating cycle
            found_violation = False
            for ce in cycle_edges:
                colors = [current[e] for e in ce]
                if len(set(colors)) == 4:
                    found_violation = True
                    # Pick a random edge in this cycle to recolor
                    eidx = random.choice(ce)
                    # Try to recolor it to match one of the other colors
                    other_colors = [current[e] for e in ce if e != eidx]
                    new_color = random.choice(other_colors)
                    current[eidx] = new_color
                    break
            
            if not found_violation:
                break
        
        return current
    
    # Strategy: Try multiple starting points and pick the best
    best_overall = None
    best_overall_count = 0
    
    # Starting point 1: Modular arithmetic (various parameters)
    modular_params = [
        (1, 1, 19), (1, 2, 19), (1, 3, 19), (2, 1, 19), (3, 1, 19),
        (1, 1, 23), (1, 2, 23), (1, 3, 23), (2, 1, 23), (3, 2, 23),
        (1, 1, 31), (1, 2, 31), (1, 3, 31), (2, 1, 31), (3, 1, 31),
        (1, 1, 17), (1, 2, 17), (1, 3, 17), (2, 3, 17), (3, 2, 17),
        (1, 1, 13), (1, 2, 13), (1, 3, 13), (2, 1, 13), (3, 1, 13),
    ]
    
    for a, b, mod in modular_params:
        if elapsed() > DEADLINE - 10:
            break
        
        coloring = gen_modular_coloring(a, b, mod)
        
        # Repair if needed
        if not is_valid_full(coloring):
            coloring = repair_coloring(coloring, timeout=min(5.0, DEADLINE - elapsed() - 5))
        
        if not is_valid_full(coloring):
            continue
        
        # Local search
        remaining = DEADLINE - elapsed() - 5
        if remaining < 3:
            break
        
        search_time = min(15.0, remaining * 0.4)
        result, count = local_search(coloring, timeout=search_time, aggressive=True)
        
        if count > best_overall_count:
            best_overall_count = count
            best_overall = result
        
        # Check if we can do more
        if elapsed() > DEADLINE - 10:
            break
    
    # Starting point 2: Random coloring (if time permits)
    if elapsed() < DEADLINE - 15 and best_overall_count < 25:
        for trial in range(5):
            if elapsed() > DEADLINE - 10:
                break
            
            coloring = gen_random_coloring()
            coloring = repair_coloring(coloring, timeout=min(5.0, DEADLINE - elapsed() - 5))
            
            if not is_valid_full(coloring):
                continue
            
            remaining = DEADLINE - elapsed() - 5
            if remaining < 3:
                break
            
            search_time = min(10.0, remaining * 0.3)
            result, count = local_search(coloring, timeout=search_time, aggressive=True)
            
            if count > best_overall_count:
                best_overall_count = count
                best_overall = result
    
    # If we have a good result, try to polish with 2-opt
    if best_overall is not None and elapsed() < DEADLINE - 10 and best_overall_count < 30:
        remaining = DEADLINE - elapsed() - 5
        if remaining > 5:
            result, count = two_opt_search(best_overall, timeout=min(15.0, remaining * 0.5))
            if count > best_overall_count:
                best_overall_count = count
                best_overall = result
    
    # Final local search pass if time remains
    if best_overall is not None and elapsed() < DEADLINE - 5:
        remaining = DEADLINE - elapsed() - 3
        if remaining > 3:
            result, count = local_search(best_overall, timeout=min(10.0, remaining), aggressive=True)
            if count > best_overall_count:
                best_overall_count = count
                best_overall = result
    
    # If all else fails, use a simple valid coloring
    if best_overall is None:
        # All edges same color - trivially valid
        best_overall = [0] * NUM_EDGES
        best_overall_count = 1
    
    return best_overall
