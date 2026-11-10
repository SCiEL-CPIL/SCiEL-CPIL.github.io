import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119

# Precompute edge index mapping
def _edge_index(i, j):
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

# Build all 4-cycles
def _build_cycles():
    cycles = []
    for a in range(N - 3):
        for b in range(a + 1, N - 2):
            for c in range(b + 1, N - 1):
                for d in range(c + 1, N):
                    cycles.append((_edge_index(a,b), _edge_index(b,c), _edge_index(c,d), _edge_index(a,d)))
                    cycles.append((_edge_index(a,b), _edge_index(b,d), _edge_index(c,d), _edge_index(a,c)))
                    cycles.append((_edge_index(a,c), _edge_index(b,c), _edge_index(b,d), _edge_index(a,d)))
    return cycles

CYCLES = _build_cycles()
NUM_CYCLES = len(CYCLES)

# Build for each edge, the list of cycles it belongs to
def _build_edge_to_cycles():
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cyc in enumerate(CYCLES):
        for e in cyc:
            edge_to_cycles[e].append(ci)
    return edge_to_cycles

EDGE_TO_CYCLES = _build_edge_to_cycles()

# Precompute position of each edge within its cycles
EDGE_POS_IN_CYCLE = {}
for ci, cyc in enumerate(CYCLES):
    for idx, e in enumerate(cyc):
        if e not in EDGE_POS_IN_CYCLE:
            EDGE_POS_IN_CYCLE[e] = {}
        EDGE_POS_IN_CYCLE[e][ci] = idx

def _count_violations(colors):
    violations = 0
    for cyc in CYCLES:
        c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
        if len({c0, c1, c2, c3}) == 4:
            violations += 1
    return violations

def construct():
    start_time = time.time()
    deadline = start_time + 82.0
    
    random.seed(42)
    np.random.seed(42)
    
    best_colors = None
    best_distinct = 0
    
    # Initialize with all zeros
    colors = np.zeros(NUM_EDGES, dtype=np.int32)
    
    # For each cycle, track the multiset of colors
    # cycle_color_counts[ci] = dict mapping color -> count
    cycle_color_counts = [{} for _ in range(NUM_CYCLES)]
    for ci, cyc in enumerate(CYCLES):
        for e in cyc:
            c = colors[e]
            cycle_color_counts[ci][c] = cycle_color_counts[ci].get(c, 0) + 1
    
    # Track violations
    cycle_violation = np.zeros(NUM_CYCLES, dtype=np.int8)
    total_violations = 0
    
    # Initialize violation status
    for ci in range(NUM_CYCLES):
        if len(cycle_color_counts[ci]) == 4:
            cycle_violation[ci] = 1
            total_violations += 1
    
    def _cycle_is_violation(ci):
        return len(cycle_color_counts[ci]) == 4
    
    def _apply_move(e, new_color):
        """Apply a move and update all tracking structures. Returns delta_violations."""
        old_color = colors[e]
        if old_color == new_color:
            return 0
        
        delta = 0
        for ci in EDGE_TO_CYCLES[e]:
            # Update cycle_color_counts
            old_count = cycle_color_counts[ci].get(old_color, 0)
            if old_count == 1:
                del cycle_color_counts[ci][old_color]
            else:
                cycle_color_counts[ci][old_color] = old_count - 1
            
            new_count = cycle_color_counts[ci].get(new_color, 0)
            cycle_color_counts[ci][new_color] = new_count + 1
            
            old_viol = 1 if len(cycle_color_counts[ci]) == 4 else 0
            # Wait, we just modified it. Need to check before and after properly.
            # Let me redo this.
            pass
        
        # Redo properly
        for ci in EDGE_TO_CYCLES[e]:
            # Compute old violation status (before change)
            old_cc = cycle_color_counts[ci].copy()
            # Remove old_color
            old_count = old_cc.get(old_color, 0)
            if old_count == 1:
                del old_cc[old_color]
            else:
                old_cc[old_color] = old_count - 1
            # Add new_color
            new_count = old_cc.get(new_color, 0)
            old_cc[new_color] = new_count + 1
            
            old_viol = 1 if len(cycle_color_counts[ci]) == 4 else 0
            new_viol = 1 if len(old_cc) == 4 else 0
            
            delta += new_viol - old_viol
            
            # Update the actual structure
            old_count_actual = cycle_color_counts[ci].get(old_color, 0)
            if old_count_actual == 1:
                del cycle_color_counts[ci][old_color]
            else:
                cycle_color_counts[ci][old_color] = old_count_actual - 1
            
            new_count_actual = cycle_color_counts[ci].get(new_color, 0)
            cycle_color_counts[ci][new_color] = new_count_actual + 1
            
            cycle_violation[ci] = 1 if len(cycle_color_counts[ci]) == 4 else 0
        
        colors[e] = new_color
        return delta
    
    def _revert_move(e, old_color):
        """Revert edge e back to old_color. Returns delta_violations."""
        current_color = colors[e]
        if current_color == old_color:
            return 0
        
        delta = 0
        for ci in EDGE_TO_CYCLES[e]:
            old_cc = cycle_color_counts[ci].copy()
            # Remove current_color
            old_count = old_cc.get(current_color, 0)
            if old_count == 1:
                del old_cc[current_color]
            else:
                old_cc[current_color] = old_count - 1
            # Add old_color
            new_count = old_cc.get(old_color, 0)
            old_cc[old_color] = new_count + 1
            
            old_viol = 1 if len(cycle_color_counts[ci]) == 4 else 0
            new_viol = 1 if len(old_cc) == 4 else 0
            
            delta += new_viol - old_viol
            
            # Update actual
            old_count_actual = cycle_color_counts[ci].get(current_color, 0)
            if old_count_actual == 1:
                del cycle_color_counts[ci][current_color]
            else:
                cycle_color_counts[ci][current_color] = old_count_actual - 1
            
            new_count_actual = cycle_color_counts[ci].get(old_color, 0)
            cycle_color_counts[ci][old_color] = new_count_actual + 1
            
            cycle_violation[ci] = 1 if len(cycle_color_counts[ci]) == 4 else 0
        
        colors[e] = old_color
        return delta
    
    # Simulated annealing
    initial_temp = 5.0
    final_temp = 0.001
    cooling_rate = 0.9998
    temp = initial_temp
    
    used_colors = set(colors.tolist())
    
    max_iter = 300000
    iter_count = 0
    no_improve_count = 0
    
    while iter_count < max_iter:
        if time.time() > deadline:
            break
        
        # Pick a random edge
        e = random.randint(0, NUM_EDGES - 1)
        old_color = colors[e]
        
        # Choose new color
        r = random.random()
        if r < 0.4 and len(used_colors) < MAX_COLOR + 1:
            # Try a new color
            unused = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
            if unused:
                new_color = random.choice(unused)
            else:
                new_color = random.randint(0, MAX_COLOR)
        elif r < 0.8:
            # Try a used color
            used_list = list(used_colors)
            if len(used_list) > 1:
                new_color = random.choice(used_list)
            else:
                new_color = random.randint(0, MAX_COLOR)
        else:
            new_color = random.randint(0, MAX_COLOR)
        
        if new_color == old_color:
            iter_count += 1
            continue
        
        # Compute violation delta
        delta_viol = 0
        for ci in EDGE_TO_CYCLES[e]:
            old_cc = cycle_color_counts[ci].copy()
            old_count = old_cc.get(old_color, 0)
            if old_count == 1:
                del old_cc[old_color]
            else:
                old_cc[old_color] = old_count - 1
            new_count = old_cc.get(new_color, 0)
            old_cc[new_color] = new_count + 1
            
            old_viol = 1 if len(cycle_color_counts[ci]) == 4 else 0
            new_viol = 1 if len(old_cc) == 4 else 0
            delta_viol += new_viol - old_viol
        
        # Acceptance
        if delta_viol == 0:
            accepted = True
        elif delta_viol < 0:
            accepted = True
        else:
            if temp <= 0.001:
                accepted = False
            else:
                prob = np.exp(-delta_viol * 3.0 / temp)
                accepted = random.random() < prob
        
        if accepted:
            # Apply move
            for ci in EDGE_TO_CYCLES[e]:
                old_count = cycle_color_counts[ci].get(old_color, 0)
                if old_count == 1:
                    del cycle_color_counts[ci][old_color]
                else:
                    cycle_color_counts[ci][old_color] = old_count - 1
                new_count = cycle_color_counts[ci].get(new_color, 0)
                cycle_color_counts[ci][new_color] = new_count + 1
                cycle_violation[ci] = 1 if len(cycle_color_counts[ci]) == 4 else 0
            colors[e] = new_color
            total_violations += delta_viol
            
            used_colors = set(colors.tolist())
            
            # Check for improvement
            if total_violations == 0:
                current_distinct = len(used_colors)
                if current_distinct > best_distinct:
                    best_distinct = current_distinct
                    best_colors = colors.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
            
            # Restart cooling if stuck
            if no_improve_count > 50000:
                temp = max(temp, initial_temp * 0.5)
                no_improve_count = 0
        else:
            no_improve_count += 1
        
        temp = max(final_temp, temp * cooling_rate)
        iter_count += 1
    
    # Final check
    if total_violations == 0:
        current_distinct = len(set(colors.tolist()))
        if current_distinct > best_distinct:
            best_distinct = current_distinct
            best_colors = colors.copy()
    
    if best_colors is not None:
        if _count_violations(best_colors.tolist()) == 0:
            return best_colors.tolist()
    
    return [0] * NUM_EDGES
