import random
import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    n = 20
    n_colors = 20
    
    # 1. Precompute all edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_edges = len(all_edges)
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    
    # 2. Precompute all 4-cycles
    # A 4-cycle is defined by 4 distinct vertices u, v, w, x.
    # The edges are (u,v), (v,w), (w,x), (x,u).
    # There are 3 distinct 4-cycles for any set of 4 vertices in a complete graph.
    all_4cycles = []
    for quad in combinations(range(n), 4):
        w, x, y, z = quad
        # Cycle 1: w-x-y-z-w
        c1 = [(w, x), (x, y), (y, z), (z, w)]
        # Cycle 2: w-x-z-y-w
        c2 = [(w, x), (x, z), (z, y), (y, w)]
        # Cycle 3: w-y-x-z-w
        c3 = [(w, y), (y, x), (x, z), (z, w)]
        
        for cyc in [c1, c2, c3]:
            try:
                edge_idxs = tuple(sorted(edge_to_idx[e] for e in cyc))
                all_4cycles.append(edge_idxs)
            except KeyError:
                pass

    num_4cycles = len(all_4cycles)
    
    # 3. Precompute edge to cycles mapping
    edge_cycles = [[] for _ in range(num_edges)]
    for ci, cyc in enumerate(all_4cycles):
        for ei in cyc:
            edge_cycles[ei].append(ci)

    # State
    # present: list of bools, is edge present?
    # colors: list of ints, color of edge (if present)
    
    # Helper to check if a cycle is rainbow given current state
    def is_cycle_rainbow(cyc):
        # cyc is a tuple of 4 edge indices
        # Check if all 4 edges are present
        # If not present, it's not a "valid" cycle in the current graph, 
        # but for violation counting, we only care about cycles that ARE fully present.
        # Wait, the constraint is "Every 4-cycle (as a subgraph) must receive four distinct colours".
        # This implies we only check 4-cycles that exist in the graph.
        # So if any edge is missing, the cycle doesn't exist, so it's not a violation.
        
        e1, e2, e3, e4 = cyc
        if not (present[e1] and present[e2] and present[e3] and present[e4]):
            return True # Not a violation because cycle doesn't exist
        
        c1 = colors[e1]
        c2 = colors[e2]
        c3 = colors[e3]
        c4 = colors[e4]
        
        return len({c1, c2, c3, c4}) == 4

    def update_cycle_status(ci, old_state, new_state):
        # This is tricky to do incrementally without storing full state history per cycle.
        # Simpler: Just re-evaluate the specific cycle.
        pass

    # We will maintain:
    # present: list of bools
    # colors: list of ints
    # viol: list of bools (True if cycle is present and NOT rainbow)
    # total_viol: int
    
    present = [False] * num_edges
    colors = [0] * num_edges
    viol = [False] * num_4cycles
    total_viol = 0
    
    # Initialization: Start with K_n and random colors
    for i in range(num_edges):
        present[i] = True
        colors[i] = random.randint(0, n_colors - 1)
        
    # Calculate initial violations
    for ci in range(num_4cycles):
        if not is_cycle_rainbow(all_4cycles[ci]):
            viol[ci] = True
            total_viol += 1
            
    best_edges_count = sum(present)
    best_present = present.copy()
    best_colors = colors.copy()
    
    # Objective function
    # Score = Edges - Lambda * Violations
    LAMBDA = 1000.0
    
    def get_score():
        return sum(present) - LAMBDA * total_viol

    current_score = get_score()
    
    # Simulated Annealing
    T = 100.0
    T_min = 0.01
    cooling_rate = 0.999
    
    # Estimate moves per second to adjust cooling
    # We want to run until deadline
    
    move_count = 0
    while time.time() < deadline:
        move_count += 1
        
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.4:
            # Add Edge
            # Pick a random edge not present
            # Optimization: Keep a list of missing edges? 
            # For n=20, num_edges=190. Random pick is fine.
            ei = random.randint(0, num_edges - 1)
            if present[ei]:
                continue
            
            # Try to add with a random color
            # Heuristic: Try colors that are frequent in neighbors? 
            # Just random for now.
            new_color = random.randint(0, n_colors - 1)
            
            # Calculate delta
            old_viol = total_viol
            old_edges = sum(present) # Expensive. Maintain count?
            # Let's maintain edge_count
            pass
            
        # This approach of sum(present) is too slow.
        # Let's refactor to maintain state variables properly.
        
        break # Break out to implement the optimized version below
        
    # --- Optimized Implementation ---
    
    present = [False] * num_edges
    colors = [0] * num_edges
    viol = [False] * num_4cycles
    total_viol = 0
    edge_count = 0
    
    # Init K_n
    for i in range(num_edges):
        present[i] = True
        colors[i] = random.randint(0, n_colors - 1)
        edge_count += 1
        
    for ci in range(num_4cycles):
        if not is_cycle_rainbow(all_4cycles[ci]):
            viol[ci] = True
            total_viol += 1
            
    best_edge_count = edge_count
    best_present = present.copy()
    best_colors = colors.copy()
    
    def get_score():
        return edge_count - LAMBDA * total_viol
        
    current_score = get_score()
    
    T = 50.0
    T_min = 0.01
    # Adjust cooling based on expected time
    # Assume ~1000 moves/sec. 145 sec -> 145000 moves.
    # (T_min/T)^(1/N) = 0.999 roughly
    cooling_rate = 0.9995
    
    missing_edges = list(range(num_edges))
    # We need to update missing_edges efficiently or just pick random and check.
    # Picking random and checking is O(1).
    
    while time.time() < deadline:
        # Choose move
        r = random.random()
        
        if r < 0.3:
            # Add Edge
            ei = random.randint(0, num_edges - 1)
            if present[ei]:
                continue
                
            new_color = random.randint(0, n_colors - 1)
            
            # Apply move
            present[ei] = True
            colors[ei] = new_color
            edge_count += 1
            
            # Update violations for cycles containing ei
            delta_viol = 0
            for ci in edge_cycles[ei]:
                was_viol = viol[ci]
                is_now_viol = not is_cycle_rainbow(all_4cycles[ci])
                if was_viol != is_now_viol:
                    viol[ci] = is_now_viol
                    if is_now_viol:
                        total_viol += 1
                        delta_viol += 1
                    else:
                        total_viol -= 1
                        delta_viol -= 1
                        
            new_score = edge_count - LAMBDA * total_viol
            delta_score = new_score - current_score
            
            if delta_score >= 0 or random.random() < math.exp(delta_score / T):
                current_score = new_score
                if edge_count > best_edge_count and total_viol == 0:
                    best_edge_count = edge_count
                    best_present = present.copy()
                    best_colors = colors.copy()
            else:
                # Revert
                present[ei] = False
                edge_count -= 1
                # Revert violations
                for ci in edge_cycles[ei]:
                    # We need to restore the previous state.
                    # This is tricky. 
                    # Better: Store old viol state for affected cycles?
                    # Or just recompute is_cycle_rainbow for revert?
                    # Recompute is cheap.
                    pass # Handled by re-evaluation in revert? 
                    # Actually, if we revert present[ei]=False, the cycle is no longer "present" 
                    # so it shouldn't be a violation.
                    # So viol[ci] should become False if it was True due to this edge?
                    # No, if the cycle was a violation before adding ei, it might still be.
                    # If it wasn't a violation before, and adding ei made it one, reverting makes it non-violation.
                    
                    # Let's just recompute the status of affected cycles on revert.
                    # It's safer.
                    pass
                    
                # To simplify and avoid bugs, let's just recompute total_viol for affected cycles on revert.
                # But we need to know the "old" total_viol.
                # Let's store old_viol for affected cycles.
                
                # This "undo" logic is getting messy. 
                # Alternative: Store the state of the 4 cycles before the move.
                
                # Let's restart the loop logic with proper undo.
                break
                
        # This block is flawed. I will write the clean version in the final output.
        
    # Clean Version Logic:
    
    present = [False] * num_edges
    colors = [0] * num_edges
    viol = [False] * num_4cycles
    total_viol = 0
    edge_count = 0
    
    for i in range(num_edges):
        present[i] = True
        colors[i] = random.randint(0, n_colors - 1)
        edge_count += 1
        
    for ci in range(num_4cycles):
        if not is_cycle_rainbow(all_4cycles[ci]):
            viol[ci] = True
            total_viol += 1
            
    best_edge_count = edge_count
    best_present = present.copy()
    best_colors = colors.copy()
    
    def get_score():
        return edge_count - LAMBDA * total_viol
        
    current_score = get_score()
    
    T = 50.0
    cooling_rate = 0.9995
    
    # Precompute cycle edge lists for fast access
    # all_4cycles[ci] is a tuple of 4 edge indices
    
    while time.time() < deadline:
        r = random.random()
        
        if r < 0.4:
            # Add Edge
            ei = random.randint(0, num_edges - 1)
            if present[ei]:
                continue
                
            new_color = random.randint(0, n_colors - 1)
            
            # Store old state of affected cycles
            affected_cycles = edge_cycles[ei]
            old_viol_states = [viol[ci] for ci in affected_cycles]
            old_total_viol = total_viol
            old_edge_count = edge_count
            old_score = current_score
            
            # Apply
            present[ei] = True
            colors[ei] = new_color
            edge_count += 1
            
            for ci in affected_cycles:
                is_now_viol = not is_cycle_rainbow(all_4cycles[ci])
                if viol[ci] != is_now_viol:
                    if is_now_viol:
                        total_viol += 1
                    else:
                        total_viol -= 1
                    viol[ci] = is_now_viol
                    
            new_score = get_score()
            delta = new_score - old_score
            
            if delta >= 0 or random.random() < math.exp(delta / T):
                current_score = new_score
                if total_viol == 0 and edge_count > best_edge_count:
                    best_edge_count = edge_count
                    best_present = present.copy()
                    best_colors = colors.copy()
            else:
                # Revert
                present[ei] = False
                edge_count = old_edge_count
                total_viol = old_total_viol
                for i, ci in enumerate(affected_cycles):
                    viol[ci] = old_viol_states[i]
                current_score = old_score
                
        elif r < 0.7:
            # Recolor
            ei = random.randint(0, num_edges - 1)
            if not present[ei]:
                continue
                
            old_color = colors[ei]
            new_color = random.randint(0, n_colors - 1)
            if new_color == old_color:
                continue
                
            affected_cycles = edge_cycles[ei]
            old_viol_states = [viol[ci] for ci in affected_cycles]
            old_total_viol = total_viol
            old_score = current_score
            
            # Apply
            colors[ei] = new_color
            
            for ci in affected_cycles:
                is_now_viol = not is_cycle_rainbow(all_4cycles[ci])
                if viol[ci] != is_now_viol:
                    if is_now_viol:
                        total_viol += 1
                    else:
                        total_viol -= 1
                    viol[ci] = is_now_viol
                    
            new_score = get_score()
            delta = new_score - old_score
            
            if delta >= 0 or random.random() < math.exp(delta / T):
                current_score = new_score
                if total_viol == 0 and edge_count > best_edge_count:
                    best_edge_count = edge_count
                    best_present = present.copy()
                    best_colors = colors.copy()
            else:
                # Revert
                colors[ei] = old_color
                total_viol = old_total_viol
                for i, ci in enumerate(affected_cycles):
                    viol[ci] = old_viol_states[i]
                current_score = old_score
                
        else:
            # Remove Edge
            ei = random.randint(0, num_edges - 1)
            if not present[ei]:
                continue
                
            affected_cycles = edge_cycles[ei]
            old_viol_states = [viol[ci] for ci in affected_cycles]
            old_total_viol = total_viol
            old_edge_count = edge_count
            old_score = current_score
            
            # Apply
            present[ei] = False
            edge_count -= 1
            
            for ci in affected_cycles:
                # If edge is removed, cycle is not present, so not a violation
                # But wait, if the cycle was a violation, it becomes non-violation?
                # Yes, because the cycle no longer exists.
                # is_cycle_rainbow returns True if not present.
                is_now_viol = not is_cycle_rainbow(all_4cycles[ci])
                if viol[ci] != is_now_viol:
                    if is_now_viol:
                        total_viol += 1
                    else:
                        total_viol -= 1
                    viol[ci] = is_now_viol
                    
            new_score = get_score()
            delta = new_score - old_score
            
            if delta >= 0 or random.random() < math.exp(delta / T):
                current_score = new_score
                if total_viol == 0 and edge_count > best_edge_count:
                    best_edge_count = edge_count
                    best_present = present.copy()
                    best_colors = colors.copy()
            else:
                # Revert
                present[ei] = True
                edge_count = old_edge_count
                total_viol = old_total_viol
                for i, ci in enumerate(affected_cycles):
                    viol[ci] = old_viol_states[i]
                current_score = old_score
                
        T *= cooling_rate
        if T < T_min:
            T = T_min

    # Build result from best solution
    result = []
    for ei in range(num_edges):
        if best_present[ei]:
            u, v = all_edges[ei]
            c = best_colors[ei]
            result.append([u, v, c])
            
    return result

