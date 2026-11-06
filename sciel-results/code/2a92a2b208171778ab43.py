import random
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 20
    num_colors = 20
    
    # Precompute all 4-cycles and their edge sets
    # A 4-cycle is defined by 4 distinct vertices {u,v,w,x} and a specific pairing of edges.
    # There are 3 distinct 4-cycles (edge sets) for each set of 4 vertices.
    all_cycles = []
    edge_to_cycles = defaultdict(list)
    
    # Map edge to index
    edge_index = {}
    all_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            e = (u, v)
            edge_index[e] = len(all_edges)
            all_edges.append(e)
            
    num_edges = len(all_edges)
    
    for u, v, w, x in itertools.combinations(range(n), 4):
        # 3 distinct cycles on 4 vertices
        # 1. u-v-w-x-u
        c1 = [(u, v), (v, w), (w, x), (x, u)]
        # 2. u-v-x-w-u
        c2 = [(u, v), (v, x), (x, w), (w, u)]
        # 3. u-w-v-x-u
        c3 = [(u, w), (w, v), (v, x), (x, u)]
        
        for cycle in [c1, c2, c3]:
            # Normalize edges for hashing
            norm_cycle = tuple(sorted((min(e), max(e)) for e in cycle))
            # Check if it's a simple cycle (4 distinct edges)
            if len(set(norm_cycle)) == 4:
                idx = len(all_cycles)
                all_cycles.append(norm_cycle)
                for e in norm_cycle:
                    edge_to_cycles[e].append(idx)
    
    num_cycles = len(all_cycles)
    
    # State variables
    # present[e] = True if edge e is in the graph
    # colors[e] = colour of edge e
    # conflict_count[cycle_idx] = 1 if cycle is present and NOT rainbow, else 0
    
    present = [False] * num_edges
    colors = [0] * num_edges
    # cycle_conflict[c] = 1 if cycle c is "bad" (present and not rainbow)
    cycle_conflict = [0] * num_cycles
    
    # Helper to check if a cycle is rainbow given current colors and presence
    # We only call this when the cycle's status might change
    
    def get_cycle_status(c_idx):
        """Returns True if the cycle is 'bad' (present and not rainbow)."""
        cycle = all_cycles[c_idx]
        # Check if all edges are present
        e1, e2, e3, e4 = cycle
        # Convert to indices
        i1, i2, i3, i4 = edge_index[e1], edge_index[e2], edge_index[e3], edge_index[e4]
        
        if not (present[i1] and present[i2] and present[i3] and present[i4]):
            return False # Not present, so not a conflict
        
        # Check if rainbow
        c1, c2, c3, c4 = colors[i1], colors[i2], colors[i3], colors[i4]
        if len({c1, c2, c3, c4}) == 4:
            return False # Rainbow, so not a conflict
        
        return True # Present and not rainbow

    # Initialize with a C4-free graph using greedy method
    # This ensures initial conflict count is 0
    
    def init_c4_free():
        random.shuffle(all_edges)
        for e in all_edges:
            i = edge_index[e]
            # Check if adding e creates a C4
            # A C4 is created if there exists a path of length 3 between the endpoints of e
            # Or more simply, check if any 4-cycle containing e becomes complete
            # Since we are building C4-free, we just check if e completes a C4.
            
            # Actually, simpler: just try to add e. If it creates a C4, skip.
            # To check if it creates a C4, we look at all 4-cycles containing e.
            # If any of them has the other 3 edges present, then adding e creates a C4.
            
            creates_c4 = False
            for c_idx in edge_to_cycles[e]:
                cycle = all_cycles[c_idx]
                other_edges = [ce for ce in cycle if ce != e]
                if all(present[edge_index[oe]] for oe in other_edges):
                    creates_c4 = True
                    break
            
            if not creates_c4:
                present[i] = True
                colors[i] = random.randint(0, num_colors - 1)
                
    init_c4_free()
    
    # Calculate initial conflicts (should be 0)
    total_conflicts = 0
    for c_idx in range(num_cycles):
        if get_cycle_status(c_idx):
            cycle_conflict[c_idx] = 1
            total_conflicts += 1
        else:
            cycle_conflict[c_idx] = 0
            
    current_edges = sum(present)
    best_edges = current_edges
    best_state = (present[:], colors[:])
    
    # Simulated Annealing
    T = 5.0
    cooling_rate = 0.999
    
    while time.time() < deadline - 5.0:
        if T < 0.01:
            T = 0.01
        
        # Random move
        move_type = random.random()
        
        if move_type < 0.6:
            # Try to add an edge
            missing_edges = [i for i in range(num_edges) if not present[i]]
            if not missing_edges:
                continue
            i = random.choice(missing_edges)
            e = all_edges[i]
            
            # Try a random color
            new_color = random.randint(0, num_colors - 1)
            
            # Calculate delta conflicts
            delta_conflicts = 0
            affected_cycles = edge_to_cycles[e]
            
            # Temporarily add edge to see effect? 
            # Better: calculate change in status for affected cycles.
            
            old_status = {}
            for c_idx in affected_cycles:
                old_status[c_idx] = cycle_conflict[c_idx]
                
            # Simulate addition
            present[i] = True
            colors[i] = new_color
            
            new_conflicts = 0
            for c_idx in affected_cycles:
                if get_cycle_status(c_idx):
                    cycle_conflict[c_idx] = 1
                    new_conflicts += 1
                else:
                    cycle_conflict[c_idx] = 0
                    
            # Restore state for now, accept/reject later
            # But we need to compare total conflicts.
            # Current total conflicts = total_conflicts - sum(old_status[c] for c in affected) + new_conflicts
            
            old_total_part = sum(old_status[c] for c in affected_cycles)
            new_total_part = new_conflicts
            delta_conflicts = new_total_part - old_total_part
            
            # Acceptance criteria
            # We want to maximize edges. 
            # Score = edges - lambda * conflicts
            # If delta_conflicts < 0, we definitely accept (gain edge, reduce conflicts)
            # If delta_conflicts == 0, accept (gain edge)
            # If delta_conflicts > 0, accept with prob exp(-delta_conflicts / T)
            
            accept = False
            if delta_conflicts <= 0:
                accept = True
            else:
                # Penalty for conflicts
                # We can also allow temporary increase in conflicts if it leads to more edges later
                # But strictly, we want to stay valid (conflicts=0) or close to it.
                # If we allow conflicts, we must ensure we return to 0.
                # Let's use a hard constraint approach: only accept if conflicts remain 0?
                # No, that's too restrictive. 
                # Let's use a soft constraint: minimize conflicts.
                # If total_conflicts + delta_conflicts == 0, accept.
                # If total_conflicts + delta_conflicts > 0, accept with prob exp(-delta_conflicts / T).
                
                # Actually, if we have 0 conflicts and add an edge creating 2 conflicts, 
                # we are in an invalid state. 
                # It is often better to stay valid. 
                # So, if current total_conflicts == 0, we should only accept if new total == 0.
                # If current total_conflicts > 0, we can accept increases to escape.
                
                if total_conflicts == 0:
                    if delta_conflicts == 0:
                        accept = True
                    else:
                        accept = False
                else:
                    accept = (random.random() < pow(2.718, -delta_conflicts / T))
                    
            if accept:
                total_conflicts += delta_conflicts
                current_edges += 1
                if current_edges > best_edges and total_conflicts == 0:
                    best_edges = current_edges
                    best_state = (present[:], colors[:])
            else:
                # Revert
                present[i] = False
                # Restore cycle conflicts
                for c_idx in affected_cycles:
                    cycle_conflict[c_idx] = old_status[c_idx]
                    
        elif move_type < 0.8:
            # Recolour an edge
            present_edges = [i for i in range(num_edges) if present[i]]
            if not present_edges:
                continue
            i = random.choice(present_edges)
            e = all_edges[i]
            old_color = colors[i]
            new_color = random.randint(0, num_colors - 1)
            if new_color == old_color:
                continue
                
            affected_cycles = edge_to_cycles[e]
            old_status = {c: cycle_conflict[c] for c in affected_cycles}
            
            colors[i] = new_color
            
            new_conflicts = 0
            for c_idx in affected_cycles:
                if get_cycle_status(c_idx):
                    cycle_conflict[c_idx] = 1
                    new_conflicts += 1
                else:
                    cycle_conflict[c_idx] = 0
                    
            old_total_part = sum(old_status.values())
            delta_conflicts = new_conflicts - old_total_part
            
            # Recolouring doesn't change edge count.
            # Accept if it reduces conflicts, or with prob exp(-delta/T) if it increases.
            if delta_conflicts < 0:
                accept = True
            elif delta_conflicts == 0:
                accept = True # Neutral, might help future moves
            else:
                accept = (random.random() < pow(2.718, -delta_conflicts / T))
                
            if accept:
                total_conflicts += delta_conflicts
                if current_edges > best_edges and total_conflicts == 0:
                    best_edges = current_edges
                    best_state = (present[:], colors[:])
            else:
                colors[i] = old_color
                for c_idx in affected_cycles:
                    cycle_conflict[c_idx] = old_status[c_idx]
                    
        else:
            # Remove an edge (to escape local minima)
            present_edges = [i for i in range(num_edges) if present[i]]
            if not present_edges:
                continue
            i = random.choice(present_edges)
            e = all_edges[i]
            
            affected_cycles = edge_to_cycles[e]
            old_status = {c: cycle_conflict[c] for c in affected_cycles}
            
            present[i] = False
            
            new_conflicts = 0
            for c_idx in affected_cycles:
                if get_cycle_status(c_idx):
                    cycle_conflict[c_idx] = 1
                    new_conflicts += 1
                else:
                    cycle_conflict[c_idx] = 0
                    
            old_total_part = sum(old_status.values())
            delta_conflicts = new_conflicts - old_total_part
            
            # Removing an edge always reduces or maintains conflicts.
            # But we lose an edge.
            # Only accept if it significantly reduces conflicts (to allow re-adding later)
            # Or with small probability.
            if delta_conflicts < 0:
                # We gain validity. Accept if we are stuck?
                # Let's accept with prob 0.5 if it reduces conflicts.
                accept = (random.random() < 0.5)
            else:
                accept = False
                
            if accept:
                total_conflicts += delta_conflicts
                current_edges -= 1
            else:
                present[i] = True
                for c_idx in affected_cycles:
                    cycle_conflict[c_idx] = old_status[c_idx]
                    
        T *= cooling_rate
        
    # Return best valid state
    present, colors = best_state
    
    result = []
    for i in range(num_edges):
        if present[i]:
            u, v = all_edges[i]
            result.append([u, v, colors[i]])
            
    return result

