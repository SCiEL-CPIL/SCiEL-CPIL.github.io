import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 20
    num_colors = 20
    
    # Generate all possible edges
    all_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.append((i, j))
            
    edge_index = {e: idx for idx, e in enumerate(all_edges)}
    num_edges = len(all_edges)
    
    # Precompute all 4-cycles in K_20
    # A 4-cycle is a set of 4 edges.
    # Vertices a, b, c, d form 3 cycles:
    # 1. (a,b), (b,c), (c,d), (d,a)
    # 2. (a,b), (b,d), (d,c), (c,a)
    # 3. (a,c), (c,b), (b,d), (d,a)
    cycles = []
    for verts in itertools.combinations(range(n), 4):
        a, b, c, d = verts
        # Cycle 1: a-b-c-d-a
        e1 = (min(a,b), max(a,b))
        e2 = (min(b,c), max(b,c))
        e3 = (min(c,d), max(c,d))
        e4 = (min(d,a), max(d,a))
        cycles.append((edge_index[e1], edge_index[e2], edge_index[e3], edge_index[e4]))
        
        # Cycle 2: a-b-d-c-a
        e1 = (min(a,b), max(a,b))
        e2 = (min(b,d), max(b,d))
        e3 = (min(d,c), max(d,c))
        e4 = (min(c,a), max(c,a))
        cycles.append((edge_index[e1], edge_index[e2], edge_index[e3], edge_index[e4]))
        
        # Cycle 3: a-c-b-d-a
        e1 = (min(a,c), max(a,c))
        e2 = (min(c,b), max(c,b))
        e3 = (min(b,d), max(b,d))
        e4 = (min(d,a), max(d,a))
        cycles.append((edge_index[e1], edge_index[e2], edge_index[e3], edge_index[e4]))
        
    num_cycles = len(cycles)
    
    # State
    # active[i] = True if edge i is present
    active = [True] * num_edges
    # colors[i] = color of edge i (if active)
    colors = [random.randint(0, num_colors - 1) for _ in range(num_edges)]
    
    best_edge_count = 0
    best_colors = colors[:]
    best_active = active[:]
    
    def check_conflicts():
        """Returns list of conflict counts for each edge"""
        conflict_counts = [0] * num_edges
        for c_idx in range(num_cycles):
            e1, e2, e3, e4 = cycles[c_idx]
            # Only check if all edges are active
            if not (active[e1] and active[e2] and active[e3] and active[e4]):
                continue
            
            c1 = colors[e1]
            c2 = colors[e2]
            c3 = colors[e3]
            c4 = colors[e4]
            
            if len(set([c1, c2, c3, c4])) < 4:
                conflict_counts[e1] += 1
                conflict_counts[e2] += 1
                conflict_counts[e3] += 1
                conflict_counts[e4] += 1
                
        return conflict_counts

    # Initial check
    current_conflicts = check_conflicts()
    total_conflicts = sum(current_conflicts)
    
    if total_conflicts == 0:
        best_edge_count = num_edges
        best_colors = colors[:]
        best_active = active[:]
    else:
        # Local Search
        # We want to minimize conflicts, and if multiple solutions, maximize edges.
        # Strategy: 
        # 1. If conflicts exist, try to recolor edges involved in conflicts.
        # 2. If recoloring doesn't help in a while, remove the most conflicting edge.
        
        stall_counter = 0
        max_stall = 5000
        last_improvement = 0
        last_total_conflicts = total_conflicts
        
        while time.time() < deadline:
            if total_conflicts == 0:
                # Valid coloring for current active set
                current_edge_count = sum(active)
                if current_edge_count > best_edge_count:
                    best_edge_count = current_edge_count
                    best_colors = colors[:]
                    best_active = active[:]
                # Try to add back a removed edge?
                # This is complex. For now, if we have 0 conflicts, we are good.
                # We can try to add edges back if we removed any.
                if current_edge_count < num_edges:
                    # Try to add a random inactive edge
                    inactive_indices = [i for i in range(num_edges) if not active[i]]
                    if inactive_indices:
                        idx = random.choice(inactive_indices)
                        active[idx] = True
                        colors[idx] = random.randint(0, num_colors - 1)
                        # Re-evaluate
                        current_conflicts = check_conflicts()
                        total_conflicts = sum(current_conflicts)
                        # If adding caused conflicts, we might revert or try to fix
                        # But let's let the loop handle the fixing
                        continue
                else:
                    # We have K_20 and it's valid. We can't do better.
                    break

            # Identify edges with conflicts
            conflicting_edges = [i for i in range(num_edges) if active[i] and current_conflicts[i] > 0]
            
            if not conflicting_edges:
                # Should not happen if total_conflicts > 0, but safety check
                break

            # Choose an edge to modify
            # Heuristic: Pick an edge with high conflict count
            # Weighted random choice
            weights = [current_conflicts[i] for i in conflicting_edges]
            total_weight = sum(weights)
            if total_weight == 0:
                break
                
            r = random.random() * total_weight
            cumulative = 0
            chosen_idx = conflicting_edges[0]
            for i, w in zip(conflicting_edges, weights):
                cumulative += w
                if r <= cumulative:
                    chosen_idx = i
                    break
            
            # Decide: Recolor or Remove?
            # If the edge has very high conflict count, removing it might be faster.
            # Let's say if conflict count > 10, remove with prob 0.5.
            if current_conflicts[chosen_idx] > 10 and random.random() < 0.5:
                # Remove edge
                active[chosen_idx] = False
                # Re-evaluate
                current_conflicts = check_conflicts()
                new_total_conflicts = sum(current_conflicts)
                
                if new_total_conflicts < last_total_conflicts:
                    last_total_conflicts = new_total_conflicts
                    stall_counter = 0
                else:
                    stall_counter += 1
                    
            else:
                # Recolor
                old_color = colors[chosen_idx]
                # Try a new random color
                new_color = random.randint(0, num_colors - 1)
                while new_color == old_color:
                    new_color = random.randint(0, num_colors - 1)
                    
                colors[chosen_idx] = new_color
                current_conflicts = check_conflicts()
                new_total_conflicts = sum(current_conflicts)
                
                if new_total_conflicts < last_total_conflicts:
                    last_total_conflicts = new_total_conflicts
                    stall_counter = 0
                else:
                    # If no improvement, revert? Or keep? 
                    # Simulated annealing style: accept worse with prob exp(-delta/T)
                    # For simplicity, let's just revert if it gets much worse
                    if new_total_conflicts > last_total_conflicts + 5:
                        colors[chosen_idx] = old_color
                        current_conflicts = check_conflicts()
                        new_total_conflicts = sum(current_conflicts)
                    else:
                        last_total_conflicts = new_total_conflicts
                    stall_counter += 1
                    
            total_conflicts = new_total_conflicts
            
            if stall_counter > max_stall:
                # Restart with randomization or remove more aggressively
                # Let's just remove the single worst edge
                worst_edge = max(range(num_edges), key=lambda i: current_conflicts[i] if active[i] else -1)
                if active[worst_edge]:
                    active[worst_edge] = False
                    current_conflicts = check_conflicts()
                    total_conflicts = sum(current_conflicts)
                    last_total_conflicts = total_conflicts
                    stall_counter = 0
                else:
                    break

    # Construct final result
    result = []
    for i in range(num_edges):
        if best_active[i]:
            u, v = all_edges[i]
            result.append([u, v, best_colors[i]])
            
    return result

# Run construct once to ensure it works? No, evaluator calls it.
# But we can test locally.
if __name__ == "__main__":
    edges = construct()
    print(f"Number of edges: {len(edges)}")
