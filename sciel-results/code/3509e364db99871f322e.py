import time
import random
import sys
from itertools import combinations

def construct():
    n = 20
    deadline = time.time() + 115  # safety margin before 120s limit
    
    # Initialize with complete graph K_n
    # Edges are represented as (u, v) with u < v
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_edges = len(all_edges)
    
    # Map edge to index for fast lookup
    edge_to_idx = {}
    for i, e in enumerate(all_edges):
        edge_to_idx[e] = i
        
    # Current state: 
    # active[i] = 1 if edge i is present, 0 if removed
    # color[i] = colour of edge i (0..19)
    
    # Random initialization
    active = [1] * num_edges
    color = [random.randint(0, 19) for _ in range(num_edges)]
    
    # Precompute 4-cycles? 
    # A 4-cycle is a set of 4 vertices {a,b,c,d} and a specific cyclic ordering.
    # In a simple graph, a 4-cycle is defined by 4 distinct vertices and 4 edges connecting them in a cycle.
    # For a set of 4 vertices, there are 3 distinct 4-cycles (the 3 ways to pair up the 4 vertices into 2 non-intersecting edges is for matching, but for cycles: 
    # Vertices 1,2,3,4. Cycles: 1-2-3-4-1, 1-2-4-3-1, 1-3-2-4-1.
    # Actually, for 4 vertices, there are 3 distinct cycles of length 4.
    # Let's list all 4-cycles in K_n.
    # Number of 4-cycles in K_n is C(n,4) * 3.
    # For n=20, C(20,4) = 4845. 4845 * 3 = 14535 cycles.
    # This is small enough to precompute and check incrementally.
    
    cycles = []
    # Generate all sets of 4 vertices
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        # The 3 distinct 4-cycles on {a,b,c,d}
        # 1: a-b-c-d-a
        cycles.append(((a,b), (b,c), (c,d), (d,a)))
        # 2: a-b-d-c-a
        cycles.append(((a,b), (b,d), (d,c), (c,a)))
        # 3: a-c-b-d-a
        cycles.append(((a,c), (c,b), (b,d), (d,a)))
        
    # Convert edge tuples to indices for faster lookup in cycle checks
    # cycle_idx[i] is a tuple of 4 edge indices
    cycles_idx = []
    for cyc in cycles:
        idx_cyc = tuple(edge_to_idx[e] for e in cyc)
        cycles_idx.append(idx_cyc)
        
    # Precompute for each edge, which cycles it participates in
    # edge_to_cycles[e_idx] = list of cycle indices
    edge_to_cycles = [[] for _ in range(num_edges)]
    for c_idx, cyc in enumerate(cycles_idx):
        for e_idx in cyc:
            edge_to_cycles[e_idx].append(c_idx)
            
    def is_rainbow(cyc_idx):
        """Check if the 4 edges in the cycle have distinct colours AND are all active."""
        e0, e1, e2, e3 = cyc_idx
        if not (active[e0] and active[e1] and active[e2] and active[e3]):
            return True # Not a cycle in current graph, so no violation
        c0, c1, c2, c3 = color[e0], color[e1], color[e2], color[e3]
        return len({c0, c1, c2, c3}) == 4

    def count_violations():
        """Count number of 4-cycles that are NOT rainbow."""
        viol = 0
        for c_idx in range(len(cycles_idx)):
            if not is_rainbow(cycles_idx[c_idx]):
                viol += 1
        return viol

    # Initial state
    current_violations = count_violations()
    current_edges = sum(active)
    
    best_violations = current_violations
    best_edges = current_edges
    best_active = active[:]
    best_color = color[:]
    
    # Simulated Annealing Parameters
    T = 1.0
    cooling = 0.9999
    end_time = time.time() + (deadline - time.time() - 5) # leave 5s for final check
    
    # Optimization: Keep track of violation count per cycle to update incrementally?
    # With 14k cycles, full recount is fast (14k * 4 lookups ~ 50k ops). 
    # Python might be slow for 14k loops * 100k steps. 
    # Let's try to optimize by only checking cycles affected by the move.
    
    # We will maintain a `violations` array of 0/1 for each cycle.
    violations = [0] * len(cycles_idx)
    # Initialize violations array
    for c_idx in range(len(cycles_idx)):
        if not is_rainbow(cycles_idx[c_idx]):
            violations[c_idx] = 1
    current_violations = sum(violations)
    
    def update_violation(c_idx):
        """Update violations[c_idx] based on current state. Returns change in total violations."""
        old = violations[c_idx]
        if not is_rainbow(cycles_idx[c_idx]):
            violations[c_idx] = 1
        else:
            violations[c_idx] = 0
        new = violations[c_idx]
        return new - old

    while time.time() < end_time:
        if T < 0.01:
            break
            
        # Choose a move
        # If we have violations, we want to reduce them.
        # If we have 0 violations, we want to add edges.
        
        if current_violations == 0:
            # Try to add a removed edge
            removed_edges = [i for i in range(num_edges) if not active[i]]
            if not removed_edges:
                # All edges present and valid? Unlikely for K20, but possible.
                # If K20 is valid, we are done.
                break
            e_idx = random.choice(removed_edges)
            # Try adding it
            old_active = active[e_idx]
            active[e_idx] = 1
            # Check affected cycles
            delta_viol = 0
            for c_idx in edge_to_cycles[e_idx]:
                delta_viol += update_violation(c_idx)
            new_violations = current_violations + delta_viol
            
            if new_violations == 0:
                # Accept
                current_violations = new_violations
                current_edges += 1
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_active = active[:]
                    best_color = color[:]
            else:
                # Reject (keep edge removed)
                active[e_idx] = 0
                # Revert violations
                for c_idx in edge_to_cycles[e_idx]:
                    update_violation(c_idx)
                # Note: update_violation returns diff, but we need to restore state.
                # Since we just reverted active, the state is back to before the try.
                # The violations array is now correct for the reverted state.
                
        else:
            # We have violations. Try to fix them.
            # Pick a random edge involved in a violation, or just a random edge to recolor/remove.
            # Strategy: Pick a random edge that is active.
            active_edges = [i for i in range(num_edges) if active[i]]
            if not active_edges:
                break
            e_idx = random.choice(active_edges)
            
            # Option 1: Recolor
            if random.random() < 0.8:
                old_color = color[e_idx]
                new_color = random.randint(0, 19)
                if new_color == old_color:
                    continue
                color[e_idx] = new_color
                
                delta_viol = 0
                for c_idx in edge_to_cycles[e_idx]:
                    delta_viol += update_violation(c_idx)
                new_violations = current_violations + delta_viol
                
                # Accept if better or with probability exp(-delta/T)
                if new_violations < current_violations:
                    current_violations = new_violations
                else:
                    # Revert
                    color[e_idx] = old_color
                    for c_idx in edge_to_cycles[e_idx]:
                        update_violation(c_idx)
            else:
                # Option 2: Remove edge
                old_active = active[e_idx]
                active[e_idx] = 0
                current_edges -= 1
                
                delta_viol = 0
                for c_idx in edge_to_cycles[e_idx]:
                    delta_viol += update_violation(c_idx)
                new_violations = current_violations + delta_viol
                
                # Removing an edge always reduces or keeps violations (since fewer cycles are active)
                # Actually, removing an edge makes any cycle containing it "inactive" and thus "valid" (not a violation).
                # So new_violations should be <= current_violations.
                current_violations = new_violations
                
                if current_edges > best_edges and current_violations == 0:
                    best_edges = current_edges
                    best_active = active[:]
                    best_color = color[:]
                    
        T *= cooling

    # Final Check: Return the best valid graph found (violations == 0)
    # If best_violations > 0, we must return a valid graph. 
    # If we never found a valid graph with violations==0, we should fall back to a known valid structure (e.g., C4-free graph).
    # But the SA should find one. If not, we return the best valid one.
    # If no valid one found, we return a star (valid).
    
    if best_violations > 0:
        # Fallback to a simple valid graph (Star)
        # Star on 20 vertices: 19 edges.
        return [[0, v, 0] for v in range(1, n)]
        
    # Construct output from best_active and best_color
    result = []
    for i in range(num_edges):
        if best_active[i]:
            u, v = all_edges[i]
            c = best_color[i]
            result.append([u, v, c])
            
    return result

# For testing
if __name__ == "__main__":
    start = time.time()
    result = construct()
    elapsed = time.time() - start
    print(f"Time: {elapsed:.2f}s, Edges: {len(result)}")
    
    # Quick validity check
    n = 20
    adj = [set() for _ in range(n)]
    edge_colors = {}
    for u, v, c in result:
        adj[u].add(v)
        adj[v].add(u)
        edge_colors[(u,v) if u<v else (v,u)] = c
        
    valid = True
    # Check all 4-cycles
    from itertools import combinations
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        # 3 cycles
        cycles = [
            ((a,b), (b,c), (c,d), (d,a)),
            ((a,b), (b,d), (d,c), (c,a)),
            ((a,c), (c,b), (b,d), (d,a))
        ]
        for cyc in cycles:
            # Check if all edges exist
            if all(edge_colors.get(e) is not None for e in cyc):
                cols = [edge_colors[e] for e in cyc]
                if len(set(cols)) != 4:
                    valid = False
                    break
        if not valid:
            break
            
    print(f"Valid Rainbow 4-cycles: {valid}")
