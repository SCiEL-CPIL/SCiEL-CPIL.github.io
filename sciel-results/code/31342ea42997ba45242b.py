"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_not_2_colorable(n, edges):
    # Exact check via backtracking with unit propagation.
    # We model each vertex color as 0/1. Each edge constraint is: not(all same).
    # We implement a simple recursive search with propagation.
    m = len(edges)
    if m == 0:
        return False  # trivially 2-colorable
    
    # Precompute edge membership for quick access
    edge_sets = [set(e) for e in edges]
    
    # We will maintain for each vertex: -1 unknown, 0 color0, 1 color1
    color = [-1] * n
    # For propagation: if an edge has 4 vertices of color 0, the 5th must be 1.
    # If an edge has 5 vertices of color 0 -> contradiction.
    
    def propagate(changes):
        # changes: list of (vertex, new_color)
        if not changes:
            return True
        queue = list(changes)
        in_queue = set(queue)
        while queue:
            v, c = queue.pop()
            in_queue.discard((v, c))
            # Check all edges containing v
            for e_set in edge_sets:
                if v not in e_set:
                    continue
                # Count colors in this edge
                cnt0 = 0
                cnt1 = 0
                unknowns = []
                for u in e_set:
                    if color[u] == 0:
                        cnt0 += 1
                    elif color[u] == 1:
                        cnt1 += 1
                    else:
                        unknowns.append(u)
                
                # Contradiction check
                if cnt0 == 5 or cnt1 == 5:
                    return False
                
                # Unit propagation
                if cnt0 == 4 and len(unknowns) == 1:
                    u = unknowns[0]
                    if color[u] == -1:
                        color[u] = 1
                        if (u, 1) not in in_queue:
                            queue.append((u, 1))
                            in_queue.add((u, 1))
                    elif color[u] == 0:
                        return False
                elif cnt1 == 4 and len(unknowns) == 1:
                    u = unknowns[0]
                    if color[u] == -1:
                        color[u] = 0
                        if (u, 0) not in in_queue:
                            queue.append((u, 0))
                            in_queue.add((u, 0))
                    elif color[u] == 1:
                        return False
        return True

    def backtrack():
        # Find first unknown vertex
        unassigned = [i for i in range(n) if color[i] == -1]
        if not unassigned:
            return True  # All assigned, check if valid (should be if no contradiction)
            # Actually, we need to ensure no edge is monochromatic.
            # But propagation handles 4-same. If all assigned, check edges.
            for e_set in edge_sets:
                c = color[e_set[0]]
                if all(color[u] == c for u in e_set):
                    return False
            return True
        
        # Heuristic: choose vertex with highest degree in remaining constraints? 
        # For simplicity, pick first.
        v = unassigned[0]
        
        # Try color 0
        old = color[v]
        color[v] = 0
        # We need to check if this assignment leads to contradiction immediately or via propagation
        # Save state for backtracking
        saved_colors = color[:]
        
        if propagate([(v, 0)]):
            if backtrack():
                return True
        
        # Restore
        color[:] = saved_colors
        color[v] = 1
        if propagate([(v, 1)]):
            if backtrack():
                return True
        
        # Restore
        color[:] = saved_colors
        color[v] = -1
        return False

    return not backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Start with a small random hypergraph
    # Try to find a minimal non-2-colorable hypergraph
    # Strategy: Random restarts with greedy edge addition
    
    best_result = None
    best_edges_count = float('inf')
    
    # Use a fixed seed for reproducibility? No, random is allowed.
    # Let's try a few random seeds.
    
    for seed in range(100):
        if time.time() > deadline - 5.0:
            break
            
        random.seed(seed)
        n = random.randint(10, 15)  # Start with moderate n
        # Initial edges: random few
        all_possible = list(combinations(range(n), 5))
        random.shuffle(all_possible)
        
        current_edges = []
        # Start with 0 edges
        # Greedy addition: add edge that breaks most colorings
        
        # We need a metric for "breaks colorings". 
        # Exact count is expensive. 
        # Approximation: Add an edge if it creates a unit propagation chain?
        # Or just random addition until not 2-colorable?
        
        # Better: Start with a known small non-2-colorable core?
        # The Fano plane is 7-uniform? No, Fano is 3-uniform.
        # For 5-uniform, the complete graph on 9 vertices is non-2-colorable.
        # Can we find a subset?
        
        # Let's try a local search:
        # 1. Start with random set of edges
        # 2. Check if 2-colorable.
        # 3. If not, try to remove edges.
        # 4. If yes, add edges.
        
        # Initialize with a random subset of size ~30
        k = random.randint(20, 40)
        current_edges = random.sample(all_possible, min(k, len(all_possible)))
        
        # Check
        if not _is_not_2_colorable(n, current_edges):
            # Try to add edges
            remaining = [e for e in all_possible if e not in current_edges]
            random.shuffle(remaining)
            
            for e in remaining:
                if time.time() > deadline - 2.0:
                    break
                new_edges = current_edges + [e]
                if not _is_not_2_colorable(n, new_edges):
                    current_edges = new_edges
                    break
        
        if not _is_not_2_colorable(n, current_edges):
            # Try to remove edges
            # Sort edges by some heuristic? Random order
            idxs = list(range(len(current_edges)))
            random.shuffle(idxs)
            
            for i in idxs:
                if time.time() > deadline - 2.0:
                    break
                # Try removing edge i
                new_edges = [current_edges[j] for j in range(len(current_edges)) if j != i]
                if not _is_not_2_colorable(n, new_edges):
                    current_edges = new_edges
                    # Restart removal loop? Or continue?
                    # Continue is safer to find local min
                    idxs = list(range(len(current_edges)))
                    random.shuffle(idxs)
                    # Reset loop? This is tricky in a single loop.
                    # Let's use a while loop for removal.
                    break
            
            # Repeat removal until no single edge can be removed
            # This can be expensive, so limit iterations
            removals = 0
            while removals < 100:
                if time.time() > deadline - 2.0:
                    break
                improved = False
                idxs = list(range(len(current_edges)))
                random.shuffle(idxs)
                for i in idxs:
                    if time.time() > deadline - 2.0:
                        break
                    new_edges = [current_edges[j] for j in range(len(current_edges)) if j != i]
                    if not _is_not_2_colorable(n, new_edges):
                        current_edges = new_edges
                        improved = True
                        break
                if not improved:
                    break
                removals += 1
        
        if not _is_not_2_colorable(n, current_edges):
            if len(current_edges) < best_edges_count:
                best_edges_count = len(current_edges)
                best_result = {'n': n, 'edges': [list(e) for e in current_edges]}
                
    if best_result:
        return best_result
    else:
        # Fallback: Return a known valid one (complete 9) if search fails
        # But we need to ensure it's valid.
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
# EVOLVE-BLOCK-END
