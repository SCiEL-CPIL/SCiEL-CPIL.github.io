"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_non_2colorable_fast(n, edges):
    """
    Fast check for non-2-colorability using bitmasks and backtracking.
    """
    if not edges:
        return False
        
    edge_set = [tuple(sorted(e)) for e in edges]
    edge_set = list(set(edge_set))
    if not edge_set:
        return False
        
    m = len(edge_set)
    vert_to_edges = [[] for _ in range(n)]
    for i, e in enumerate(edge_set):
        for v in e:
            vert_to_edges[v].append(i)
            
    # Precompute edge masks for quick count updates? 
    # Actually, counts are easier to update incrementally.
    
    def solve():
        colors = [-1] * n
        count_0 = [0] * m
        count_1 = [0] * m
        
        def backtrack():
            # Unit Propagation
            changed = True
            while changed:
                changed = False
                for i, e in enumerate(edge_set):
                    c0 = count_0[i]
                    c1 = count_1[i]
                    if c0 == 5 or c1 == 5:
                        return False
                    if c0 == 4 and c1 == 0:
                        # Find unassigned vertex in this edge
                        for v in e:
                            if colors[v] == -1:
                                colors[v] = 1
                                count_1[i] += 1
                                for j in vert_to_edges[v]:
                                    if j != i:
                                        count_1[j] += 1
                                changed = True
                                break
                    elif c1 == 4 and c0 == 0:
                        for v in e:
                            if colors[v] == -1:
                                colors[v] = 0
                                count_0[i] += 1
                                for j in vert_to_edges[v]:
                                    if j != i:
                                        count_0[j] += 1
                                changed = True
                                break
                                
            # Check if all assigned
            if all(c != -1 for c in colors):
                return True # A valid coloring was found
            
            # Branch
            # Heuristic: Pick vertex involved in most "critical" edges (4 assigned)
            unassigned = [v for v in range(n) if colors[v] == -1]
            if not unassigned:
                return True
                
            best_v = -1
            max_crit = 0
            for v in unassigned:
                crit = 0
                for j in vert_to_edges[v]:
                    c0 = count_0[j]
                    c1 = count_1[j]
                    if c0 + c1 == 4:
                        crit += 1
                if crit > max_crit:
                    max_crit = crit
                    best_v = v
                    
            if best_v == -1:
                best_v = unassigned[0]
                
            v = best_v
            
            # Save state
            old_colors = colors[:]
            old_c0 = count_0[:]
            old_c1 = count_1[:]
            
            # Try color 0
            colors[v] = 0
            for j in vert_to_edges[v]:
                count_0[j] += 1
            if backtrack():
                return True
                
            # Restore state
            colors[:] = old_colors
            count_0[:] = old_c0
            count_1[:] = old_c1
            
            # Try color 1
            colors[v] = 1
            for j in vert_to_edges[v]:
                count_1[j] += 1
            if backtrack():
                return True
                
            return False
            
        return backtrack()
        
    return not solve()

def construct():
    """
    Return a non-2-colorable 5-uniform hypergraph with minimal edges.
    Uses greedy edge deletion from a complete hypergraph to minimize edge count.
    """
    deadline = time.time() + 140
    best_m = float('inf')
    best_graph = None
    
    # Try different n values
    # Start with n=12 as it's a sweet spot for 5-uniform
    for n in range(10, 16):
        if time.time() > deadline - 10:
            break
            
        all_edges = list(combinations(range(n), 5))
        if len(all_edges) < 10:
            continue
        
        # Start with complete hypergraph
        current_edges = list(all_edges)
        
        # Greedy deletion
        # We want to remove edges while maintaining non-2-colorability
        # To speed up, we can try removing multiple edges at once or use a smarter heuristic
        # But simple random deletion is robust.
        
        # Shuffle edges to try removing in random order
        edge_indices = list(range(len(current_edges)))
        random.shuffle(edge_indices)
        
        # We'll iterate through the shuffled list and try to remove each edge
        # If removal keeps it non-2-colorable, we remove it permanently.
        # This is O(m * check_cost).
        
        # Optimization: We can restart this process a few times with different shuffles
        # if time permits.
        
        for _ in range(3): # 3 restarts of the deletion process
            if time.time() > deadline - 5:
                break
                
            # Reset to complete
            current_edges = list(all_edges)
            edge_indices = list(range(len(current_edges)))
            random.shuffle(edge_indices)
            
            # Try to remove each edge
            # We iterate backwards or just go through the list
            # If we remove an edge, we don't add it back in this pass.
            # This is a greedy "knapsack" like approach.
            
            # To make it faster, we can skip checking if the current size is already >= best_m
            # But we don't know best_m for this n yet.
            
            # Let's just do the pass.
            # Note: Removing edges makes the hypergraph sparser, which might make it 2-colorable.
            # So we check after each removal.
            
            # To speed up, we can try to remove a batch? No, one by one is safer.
            
            # We can also prioritize edges that are "redundant".
            # But random is fine for now.
            
            # We'll use a set for current edges for O(1) removal?
            # No, list is fine, we just mark as removed or build a new list.
            
            # Let's use a list of active indices.
            active = [True] * len(current_edges)
            
            for idx in edge_indices:
                if not active[idx]:
                    continue
                if time.time() > deadline - 5:
                    break
                    
                # Try removing edge idx
                active[idx] = False
                # Build list of active edges for checker
                # This is O(m) every time, which is slow.
                # Better: pass the list of active edges directly?
                # The checker takes a list of edges.
                
                # Optimization: Only check if the number of edges is still > some lower bound?
                # No, we want to minimize.
                
                # To avoid O(m) list construction, we can modify the checker to take a mask?
                # Or just accept the cost if m is small.
                # m starts at 792 for n=12.
                # 792 checks * 3 restarts * 5 n-values = ~12000 checks.
                # If each check is 1ms, that's 12s. Feasible.
                
                edges_to_check = [current_edges[i] for i in range(len(current_edges)) if active[i]]
                
                if _is_non_2colorable_fast(n, edges_to_check):
                    # Keep it removed
                    if len(edges_to_check) < best_m:
                        best_m = len(edges_to_check)
                        best_graph = {'n': n, 'edges': [list(e) for e in edges_to_check]}
                else:
                    # Add it back
                    active[idx] = True

    if best_graph:
        return best_graph
    else:
        # Fallback: Complete hypergraph on 9 vertices
        fallback_n = 9
        fallback_edges = [list(e) for e in combinations(range(fallback_n), 5)]
        return {'n': fallback_n, 'edges': fallback_edges}
# EVOLVE-BLOCK-END
