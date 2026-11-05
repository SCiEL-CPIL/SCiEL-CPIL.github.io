import random
import time
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Checks if the hypergraph (n, edges) is 2-colorable.
    Returns True if 2-colorable, False otherwise.
    """
    if not edges:
        return True
        
    # Preprocess edges: convert to frozensets for faster lookup? 
    # Or keep as tuples.
    # We need to track which vertices are in which edges for propagation.
    
    # Map vertex to list of edge indices
    vertex_to_edges = {v: [] for v in range(n)}
    for i, e in enumerate(edges):
        for v in e:
            vertex_to_edges[v].append(i)
            
    # State: colors[v] = 0, 1, or -1 (unassigned)
    colors = [-1] * n
    
    # Queue for unit propagation
    # When a vertex is assigned, check all its edges.
    # If an edge has 4 assigned vertices, the 5th is determined.
    # If an edge has 5 assigned vertices, check if sum is 1..4.
    
    def propagate(v, c):
        """
        Assign color c to vertex v.
        Returns False if contradiction found.
        Modifies colors and queue.
        """
        if colors[v] != -1:
            if colors[v] != c:
                return False
            else:
                return True
                
        colors[v] = c
        queue = [(v, c)]
        
        while queue:
            u, cu = queue.pop(0)
            for e_idx in vertex_to_edges[u]:
                e = edges[e_idx]
                # Check this edge
                assigned = 0
                sum_c = 0
                unassigned_v = -1
                unassigned_count = 0
                
                for w in e:
                    if colors[w] != -1:
                        assigned += 1
                        sum_c += colors[w]
                    else:
                        unassigned_count += 1
                        if unassigned_count == 1:
                            unassigned_v = w
                        else:
                            unassigned_v = -2 # more than 1 unassigned
                            
                if assigned == 5:
                    # All assigned
                    if sum_c == 0 or sum_c == 5:
                        return False
                elif assigned == 4:
                    # One unassigned
                    if unassigned_v != -1:
                        # Determine color of unassigned_v
                        # sum_c + c_unassigned must be in {1, 2, 3, 4}
                        # So c_unassigned cannot be -sum_c mod 2?
                        # No, it's binary.
                        # If sum_c == 0, c_unassigned must be 1 (sum 1)
                        # If sum_c == 1, c_unassigned can be 0 (sum 1) or 1 (sum 2) -> No constraint?
                        # Wait. The condition is NOT (0) and NOT (5).
                        # So sum must be 1, 2, 3, 4.
                        # If sum_c == 0, c_unassigned MUST be 1. (0+1=1 ok, 0+0=0 bad)
                        # If sum_c == 4, c_unassigned MUST be 0. (4+0=4 ok, 4+1=5 bad)
                        # If sum_c == 1, c_unassigned can be 0 (sum 1) or 1 (sum 2). No forced move.
                        # If sum_c == 2, c_unassigned can be 0 (sum 2) or 1 (sum 3). No forced move.
                        # If sum_c == 3, c_unassigned can be 0 (sum 3) or 1 (sum 4). No forced move.
                        
                        if sum_c == 0:
                            if colors[unassigned_v] != -1:
                                if colors[unassigned_v] != 1: return False
                            else:
                                colors[unassigned_v] = 1
                                queue.append((unassigned_v, 1))
                        elif sum_c == 4:
                            if colors[unassigned_v] != -1:
                                if colors[unassigned_v] != 0: return False
                            else:
                                colors[unassigned_v] = 0
                                queue.append((unassigned_v, 0))
                        else:
                            # No forced move, but check if unassigned is already assigned?
                            # Handled above.
                            pass
                            
        return True

    # Backtracking
    def backtrack(idx):
        if idx == n:
            return True
            
        v = idx
        # Try color 0
        colors_backup = colors[:]
        if propagate(v, 0):
            if backtrack(idx + 1):
                return True
        colors[:] = colors_backup
        
        # Try color 1
        colors_backup = colors[:]
        if propagate(v, 1):
            if backtrack(idx + 1):
                return True
        colors[:] = colors_backup
        
        return False

    # Heuristic: Order vertices by degree (descending)
    # But for simplicity, let's just use 0..n-1.
    # Actually, ordering by degree helps propagation.
    # Let's stick to simple 0..n-1 for now, but maybe optimize if slow.
    
    return backtrack(0)

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s limit, 20s margin
    
    best_edges = None
    best_n = 9
    best_edge_count = 1000
    
    # Initial solution: Complete graph on 9 vertices
    # This is definitely non-2-colorable
    init_edges = [list(e) for e in combinations(range(9), 5)]
    if not is_2_colorable(9, init_edges):
        best_edges = init_edges
        best_n = 9
        best_edge_count = len(init_edges)
        
    # Try to improve by pruning
    # We will run a loop that tries to prune the best solution found so far
    # And also try new random initializations
    
    while time.time() < deadline:
        # 1. Prune current best
        if best_edges:
            current_edges = list(best_edges)
            current_n = best_n
            
            # Greedy deletion
            changed = True
            while changed and time.time() < deadline:
                changed = False
                # Shuffle to avoid bias
                random.shuffle(current_edges)
                for i in range(len(current_edges)):
                    if time.time() > deadline: break
                    e_to_remove = current_edges[i]
                    trial_edges = current_edges[:i] + current_edges[i+1:]
                    if not is_2_colorable(current_n, trial_edges):
                        # Still non-2-colorable, so removal is valid
                        current_edges = trial_edges
                        changed = True
                        # Update best if better
                        if len(current_edges) < best_edge_count:
                            best_edges = list(current_edges)
                            best_edge_count = len(current_edges)
                        break # Restart shuffle after a successful removal? Or continue?
                        # Breaking restarts the loop with the new list.
            
            # If no changes, try a random restart or new initialization
            # To escape local minima, we can try adding a random edge and then pruning
            # But adding edges increases count, so we need to remove more later.
            
        # 2. Try a new random initialization
        # Pick a random n between 10 and 26
        n = random.randint(10, 26)
        edges = []
        # Add random edges until non-2-colorable
        max_edges_to_add = 200
        for _ in range(max_edges_to_add):
            if time.time() > deadline: break
            e = list(random.sample(range(n), 5))
            edges.append(e)
            if not is_2_colorable(n, edges):
                # Found a non-2-colorable one
                if len(edges) < best_edge_count:
                    best_edges = list(edges)
                    best_n = n
                    best_edge_count = len(edges)
                break
        # If we didn't find one in max_edges, discard this n
        
    if best_edges is None:
        # Fallback to initial
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
        
    return {'n': best_n, 'edges': best_edges}
