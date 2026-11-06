import time
import random
from itertools import combinations

def is_2_colorable_fast(n, edges):
    """
    Check if a 5-uniform hypergraph is 2-colorable.
    Returns True if 2-colorable, False otherwise.
    Uses backtracking with unit propagation and a trail for efficient undo.
    """
    if not edges:
        return True
    
    num_edges = len(edges)
    
    # Precompute edge data
    edge_vertices = [list(e) for e in edges]
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edge_vertices):
        for v in e:
            vertex_edges[v].append(i)
    
    # State
    c0 = [0] * num_edges
    c1 = [0] * num_edges
    color = [-1] * n
    
    # Trail for undo
    trail = [] # List of (vertex, old_color, edge_indices_affected, old_c0s, old_c1s) ? 
    # Simpler: just record (vertex, color) and re-calculate? No, that's slow.
    # Standard approach: Record (vertex, new_color) and the list of edges whose counts changed?
    # Even simpler for 5-uniform: 
    # When we assign v to c, we update counts for all edges incident to v.
    # To undo, we just decrement those counts.
    # We don't need to save old counts if we know which edges were touched.
    
    def propagate():
        # Returns False if conflict, True otherwise.
        # Uses a stack of edges to check
        # We need to handle unit propagation: if 4 are 0, 5th must be 1.
        
        # We'll use a queue of edges to check.
        # Initially, we don't have a specific trigger, so we check all? 
        # No, propagation is triggered by assignments.
        # In this implementation, we call propagate after each assignment in backtrack.
        # But unit propagation can cascade.
        
        # Let's implement a simple loop that checks all edges? 
        # That is O(M) per step. 
        # Better: maintain a list of "dirty" edges.
        
        # For simplicity and speed in Python, let's stick to the current structure but optimize the undo.
        # The current code copies arrays. That is slow.
        
        # Let's use a recursive backtrack with explicit undo of counts.
        
        # We'll modify the state directly and undo by decrementing.
        
        # To find the unassigned vertex in an edge quickly:
        # We can just loop through the 5 vertices.
        
        # We'll implement a stack-based propagation.
        # However, implementing a full incremental SAT solver with trail is complex.
        # Given the small size (n<=26, m<=200), the current O(M) copy might be acceptable if the search tree is small.
        # But if the search tree is large, it's too slow.
        
        # Let's try to optimize the current code's undo mechanism.
        # Instead of copying c0, c1, we can just decrement the counts for the edges incident to the vertex.
        
        return True # Placeholder, logic below

    # The current code's `propagate` is called after assignment.
    # It uses a stack.
    # The issue is `saved_c0 = c0[:]`.
    
    # Let's rewrite the core search to use a trail of (vertex, color) and just decrement counts on undo.
    
    def backtrack():
        # Find most constrained edge
        best_edge = -1
        best_unassigned = 6
        best_max_color = 0
        
        for i in range(num_edges):
            if c0[i] == 5 or c1[i] == 5:
                return False
            unassigned = 5 - c0[i] - c1[i]
            max_color = max(c0[i], c1[i])
            
            if unassigned < best_unassigned or (unassigned == best_unassigned and max_color > best_max_color):
                best_unassigned = unassigned
                best_max_color = max_color
                best_edge = i
        
        if best_edge == -1:
            return True
            
        if best_unassigned == 0:
            return False

        # Get unassigned vertices in best_edge
        candidates = []
        for v in edge_vertices[best_edge]:
            if color[v] == -1:
                candidates.append(v)
                
        if not candidates:
            return True
        
        # Heuristic: try color 0 then 1, or random?
        # Randomization helps escape symmetry.
        random.shuffle(candidates)
        
        for v in candidates:
            # Try color 0
            color[v] = 0
            affected_edges = vertex_edges[v]
            for e_idx in affected_edges:
                c0[e_idx] += 1
                if c0[e_idx] == 5:
                    break
            else:
                # Check for propagation/conflicts
                if propagate():
                    if backtrack():
                        return True
            # Undo
            for e_idx in affected_edges:
                c0[e_idx] -= 1
            color[v] = -1
            
            # Try color 1
            color[v] = 1
            for e_idx in affected_edges:
                c1[e_idx] += 1
                if c1[e_idx] == 5:
                    break
            else:
                if propagate():
                    if backtrack():
                        return True
            # Undo
            for e_idx in affected_edges:
                c1[e_idx] -= 1
            color[v] = -1
            
        return False
        
    def propagate():
        # Check for unit clauses and conflicts
        # We need to check if any edge has 4 of one color.
        # To do this efficiently, we should only check edges that were affected by recent changes.
        # However, since we don't track "recent" easily in this simple structure, 
        # and n is small, we can scan all edges.
        # M is at most 200. Scanning 200 edges is fast.
        
        for i in range(num_edges):
            if c0[i] == 5 or c1[i] == 5:
                return False
            
            # Unit propagation
            if c0[i] == 4 and c1[i] == 0:
                # Find unassigned
                unassigned_v = -1
                for v in edge_vertices[i]:
                    if color[v] == -1:
                        unassigned_v = v
                        break
                if unassigned_v != -1:
                    # Assign to 1
                    color[unassigned_v] = 1
                    c1[i] += 1
                    # Update other edges
                    for other_e in vertex_edges[unassigned_v]:
                        c1[other_e] += 1
                        if c1[other_e] == 5:
                            return False
                    # We must continue propagating from this new assignment
                    # This simple loop doesn't handle cascading well without a stack.
                    # Let's use a stack.
        
        # The above simple loop is flawed for cascading.
        # Let's use a stack for propagation.
        
        # Re-implement propagate with stack
        # Since we are inside a function that is called repeatedly, 
        # and we modify global state, we need to be careful.
        
        # Actually, the previous code's approach with a stack was better.
        # Let's stick to the previous logic but fix the undo.
        
        # The previous code:
        # 1. Assigns v.
        # 2. Calls propagate().
        # 3. Propagate uses a stack to find forced assignments.
        # 4. If success, recurse.
        # 5. Undo by restoring c0, c1, color.
        
        # The bottleneck was `c0[:]`.
        # We can avoid copying by just decrementing on undo.
        
        # So, we need `propagate` to also record what it did so we can undo it?
        # Or, we can just let `backtrack` handle the undo of the *initial* assignment,
        # and `propagate` just returns True/False, and if True, we recurse.
        # But `propagate` makes changes to `color`, `c0`, `c1`.
        # If `backtrack` returns False, we need to undo *all* changes made by `propagate` and the initial assignment.
        
        # This is tricky without a trail.
        
        # Alternative: Use a trail.
        # Trail entries: (vertex, old_color)
        # When we assign, we push to trail.
        # When we undo, we pop and restore.
        # But we also need to restore c0, c1.
        # If we record the edges affected and the delta, we can restore.
        
        # Let's use a simpler approach:
        # Since the graph is small, copying might actually be faster in Python than complex trail management if the depth is small.
        # But if depth is large, it's slow.
        
        # Let's try to optimize the existing code by using `array` or `list` slicing more efficiently?
        # Or just accept the copy but make the search smarter.
        
        # Given the time constraint, I will provide a version that uses a **Trail** for the colors and counts.
        
        # State:
        # color: list of int
        # c0, c1: list of int
        # trail: list of (vertex, color, edge_indices, old_c0s, old_c1s) is too big.
        
        # Better Trail:
        # Each assignment of vertex v to color c affects edges E_v.
        # We push (v, c, E_v) to trail.
        # To undo: for each e in E_v, decrement c0 or c1. Set color[v] = -1.
        
        # This works!
        
        pass

    # I will implement the Trail-based checker.
    
    trail = []
    
    def assign(v, c):
        color[v] = c
        affected = vertex_edges[v]
        trail.append((v, c, affected))
        for e in affected:
            if c == 0:
                c0[e] += 1
            else:
                c1[e] += 1
            if (c == 0 and c0[e] == 5) or (c == 1 and c1[e] == 5):
                return False
        return True

    def undo_assign():
        v, c, affected = trail.pop()
        for e in affected:
            if c == 0:
                c0[e] -= 1
            else:
                c1[e] -= 1
        color[v] = -1

    def propagate():
        # Returns False if conflict, True otherwise.
        # We need to check all edges for unit clauses.
        # To be efficient, we should only check edges that might have changed.
        # But without tracking, we scan all.
        
        # Optimization: Only scan edges if we just assigned something?
        # Yes, propagate is called after assign.
        
        # We'll use a stack of edges to check.
        # Initially, we don't know which edges are "dirty" from the last assign.
        # But we know `affected` from the last assign?
        # No, `assign` doesn't return the list.
        
        # Let's modify `assign` to return the affected edges?
        # Or just scan all edges. M=200 is small.
        
        for i in range(num_edges):
            if c0[i] == 5 or c1[i] == 5:
                return False
            
            if c0[i] == 4 and c1[i] == 0:
                # Find unassigned
                unassigned_v = -1
                for v in edge_vertices[i]:
                    if color[v] == -1:
                        unassigned_v = v
                        break
                if unassigned_v != -1:
                    if not assign(unassigned_v, 1):
                        return False
                    # Recurse? No, just continue loop?
                    # The new assignment might force others.
                    # We should restart the scan or use a stack.
                    # Let's use a stack.
        
        # The simple loop above doesn't handle cascading properly if the new assignment creates another unit clause.
        # We need a stack.
        
        # Let's implement propagate with a stack of edges to check.
        # We'll push all edges initially? No, that's O(M).
        # We'll push edges affected by the last assignment?
        # We don't have that info here.
        
        # Let's just scan all edges in a loop until no changes.
        # Since n is small, this should be fine.
        
        changed = True
        while changed:
            changed = False
            for i in range(num_edges):
                if c0[i] == 5 or c1[i] == 5:
                    return False
                
                if c0[i] == 4 and c1[i] == 0:
                    unassigned_v = -1
                    for v in edge_vertices[i]:
                        if color[v] == -1:
                            unassigned_v = v
                            break
                    if unassigned_v != -1:
                        if not assign(unassigned_v, 1):
                            return False
                        changed = True
                        # Break to restart scan? Or continue?
                        # Restarting is safer.
                        break
                elif c1[i] == 4 and c0[i] == 0:
                    unassigned_v = -1
                    for v in edge_vertices[i]:
                        if color[v] == -1:
                            unassigned_v = v
                            break
                    if unassigned_v != -1:
                        if not assign(unassigned_v, 0):
                            return False
                        changed = True
                        break
        
        return True

    def backtrack():
        # Find most constrained edge
        best_edge = -1
        best_unassigned = 6
        best_max_color = 0
        
        for i in range(num_edges):
            if c0[i] == 5 or c1[i] == 5:
                return False
            unassigned = 5 - c0[i] - c1[i]
            max_color = max(c0[i], c1[i])
            
            if unassigned < best_unassigned or (unassigned == best_unassigned and max_color > best_max_color):
                best_unassigned = unassigned
                best_max_color = max_color
                best_edge = i
        
        if best_edge == -1:
            return True
            
        if best_unassigned == 0:
            return False

        candidates = []
        for v in edge_vertices[best_edge]:
            if color[v] == -1:
                candidates.append(v)
                
        if not candidates:
            return True
        
        random.shuffle(candidates)
        
        for v in candidates:
            # Try 0
            if assign(v, 0) and propagate():
                if backtrack():
                    return True
            undo_assign()
            
            # Try 1
            if assign(v, 1) and propagate():
                if backtrack():
                    return True
            undo_assign()
            
        return False

    return backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_edges = None
    best_n = 0
    best_m = 10**9
    
    for n in range(9, 15):
        if time.time() > deadline:
            break
            
        all_possible = list(combinations(range(n), 5))
        
        if is_2_colorable_fast(n, [list(e) for e in all_possible]):
            continue
            
        current_edges = [list(e) for e in all_possible]
        current_m = len(current_edges)
        
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            order = list(range(len(current_edges)))
            random.shuffle(order)
            
            improved = True
            while improved:
                improved = False
                for idx in order:
                    if time.time() > deadline:
                        break
                    if idx >= len(current_edges):
                        continue
                        
                    new_edges = current_edges[:idx] + current_edges[idx+1:]
                    
                    if not is_2_colorable_fast(n, new_edges):
                        current_edges = new_edges
                        current_m = len(current_edges)
                        improved = True
                        order = list(range(len(current_edges)))
                        random.shuffle(order)
                        break
        
        if current_m < best_m:
            best_m = current_m
            best_edges = current_edges[:]
            best_n = n
    
    if best_edges is None:
        n = 9
        best_edges = [list(e) for e in combinations(range(n), 5)]
        best_n = n
        
    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    res = construct()
    print(f"Found hypergraph with n={res['n']}, m={len(res['edges'])}")
