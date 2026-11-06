"""Direct construction of a small 5-uniform hypergraph without Property B."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
import sys

# Deadline guard: kill long-running work well before the evaluator limit.
_START_TIME = time.time()
_TIME_LIMIT = 115.0  # safety margin under 120s

def construct():
    """
    Return a 5-uniform hypergraph on n vertices (5 <= n <= 26, <=200 edges)
    that has no proper 2-colouring, minimizing the number of edges.
    
    Strategy:
    1. Start with K_9^5 (126 edges).
    2. Use a fast backtracking solver with worklist-based unit propagation.
    3. Greedily remove edges that do not affect non-2-colorability.
    """
    n = 9
    all_edges = [tuple(e) for e in combinations(range(n), 5)]
    edges = list(all_edges)
    
    # Fast 2-colorability checker using worklist-based propagation
    def is_2_colorable(n, edges):
        if not edges:
            return True
        
        m = len(edges)
        
        # Precompute for each vertex, the list of edges it belongs to
        # And for each edge, the list of vertices it contains
        vert_edges = [[] for _ in range(n)]
        edge_verts = [list(e) for e in edges]
        
        for ei, e in enumerate(edges):
            for v in e:
                vert_edges[v].append(ei)
        
        # State:
        # color: -1 (unassigned), 0, 1
        color = [-1] * n
        # counts: for each edge, count of 0s and 1s
        c0 = [0] * m
        c1 = [0] * m
        # unassigned count per edge
        c_un = [5] * m
        
        # Worklist of edges that need checking
        worklist = []
        
        # Helper to assign a vertex, returns (success, list_of_affected_edges)
        def assign(v, c):
            # Returns (False, []) if conflict, (True, affected) otherwise
            affected = []
            for ei in vert_edges[v]:
                if c == 0:
                    c0[ei] += 1
                else:
                    c1[ei] += 1
                c_un[ei] -= 1
                
                # Check conflict: 5 of one color
                if c0[ei] == 5 or c1[ei] == 5:
                    return (False, [])
                
                affected.append(ei)
            
            color[v] = c
            return (True, affected)
        
        def propagate():
            """
            Worklist-based unit propagation.
            Returns False if conflict, True otherwise.
            """
            if not worklist:
                return True
            
            while worklist:
                ei = worklist.pop()
                
                # Check conflict
                if c0[ei] == 5 or c1[ei] == 5:
                    return False
                
                # Check unit constraints
                if c_un[ei] > 0:
                    e = edge_verts[ei]
                    
                    # Count unassigned
                    unassigned = []
                    for v in e:
                        if color[v] == -1:
                            unassigned.append(v)
                    
                    # If all remaining unassigned must be same color
                    if c0[ei] == 4 and c1[ei] == 0 and c_un[ei] == 1:
                        # The one unassigned must be 1
                        if len(unassigned) == 1:
                            v = unassigned[0]
                            if color[v] != -1:
                                if color[v] != 1:
                                    return False
                            else:
                                ok, aff = assign(v, 1)
                                if not ok:
                                    return False
                                worklist.extend(aff)
                    elif c1[ei] == 4 and c0[ei] == 0 and c_un[ei] == 1:
                        # The one unassigned must be 0
                        if len(unassigned) == 1:
                            v = unassigned[0]
                            if color[v] != -1:
                                if color[v] != 0:
                                    return False
                            else:
                                ok, aff = assign(v, 0)
                                if not ok:
                                    return False
                                worklist.extend(aff)
                    elif c0[ei] == 4 and c1[ei] == 0 and c_un[ei] > 1:
                        # All unassigned must be 1
                        for v in unassigned:
                            if color[v] == -1:
                                ok, aff = assign(v, 1)
                                if not ok:
                                    return False
                                worklist.extend(aff)
                    elif c1[ei] == 4 and c0[ei] == 0 and c_un[ei] > 1:
                        # All unassigned must be 0
                        for v in unassigned:
                            if color[v] == -1:
                                ok, aff = assign(v, 0)
                                if not ok:
                                    return False
                                worklist.extend(aff)
            
            return True
        
        # Backtracking with state saving
        def backtrack():
            # Check time
            if time.time() - _START_TIME > _TIME_LIMIT:
                return True  # Assume colourable to be safe
            
            if not propagate():
                return False
            
            # Check if all assigned
            unassigned = [i for i in range(n) if color[i] == -1]
            if not unassigned:
                # All edges should be satisfied (mixed) - already checked by propagate
                return True
            
            # Choose vertex: most constrained (involved in most edges with small c_un)
            # Score: sum over edges of (1 if edge has unassigned else 0), weighted by urgency
            best_v = -1
            best_score = -1
            for v in unassigned:
                score = 0
                for ei in vert_edges[v]:
                    if c_un[ei] > 0:
                        # Prioritize edges that are close to being forced
                        urgency = 5 - c_un[ei]
                        score += urgency * 10 + 1
                if score > best_score:
                    best_score = score
                    best_v = v
            
            # Save state
            color_backup = color[:]
            c0_backup = c0[:]
            c1_backup = c1[:]
            c_un_backup = c_un[:]
            worklist_backup = worklist[:]
            
            # Try color 0
            ok, aff = assign(best_v, 0)
            if ok:
                worklist.extend(aff)
                if backtrack():
                    return True
            
            # Restore
            color[:] = color_backup
            c0[:] = c0_backup
            c1[:] = c1_backup
            c_un[:] = c_un_backup
            worklist[:] = worklist_backup
            
            # Try color 1
            ok, aff = assign(best_v, 1)
            if ok:
                worklist.extend(aff)
                if backtrack():
                    return True
            
            return False
        
        # Initialize worklist with all edges
        worklist = list(range(m))
        return backtrack()

    # Greedy removal
    # We maintain a list of active edge indices
    active = list(range(len(edges)))
    
    # We iterate through the active list. If we remove an edge, we continue.
    # To be more effective, we can shuffle the order or try multiple passes.
    
    passes = 0
    while True:
        passes += 1
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
        
        removed_in_pass = False
        new_active = []
        
        for idx in active:
            if time.time() - _START_TIME > _TIME_LIMIT - 5:
                break
            
            # Try removing edge idx
            candidate_active = [i for i in active if i != idx]
            
            # Check if hypergraph with candidate_active is 2-colorable
            cand_edges = [edges[i] for i in candidate_active]
            
            if not is_2_colorable(n, cand_edges):
                # It is NOT 2-colorable, so we can remove idx
                removed_in_pass = True
                # Do not add idx to new_active
            else:
                # It IS 2-colorable, so we must keep idx
                new_active.append(idx)
        
        if not removed_in_pass:
            break
        
        active = new_active
    
    final_edges = [edges[i] for i in active]
    
    # Verify
    assert 5 <= n <= 26
    assert len(final_edges) <= 200
    assert len(set(tuple(sorted(e)) for e in final_edges)) == len(final_edges)
    for e in final_edges:
        assert len(e) == 5
        assert len(set(e)) == 5
        assert all(0 <= v < n for v in e)
    
    # Final check
    if is_2_colorable(n, final_edges):
        # Fallback
        return {'n': n, 'edges': all_edges}
    
    return {'n': n, 'edges': final_edges}
# EVOLVE-BLOCK-END
