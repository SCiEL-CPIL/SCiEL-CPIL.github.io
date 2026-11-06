"""Direct construction of a small 5-uniform hypergraph without Property B."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

# Deadline guard: kill long-running work well before the evaluator limit.
_START_TIME = time.time()
_TIME_LIMIT = 115.0  # safety margin under 120s

def construct():
    """
    Return a 5-uniform hypergraph on n vertices (5 <= n <= 26, <=200 edges)
    that has no proper 2-colouring, minimizing the number of edges.
    
    We use a constructive approach:
    1. Start with the complete 5-graph on 9 vertices (126 edges) as a guaranteed
       non-2-colourable base.
    2. Greedily remove edges that are 'redundant' — i.e., whose removal leaves
       the hypergraph still non-2-colourable.
    
    To check non-2-colourability efficiently, we use a SAT-style backtracking
    with unit propagation. For each edge (5 vertices), the constraint is that
    the 5 vertices cannot be all same colour, i.e., at least one is 0 and at
    least one is 1. This is equivalent to: NOT(all 0) AND NOT(all 1), which
    in CNF is (v1∨v2∨v3∨v4∨v5) ∧ (¬v1∨¬v2∨¬v3∨¬v4∨¬v5).
    
    However, for the greedy removal, we can use a faster heuristic check:
    we try to find a 2-colouring via backtracking with unit propagation.
    If no colouring exists, the hypergraph is non-2-colourable.
    """
    # Base: complete 5-uniform hypergraph on 9 vertices
    n = 9
    all_edges = [list(e) for e in combinations(range(n), 5)]
    edges = list(all_edges)  # mutable list
    
    # Helper: check if hypergraph (n, edges) has a proper 2-colouring
    # Returns True if it IS 2-colourable (i.e., a valid colouring exists)
    def is_2_colorable(n, edges):
        if not edges:
            return True
        
        # Build adjacency: for each vertex, list of edge indices containing it
        vert_edges = [[] for _ in range(n)]
        for ei, e in enumerate(edges):
            for v in e:
                vert_edges[v].append(ei)
        
        # State for incremental propagation
        assignment = [-1] * n
        edge_state = [[0, 0, 5] for _ in range(len(edges))]
        
        def propagate():
            changed = True
            while changed:
                changed = False
                for ei, e in enumerate(edges):
                    c0, c1, c_un = edge_state[ei]
                    
                    if c_un == 0:
                        if c0 == 5 or c1 == 5:
                            return False
                        continue
                    
                    if c_un == 1:
                        if c0 == 4:
                            for v in e:
                                if assignment[v] == -1:
                                    assignment[v] = 1
                                    for ei2 in vert_edges[v]:
                                        edge_state[ei2][1] += 1
                                        edge_state[ei2][2] -= 1
                                    changed = True
                                    break
                        elif c1 == 4:
                            for v in e:
                                if assignment[v] == -1:
                                    assignment[v] = 0
                                    for ei2 in vert_edges[v]:
                                        edge_state[ei2][0] += 1
                                        edge_state[ei2][2] -= 1
                                    changed = True
                                    break
            return True

        import sys
        sys.setrecursionlimit(10000)
        
        def backtrack():
            if time.time() - _START_TIME > _TIME_LIMIT:
                return True
            
            if not propagate():
                return False
            
            unassigned = [i for i in range(n) if assignment[i] == -1]
            if not unassigned:
                for ei, e in enumerate(edges):
                    c0, c1, c_un = edge_state[ei]
                    if c_un == 0 and (c0 == 5 or c1 == 5):
                        return False
                return True
            
            # Heuristic: pick vertex with highest degree
            best_v = max(unassigned, key=lambda v: len(vert_edges[v]))
            
            # Try 0
            assignment[best_v] = 0
            for ei in vert_edges[best_v]:
                edge_state[ei][0] += 1
                edge_state[ei][2] -= 1
            
            if backtrack():
                return True
            
            # Undo 0
            assignment[best_v] = -1
            for ei in vert_edges[best_v]:
                edge_state[ei][0] -= 1
                edge_state[ei][2] += 1
            
            # Try 1
            assignment[best_v] = 1
            for ei in vert_edges[best_v]:
                edge_state[ei][1] += 1
                edge_state[ei][2] -= 1
            
            if backtrack():
                return True
            
            # Undo 1
            assignment[best_v] = -1
            for ei in vert_edges[best_v]:
                edge_state[ei][1] -= 1
                edge_state[ei][2] += 1
            
            return False
        
        return backtrack()
    
    # Greedy removal: try to remove each edge if the hypergraph remains non-2-colourable
    # Process edges in a deterministic order
    remaining = list(range(len(edges)))
    removed = set()
    
    # Sort edges by some heuristic (e.g., original index)
    for idx in range(len(edges)):
        if idx in removed:
            continue
        if time.time() - _START_TIME > _TIME_LIMIT - 5:
            break
        
        # Try removing this edge
        candidate_edges = [edges[i] for i in remaining if i != idx and i not in removed]
        
        # Quick check: if candidate_edges is empty, removing leaves 0 edges -> 2-colourable
        if not candidate_edges:
            continue
        
        # Check if still non-2-colourable
        if not is_2_colorable(n, candidate_edges):
            removed.add(idx)
    
    final_edges = [edges[i] for i in range(len(edges)) if i not in removed]
    
    # Verify the result is valid
    assert 5 <= n <= 26
    assert len(final_edges) <= 200
    assert len(set(tuple(sorted(e)) for e in final_edges)) == len(final_edges)
    for e in final_edges:
        assert len(e) == 5
        assert len(set(e)) == 5
        assert all(0 <= v < n for v in e)
    
    # Final verification that it's NOT 2-colourable
    if is_2_colorable(n, final_edges):
        # Fallback: return the complete graph
        return {'n': n, 'edges': all_edges}
    
    return {'n': n, 'edges': final_edges}
# EVOLVE-BLOCK-END
