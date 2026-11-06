"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_not_2_colorable(n, edges):
    """Check if hypergraph is NOT 2-colorable using backtracking with unit propagation."""
    edge_set = [tuple(sorted(e)) for e in edges]
    
    # State: assignment[v] = -1 (unassigned), 0 or 1
    
    def backtrack2(assignment_state):
        """Backtracking with explicit state."""
        # Find first unassigned vertex
        idx = -1
        for i in range(n):
            if assignment_state[i] == -1:
                idx = i
                break
        if idx == -1:
            # All assigned, check
            for e in edge_set:
                colors = set(assignment_state[v] for v in e)
                if len(colors) == 1:
                    return False
            return True
        
        for color in [0, 1]:
            new_state = assignment_state[:]
            new_state[idx] = color
            # Unit propagation
            changed = True
            valid = True
            while changed and valid:
                changed = False
                for e in edge_set:
                    c0 = sum(1 for v in e if new_state[v] == 0)
                    c1 = sum(1 for v in e if new_state[v] == 1)
                    unassigned = [v for v in e if new_state[v] == -1]
                    
                    if c0 == 5 or c1 == 5:
                        valid = False
                        break
                    if c0 == 4 and len(unassigned) == 1:
                        new_state[unassigned[0]] = 1
                        changed = True
                    elif c1 == 4 and len(unassigned) == 1:
                        new_state[unassigned[0]] = 0
                        changed = True
            
            if valid:
                if backtrack2(new_state):
                    return True
        return False
    
    return not backtrack2([-1] * n)

def _greedy_construct(n, num_edges):
    """Construct a hypergraph greedily, adding edges that are hardest to 2-color."""
    edges = []
    all_edges = list(combinations(range(n), 5))
    random.shuffle(all_edges)
    
    for e in all_edges:
        if len(edges) >= num_edges:
            break
        candidate = edges + [e]
        if _is_not_2_colorable(n, candidate):
            edges = candidate
    
    return edges if len(edges) == num_edges else None

def construct():
    """Find a small non-2-colorable 5-uniform hypergraph."""
    start_time = time.time()
    deadline = 140  # Safety margin before 150s limit
    
    best_result = None
    best_edges = 999
    
    # Known lower bound: at most 28 edges is always 2-colorable
    # Best known: 51 edges
    # Try to find constructions with fewer edges
    
    # Strategy: Start with known small constructions and try to improve
    # First, try to find a valid construction with a reasonable number of edges
    for n in range(9, 18):
        if time.time() > deadline:
            break
        
        # Try different edge counts, starting from lower bounds
        # For n=9, we know 126 edges works (complete)
        # Try to find smaller edge counts
        
        # Use greedy construction to find minimal edge count
        for num_edges in range(30, min(127, n*(n-1)*(n-2)*(n-3)*(n-4)//120 + 1), 3):
            if time.time() > deadline:
                break
            
            # Try greedy construction
            result = _greedy_construct(n, num_edges)
            if result is not None and num_edges < best_edges:
                best_result = {'n': n, 'edges': [list(e) for e in result]}
                best_edges = num_edges
                break
    
    if best_result is not None:
        return best_result
    
    # Fallback: complete hypergraph on 9 vertices
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
# EVOLVE-BLOCK-END
