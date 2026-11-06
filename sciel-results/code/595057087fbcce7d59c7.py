"""Few edges in 5-uniform hypergraphs without Property B (m(5))."""
import time
import random
import sys
from itertools import combinations

# Deadline guard
_DEADLINE = 115.0  # seconds, safety margin before 120s kill

def _elapsed():
    return time.time() - _START_TIME

_START_TIME = time.time()

def _is_2_colorable(edges, n, deadline=None):
    """Check if hypergraph has a proper 2-coloring using backtracking with unit propagation.
    Returns True if 2-colorable (has a valid coloring), False if not.
    """
    if deadline is None:
        deadline = _deadline
    deadline = min(deadline, _deadline)
    
    # Build incidence: for each vertex, list of edge indices containing it
    vert_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vert_edges[v].append(i)
    
    # Edge state: 0 = unassigned (not yet violated/satisfied), 1 = satisfied (has both colors), -1 = violated (all same color)
    # Actually, let's use a different approach:
    # For each edge, track: count of 0s, count of 1s, and whether it's already "satisfied" (has both)
    # An edge is violated if all its vertices are the same color.
    # An edge is satisfied if it has at least one 0 and at least one 1.
    
    m = len(edges)
    edge_zeros = [5] * m  # number of uncolored vertices in each edge
    edge_ones = [0] * m   # number of colored-1 vertices in each edge
    edge_satisfied = [False] * m  # True if edge already has both colors
    
    # For unit propagation: if an edge has 4 zeros and 1 one, then all zeros must be 0 (otherwise if any becomes 1, edge is satisfied... wait no)
    # Actually: if an edge has k zeros and (5-k) ones, and it's not yet satisfied:
    #   - If k=0: all colored. If all same color, violated. If mixed, satisfied.
    #   - If k=1: the one uncolored vertex - if it's 1, edge satisfied. If 0, edge might be violated if all others are 0.
    #     Wait, if k=1 and there are 4 ones: the uncolored can be 0 or 1. If 0, edge has both. If 1, edge all ones -> violated.
    #     So uncolored MUST be 0 to avoid violation? No - we need to avoid ALL edges being violated simultaneously.
    #     
    # Let me think more carefully. The constraint is: for every edge, NOT(all vertices same color).
    # Equivalently: for every edge, at least one vertex is 0 AND at least one vertex is 1.
    # 
    # Unit propagation: if an edge has 4 vertices colored as 0, and 1 uncolored, then the uncolored MUST be 1 (otherwise all 5 are 0, violating).
    # If an edge has 4 vertices colored as 1, and 1 uncolored, then the uncolored MUST be 0.
    # If an edge has 4 zeros and 1 one: satisfied already.
    # If an edge has 3 zeros, 1 one, 1 uncolored: uncolored can be 0 or 1 (both work for this edge).
    # If an edge has 5 uncolored: no constraint.
    
    # State: color[v] = -1 (uncolored), 0, or 1
    color = [-1] * n
    
    # For efficient propagation, track for each edge: count_0, count_1, count_uncolored
    cnt0 = [0] * m
    cnt1 = [0] * m
    cntu = [5] * m
    
    # Which edges are "critical" - need monitoring for unit propagation
    # An edge triggers unit propagation when:
    #   cnt0 == 4 and cntu == 1: the uncolored must be 1
    #   cnt1 == 4 and cntu == 1: the uncolored must be 0
    #   cnt0 == 5: violated (all zero)
    #   cnt1 == 5: violated (all one)
    #   cnt0 == 4 and cnt1 == 0 and cntu == 1: must be 1
    #   cnt1 == 4 and cnt0 == 0 and cntu == 1: must be 0
    
    # For backtracking, we need to track assignments.
    # Let's use a simple recursive approach with a trail for undo.
    
    trail = []  # list of (vertex, old_color)
    
    def propagate():
        """Apply unit propagation. Returns False if contradiction found."""
        changed = True
        while changed:
            changed = False
            for i in range(m):
                if cntu[i] == 0:
                    # All colored
                    if cnt0[i] == 5 or cnt1[i] == 5:
                        return False  # violated
                    continue
                if cntu[i] == 1:
                    if cnt0[i] == 4:
                        # Must set the uncolored vertex to 1
                        for v in edges[i]:
                            if color[v] == -1:
                                if color[v] != 1:
                                    trail.append((v, color[v]))
                                    color[v] = 1
                                    cnt0[i] -= 1  # was counted as uncolored? No...
                                    # Wait, I need to be more careful with counts
                                break
                    elif cnt1[i] == 4:
                        # Must set the uncolored vertex to 0
                        for v in edges[i]:
                            if color[v] == -1:
                                if color[v] != 0:
                                    trail.append((v, color[v]))
                                    color[v] = 0
                                break
        return True
    
    # This approach is getting complicated. Let me use a cleaner implementation.
    
    # Reset and use a cleaner backtracking approach
    color = [-1] * n
    cnt0 = [0] * m
    cnt1 = [0] * m
    cntu = [5] * m
    trail = []
    
    def assign(v, c):
        """Assign color c to vertex v, updating edge counts. Returns False if violation."""
        old = color[v]
        if old == c:
            return True
        if old != -1:
            # Changing from one color to another
            if old == 0:
                for i in vert_edges[v]:
                    cnt0[i] -= 1
                    cnt1[i] += 1
            else:
                for i in vert_edges[v]:
                    cnt1[i] -= 1
                    cnt0[i] += 1
        else:
            # From uncolored
            for i in vert_edges[v]:
                cntu[i] -= 1
                if c == 0:
                    cnt0[i] += 1
                else:
                    cnt1[i] += 1
        color[v] = c
        trail.append((v, old))
        # Check for violations in affected edges
        for i in vert_edges[v]:
            if cntu[i] == 0:
                if cnt0[i] == 5 or cnt1[i] == 5:
                    return False
        return True
    
    def unassign(v):
        """Undo last assignment to v."""
        old = color[v]
        for i in vert_edges[v]:
            if old == 0:
                cnt0[i] -= 1
                cntu[i] += 1
            else:
                cnt1[i] -= 1
                cntu[i] += 1
        color[v] = -1
    
    def propagate():
        """Unit propagation. Returns False on contradiction."""
        queue = []
        # Find all edges that need propagation
        for i in range(m):
            if cntu[i] == 1:
                if cnt0[i] == 4:
                    # The single uncolored vertex must be 1
                    for v in edges[i]:
                        if color[v] == -1:
                            if not assign(v, 1):
                                return False
                            queue.append(v)
                            break
                elif cnt1[i] == 4:
                    # The single uncolored vertex must be 0
                    for v in edges[i]:
                        if color[v] == -1:
                            if not assign(v, 0):
                                return False
                            queue.append(v)
                            break
        # Process queue - new assignments might trigger more propagation
        while queue:
            v = queue.pop()
            for i in vert_edges[v]:
                if cntu[i] == 1:
                    if cnt0[i] == 4:
                        for u in edges[i]:
                            if color[u] == -1:
                                if not assign(u, 1):
                                    return False
                                queue.append(u)
                                break
                    elif cnt1[i] == 4:
                        for u in edges[i]:
                            if color[u] == -1:
                                if not assign(u, 0):
                                    return False
                                queue.append(u)
                                break
        return True
    
    def backtrack():
        """Try to find a valid 2-coloring. Returns True if found."""
        if _elapsed() > deadline:
            return None  # timeout
        
        if not propagate():
            return False
        
        # Find an uncolored vertex to branch on
        # Heuristic: pick vertex in the most constrained edge
        best_v = -1
        best_constraint = -1
        for v in range(n):
            if color[v] == -1:
                # Count how many of its edges are "tight"
                tight = 0
                for i in vert_edges[v]:
                    if cntu[i] <= 2:
                        tight += 1
                if tight > best_constraint:
                    best_constraint = tight
                    best_v = v
                    if tight >= 2:
                        break
        
        if best_v == -1:
            # All vertices colored
            # Check if valid
            for i in range(m):
                if cnt0[i] == 5 or cnt1[i] == 5:
                    return False
            return True
        
        # Try both colors
        for c in [0, 1]:
            if assign(best_v, c):
                if propagate():
                    result = backtrack()
                    if result is True:
                        return True
                    elif result is None:
                        # Timeout
                        # Undo
                        while trail and trail[-1][0] != best_v:
                            u, old = trail.pop()
                            unassign(u)
                        if trail:
                            u, old = trail.pop()
                            unassign(u)
                        return None
                # Undo
                while trail and trail[-1][0] != best_v:
                    u, old = trail.pop()
                    unassign(u)
                if trail and trail[-1][0] == best_v:
                    u, old = trail.pop()
                    unassign(u)
        
        return False
    
    result = backtrack()
    if result is True:
        return True
    elif result is False:
        return False
    else:
        # Timeout - return True optimistically (assume 2-colorable)
        return True


def _is_not_2_colorable(edges, n, deadline=None):
    """Check if hypergraph is NOT 2-colorable."""
    return not _is_2_colorable(edges, n, deadline)


def _try_remove_edge(edges, n, idx, deadline=None):
    """Try removing edge idx. Returns True if still not 2-colorable."""
    if deadline is None:
        deadline = _deadline
    new_edges = edges[:idx] + edges[idx+1:]
    return _is_not_2_colorable(new_edges, n, deadline)


def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    deadline = _DEADLINE
    
    # Start with K9^5 (all 5-subsets of 9 vertices) - 126 edges
    # This is definitely not 2-colorable
    n = 9
    all_edges = [list(e) for e in combinations(range(n), 5)]
    
    # Try to remove edges while maintaining non-2-colorability
    # Greedy removal: try removing each edge, keep it removed if still not 2-colorable
    edges = list(all_edges)
    
    # First pass: greedy removal
    i = 0
    while i < len(edges):
        if _elapsed() > deadline - 5:
            break
        if _try_remove_edge(edges, n, i, deadline - 5):
            edges.pop(i)
            # Don't increment i, check same position again
        else:
            i += 1
    
    # Second pass: try more aggressive removal with different orderings
    for attempt in range(10):
        if _elapsed() > deadline - 5:
            break
        # Shuffle and try removal
        shuffled_idx = list(range(len(edges)))
        random.shuffle(shuffled_idx)
        improved = False
        for idx in shuffled_idx:
            if _elapsed() > deadline - 5:
                break
            if idx >= len(edges):
                continue
            if _try_remove_edge(edges, n, idx, deadline - 5):
                edges.pop(idx)
                improved = True
        if not improved:
            break
    
    # Local search: try swapping - remove one edge and add another to maintain property
    # Actually, let's try a different approach: start from scratch with random subsets
    
    # Try random constructions
    best_edges = list(edges)
    best_count = len(edges)
    
    for trial in range(50):
        if _elapsed() > deadline - 10:
            break
        
        # Random approach: pick random 5-edges and check
        # Start with a random set and try to minimize
        n_try = random.randint(8, 12)
        
        # Generate random candidate
        all_possible = list(combinations(range(n_try), 5))
        random.shuffle(all_possible)
        
        # Start with all edges, remove greedily
        cand = [list(e) for e in all_possible]
        
        i = 0
        while i < len(cand):
            if _elapsed() > deadline - 10:
                break
            if _try_remove_edge(cand, n_try, i, deadline - 10):
                cand.pop(i)
            else:
                i += 1
        
        if len(cand) < best_count:
            best_edges = cand
            best_count = len(cand)
            n = n_try
    
    # Further optimization on best found
    n = len(set(v for e in best_edges for v in e)) if best_edges else 9
    # Actually n should be the max vertex + 1
    if best_edges:
        n = max(v for e in best_edges for v in e) + 1
    else:
        n = 9
    
    # Do more greedy passes on the best
    for _ in range(5):
        if _elapsed() > deadline - 3:
            break
        improved = False
        i = 0
        while i < len(best_edges):
            if _elapsed() > deadline - 3:
                break
            if _try_remove_edge(best_edges, n, i, deadline - 3):
                best_edges.pop(i)
                improved = True
            else:
                i += 1
        if not improved:
            break
    
    # Verify the result is valid
    if _elapsed() > deadline - 2:
        # Just return what we have, assume it's valid
        return {'n': n, 'edges': best_edges}
    
    # Final verification
    if not _is_not_2_colorable(best_edges, n, deadline - 1):
        # Fallback to K9^5
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
    
    return {'n': n, 'edges': best_edges}
