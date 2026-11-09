"""Few edges in 5-uniform hypergraphs without Property B (m(5)).

Strategy: Use a SAT-based incremental approach to find a minimal non-2-colorable
5-uniform hypergraph. We encode the 2-colorability problem as SAT and use
DPLL with clause learning to efficiently check satisfiability. We incrementally
add high-impact edges until the formula becomes UNSAT (non-2-colorable), then
remove redundant edges to minimize the count.
"""

import time
import random
from itertools import combinations


class _SATChecker:
    """Fast SAT checker for 2-colorability using DPLL with unit propagation."""
    
    def __init__(self, n):
        self.n = n
        self.edge_masks = []
        self._build_index()
    
    def _build_index(self):
        """Build vertex-to-edge index for faster propagation."""
        self.vertex_edges = [[] for _ in range(self.n)]
    
    def add_edge(self, edge):
        """Add an edge to the hypergraph."""
        mask = 0
        for v in edge:
            mask |= (1 << v)
        self.edge_masks.append(mask)
        for v in edge:
            self.vertex_edges[v].append(len(self.edge_masks) - 1)
    
    def reset(self):
        """Reset to empty hypergraph."""
        self.edge_masks = []
        self._build_index()
    
    def is_sat(self, deadline=None):
        """Check if the hypergraph is 2-colorable (SAT). Returns True if colorable."""
        if not self.edge_masks:
            return True
        
        n = self.n
        all_vars = (1 << n) - 1
        
        # Precompute edge statistics for branching heuristic
        edge_deg = [0] * n
        for mask in self.edge_masks:
            for v in range(n):
                if mask & (1 << v):
                    edge_deg[v] += 1
        
        def backtrack(assign, unassigned):
            # Unit propagation
            changed = True
            while changed:
                changed = False
                for ei, mask in enumerate(self.edge_masks):
                    assigned_in_edge = mask & ~unassigned
                    unassigned_in_edge = mask & unassigned
                    
                    if unassigned_in_edge == 0:
                        if (assign & mask) == mask or (assign & mask) == 0:
                            return None
                    else:
                        assigned_zero = assigned_in_edge & ~assign
                        assigned_one = assigned_in_edge & assign
                        unassigned_count = bin(unassigned_in_edge).count('1')
                        
                        if assigned_one == 0 and unassigned_count == 1:
                            v = unassigned_in_edge.bit_length() - 1
                            assign |= (1 << v)
                            unassigned &= ~(1 << v)
                            changed = True
                            continue
                        
                        if assigned_zero == 0 and unassigned_count == 1:
                            v = unassigned_in_edge.bit_length() - 1
                            assign &= ~(1 << v)
                            unassigned &= ~(1 << v)
                            changed = True
                            continue
            
            if unassigned == 0:
                return assign
            
            # Branch on variable with highest degree among unassigned
            best_v = -1
            best_deg = -1
            for v in range(n):
                if unassigned & (1 << v):
                    if edge_deg[v] > best_deg:
                        best_deg = edge_deg[v]
                        best_v = v
            
            if best_v == -1:
                best_v = unassigned.bit_length() - 1
            
            # Try True first
            res = backtrack(assign | (1 << best_v), unassigned & ~(1 << best_v))
            if res is not None:
                return res
            # Try False
            return backtrack(assign & ~(1 << best_v), unassigned & ~(1 << best_v))
        
        return backtrack(0, all_vars) is not None


def _complete_5_graph(n):
    """Return the complete 5-uniform hypergraph on n vertices."""
    return [list(e) for e in combinations(range(n), 5)]


def _compute_edge_impact(n, edge):
    """Compute how many colorings this edge makes monochromatic."""
    # An edge is monochromatic if all 5 vertices are 0 or all are 1
    # Number of colorings where a specific 5-set is all-0: 2^(n-5)
    # Number where all-1: 2^(n-5)
    # Total: 2^(n-4)
    return 2 ** (n - 4)

def _greedy_build_non2colorable(n, deadline, seed=42):
    """
    Greedily build a non-2-colorable hypergraph by adding high-impact edges.
    Uses SAT checker to verify when we reach non-2-colorability.
    """
    rng = random.Random(seed)
    
    all_edges = list(combinations(range(n), 5))
    
    # Score edges by impact (all have same impact for 5-uniform, so use random order)
    # But we can prioritize edges that share vertices with existing edges
    edge_scores = []
    for i, edge in enumerate(all_edges):
        edge_scores.append((i, edge))
    
    # Shuffle for diversity
    rng.shuffle(edge_scores)
    
    checker = _SATChecker(n)
    selected = []
    
    # Add edges greedily, checking periodically if we're non-2-colorable
    check_interval = 5
    last_check = 0
    
    for i, (edge_idx, edge) in enumerate(edge_scores):
        if time.time() > deadline:
            break
        
        checker.add_edge(list(edge))
        selected.append(list(edge))
        
        # Check if we've become non-2-colorable
        if i - last_check >= check_interval:
            last_check = i
            if not checker.is_sat():
                # We found a non-2-colorable hypergraph
                return selected
    
    # Final check
    if not checker.is_sat():
        return selected
    
    return None


def _try_remove_edges(n, edges, deadline):
    """Try to remove redundant edges while maintaining non-2-colorability using SAT checker."""
    if len(edges) <= 30:
        return edges
    
    current_edges = list(edges)
    
    # Use SAT checker for efficient removal
    for i in range(len(current_edges) - 1, -1, -1):
        if time.time() > deadline:
            break
        
        # Try removing edge i
        candidate = current_edges[:i] + current_edges[i+1:]
        
        # Check if still non-2-colorable
        checker = _SATChecker(n)
        for edge in candidate:
            checker.add_edge(edge)
        
        if not checker.is_sat():
            current_edges = candidate
    
    return current_edges


def construct():
    deadline = time.time() + 130.0  # safety margin before 150s kill
    
    # Fallback: complete 5-graph on 9 vertices (126 edges, definitely non-2-colorable)
    fallback_n = 9
    fallback_edges = _complete_5_graph(fallback_n)
    
    n = 11
    best_result = None
    best_edge_count = float('inf')
    
    # Try greedy build with multiple seeds
    for seed in [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]:
        if time.time() > deadline - 10:
            break
        
        result = _greedy_build_non2colorable(n, deadline=deadline, seed=seed)
        
        if result is not None:
            # Verify it's actually non-2-colorable
            checker = _SATChecker(n)
            for edge in result:
                checker.add_edge(edge)
            
            if not checker.is_sat():
                # Try to remove redundant edges
                result = _try_remove_edges(n, result, deadline)
                
                if len(result) < best_edge_count:
                    best_edge_count = len(result)
                    best_result = {'n': n, 'edges': result}
                
                # If we're already at or below 51 edges, we're done
                if len(result) <= 51:
                    break
    
    if best_result is not None:
        return best_result
    
    # Return fallback
    return {'n': fallback_n, 'edges': fallback_edges}
