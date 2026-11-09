import sys
import time
import random
from itertools import combinations

P = {'n': 15}

def _construct_graph(n):
    """
    Construct a graph on n vertices maximizing edges such that
    no 3-edge path has endpoints of equal degree.
    Uses conflict-graph approach: identify edge pairs that cause violations,
    then find maximum subset avoiding all conflicts.
    """
    start_time = time.time()
    deadline = 80.0  # Leave 10s margin for 90s limit
    
    # Enumerate all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_set = set(all_edges)
    edge_index = {e: i for i, e in enumerate(all_edges)}
    m = len(all_edges)
    
    def compute_degrees(selected_edges):
        deg = [0] * n
        for (u, v) in selected_edges:
            deg[u] += 1
            deg[v] += 1
        return deg
    
    def find_violations(selected_edges):
        """
        Find all 3-edge paths s-u-v-t where deg(s) == deg(t).
        Returns list of violating edge triples.
        """
        deg = compute_degrees(selected_edges)
        adj = set(selected_edges)
        violations = []
        
        # For each edge (u,v) in selected_edges
        for (u, v) in selected_edges:
            # s is neighbor of u (not v), t is neighbor of v (not u)
            # Need s != t, and (s,u), (u,v), (v,t) all in selected_edges
            for s in range(n):
                if s == u or s == v:
                    continue
                if (min(s,u), max(s,u)) not in adj:
                    continue
                for t in range(n):
                    if t == s or t == u or t == v:
                        continue
                    if (min(t,v), max(t,v)) not in adj:
                        continue
                    if deg[s] == deg[t]:
                        # Violation: edges (s,u), (u,v), (v,t)
                        violations.append(((s,u), (u,v), (v,t)))
        
        return violations
    
    def is_valid(selected_edges):
        """Check if selected_edges form a valid graph."""
        return len(find_violations(selected_edges)) == 0
    
    def count_edges(selected_edges):
        return len(selected_edges)
    
    # Start with complete graph
    current_edges = set(all_edges)
    
    # Greedy removal: repeatedly remove edges that are in violations
    # To maximize remaining edges, remove the edge that appears in most violations
    best_edges = set()
    best_count = 0
    
    # Fallback: start with empty graph, then add edges greedily
    # This ensures we always have a valid solution
    current_edges = set()
    
    # Try to add edges one by one, in order of which additions cause fewest new violations
    # We'll use a heuristic: try all edges, pick the one that adds most edges while staying valid
    
    # Better approach: Start with complete graph, remove edges strategically
    # Let's try multiple strategies and keep the best
    
    strategies_results = []
    
    # Strategy 1: Greedy removal from complete graph
    current_edges = set(all_edges)
    for iteration in range(5000):
        if time.time() - start_time > deadline:
            break
        
        if is_valid(current_edges):
            strategies_results.append(set(current_edges))
            break
        
        violations = find_violations(current_edges)
        if not violations:
            strategies_results.append(set(current_edges))
            break
        
        # Count how many times each edge appears in violations
        edge_violation_count = {}
        for (e1, e2, e3) in violations:
            for e in [e1, e2, e3]:
                e = (min(e[0], e[1]), max(e[0], e[1]))
                edge_violation_count[e] = edge_violation_count.get(e, 0) + 1
        
        # Remove the edge with highest violation count
        if edge_violation_count:
            max_count = max(edge_violation_count.values())
            candidates = [e for e, c in edge_violation_count.items() if c == max_count and e in current_edges]
            if candidates:
                # Remove a random candidate for diversity
                to_remove = random.choice(candidates)
                current_edges.discard(to_remove)
    
    if current_edges:
        if is_valid(current_edges):
            strategies_results.append(set(current_edges))
    
    # Strategy 2: Greedy addition from empty graph
    current_edges = set()
    for iteration in range(5000):
        if time.time() - start_time > deadline:
            break
        
        # Try adding each edge not yet added
        best_addition = None
        best_new_count = len(current_edges)
        
        for e in all_edges:
            if e in current_edges:
                continue
            # Try adding e
            trial_edges = current_edges | {e}
            if is_valid(trial_edges):
                # Check if this addition allows more future additions
                # For now, just accept any valid addition
                best_addition = e
                best_new_count = len(trial_edges)
                break
        
        if best_addition is None:
            break
        
        current_edges.add(best_addition)
    
    if is_valid(current_edges):
        strategies_results.append(set(current_edges))
    
    # Strategy 3: Randomized local search
    # Start from a random valid graph, then try to add edges
    for restart in range(100):
        if time.time() - start_time > deadline:
            break
        
        # Random initial graph
        current_edges = set()
        for e in all_edges:
            if random.random() < 0.3:
                current_edges.add(e)
        
        # Ensure valid
        if not is_valid(current_edges):
            # Remove violating edges
            while not is_valid(current_edges) and current_edges:
                violations = find_violations(current_edges)
                if not violations:
                    break
                # Remove random edge from a violation
                e1, e2, e3 = random.choice(violations)
                random_choice = random.choice([e1, e2, e3])
                current_edges.discard((min(random_choice[0], random_choice[1]), max(random_choice[0], random_choice[1])))
        
        # Try to add edges
        improved = True
        while improved and time.time() - start_time < deadline:
            improved = False
            for e in all_edges:
                if e in current_edges:
                    continue
                trial_edges = current_edges | {e}
                if is_valid(trial_edges):
                    current_edges.add(e)
                    improved = True
                    break
        
        if is_valid(current_edges):
            strategies_results.append(set(current_edges))
    
    # Pick the best result
    if not strategies_results:
        # Fallback: star graph
        fallback_edges = set()
        for v in range(1, n):
            fallback_edges.add((0, v))
        strategies_results.append(fallback_edges)
    
    best_edges = max(strategies_results, key=len)
    
    # Final validation
    if not is_valid(best_edges):
        # Emergency fallback
        best_edges = set()
        for v in range(1, n):
            best_edges.add((0, v))
    
    edges = [[u, v] for (u, v) in best_edges]
    return edges

def construct():
    edges = _construct_graph(P['n'])
    result = []
    for e in edges:
        u, v = int(e[0]), int(e[1])
        result.append([u, v])
    return result
