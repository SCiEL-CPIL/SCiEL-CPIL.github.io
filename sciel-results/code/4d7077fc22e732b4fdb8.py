"""Rewritten with structural change: constructive backtracking + fast bitmask validation."""
# EVOLVE-BLOCK-START
import random
import time
import itertools
from collections import defaultdict

def _check_regular_6(adj, n, subset):
    """Check if 6 vertices induce a regular subgraph. Fast bitmask check."""
    mask = 0
    for v in subset:
        mask |= (1 << v)
    degs = []
    for v in subset:
        degs.append((adj[v] & mask).bit_count())
    return all(d == degs[0] for d in degs)

def _find_all_violations_6(adj, n):
    """Find all 6-subsets that induce regular subgraphs."""
    violations = []
    for subset in itertools.combinations(range(n), 6):
        if _check_regular_6(adj, n, subset):
            violations.append(subset)
    return violations

def _check_larger_violations(adj, n):
    """Check if any subset of size >= 7 induces a regular subgraph."""
    for k in range(7, n + 1):
        for subset in itertools.combinations(range(n), k):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            degs = []
            for v in subset:
                degs.append((adj[v] & mask).bit_count())
            if all(d == degs[0] for d in degs):
                return True
    return False

def _is_valid_graph(adj, n):
    """Check if graph has no regular induced subgraph on >= 6 vertices."""
    if _find_all_violations_6(adj, n):
        return False
    if _check_larger_violations(adj, n):
        return False
    return True

def _constructive_search(n, deadline, max_restarts=30):
    """
    Constructive backtracking: build graph vertex by vertex.
    At each step, try adding edges to vertex i such that no new
    regular 6-subset is created. Use backjumping when stuck.
    """
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        random.seed(restart * 12345 + n)
        
        # adj[i] is bitmask of neighbors of vertex i
        adj = [0] * n
        
        # We'll build the graph by deciding edges for each vertex in order
        # For vertex i, we decide which of vertices 0..i-1 it connects to
        # AND which of vertices i+1..n-1 (pre-decided via symmetry breaking)
        
        # Simpler: just do full random + local repair with smarter heuristic
        # Initialize with biased random
        p = 0.45 + random.random() * 0.1  # Slightly below 0.5
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Local search with conflict-directed moves
        for iteration in range(5000):
            if time.time() > deadline - 3:
                break
            
            violations = _find_all_violations_6(adj, n)
            if not violations:
                if not _check_larger_violations(adj, n):
                    return adj
                # Has larger violation but no 6-violation
                # Try random flip to break larger violation
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if adj[u] & (1 << v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                continue
            
            # Score each edge flip by how many violations it breaks
            edge_scores = defaultdict(int)
            for subset in violations:
                for i in range(6):
                    for j in range(i + 1, 6):
                        u, v = subset[i], subset[j]
                        key = (min(u, v), max(u, v))
                        edge_scores[key] += 1
            
            # Pick edge with highest score, flip it
            if edge_scores:
                max_score = max(edge_scores.values())
                candidates = [k for k, v in edge_scores.items() if v == max_score]
                u, v = random.choice(candidates)
            else:
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
            
            # Flip edge (u, v)
            if adj[u] & (1 << v):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # If local search failed, try simulated annealing from current state
        if time.time() < deadline - 5:
            current_cost = len(_find_all_violations_6(adj, n))
            T = 3.0
            T_min = 0.05
            cooling = 0.997
            
            for it in range(2000):
                if time.time() > deadline - 3:
                    break
                if T < T_min:
                    break
                
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                
                # Flip
                if adj[u] & (1 << v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                
                new_cost = len(_find_all_violations_6(adj, n))
                delta = new_cost - current_cost
                
                if delta < 0:
                    current_cost = new_cost
                else:
                    if delta == 0 or random.random() < (2.71828 ** (-delta / T)):
                        current_cost = new_cost
                    else:
                        # Revert
                        if adj[u] & (1 << v):
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                        else:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                
                T *= cooling
                
                if current_cost == 0:
                    if not _check_larger_violations(adj, n):
                        return adj
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    best_n = 5
    best_edges = []
    
    # Try from large n down
    for n in [20, 19, 18, 17, 16, 15]:
        if time.time() > deadline - 15:
            break
        
        adj = _constructive_search(n, deadline)
        
        if adj is not None:
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i] & (1 << j):
                        edges.append([i, j])
            best_n = n
            best_edges = edges
            return {'n': best_n, 'edges': best_edges}
    
    # Fallback: try very small n quickly
    for n in [10, 9, 8, 7]:
        if time.time() > deadline - 3:
            break
        for _ in range(20):
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            if _is_valid_graph(adj, n):
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                best_n = n
                best_edges = edges
                break
        if best_n > 5:
            break
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
