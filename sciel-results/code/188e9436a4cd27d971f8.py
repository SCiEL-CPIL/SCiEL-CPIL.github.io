import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 280  # 20s safety margin for evaluator overhead
    
    def paley_graph(n):
        """Generate Paley graph on n vertices (n must be prime, n ≡ 1 mod 4)."""
        if n < 2:
            return []
        # Check if n is prime (required for Paley graph)
        def is_prime(x):
            if x < 2:
                return False
            if x == 2:
                return True
            if x % 2 == 0:
                return False
            for i in range(3, int(x**0.5) + 1, 2):
                if x % i == 0:
                    return False
            return True
        if not is_prime(n) or n % 4 != 1:
            return []
        
        # Quadratic residues mod n
        residues = set()
        for x in range(1, n):
            residues.add((x * x) % n)
        
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % n in residues:
                    edges.append((i, j))
        return edges
    
    def check_valid(n, edges):
        """Check if graph has no K5 and no independent set of size 5.
        Returns (is_valid, violation_type, violation_vertices)."""
        # Build adjacency set for fast lookup
        edge_set = set(edges)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check for K5
        for combo in combinations(range(n), 5):
            is_clique = True
            for i in range(5):
                for j in range(i + 1, 5):
                    u, v = combo[i], combo[j]
                    if (min(u,v), max(u,v)) not in edge_set:
                        is_clique = False
                        break
                if not is_clique:
                    break
            if is_clique:
                return (False, "clique", combo)
        
        # Check for independent set of size 5
        for combo in combinations(range(n), 5):
            is_independent = True
            for i in range(5):
                for j in range(i + 1, 5):
                    u, v = combo[i], combo[j]
                    if (min(u,v), max(u,v)) in edge_set:
                        is_independent = False
                        break
                if not is_independent:
                    break
            if is_independent:
                return (False, "independent", combo)
        
        return (True, None, None)
    
    def find_violation_fast(n, edge_set):
        """Find a K5 or independent 5-set. Returns (type, vertices) or None."""
        # Build adjacency lists
        adj = [set() for _ in range(n)]
        non_adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        for i in range(n):
            for j in range(n):
                if i != j and j not in adj[i]:
                    non_adj[i].add(j)
        
        # Check for K5
        for combo in combinations(range(n), 5):
            ok = True
            for i in range(5):
                for j in range(i+1, 5):
                    if combo[j] not in adj[combo[i]]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return ("clique", combo)
        
        # Check for independent set
        for combo in combinations(range(n), 5):
            ok = True
            for i in range(5):
                for j in range(i+1, 5):
                    if combo[j] in adj[combo[i]]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return ("independent", combo)
        
        return None
    
    # Try to build up from smaller valid graphs
    # Start with known valid Paley graphs
    best_n = 0
    best_edges = []
    
    # Known valid Paley graphs
    for q in [37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        if q > 46:  # R(5,5) <= 46
            break
        edges = paley_graph(q)
        if not edges:
            continue
        valid, vtype, vset = check_valid(q, edges)
        if valid:
            if q > best_n:
                best_n = q
                best_edges = edges
            break  # Only need the largest valid one
    
    if best_n == 0:
        # Fallback: Paley 37
        best_edges = paley_graph(37)
        best_n = 37
    
    # Try to extend using greedy vertex addition with repair
    current_n = best_n
    current_edges = list(best_edges)
    
    # Build adjacency structure
    def build_adj(n, edges):
        edge_set = set(edges)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return edge_set, adj
    
    edge_set, adj = build_adj(current_n, current_edges)
    
    # Try to add vertices greedily
    max_attempts = 20
    for attempt in range(max_attempts):
        if time.time() > deadline - 10:
            break
        
        target_n = current_n + 1
        if target_n > 46:
            break
        
        # Try multiple random connection patterns
        found_valid = False
        for trial in range(50):
            if time.time() > deadline - 10:
                break
            
            new_edges = list(current_edges)
            new_adj = [set(s) for s in adj]
            new_adj.append(set())
            
            # Decide connections for new vertex (index current_n)
            for i in range(current_n):
                if random.random() < 0.5:
                    new_edges.append((i, current_n))
                    new_adj[i].add(current_n)
                    new_adj[current_n].add(i)
            
            # Check validity
            valid, vtype, vset = check_valid(target_n, new_edges)
            if valid:
                current_n = target_n
                current_edges = new_edges
                edge_set, adj = build_adj(current_n, current_edges)
                found_valid = True
                break
        
        if not found_valid:
            # Try local search to fix violations
            # Find violation
            violation = find_violation_fast(current_n, edge_set)
            if violation is None:
                break  # Already valid
            
            vtype, vset = violation
            # Try to fix by modifying edges
            fixed = False
            for fix_attempt in range(100):
                if time.time() > deadline - 10:
                    break
                
                if vtype == "clique":
                    # Remove a random edge from the clique
                    pairs = list(combinations(vset, 2))
                    random.shuffle(pairs)
                    u, v = pairs[0]
                    u, v = min(u,v), max(u,v)
                    if (u,v) in edge_set:
                        edge_set.remove((u,v))
                        adj[u].discard(v)
                        adj[v].discard(u)
                        current_edges = list(edge_set)
                        fixed = True
                        break
                else:
                    # Add a random non-edge from the independent set
                    pairs = list(combinations(vset, 2))
                    random.shuffle(pairs)
                    u, v = pairs[0]
                    u, v = min(u,v), max(u,v)
                    if (u,v) not in edge_set:
                        edge_set.add((u,v))
                        adj[u].add(v)
                        adj[v].add(u)
                        current_edges = list(edge_set)
                        fixed = True
                        break
            
            if not fixed:
                break
    
    # Final verification
    valid, _, _ = check_valid(current_n, current_edges)
    if not valid:
        # Return the last known valid graph
        return best_n, best_edges
    
    return current_n, current_edges
