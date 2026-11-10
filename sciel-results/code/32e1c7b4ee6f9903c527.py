import time
import random
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try to find a valid graph for n from 20 down to 5
    for n in range(20, 4, -1):
        # Time budget for this n
        time_budget = 20.0 if n > 10 else 5.0
        target_end = start_time + time_budget
        
        # Local search
        best_graph = None
        best_violations = float('inf')
        
        for restart in range(50):  # Max 50 restarts
            if time.time() > target_end:
                break
                
            # Initialize random graph
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            current_violations = count_violations(adj, n)
            if current_violations == 0:
                return {'n': n, 'edges': get_edges(adj)}
            
            if current_violations < best_violations:
                best_violations = current_violations
                best_graph = adj[:]
            
            # Local search steps
            steps = 0
            while time.time() < target_end and steps < 1000:
                steps += 1
                violations = find_violations(adj, n)
                if not violations:
                    return {'n': n, 'edges': get_edges(adj)}
                
                # Pick a random violation
                S = random.choice(violations)
                mask = 0
                for v in S:
                    mask |= (1 << v)
                
                # Try flipping edges within S
                best_flip = None
                best_score = current_violations
                found_improvement = False
                
                for i_idx, u in enumerate(S):
                    for v in S[i_idx+1:]:
                        # Flip edge (u, v)
                        if (adj[u] >> v) & 1:
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                        else:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                        
                        new_violations = count_violations(adj, n)
                        
                        # Flip back
                        if (adj[u] >> v) & 1:
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                        else:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            
                        if new_violations < best_score:
                            best_score = new_violations
                            best_flip = (u, v)
                            found_improvement = True
                
                if found_improvement:
                    u, v = best_flip
                    if (adj[u] >> v) & 1:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    current_violations = best_score
                    if current_violations < best_violations:
                        best_violations = current_violations
                        best_graph = adj[:]
                else:
                    # Random move to escape local minimum
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u != v:
                        if (adj[u] >> v) & 1:
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                        else:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                        current_violations = count_violations(adj, n)
                        if current_violations < best_violations:
                            best_violations = current_violations
                            best_graph = adj[:]

    # Fallback: return the best graph found if it was valid, otherwise n=5
    if best_graph is not None and count_violations(best_graph, len(best_graph)) == 0:
        return {'n': len(best_graph), 'edges': get_edges(best_graph)}
    
    return {'n': 5, 'edges': []}

def get_edges(adj):
    edges = []
    n = len(adj)
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    return edges

def count_violations(adj, n):
    """Count number of subsets of size >= 6 that induce a regular graph."""
    total = 0
    for k in range(6, n + 1):
        for S in combinations(range(n), k):
            if is_regular(adj, S):
                total += 1
        if total > 0:
            # Optimization: if we already have violations, we can stop early 
            # if we just need the count for comparison, but for finding 
            # specific violations we need the list. 
            # However, for the "is it valid" check, 0 is the target.
            # For optimization, counting all is safer for scoring.
            # But if total is large, we might want to stop.
            # Let's just count all for simplicity, as n is small.
            pass
    return total

def find_violations(adj, n):
    """Return list of violating subsets."""
    violations = []
    for k in range(6, n + 1):
        for S in combinations(range(n), k):
            if is_regular(adj, S):
                violations.append(S)
        if violations:
            # If we find violations at size k, we can stop checking larger k
            # for the purpose of "picking a violation to fix", as fixing 
            # smaller ones is usually more effective.
            # However, a larger violation might be the only one.
            # But typically, if there are many small ones, fix those first.
            break
    return violations

def is_regular(adj, S):
    """Check if subset S induces a regular graph."""
    mask = 0
    for v in S:
        mask |= (1 << v)
    
    degrees = []
    for v in S:
        deg = (adj[v] & mask).bit_count()
        degrees.append(deg)
    
    return len(set(degrees)) == 1
