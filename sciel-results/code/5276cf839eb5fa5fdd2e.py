import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit
    
    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Precompute all triples of vertices (i, j, k) with i < j < k
    # A violation occurs when |N(i) ∩ N(j) ∩ N(k)| >= 3
    # Equivalently: there exist 3 common neighbors
    # We track this via hyperedges: for each triple (i,j,k), if it has >=3 common neighbors, it's violated
    
    # Precompute all vertex triples
    all_triples = list(combinations(range(n), 3))
    
    def count_violations_fast(adj):
        """Count violations: triples with >= 3 common neighbors."""
        v_count = 0
        for i, j, k in all_triples:
            common = adj[i] & adj[j] & adj[k]
            if common.bit_count() >= 3:
                v_count += 1
        return v_count

    def get_edge_count(adj):
        return sum(x.bit_count() for x in adj) // 2

    def find_violating_triples(adj):
        """Return list of violating triples."""
        violations = []
        for i, j, k in all_triples:
            common = adj[i] & adj[j] & adj[k]
            if common.bit_count() >= 3:
                violations.append((i, j, k, common))
        return violations

    def repair_adj(adj):
        """Remove edges to eliminate all K3,3 violations."""
        max_iter = 10000
        while max_iter > 0:
            max_iter -= 1
            violations = find_violating_triples(adj)
            if not violations:
                break
            # Pick a random violating triple
            i, j, k, common = random.choice(violations)
            # Get list of common neighbors
            c_list = [b for b in range(n) if common & (1 << b)]
            if not c_list:
                continue
            c = random.choice(c_list)
            # Remove edge between c and one of i, j, k
            v_to_remove = random.choice([i, j, k])
            adj[v_to_remove] &= ~(1 << c)
            adj[c] &= ~(1 << v_to_remove)
        return adj

    def spectral_init():
        """Initialize using spectral methods to get a dense starting point."""
        # Use a Turán-type construction: partition vertices into parts
        # For K3,3-free, we can use complete multipartite graphs
        # T(20, 5) has 5 parts of size 4, edges = C(20,2) - 5*C(4,2) = 190 - 30 = 160
        # But complete multipartite with 5 parts IS K3,3-free? Let's check.
        # In T(20,5), any 3 vertices from different parts have all other 16 as common neighbors -> violation!
        # So complete multipartite is NOT K3,3-free for many parts.
        
        # Instead, use a random initialization with moderate density
        adj = [0] * n
        density = random.uniform(0.5, 0.7)
        for u, v in all_edges:
            if random.random() < density:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        adj = repair_adj(adj)
        return adj

    def greedy_fill(adj):
        """Try to add edges greedily."""
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        random.shuffle(missing)
        for u, v in missing:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if count_violations_fast(adj) > 0:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        return adj

    def max_sat_local_search(adj, max_time):
        """
        Max-SAT style local search:
        - Each edge is a variable (1 = included, 0 = not)
        - Each violating triple is a clause that must be satisfied
        - Objective: maximize sum of edge variables - penalty * sum of violated clauses
        
        We use weighted local search with edge flips (add/remove).
        """
        ls_start = time.time()
        current_adj = adj[:]
        current_edges = get_edge_count(current_adj)
        
        # Ensure valid start
        if count_violations_fast(current_adj) > 0:
            current_adj = repair_adj(current_adj)
            current_edges = get_edge_count(current_adj)
        
        best_adj = current_adj[:]
        best_edges = current_edges
        
        # Precompute for each edge (u,v), the list of triples that could be affected
        # When we add/remove edge (u,v), only triples containing u or v are affected
        # For triple (i,j,k), it's violated if |N(i)∩N(j)∩N(k)| >= 3
        
        # For efficiency, track violation count per triple
        # But that's 1140 triples, so let's just recompute affected ones
        
        # For each vertex pair (u,v), store which triples involve u or v
        # Actually, let's precompute for each edge (u,v), the set of triples that include u or v
        # When we flip edge (u,v), a triple (i,j,k) is affected if u or v is in {i,j,k}
        
        # Precompute: for each vertex x, list of triples containing x
        triples_by_vertex = {v: [] for v in range(n)}
        for tri in all_triples:
            for v in tri:
                triples_by_vertex[v].append(tri)
        
        # For each edge (u,v), the affected triples are those containing u or v
        # Deduplicate
        edge_affected_triples = {}
        for u, v in all_edges:
            affected = set(triples_by_vertex[u] + triples_by_vertex[v])
            edge_affected_triples[(u,v)] = list(affected)
        
        # Simulated annealing with Max-SAT weighting
        T = 2.0
        penalty = 5.0  # Penalty per violation
        
        while time.time() - ls_start < max_time:
            T = max(0.05, T * 0.997)
            penalty = max(1.0, penalty * 0.9995)
            
            # Choose a random edge to flip
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            
            edge = (min(u,v), max(u,v))
            u, v = edge
            
            is_present = bool(current_adj[u] & (1 << v))
            
            # Compute delta in edge count
            if is_present:
                delta_edges = -1
            else:
                delta_edges = 1
            
            # Compute delta in violations for affected triples
            # Before flip
            old_violations_affected = 0
            for tri in edge_affected_triples[edge]:
                common = current_adj[tri[0]] & current_adj[tri[1]] & current_adj[tri[2]]
                if common.bit_count() >= 3:
                    old_violations_affected += 1
            
            # Apply flip
            if is_present:
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
            else:
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
            
            # After flip
            new_violations_affected = 0
            for tri in edge_affected_triples[edge]:
                common = current_adj[tri[0]] & current_adj[tri[1]] & current_adj[tri[2]]
                if common.bit_count() >= 3:
                    new_violations_affected += 1
            
            delta_violations = new_violations_affected - old_violations_affected
            delta_score = delta_edges - penalty * delta_violations
            
            # Acceptance criterion
            if delta_score > 0:
                accept = True
            else:
                if delta_score < -30:
                    accept = False
                else:
                    accept = random.random() < np.exp(delta_score / T)
            
            if accept:
                current_edges += delta_edges
                # Update best if valid
                if delta_violations == 0 and old_violations_affected == 0:
                    # Check if globally valid
                    if count_violations_fast(current_adj) == 0 and current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = current_adj[:]
                elif count_violations_fast(current_adj) == 0 and current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = current_adj[:]
            else:
                # Revert
                if is_present:
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                else:
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
        
        return best_adj, best_edges

    # Main search
    best_adj = [0] * n
    best_edges = 0
    
    # Multiple restarts with different strategies
    num_restarts = 30
    time_per_restart = (deadline - start_time - 5) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Initialize
        adj = spectral_init()
        adj = greedy_fill(adj)
        edges = get_edge_count(adj)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
        
        # Run Max-SAT local search
        remaining = deadline - time.time() - 2
        sa_time = max(0.1, remaining / (num_restarts - r))
        adj, edges = max_sat_local_search(adj, sa_time)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
    
    # Final greedy polish
    if time.time() < deadline - 1:
        best_adj = greedy_fill(best_adj)
        best_edges = get_edge_count(best_adj)
    
    # Convert to list of lists
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result.append([i, j])
                
    return result

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
