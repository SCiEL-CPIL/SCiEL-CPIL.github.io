"""Direct bootstrap for R(5,5): extend Paley(37) with a time-boxed heuristic search."""
import time
import random
import sys
from itertools import combinations

# EVOLVE-BLOCK-START

DEADLINE = time.time() + 270  # safety margin: 330s hard, aim to finish by 270s


def _build_paley37():
    """Return adjacency bitset list for Paley graph on 37 vertices (vertices 0..36)."""
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj


def _find_k4_sets(adj, n):
    """Find all K4 cliques in graph with n vertices. Return list of 4-tuples."""
    k4_list = []
    for i in range(n):
        neigh_i = [j for j in range(n) if j > i and (adj[i] & (1 << j))]
        for j in neigh_i:
            neigh_ij = [k for k in range(n) if k > j and (adj[i] & (1 << k)) and (adj[j] & (1 << k))]
            for k1 in range(len(neigh_ij)):
                a = neigh_ij[k1]
                for k2 in range(k1 + 1, len(neigh_ij)):
                    b = neigh_ij[k2]
                    if adj[a] & (1 << b):
                        k4_list.append((i, j, a, b))
    return k4_list


def _find_ind4_sets(adj, n):
    """Find all independent 4-sets in graph with n vertices. Return list of 4-tuples."""
    ind4_list = []
    for i in range(n):
        non_neigh_i = [j for j in range(n) if j > i and not (adj[i] & (1 << j))]
        for j in non_neigh_i:
            non_neigh_ij = [k for k in range(n) if k > j and not (adj[i] & (1 << k)) and not (adj[j] & (1 << k))]
            for k1 in range(len(non_neigh_ij)):
                a = non_neigh_ij[k1]
                for k2 in range(k1 + 1, len(non_neigh_ij)):
                    b = non_neigh_ij[k2]
                    if not (adj[a] & (1 << b)):
                        ind4_list.append((i, j, a, b))
    return ind4_list


def _try_add_vertex(adj, n, k4_list, ind4_list, max_attempts=5000):
    """Try to add vertex n (new index = n) by choosing edges to 0..n-1.
    Returns new_adj or None if failed.
    Uses a smarter heuristic: start with a balanced assignment, then do local search."""
    new_n = n + 1
    
    # First, try a simple heuristic: for each vertex, assign edge based on degree
    # Vertices with high degree should tend to be non-adjacent to new vertex (to avoid K5)
    # Vertices with low degree should tend to be adjacent to new vertex (to avoid Ind5)
    
    # Compute degrees
    degrees = [bin(adj[i]).count('1') for i in range(n)]
    avg_degree = sum(degrees) / n if n > 0 else 0
    
    # Try multiple strategies
    strategies = []
    
    # Strategy 1: Random
    for _ in range(100):
        strategies.append([random.choice([0, 1]) for _ in range(n)])
    
    # Strategy 2: Degree-based heuristic
    for trial in range(100):
        assign = []
        for v in range(n):
            # Higher degree -> less likely to connect (to avoid K5)
            # Lower degree -> more likely to connect (to avoid Ind5)
            p = 0.5 + (avg_degree - degrees[v]) / (2 * avg_degree + 1) * 0.3
            p = max(0.1, min(0.9, p))
            assign.append(1 if random.random() < p else 0)
        strategies.append(assign)
    
    # Strategy 3: Complement of degree-based
    for trial in range(100):
        assign = []
        for v in range(n):
            p = 0.5 + (degrees[v] - avg_degree) / (2 * avg_degree + 1) * 0.3
            p = max(0.1, min(0.9, p))
            assign.append(1 if random.random() < p else 0)
        strategies.append(assign)
    
    # Check each strategy
    for assign in strategies:
        # Check K5 constraints
        valid = True
        for (a, b, c, d) in k4_list:
            if assign[a] and assign[b] and assign[c] and assign[d]:
                valid = False
                break
        if not valid:
            continue
        # Check independent 5 constraints
        for (a, b, c, d) in ind4_list:
            if not assign[a] and not assign[b] and not assign[c] and not assign[d]:
                valid = False
                break
        if valid:
            # Build new adjacency
            new_adj = adj[:] + [0]
            for v in range(n):
                if assign[v]:
                    new_adj[new_n] |= (1 << v)
                    new_adj[v] |= (1 << new_n)
            return new_adj
    
    # If simple strategies failed, try local search
    # Start with a random assignment and flip bits to reduce violations
    for attempt in range(200):
        if time.time() > DEADLINE - 30:
            break
        
        # Initialize with random or degree-based
        if attempt % 2 == 0:
            assign = [random.choice([0, 1]) for _ in range(n)]
        else:
            assign = []
            for v in range(n):
                p = 0.5 + (avg_degree - degrees[v]) / (2 * avg_degree + 1) * 0.3
                p = max(0.1, min(0.9, p))
                assign.append(1 if random.random() < p else 0)
        
        # Local search: flip bits to reduce total violations
        for iteration in range(200):
            if time.time() > DEADLINE - 30:
                break
            
            # Count violations
            k5_violations = 0
            ind5_violations = 0
            for (a, b, c, d) in k4_list:
                if assign[a] and assign[b] and assign[c] and assign[d]:
                    k5_violations += 1
            for (a, b, c, d) in ind4_list:
                if not assign[a] and not assign[b] and not assign[c] and not assign[d]:
                    ind5_violations += 1
            
            total_violations = k5_violations + ind5_violations
            if total_violations == 0:
                # Build new adjacency
                new_adj = adj[:] + [0]
                for v in range(n):
                    if assign[v]:
                        new_adj[new_n] |= (1 << v)
                        new_adj[v] |= (1 << new_n)
                return new_adj
            
            # Try flipping one bit
            best_flip = None
            best_score = total_violations
            
            # Try flipping each vertex
            for v in random.sample(range(n), min(n, 20)):
                old_val = assign[v]
                assign[v] = 1 - old_val
                
                # Recount violations (approximate: only check constraints involving v)
                new_k5 = 0
                new_ind5 = 0
                for (a, b, c, d) in k4_list:
                    if v in (a, b, c, d):
                        if assign[a] and assign[b] and assign[c] and assign[d]:
                            new_k5 += 1
                for (a, b, c, d) in ind4_list:
                    if v in (a, b, c, d):
                        if not assign[a] and not assign[b] and not assign[c] and not assign[d]:
                            new_ind5 += 1
                
                # Approximate total: use old total minus violations not involving v plus new violations involving v
                # This is approximate but fast
                approx_score = new_k5 + new_ind5 + (total_violations - k5_violations - ind5_violations + new_k5 + new_ind5)
                # Simpler: just use the count of violations involving v
                if new_k5 + new_ind5 < best_score:
                    best_score = new_k5 + new_ind5
                    best_flip = v
                
                assign[v] = old_val
            
            if best_flip is not None:
                assign[best_flip] = 1 - assign[best_flip]
            else:
                # Stuck, try a random flip
                v = random.randrange(n)
                assign[v] = 1 - assign[v]
    
    return None


def construct():
    adj = _build_paley37()
    n = 37
    
    best_n = 37
    best_adj = adj[:]
    
    # Try to extend the graph by adding vertices one at a time
    for target_n in range(38, 43):
        if time.time() > DEADLINE:
            break
        
        # Recompute k4 and ind4 for current graph
        k4_list = _find_k4_sets(adj, n)
        ind4_list = _find_ind4_sets(adj, n)
        
        # Try adding vertex with index n
        new_adj = _try_add_vertex(adj, n, k4_list, ind4_list)
        
        if new_adj is not None:
            adj = new_adj
            n = target_n
            best_n = n
            best_adj = adj[:]
            print(f"Extended to n={n}")
        else:
            # Try a few more times with different random seeds
            for retry in range(5):
                if time.time() > DEADLINE:
                    break
                new_adj = _try_add_vertex(adj, n, k4_list, ind4_list)
                if new_adj is not None:
                    adj = new_adj
                    n = target_n
                    best_n = n
                    best_adj = adj[:]
                    print(f"Extended to n={n} (retry {retry})")
                    break
            else:
                break
    
    # Build edge list from best_adj
    n_best = best_n
    edges = []
    for i in range(n_best):
        for j in range(i + 1, n_best):
            if best_adj[i] & (1 << j):
                edges.append((i, j))
    
    return n_best, edges

# EVOLVE-BLOCK-END
