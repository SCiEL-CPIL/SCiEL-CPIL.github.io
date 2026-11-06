import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    def check_time():
        return time.time() < deadline

    # Try to find a 42-vertex graph
    # If we find one, return it.
    # If not, fall back to Paley 37.
    
    def get_paley_graph(n):
        q = n
        residues = {(x * x) % q for x in range(1, q)}
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % q in residues:
                    adj[i].add(j)
                    adj[j].add(i)
        return adj

    def is_valid(adj, n):
        # Check for K5 and Ind 5
        for combo in combinations(range(n), 5):
            is_clique = True
            is_indep = True
            for u, v in combinations(combo, 2):
                if v not in adj[u]:
                    is_clique = False
                else:
                    is_indep = False
                if not is_clique and not is_indep:
                    break
            if is_clique or is_indep:
                return False
        return True

    def count_k3_in_set(adj, node_set):
        """Count K3s in the induced subgraph of node_set"""
        count = 0
        # Convert to list for index access if needed, but set intersection is fine
        for u in node_set:
            for v in node_set:
                if u < v and v in adj[u]:
                    # Check if there is a w in node_set connected to both
                    common = adj[u] & adj[v] & node_set
                    if common:
                        count += len(common)
        return count // 3

    def count_ind3_in_set(adj, n, node_set):
        """Count independent 3-sets in node_set"""
        count = 0
        # An independent set is a clique in the complement graph
        # Or just check all triples
        # For efficiency, we can iterate
        for u in node_set:
            for v in node_set:
                if u < v and v not in adj[u]:
                    # Check if there is a w in node_set NOT connected to u or v
                    # w must be in node_set, w != u, w != v
                    # w not in adj[u] and w not in adj[v]
                    non_neighbors_u = set(range(n)) - adj[u] - {u}
                    non_neighbors_v = set(range(n)) - adj[v] - {v}
                    common_non = non_neighbors_u & non_neighbors_v & node_set
                    if common_non:
                        count += len(common_non)
        return count // 3

    # The above count functions are O(N^3) or worse.
    # Let's optimize count_k3_in_set.
    # For a set S, K3s are triangles.
    # We can iterate over edges in S and count common neighbors in S.
    
    def fast_count_k3(adj, node_set):
        count = 0
        # Iterate over pairs in node_set
        # This is still O(M^2 * M)
        # Let's just use the simple one, M is small.
        for u in node_set:
            for v in node_set:
                if u < v and v in adj[u]:
                    common = adj[u] & adj[v] & node_set
                    count += len(common)
        return count // 3

    def fast_count_ind3(adj, n, node_set):
        # Independent 3-set: u, v, w such that no edges between them.
        # This is equivalent to K3 in the complement.
        # Let's just check triples.
        count = 0
        nodes = list(node_set)
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                if nodes[j] not in adj[nodes[i]]:
                    # Check k
                    for k in range(j+1, len(nodes)):
                        if nodes[k] not in adj[nodes[i]] and nodes[k] not in adj[nodes[j]]:
                            count += 1
        return count

    # These are too slow for 300k flips.
    # We need a faster delta.
    
    # Let's try a different strategy:
    # Use the Paley graph (37) and try to add 5 vertices.
    # The previous program failed at this.
    # Why? Because adding a vertex requires finding a subset S of 37 vertices
    # such that S has no K4 and complement of S has no Ind 4.
    
    # Let's try to find such a subset S using Local Search on the 37 vertices.
    # Objective: Minimize (K4s in S) + (Ind 4s in complement of S).
    
    def find_valid_subset(adj, n):
        """Find a subset S of range(n) such that S has no K4 and comp(S) has no Ind 4."""
        # Use Simulated Annealing on the subset S.
        # S is a bitmask of n bits.
        
        # Precompute all K4s and Ind 4s in the graph.
        k4s = []
        ind4s = []
        for combo in combinations(range(n), 4):
            is_clique = True
            is_indep = True
            for u, v in combinations(combo, 2):
                if v not in adj[u]:
                    is_clique = False
                else:
                    is_indep = False
                if not is_clique and not is_indep:
                    break
            if is_clique:
                k4s.append(combo)
            if is_indep:
                ind4s.append(combo)
        
        # Objective: Count K4s fully in S + Count Ind 4s fully in complement of S.
        
        def count_violations(S_mask):
            violations = 0
            # K4s in S
            for k4 in k4s:
                # Check if all 4 are in S_mask
                if all((S_mask >> v) & 1 for v in k4):
                    violations += 1
            # Ind 4s in complement of S
            # Complement mask
            comp_mask = ((1 << n) - 1) ^ S_mask
            for ind4 in ind4s:
                if all((comp_mask >> v) & 1 for v in ind4):
                    violations += 1
            return violations

        # Initial random S
        S_mask = 0
        for i in range(n):
            if random.random() < 0.5:
                S_mask |= (1 << i)
        
        current_violations = count_violations(S_mask)
        best_violations = current_violations
        best_mask = S_mask
        
        T = 10.0
        cooling_rate = 0.999
        
        while check_time():
            # Flip a random bit
            i = random.randint(0, n - 1)
            new_S_mask = S_mask ^ (1 << i)
            new_violations = count_violations(new_S_mask)
            
            delta = new_violations - current_violations
            
            if delta < 0 or random.random() < pow(2.718, -delta / T):
                S_mask = new_S_mask
                current_violations = new_violations
                
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_mask = S_mask
                    
                    if best_violations == 0:
                        # Found a valid subset
                        S = [i for i in range(n) if (best_mask >> i) & 1]
                        return S
            
            T *= cooling_rate
            if T < 0.1:
                break
        
        return None

    # Try to extend the Paley graph
    n_paley = 37
    adj_paley = get_paley_graph(n_paley)
    
    # We want to add 5 vertices.
    # For each new vertex, we need a valid subset of the *current* graph.
    # But the current graph changes as we add vertices.
    
    # Let's try to add vertices one by one.
    adj = adj_paley
    n = n_paley
    
    added = 0
    while added < 5 and check_time():
        S = find_valid_subset(adj, n)
        if S is None:
            break
        
        # Add new vertex n
        new_v = n
        adj.append(set(S))
        for v in S:
            adj[v].add(new_v)
        n += 1
        added += 1
        
    # Check if the resulting graph is valid
    if is_valid(adj, n):
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
    
    # If failed, return Paley
    n = n_paley
    adj = get_paley_graph(n)
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
    return n, edges
