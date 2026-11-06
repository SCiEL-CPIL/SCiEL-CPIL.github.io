import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Fallback: Paley graph on 17 vertices
    def get_paley_17():
        squares = {x * x % 17 for x in range(1, 17)}
        edges = []
        for i in range(17):
            for j in range(i + 1, 17):
                if (j - i) % 17 in squares:
                    edges.append([i, j])
        return {'n': 17, 'edges': edges}

    best_result = get_paley_17()
    
    # Target n values to try, descending
    targets = [36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20]
    
    for n in targets:
        if time.time() > deadline:
            break
        
        # Try to find a valid graph for this n
        result = try_find_graph(n, deadline)
        if result:
            # Verify the result before accepting
            if is_valid(result):
                best_result = result
                # If we found 35 or 36, we can stop as it's optimal or near-optimal
                if n >= 35:
                    return best_result
                # Continue to try for larger n? No, we iterate descending.
                # Since we iterate descending, if we find 35, 36 already failed.
                # So we can return 35 immediately.
                return best_result

    return best_result

def is_valid(graph):
    n = graph['n']
    edges = graph['edges']
    # Build adjacency matrix using bitsets for speed
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Check K4
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                common_ij = adj[i] & adj[j]
                # Remove i and j from common
                common_ij &= ~(1 << i)
                common_ij &= ~(1 << j)
                # Iterate over pairs in common_ij
                # We need to check if any two vertices in common_ij are adjacent
                # This is equivalent to checking if the induced subgraph on common_ij has an edge
                # Let's iterate through bits of common_ij
                bits = []
                temp = common_ij
                while temp:
                    lsb = temp & (-temp)
                    bits.append(lsb.bit_length() - 1)
                    temp ^= lsb
                for k_idx in range(len(bits)):
                    k = bits[k_idx]
                    for l_idx in range(k_idx + 1, len(bits)):
                        l = bits[l_idx]
                        if adj[k] & (1 << l):
                            return False
    
    # Check Independent 6-set
    # Equivalent to K6 in complement
    # Complement adjacency: ~adj[i] & ((1<<n)-1) & ~(1<<i)
    mask = (1 << n) - 1
    comp_adj = [0] * n
    for i in range(n):
        comp_adj[i] = (~adj[i]) & mask & ~(1 << i)
    
    for i in range(n):
        for j in range(i + 1, n):
            if comp_adj[i] & (1 << j):
                common_ij = comp_adj[i] & comp_adj[j]
                common_ij &= ~(1 << i)
                common_ij &= ~(1 << j)
                bits = []
                temp = common_ij
                while temp:
                    lsb = temp & (-temp)
                    bits.append(lsb.bit_length() - 1)
                    temp ^= lsb
                for k_idx in range(len(bits)):
                    k = bits[k_idx]
                    for l_idx in range(k_idx + 1, len(bits)):
                        l = bits[l_idx]
                        if comp_adj[k] & (1 << l):
                            common_kl = comp_adj[k] & comp_adj[l]
                            common_kl &= ~(1 << i)
                            common_kl &= ~(1 << j)
                            common_kl &= ~(1 << k)
                            common_kl &= ~(1 << l)
                            # Need one more vertex m connected to i,j,k,l in complement
                            # So m must be in common_kl and also in comp_adj[i], comp_adj[j]
                            # But common_kl is already intersection of comp_adj[k] and comp_adj[l]
                            # We need m such that comp_adj[m] has i,j,k,l
                            # Let's just iterate bits of common_kl
                            temp2 = common_kl
                            while temp2:
                                lsb2 = temp2 & (-temp2)
                                m = lsb2.bit_length() - 1
                                # Check if m is connected to i, j in complement
                                if comp_adj[m] & (1 << i) and comp_adj[m] & (1 << j):
                                    return False
                                temp2 ^= lsb2
    return True

def try_find_graph(n, deadline):
    # Structural change: Use a SAT-inspired local search with weighted violations
    # and multiple diverse restarts.
    
    # We will use a bitset representation for adjacency for speed.
    # adj[i] is a bitmask of neighbors of i.
    
    def bitset_to_edges(adj, n):
        edges = []
        for i in range(n):
            temp = adj[i] & ((1 << i) - 1)  # Only bits lower than i
            while temp:
                lsb = temp & (-temp)
                j = lsb.bit_length() - 1
                edges.append([i, j])
                temp ^= lsb
        return edges

    def count_violations(adj, n):
        # Returns (k4_count, i6_count)
        mask = (1 << n) - 1
        k4_count = 0
        i6_count = 0
        
        # Count K4s
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    common = adj[i] & adj[j] & ~(1 << i) & ~(1 << j)
                    # Count edges in common
                    # This is expensive. Let's just check for existence for violation finding,
                    # but for counting, we might need a faster way or just skip counting
                    # and just find one violation.
                    # Let's just find one violation for the search step.
                    pass
        
        # Let's just find one violation of each type to guide the flip.
        # Return a tuple (viol_type, viol_vertices) or None
        # Actually, let's return a list of violations to choose from.
        
        violations = []
        
        # Find K4s
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    common = adj[i] & adj[j] & ~(1 << i) & ~(1 << j)
                    # Find an edge in common
                    temp = common
                    while temp:
                        lsb = temp & (-temp)
                        k = lsb.bit_length() - 1
                        temp ^= lsb
                        common_k = common & adj[k] & ~(1 << k)
                        if common_k:
                            # Find a vertex l in common_k
                            lsb2 = common_k & (-common_k)
                            l = lsb2.bit_length() - 1
                            violations.append(('K4', (i, j, k, l)))
                            break
                    if violations:
                        break
            if violations:
                break
        
        if not violations:
            # Find Independent 6-sets
            comp_adj = [0] * n
            for i in range(n):
                comp_adj[i] = (~adj[i]) & mask & ~(1 << i)
            
            # Use a heuristic search for independent set of size 6
            # Greedy: pick vertex with max degree in complement?
            # Or just iterate.
            # Let's use a simple backtracking with ordering.
            
            def find_indep_6(comp_adj, n):
                # Order by degree in complement descending
                degrees = [bin(comp_adj[i]).count('1') for i in range(n)]
                order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
                
                def backtrack(current, candidates):
                    if len(current) == 6:
                        return current
                    if not candidates:
                        return None
                    if len(candidates) + len(current) < 6:
                        return None
                    
                    # Pick a vertex
                    # Heuristic: pick the one with highest degree in the induced subgraph?
                    # For simplicity, pick the first one that allows progress.
                    for idx, v in enumerate(candidates):
                        # Check if v is independent from current
                        valid = True
                        for u in current:
                            if not (comp_adj[u] & (1 << v)):
                                valid = False
                                break
                        if valid:
                            new_candidates = [w for w in candidates[idx+1:] if comp_adj[v] & (1 << w)]
                            res = backtrack(current + [v], new_candidates)
                            if res:
                                return res
                    return None
                
                return backtrack([], order)
            
            indep = find_indep_6(comp_adj, n)
            if indep:
                violations.append(('I6', tuple(indep)))
        
        return violations

    def flip_edge(adj, u, v, n):
        if adj[u] & (1 << v):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

    max_restarts = 10
    max_iters_per_restart = 2000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            return None
        
        # Initialize with random graph p=0.5
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Local Search
        for it in range(max_iters_per_restart):
            if time.time() > deadline:
                return None
            
            violations = count_violations(adj, n)
            if not violations:
                edges = bitset_to_edges(adj, n)
                return {'n': n, 'edges': edges}
            
            # Pick a random violation
            vtype, verts = random.choice(violations)
            
            if vtype == 'K4':
                i, j, k, l = verts
                # Random edge to remove
                pairs = [(i,j), (i,k), (i,l), (j,k), (j,l), (k,l)]
                u, v = random.choice(pairs)
                flip_edge(adj, u, v, n)
            else: # I6
                i, j, k, l, m, p = verts
                # Random non-edge to add
                pairs = [(i,j), (i,k), (i,l), (i,m), (i,p), (j,k), (j,l), (j,m), (j,p), (k,l), (k,m), (k,p), (l,m), (l,p), (m,p)]
                u, v = random.choice(pairs)
                flip_edge(adj, u, v, n)
        
    return None
