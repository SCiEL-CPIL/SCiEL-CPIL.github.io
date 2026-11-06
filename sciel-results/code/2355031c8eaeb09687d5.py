import time
from math import gcd
from fractions import Fraction

def construct():
    deadline = time.time() + 55
    
    N = 256
    nodes = list(range(2, N + 1))
    n = len(nodes)
    
    # Build adjacency: adj[i] is a bitset of nodes compatible with node i
    # Two nodes are compatible if lcm(a,b) > 256
    adj = [0] * n
    
    for i in range(n):
        a = nodes[i]
        for j in range(n):
            if i == j:
                continue
            b = nodes[j]
            g = gcd(a, b)
            # lcm(a,b) = a*b/g > 256  <=>  a*b > 256*g
            if a * b > 256 * g:
                adj[i] |= (1 << j)
    
    # Weights: 1/x for each node
    weights = [1.0 / x for x in nodes]
    
    # Maximum weight clique via branch and bound with bitsets
    # State: (candidates_bitset, current_weight, current_clique)
    
    best_weight = 0.0
    best_clique = []
    
    # Greedy coloring upper bound for a set of candidate vertices
    # We color the subgraph induced by candidates to get an upper bound on clique weight
    def greedy_color_bound(candidates, weights):
        """
        Returns upper bound on max weight clique in candidates.
        Uses greedy coloring on the candidate subgraph.
        """
        if candidates == 0:
            return 0.0
        
        # Get list of vertices in candidates
        verts = []
        v = candidates
        while v:
            lowbit = v & (-v)
            idx = lowbit.bit_length() - 1
            verts.append(idx)
            v ^= lowbit
        
        if not verts:
            return 0.0
        
        # Greedy coloring: assign colors greedily
        # For upper bound on weighted clique, we need to be careful
        # Standard approach: color the complement (conflict) graph, but here
        # we want clique in original graph = independent set in complement
        # Actually for max weight clique, we can use:
        # Order vertices, assign each to first available color
        # The max weight clique has weight <= sum of max weight per color class
        # But this isn't tight for weights. Let's use a simpler bound.
        
        # Simpler: just return sum of all weights as a loose upper bound
        # Or use a better heuristic: process vertices in order, maintain color classes
        # Upper bound = sum over color classes of max weight in that class
        
        # For efficiency, let's use a simple bound: sum of top k weights where
        # k is the chromatic number upper bound from greedy coloring
        # Actually, let's just compute sum of weights for now and refine
        
        # Better approach: use the fact that in a proper coloring, each color class
        # is an independent set, so a clique can have at most one vertex per color.
        # Thus max weight clique <= sum of max weight in each color class.
        
        color_of = [-1] * n
        num_colors = 0
        color_max_weight = []
        
        for v_idx in verts:
            # Find a color not used by any neighbor of v_idx in candidates
            used_colors = set()
            neighbors = adj[v_idx] & candidates
            nb = neighbors
            while nb:
                lowbit = nb & (-nb)
                nidx = lowbit.bit_length() - 1
                if color_of[nidx] >= 0:
                    used_colors.add(color_of[nidx])
                nb ^= lowbit
            
            # Assign smallest available color
            c = 0
            while c in used_colors:
                c += 1
            
            if c >= len(color_max_weight):
                color_max_weight.append(0.0)
            
            if weights[v_idx] > color_max_weight[c]:
                color_max_weight[c] = weights[v_idx]
            
            color_of[v_idx] = c
            num_colors = max(num_colors, c + 1)
        
        return sum(color_max_weight[:num_colors])
    
    # Branch and bound
    def branch_and_bound(candidates, current_weight, current_clique):
        nonlocal best_weight, best_clique
        
        if time.time() > deadline - 2:
            return
        
        if candidates == 0:
            if current_weight > best_weight:
                best_weight = current_weight
                best_clique = list(current_clique)
            return
        
        # Upper bound check
        ub = current_weight + greedy_color_bound(candidates, weights)
        if ub <= best_weight + 1e-12:
            return
        
        # Get vertices in candidates
        verts = []
        v = candidates
        while v:
            lowbit = v & (-v)
            idx = lowbit.bit_length() - 1
            verts.append(idx)
            v ^= lowbit
        
        # Sort by weight descending for better pruning
        verts.sort(key=lambda x: weights[x], reverse=True)
        
        for v_idx in verts:
            if time.time() > deadline - 2:
                return
            
            # Check if v_idx is still in candidates
            if not (candidates & (1 << v_idx)):
                continue
            
            # Add v_idx to clique
            current_clique.append(v_idx)
            new_weight = current_weight + weights[v_idx]
            
            # New candidates are neighbors of v_idx that are still in candidates
            new_candidates = candidates & adj[v_idx]
            
            branch_and_bound(new_candidates, new_weight, current_clique)
            
            current_clique.pop()
            
            # Remove v_idx from candidates for subsequent iterations
            candidates &= ~(1 << v_idx)
    
    # Start with all nodes as candidates
    all_candidates = (1 << n) - 1
    branch_and_bound(all_candidates, 0.0, [])
    
    if not best_clique:
        # Fallback: pick the single best node
        best_idx = max(range(n), key=lambda i: weights[i])
        best_clique = [best_idx]
    
    # Convert back to node values
    result = [nodes[i] for i in best_clique]
    
    # Verify the result is valid
    if len(result) > 1:
        valid = True
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                a, b = result[i], result[j]
                g = gcd(a, b)
                if a * b <= 256 * g:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback to greedy
            greedy_set = []
            for i in nodes:
                ok = True
                for y in greedy_set:
                    if i * y <= 256 * gcd(i, y):
                        ok = False
                        break
                if ok:
                    greedy_set.append(i)
            result = greedy_set
    
    return result
