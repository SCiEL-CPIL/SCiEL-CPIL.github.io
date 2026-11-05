import time
import random
from math import gcd
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """
    Find a large subset of [1, 128] with no three distinct elements whose
    three pairwise gcds are equal.
    """
    N = 128
    deadline = time.time() + 85.0  # Safety margin (limit 90s)
    
    # Precompute GCDs for all pairs in [1, 128]
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute bad edges: triples (a, b, c) where gcd(a,b) == gcd(a,c) == gcd(b,c)
    # and a < b < c
    bad_edges = []
    bad_edge_set = set()
    
    # For each possible gcd value d, find all pairs with gcd d
    # Then check if any three form a bad triple
    from collections import defaultdict
    pairs_by_gcd = defaultdict(list)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = gcd_table[i][j]
            pairs_by_gcd[g].append((i, j))
    
    # For each gcd value, check all triples of pairs that share a common element
    # Actually, we need triples (a,b,c) where all three pairwise gcds are equal
    # Let's just enumerate all triples directly - there are C(128,3) = 341,824
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd_table[a][b]
            for c in range(b + 1, N + 1):
                g_ac = gcd_table[a][c]
                g_bc = gcd_table[b][c]
                if g_ab == g_ac == g_bc:
                    bad_edges.append((a, b, c))
                    bad_edge_set.add((a, b, c))
    
    # For each vertex, precompute which other vertices it forms bad edges with
    # vertex_conflicts[v] = set of (u, w) pairs such that (v, u, w) is a bad edge
    # Better: for each vertex v, and each other vertex u, count how many w exist
    # such that (v, u, w) is a bad edge. This is the "degree" in the hypergraph.
    
    # For efficient conflict checking when adding x to a set S:
    # x conflicts with S if there exist a, b in S such that (x, a, b) is a bad edge
    # Precompute: for each x, for each a, the set of b such that (x, a, b) is a bad edge
    # Let's use: conflicts[x][a] = set of b where {x,a,b} is a bad edge
    
    # Build adjacency: for each vertex, list of (other1, other2) bad edges containing it
    vertex_edges = defaultdict(list)
    for (a, b, c) in bad_edges:
        vertex_edges[a].append((b, c))
        vertex_edges[b].append((a, c))
        vertex_edges[c].append((a, b))
    
    # For fast conflict checking: when adding x to S, check if any (u,v) in S
    # forms a bad edge with x. Precompute for each x, a set of frozenset({u,v}) 
    # that conflict with x.
    
    # Actually, let's precompute for each pair (u,v) the set of x that form bad edges with u,v
    # pair_to_x[(u,v)] = set of x such that {x,u,v} is a bad edge
    # Then adding x to S conflicts if there exists (u,v) in S such that x in pair_to_x[(u,v)]
    
    # Alternative: for each x, precompute conflict_pairs[x] = set of (u,v) pairs
    # such that {x,u,v} is a bad edge. Then x conflicts with S if any pair in 
    # conflict_pairs[x] is a subset of S.
    
    conflict_pairs = defaultdict(set)
    for (a, b, c) in bad_edges:
        conflict_pairs[a].add((b, c))
        conflict_pairs[a].add((c, b))  # Keep both orders for consistency? No, let's keep sorted
        conflict_pairs[b].add((a, c))
        conflict_pairs[b].add((c, a))
        conflict_pairs[c].add((a, b))
        conflict_pairs[c].add((b, a))
    
    # Normalize: store as sorted tuples
    conflict_pairs_normalized = defaultdict(set)
    for x in range(1, N + 1):
        for (u, v) in conflict_pairs[x]:
            if u < v:
                conflict_pairs_normalized[x].add((u, v))
            else:
                conflict_pairs_normalized[x].add((v, u))
    
    def is_conflict(x, S):
        """Check if x can be added to S without creating a bad triple."""
        if x in S:
            return True
        for (u, v) in conflict_pairs_normalized[x]:
            if u in S and v in S:
                return True
        return False
    
    def greedy_min_degree():
        """Greedy: always pick the element with minimum conflict degree in remaining set."""
        remaining = set(range(1, N + 1))
        selected = set()
        
        while remaining:
            best_v = None
            best_deg = float('inf')
            for v in remaining:
                # Degree = number of other vertices in remaining that conflict with v
                # i.e., number of u in remaining such that is_conflict(v, selected | {u}) 
                # Actually, we want the vertex that causes fewest conflicts
                # Let's count how many pairs (u,w) in remaining form bad edges with v
                deg = 0
                for (u, w) in conflict_pairs_normalized[v]:
                    if u in remaining and w in remaining:
                        deg += 1
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            
            selected.add(best_v)
            remaining.discard(best_v)
            
            # Remove from remaining any vertex that now conflicts with selected
            to_remove = set()
            for v in remaining:
                if is_conflict(v, selected):
                    to_remove.add(v)
            remaining -= to_remove
        
        return selected
    
    def local_search(selected, max_iters=200):
        """Local search: try adding vertices, then 2-1 exchanges."""
        selected = set(selected)
        improved = True
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            # Try adding any excluded vertex
            excluded = [x for x in range(1, N + 1) if x not in selected]
            random.shuffle(excluded)
            for x in excluded:
                if not is_conflict(x, selected):
                    selected.add(x)
                    improved = True
                    break
            
            # If no add, try 2-1 exchange
            if not improved:
                # Try removing 1 and adding 2
                sel_list = list(selected)
                random.shuffle(sel_list)
                for x in sel_list:
                    temp = selected - {x}
                    added = []
                    for y in excluded:
                        if y != x and not is_conflict(y, temp | set(added)):
                            added.append(y)
                            if len(added) >= 2:
                                break
                    if len(added) >= 2:
                        new_set = temp | set(added)
                        if len(new_set) > len(selected):
                            selected = new_set
                            improved = True
                            break
                    elif len(added) == 1:
                        # 1-for-1 swap
                        new_set = temp | set(added)
                        # Check if we can add more
                        further = []
                        for z in excluded:
                            if z != x and z not in added and not is_conflict(z, new_set | set(further)):
                                further.append(z)
                                if len(further) >= 1:
                                    break
                        if further:
                            selected = new_set | set(further)
                            improved = True
                            break
        
        return selected
    
    # Run multiple strategies
    best_set = set()
    best_size = 0
    
    # Strategy 1: Min-degree greedy
    s1 = greedy_min_degree()
    s1 = local_search(s1)
    best_size = len(s1)
    best_set = s1
    
    if time.time() > deadline:
        return sorted(best_set) if best_set else [1]
    
    # Strategy 2: Randomized greedy + local search
    random.seed(123)
    for trial in range(200):
        if time.time() > deadline:
            break
        
        # Randomized greedy
        remaining = set(range(1, N + 1))
        current = set()
        
        while remaining:
            # Pick randomly among low-degree vertices
            candidates = list(remaining)
            random.shuffle(candidates)
            # Sort by degree (approximate)
            def deg(v):
                count = 0
                for (u, w) in conflict_pairs_normalized[v]:
                    if u in remaining and w in remaining:
                        count += 1
                return count
            candidates.sort(key=deg)
            # Pick from top k
            k = min(5, len(candidates))
            chosen = random.choice(candidates[:k])
            current.add(chosen)
            remaining.discard(chosen)
            
            to_remove = set()
            for v in remaining:
                if is_conflict(v, current):
                    to_remove.add(v)
            remaining -= to_remove
        
        current = local_search(current)
        if len(current) > best_size:
            best_size = len(current)
            best_set = current
    
    # Final polish
    if time.time() < deadline:
        final = local_search(best_set, max_iters=500)
        if len(final) > best_size:
            best_size = len(final)
            best_set = final
    
    if not best_set:
        best_set = {1}
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
