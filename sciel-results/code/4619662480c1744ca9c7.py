import time
import itertools
import random

N = 18

def is_k4_free(adj):
    """Check if graph with adjacency bitmasks is K4-free."""
    for a in range(N):
        na = adj[a]
        for b in range(a + 1, N):
            if not (na >> b) & 1:
                continue
            nb = adj[b]
            # For each common neighbor c of a and b (c > b to avoid duplicates),
            # check if a, b, c form a triangle and if that triangle has a common neighbor d
            common_ab = na & nb
            # Only consider c > b
            common_ab &= ~((1 << (b + 1)) - 1)
            c_mask = common_ab
            while c_mask:
                c_bit = c_mask & -c_mask
                c = c_bit.bit_length() - 1
                c_mask ^= c_bit
                nc = adj[c]
                # Check if a, b, c form a triangle (need edge a-c and b-c)
                if (na & nc) and (nb & nc):
                    # Triangle a,b,c exists. Check if there's a common neighbor d
                    # d must connect to all of a, b, c
                    if (na & nb & nc):
                        return False
    return True

def max_triangle_free_induced(adj):
    """Find the size of the largest triangle-free induced subgraph."""
    # Brute force over all subsets, from largest to smallest
    # For a subset S, it's triangle-free iff for every edge (u,v) in S,
    # N(u) ∩ N(v) ∩ S = ∅
    
    # Precompute for efficiency
    # Iterate k from N down to 1
    for k in range(N, 0, -1):
        for subset in itertools.combinations(range(N), k):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            is_tf = True
            # Check all pairs in subset
            for i, u in enumerate(subset):
                au = adj[u]
                for v in subset[i + 1:]:
                    if (au >> v) & 1:
                        # Edge (u,v) exists. Check if there's a common neighbor in subset
                        common = au & adj[v] & mask
                        if common:
                            is_tf = False
                            break
                if not is_tf:
                    break
            if is_tf:
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 150s limit, 20s safety margin
    
    # Start with Turán graph T_3(18): three parts of 6 vertices each,
    # complete bipartite between each pair of parts, no edges within parts.
    # Parts: 0-5, 6-11, 12-17
    # This is K4-free (any 4 vertices have two in the same part, not adjacent)
    # Largest triangle-free induced subgraph: one part (6 vertices) is triangle-free.
    # Actually, we can take one part entirely (6 vertices, no edges, triangle-free).
    # Can we do better? A triangle-free induced subgraph in T_3(18):
    # If we take vertices from multiple parts, we get edges between parts.
    # To be triangle-free, we can't have a triangle. In T_3, a triangle would need
    # one vertex from each of the three parts. So we can take all vertices from two parts
    # (12 vertices) but that has edges between the two parts, forming many triangles?
    # No: if we take all from part A and all from part B, edges only between A and B.
    # A triangle needs 3 mutually adjacent vertices. With only A-B edges, no triangle.
    # So A ∪ B (12 vertices) is triangle-free!
    # Wait, that means L(T_3(18)) = 12.
    
    # Hmm, so T_3(18) has L = 12. Score = 18/12 = 1.5.
    
    # The random search found L = 15 (score 1.2). T_3 gives 1.5 which is better.
    
    best_l = 12
    best_adj = None
    
    # Build T_3(18)
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    adj = [0] * N
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
    
    if is_k4_free(adj):
        l = max_triangle_free_induced(adj)
        if l < best_l:
            best_l = l
            best_adj = adj[:]
    
    # Try to improve by searching near T_3(18)
    # Local search: try adding/removing edges while maintaining K4-free
    
    while time.time() < deadline:
        # Random perturbation of T_3
        trial_adj = best_adj[:]
        
        # Randomly flip some edges
        for _ in range(random.randint(1, 5)):
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i != j:
                bit_i = 1 << i
                bit_j = 1 << j
                # Flip edge (i,j)
                if trial_adj[i] & bit_j:
                    trial_adj[i] &= ~bit_j
                    trial_adj[j] &= ~bit_i
                else:
                    trial_adj[i] |= bit_j
                    trial_adj[j] |= bit_i
        
        # Repair K4-freeness
        for _ in range(50):
            if is_k4_free(trial_adj):
                break
            # Find a K4 and remove an edge
            found = False
            for a in range(N):
                na = trial_adj[a]
                for b in range(a + 1, N):
                    if not (na >> b) & 1:
                        continue
                    nb = trial_adj[b]
                    common_ab = na & nb
                    common_ab &= ~((1 << (b + 1)) - 1)
                    c_mask = common_ab
                    while c_mask:
                        c_bit = c_mask & -c_mask
                        c = c_bit.bit_length() - 1
                        c_mask ^= c_bit
                        nc = trial_adj[c]
                        if (na & nc) and (nb & nc):
                            common = na & nb & nc
                            if common:
                                d_bit = common & -common
                                d = d_bit.bit_length() - 1
                                # Remove edge (a, d)
                                trial_adj[a] &= ~(1 << d)
                                trial_adj[d] &= ~(1 << a)
                                found = True
                                break
                    if found:
                        break
                if found:
                    break
            if not found:
                break
        
        if is_k4_free(trial_adj):
            l = max_triangle_free_induced(trial_adj)
            if l < best_l:
                best_l = l
                best_adj = trial_adj[:]
    
    # Convert to edge list
    edges = []
    if best_adj is None:
        # Fallback
        for i in range(3):
            for j in range(i + 1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        edges.append([a, b])
    else:
        for i in range(N):
            for j in range(i + 1, N):
                if best_adj[i] & (1 << j):
                    edges.append([i, j])
    
    return edges
