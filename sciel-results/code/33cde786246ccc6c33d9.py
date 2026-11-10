import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n = 18
    
    def is_k4_free(adj):
        # adj is a list of bitmasks
        # Check all 4-cliques
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    continue
                for k in range(j + 1, n):
                    if not (adj[i] & (1 << k)) or not (adj[j] & (1 << k)):
                        continue
                    # Check if l forms a K4 with i, j, k
                    # Candidates for l must be neighbors of i, j, k
                    # l > k to avoid duplicates
                    candidates = adj[i] & adj[j] & adj[k]
                    # Remove i, j, k themselves
                    candidates &= ~(1 << i) & ~(1 << j) & ~(1 << k)
                    # Check if any l > k is in candidates
                    # Actually, just check if there is any l in candidates with l > k
                    # Or simply, if candidates has any bit set for l > k
                    # We can just check if candidates & (sum(1<<l for l in range(k+1, n))) is non-zero
                    # Or simpler: iterate l
                    for l in range(k + 1, n):
                        if (adj[i] & (1 << l)) and (adj[j] & (1 << l)) and (adj[k] & (1 << l)):
                            return False
        return True

    def get_triangles(adj):
        triangles = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    continue
                common = adj[i] & adj[j] & ~(1 << i) & ~(1 << j)
                # k must be > j to avoid duplicates? No, just check all k
                # But to avoid duplicates, enforce i < j < k
                # So k > j
                k_mask = common & (sum(1 << x for x in range(j + 1, n)))
                while k_mask:
                    # Extract lowest set bit
                    lsb = k_mask & -k_mask
                    k = lsb.bit_length() - 1
                    triangles.append((i, j, k))
                    k_mask ^= lsb
        return triangles

    def max_triangle_free_size(adj, triangles):
        # Check subsets of size k from n down to 1
        # If triangles is empty, return n
        if not triangles:
            return n
        
        # Precompute triangle bitmasks
        tri_masks = [0] * len(triangles)
        for idx, (i, j, k) in enumerate(triangles):
            tri_masks[idx] = (1 << i) | (1 << j) | (1 << k)
            
        # Iterate k from n down to 1
        for k in range(n, 0, -1):
            # Check all subsets of size k
            for subset in itertools.combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                # Check if mask contains any triangle
                contains_tri = False
                for tm in tri_masks:
                    if (mask & tm) == tm:
                        contains_tri = True
                        break
                if not contains_tri:
                    return k
        return 0

    def random_3_regular_graph(n):
        # Generate a random 3-regular graph
        # Use configuration model or simple swapping
        # For n=18, simple random graph and adjust degrees might be hard.
        # Let's try random graph with p=0.5 and check K4-free, then maybe modify.
        # Better: Start with a random graph, remove K4s.
        
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Remove K4s
        # Greedy removal
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[i] & (1 << j)):
                        continue
                    for k in range(j + 1, n):
                        if not (adj[i] & (1 << k)) or not (adj[j] & (1 << k)):
                            continue
                        for l in range(k + 1, n):
                            if (adj[i] & (1 << l)) and (adj[j] & (1 << l)) and (adj[k] & (1 << l)):
                                # Remove edge (i, l)
                                adj[i] &= ~(1 << l)
                                adj[l] &= ~(1 << i)
                                changed = True
                                break
                        if changed:
                            break
                    if changed:
                        break
                if changed:
                    break
        return adj

    best_score = 12
    best_edges = None
    
    # Seed construction: K6,6,6
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    seed_edges = [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
    
    # Try to find a better graph
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        adj = random_3_regular_graph(n)
        
        if not is_k4_free(adj):
            continue
            
        triangles = get_triangles(adj)
        s = max_triangle_free_size(adj, triangles)
        
        if s < best_score:
            best_score = s
            # Convert adj to edges
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i] & (1 << j):
                        edges.append([i, j])
            best_edges = edges
            
        if best_score <= 10: # Likely optimal or near optimal
            break
            
    if best_edges is not None:
        return best_edges
    else:
        return seed_edges

# EVOLVE-BLOCK-START
# The construct function is defined above.
# We need to wrap it or just define it here.
# The prompt asks to rewrite the program.
# I will place the whole logic inside the EVOLVE-BLOCK.
# EVOLVE-BLOCK-END
