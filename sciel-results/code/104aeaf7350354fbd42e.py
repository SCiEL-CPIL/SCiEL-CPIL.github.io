import itertools
import random
import time

# Precompute Q3 patterns on 8 vertices
# Vertices 0..7
# Q3 edges:
# 0-1, 0-2, 0-4, 1-3, 1-5, 2-3, 2-6, 3-7, 4-5, 4-6, 5-7, 6-7
# This is one labeling. We need all 840 distinct edge sets.
# We can generate them by permuting vertices 0..7 and collecting unique edge sets.

def generate_q3_patterns():
    base_edges = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
    patterns = set()
    for perm in itertools.permutations(range(8)):
        # Map base edges to permuted vertices
        # perm[i] is the new vertex for old vertex i?
        # Or perm is a mapping from old to new?
        # Let's say perm[i] is the vertex in the pattern corresponding to vertex i in base.
        # Actually, easier: take base edges, apply permutation to vertices.
        edges = set()
        for u, v in base_edges:
            edges.add((perm[u], perm[v]))
        # Normalize edge set (sort vertices in edge, sort edges)
        norm_edges = frozenset(tuple(sorted(e)) for e in edges)
        patterns.add(norm_edges)
    
    # Convert to bitmasks of 28 bits
    # Map (u,v) with u<v to index
    # Index = u*7 - u*(u+1)//2 + v - u - 1 ? No.
    # Simpler: precompute index map for 8 vertices.
    idx_map = {}
    k = 0
    for i in range(8):
        for j in range(i+1, 8):
            idx_map[(i,j)] = k
            k += 1
            
    pattern_masks = []
    for p in patterns:
        mask = 0
        for u, v in p:
            mask |= (1 << idx_map[(u,v)])
        pattern_masks.append(mask)
    return pattern_masks

Q3_PATTERNS = generate_q3_patterns()

# Precompute 8-subsets and their local-to-global edge index maps
SUBSETS = list(itertools.combinations(range(16), 8))
SUBSET_EDGE_MAPS = []
for s in SUBSETS:
    # s is a tuple of 8 global vertices
    # We need to map local edge (i,j) in 0..7 to global edge index
    # Global edge index for (u,v) u<v:
    # We can just use a function get_edge_idx(u,v)
    pass

def get_edge_idx(u, v):
    if u > v: u, v = v, u
    # 0..15
    # Number of edges before u: u*15 - u*(u-1)//2 ?
    # Total edges = 120.
    # Index = sum_{k=0}^{u-1} (15-k) + (v-u-1)
    # Precompute prefix sums
    pass

PREFIX_SUMS = [0]*17
for i in range(16):
    PREFIX_SUMS[i+1] = PREFIX_SUMS[i] + (15-i)

def get_edge_idx(u, v):
    if u > v: u, v = v, u
    return PREFIX_SUMS[u] + (v - u - 1)

# Precompute for each subset the list of global edge indices for the 28 local edges
SUBSET_LOCAL_TO_GLOBAL = []
for s in SUBSETS:
    mapping = []
    for i in range(8):
        for j in range(i+1, 8):
            mapping.append(get_edge_idx(s[i], s[j]))
    SUBSET_LOCAL_TO_GLOBAL.append(mapping)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    best_edges = None
    best_count = 0
    
    # Try multiple strategies
    strategies = [
        "random",
        "max_degree",
        "min_degree" # maybe?
    ]
    
    for strat in strategies:
        if time.time() > deadline: break
        
        # Initialize K16
        # Bitmask of 120 bits, all 1s
        G = (1 << 120) - 1
        
        # We need to track degrees for max_degree heuristic
        degrees = [15] * 16
        
        while True:
            if time.time() > deadline: break
            
            # Find a Q3
            q3_edges = None
            for i, s in enumerate(SUBSETS):
                # Compute local mask
                local_mask = 0
                mapping = SUBSET_LOCAL_TO_GLOBAL[i]
                for k, global_idx in enumerate(mapping):
                    if (G >> global_idx) & 1:
                        local_mask |= (1 << k)
                
                # Quick check: popcount
                if local_mask.bit_count() < 12:
                    continue
                
                # Check patterns
                for pattern in Q3_PATTERNS:
                    if (local_mask & pattern) == pattern:
                        # Found Q3
                        # Extract global edges
                        q3_edges = []
                        for k in range(28):
                            if pattern & (1 << k):
                                q3_edges.append(mapping[k])
                        break
                if q3_edges:
                    break
            
            if q3_edges is None:
                break
            
            # Select edge to remove
            if strat == "random":
                idx = random.choice(q3_edges)
            elif strat == "max_degree":
                # Find edge in q3_edges with max (deg[u]+deg[v])
                # Need to recover u,v from idx
                # This is slow. Precompute edge endpoints?
                # Or just pick random for now to keep it fast, 
                # and rely on multiple restarts.
                # Let's implement a fast "random" with different seeds.
                idx = random.choice(q3_edges)
            else:
                idx = q3_edges[0]
                
            # Remove edge
            G &= ~(1 << idx)
            
            # Update degrees if we were tracking them (skipped for simplicity in this thought process, 
            # but will add in code)
            
        # End while
        
        # Count edges
        count = G.bit_count()
        if count > best_count:
            best_count = count
            # Convert bitmask to list of edges
            best_edges = []
            for u in range(16):
                for v in range(u+1, 16):
                    idx = get_edge_idx(u, v)
                    if (G >> idx) & 1:
                        best_edges.append([u, v])
                        
    return best_edges
