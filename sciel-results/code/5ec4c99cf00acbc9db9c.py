"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 15}

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Precompute edge indices
    edge_list = []
    edge_to_idx = {}
    for u in range(n):
        for v in range(u+1, n):
            edge_to_idx[(u, v)] = len(edge_list)
            edge_list.append((u, v))
    E = len(edge_list)  # 105
    
    # For fast validity: for each edge (u,v), precompute which other edges
    # share u or v (these are the "adjacent" edges that form 3-paths)
    # edge_neighbors[i] = list of (other_edge_idx, endpoint_s, endpoint_t)
    # where edge i is (u,v), other edge shares u giving s, or shares v giving t
    # Actually, for path s-u-v-t: s is neighbor of u (not v), t is neighbor of v (not u)
    # So for edge (u,v), the relevant pairs are (s in N(u)\{v}, t in N(v)\{u})
    
    # Fast validity check using degree classes
    def is_valid_fast(adj, deg):
        """Check validity given adjacency lists and degree array."""
        # For each edge (u,v), check if there exists s in N(u)\{v}, t in N(v)\{u}
        # with s != t and deg[s] == deg[t]
        for i in range(E):
            u, v = edge_list[i]
            if v not in adj[u]:
                continue
            # Get neighbors of u except v, and neighbors of v except u
            ns = [x for x in adj[u] if x != v]
            nt = [x for x in adj[v] if x != u]
            if not ns or not nt:
                continue
            # Build degree-to-vertices map for ns
            deg_to_s = {}
            for s in ns:
                ds = deg[s]
                if ds not in deg_to_s:
                    deg_to_s[ds] = []
                deg_to_s[ds].append(s)
            # For each t in nt, check if any s in ns with deg[s]==deg[t] and s!=t
            for t in nt:
                dt = deg[t]
                if dt in deg_to_s:
                    for s in deg_to_s[dt]:
                        if s != t:
                            return False
        return True
    
    def get_adj_deg(mask):
        """Build adjacency and degree from mask."""
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for i in range(E):
            if mask & (1 << i):
                u, v = edge_list[i]
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
        return adj, deg
    
    def is_valid_mask(mask):
        adj, deg = get_adj_deg(mask)
        return is_valid_fast(adj, deg)
    
    # Build initial graph: K_{7,8}
    A = set(range(7))
    B = set(range(7, 15))
    
    init_mask = 0
    for a in A:
        for b in B:
            init_mask |= (1 << edge_to_idx[(a, b)])
    
    # Check if K_{7,8} is valid
    if is_valid_mask(init_mask):
        best_mask = init_mask
        best_count = 56
    else:
        # Find best valid subset greedily
        best_mask = 0
        best_count = 0
        for i in range(E):
            new_mask = best_mask | (1 << i)
            if is_valid_mask(new_mask):
                best_mask = new_mask
                best_count += 1
    
    # Try adding any edge (multiple passes)
    for _ in range(20):
        if time.time() > deadline - 5:
            break
        changed = False
        for i in range(E):
            if best_mask & (1 << i):
                continue
            new_mask = best_mask | (1 << i)
            if is_valid_mask(new_mask):
                best_mask = new_mask
                best_count += 1
                changed = True
        if not changed:
            break
    
    # Local search with swaps
    rng = random.Random(42)
    for iteration in range(3000):
        if time.time() > deadline - 3:
            break
        
        present = [i for i in range(E) if best_mask & (1 << i)]
        absent = [i for i in range(E) if not (best_mask & (1 << i))]
        
        if not present or not absent:
            break
        
        improved = False
        
        # Try single add
        for i in absent:
            new_mask = best_mask | (1 << i)
            if is_valid_mask(new_mask):
                best_mask = new_mask
                best_count += 1
                improved = True
                break
        
        if improved:
            continue
        
        # Try 1-for-2 swap
        if len(absent) >= 2 and len(present) >= 1:
            attempts = min(800, max(200, len(absent) * 3))
            for _ in range(attempts):
                if time.time() > deadline - 3:
                    break
                ri = rng.choice(present)
                a1 = rng.choice(absent)
                a2 = rng.choice(absent)
                if a1 == a2:
                    continue
                new_mask = (best_mask & ~(1 << ri)) | (1 << a1) | (1 << a2)
                if is_valid_mask(new_mask):
                    best_mask = new_mask
                    best_count += 1
                    improved = True
                    break
        
        if not improved and iteration > 300:
            # Perturb and re-greedy
            if iteration % 50 == 0:
                new_mask = best_mask
                k = min(5, best_count)
                for _ in range(k):
                    ri = rng.choice(present)
                    new_mask &= ~(1 << ri)
                # Re-add greedily
                for i in range(E):
                    if new_mask & (1 << i):
                        continue
                    try_mask = new_mask | (1 << i)
                    if is_valid_mask(try_mask):
                        new_mask = try_mask
                new_count = bin(new_mask).count('1')
                if new_count > best_count:
                    best_mask = new_mask
                    best_count = new_count
    
    # Final greedy pass
    for _ in range(10):
        if time.time() > deadline - 1:
            break
        changed = False
        for i in range(E):
            if best_mask & (1 << i):
                continue
            new_mask = best_mask | (1 << i)
            if is_valid_mask(new_mask):
                best_mask = new_mask
                best_count += 1
                changed = True
        if not changed:
            break
    
    # Convert result
    result = []
    for i in range(E):
        if best_mask & (1 << i):
            u, v = edge_list[i]
            result.append([int(u), int(v)])
    result.sort()
    return result
# EVOLVE-BLOCK-END
