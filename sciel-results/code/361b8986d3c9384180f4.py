import time
import random
import sys

DEADLINE = 330.0
SAFETY_MARGIN = 5.0

def _paley_graph(q):
    """Construct Paley graph on q vertices. Returns list of bitmasks."""
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return q, adj

def _has_k4_in_subset(adj, n, mask):
    """Check if the subgraph induced by vertices in 'mask' contains a K4."""
    if mask.bit_count() < 4:
        return False
    
    verts = []
    m = mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        verts.append(v)
        m ^= lsb
    
    # Check all combinations of 4
    # Optimization: Early exit
    for i in range(len(verts)):
        vi = verts[i]
        for j in range(i+1, len(verts)):
            vj = verts[j]
            if not (adj[vi] & (1 << vj)):
                continue
            for k in range(j+1, len(verts)):
                vk = verts[k]
                if not (adj[vi] & (1 << vk)) or not (adj[vj] & (1 << vk)):
                    continue
                for l in range(k+1, len(verts)):
                    vl = verts[l]
                    if (adj[vi] & (1 << vl)) and (adj[vj] & (1 << vl)) and (adj[vk] & (1 << vl)):
                        return True
    return False

def _has_i4_in_subset(adj, n, mask):
    """Check if the subgraph induced by vertices in 'mask' contains an I4."""
    if mask.bit_count() < 4:
        return False
    
    verts = []
    m = mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        verts.append(v)
        m ^= lsb
        
    for i in range(len(verts)):
        vi = verts[i]
        for j in range(i+1, len(verts)):
            vj = verts[j]
            if (adj[vi] & (1 << vj)):
                continue
            for k in range(j+1, len(verts)):
                vk = verts[k]
                if (adj[vi] & (1 << vk)) or (adj[vj] & (1 << vk)):
                    continue
                for l in range(k+1, len(verts)):
                    vl = verts[l]
                    if (adj[vi] & (1 << vl)) or (adj[vj] & (1 << vl)) or (adj[vk] & (1 << vl)):
                        return True
    return False

def _count_k4s_in_subset(adj, n, mask):
    """Count K4s in subset. Used for SA cost."""
    count = 0
    if mask.bit_count() < 4:
        return 0
    verts = []
    m = mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        verts.append(v)
        m ^= lsb
        
    for i in range(len(verts)):
        vi = verts[i]
        for j in range(i+1, len(verts)):
            vj = verts[j]
            if not (adj[vi] & (1 << vj)):
                continue
            for k in range(j+1, len(verts)):
                vk = verts[k]
                if not (adj[vi] & (1 << vk)) or not (adj[vj] & (1 << vk)):
                    continue
                for l in range(k+1, len(verts)):
                    vl = verts[l]
                    if (adj[vi] & (1 << vl)) and (adj[vj] & (1 << vl)) and (adj[vk] & (1 << vl)):
                        count += 1
    return count

def _count_i4s_in_subset(adj, n, mask):
    """Count I4s in subset. Used for SA cost."""
    count = 0
    if mask.bit_count() < 4:
        return 0
    verts = []
    m = mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        verts.append(v)
        m ^= lsb
        
    for i in range(len(verts)):
        vi = verts[i]
        for j in range(i+1, len(verts)):
            vj = verts[j]
            if (adj[vi] & (1 << vj)):
                continue
            for k in range(j+1, len(verts)):
                vk = verts[k]
                if (adj[vi] & (1 << vk)) or (adj[vj] & (1 << vk)):
                    continue
                for l in range(k+1, len(verts)):
                    vl = verts[l]
                    if (adj[vi] & (1 << vl)) or (adj[vj] & (1 << vl)) or (adj[vk] & (1 << vl)):
                        count += 1
    return count

def _try_add_vertex(adj, n, time_start):
    """
    Try to find a valid neighborhood S for a new vertex.
    S must be K4-free in G[S].
    Complement of S must be I4-free in G[comp(S)].
    """
    deadline = time_start + (DEADLINE - SAFETY_MARGIN)
    full_mask = (1 << n) - 1
    
    # Initialize S randomly
    best_S = None
    best_cost = float('inf')
    
    # Run SA for a few seconds or until solution found
    T = 1.0
    current_S = 0
    for i in range(n):
        if random.random() < 0.5:
            current_S |= (1 << i)
            
    current_cost = _count_k4s_in_subset(adj, n, current_S) + _count_i4s_in_subset(adj, n, full_mask ^ current_S)
    if current_cost == 0:
        return current_S
        
    steps = 0
    while time.time() < deadline:
        steps += 1
        if steps % 1000 == 0:
            T *= 0.95
            
        # Pick random vertex to flip
        v = random.randint(0, n - 1)
        bit = 1 << v
        new_S = current_S ^ bit
        
        new_cost = _count_k4s_in_subset(adj, n, new_S) + _count_i4s_in_subset(adj, n, full_mask ^ new_S)
        
        delta = new_cost - current_cost
        if delta < 0 or random.random() < pow(2.71828, -delta / T):
            current_S = new_S
            current_cost = new_cost
            
            if current_cost == 0:
                return current_S
                
            if current_cost < best_cost:
                best_cost = current_cost
                best_S = current_S
        else:
            # If T is very low and we haven't found 0, restart with best_S?
            if T < 0.01:
                current_S = best_S
                current_cost = best_cost
                T = 1.0
                # Maybe perturb best_S
                for _ in range(5):
                    v = random.randint(0, n - 1)
                    current_S ^= (1 << v)
                current_cost = _count_k4s_in_subset(adj, n, current_S) + _count_i4s_in_subset(adj, n, full_mask ^ current_S)

    return best_S if best_cost == 0 else None

def _is_55_free(adj, n):
    """Check if graph is (5,5)-free using bitsets."""
    if n < 5:
        return True
    
    # Check K5
    for i in range(n):
        for j in range(i+1, n):
            if (adj[i] >> j) & 1:
                common = adj[i] & adj[j]
                mask = common
                while mask:
                    lsb = mask & -mask
                    k = lsb.bit_length() - 1
                    mask ^= lsb
                    if k <= j: continue
                    C = adj[k] & common
                    if C & (C - 1):
                        mask_c = C
                        while mask_c:
                            lsb_c = mask_c & -mask_c
                            u = lsb_c.bit_length() - 1
                            mask_c ^= lsb_c
                            if (adj[u] & C):
                                return False
    
    # Check I5
    comp_adj = [0] * n
    full_mask = (1 << n) - 1
    for i in range(n):
        comp_adj[i] = full_mask ^ (1 << i) ^ adj[i]
        
    for i in range(n):
        for j in range(i+1, n):
            if (comp_adj[i] >> j) & 1:
                common = comp_adj[i] & comp_adj[j]
                mask = common
                while mask:
                    lsb = mask & -mask
                    k = lsb.bit_length() - 1
                    mask ^= lsb
                    if k <= j: continue
                    C = comp_adj[k] & common
                    if C & (C - 1):
                        mask_c = C
                        while mask_c:
                            lsb_c = mask_c & -mask_c
                            u = lsb_c.bit_length() - 1
                            mask_c ^= lsb_c
                            if (comp_adj[u] & C):
                                return False
    return True

def _to_edges(adj, n):
    """Convert bitset adjacency to list of (i, j) edges."""
    edges = []
    for i in range(n):
        mask = adj[i] & ~((1 << (i+1)) - 1)
        while mask:
            lsb = mask & -mask
            j = lsb.bit_length() - 1
            edges.append((i, j))
            mask ^= lsb
    return edges

def construct():
    """Construct a (5,5)-free graph. Returns (n, edges)."""
    time_start = time.time()
    random.seed(42)
    
    # Start with Paley graph on 37 vertices
    q, adj = _paley_graph(37)
    n = q
    best_n = n
    best_adj = adj[:]
    
    if not _is_55_free(adj, n):
        raise ValueError("Seed graph is not (5,5)-free")
        
    # Try to extend the graph
    # We will try to add vertices one by one.
    # If we get stuck, we will try to flip edges to create space for a new vertex.
    
    max_attempts = 50
    for attempt in range(max_attempts):
        if time.time() - time_start > DEADLINE - SAFETY_MARGIN:
            break
            
        # Try to add a vertex to the current graph
        S = _try_add_vertex(adj, n, time_start)
        
        if S is not None:
            # Add new vertex n
            new_adj = adj[:] + [0]
            for i in range(n):
                if (S >> i) & 1:
                    new_adj[i] |= (1 << n)
                    new_adj[n] |= (1 << i)
            
            adj = new_adj
            n += 1
            best_n = n
            best_adj = adj[:]
            print(f"Extended to {n} vertices", file=sys.stderr)
            
            # Reset attempts counter? Or just continue.
            # If we can add another, great.
        else:
            # Stuck. Try to modify the graph to allow extension.
            # Flip a random edge, check if still valid, then retry.
            # Limit the number of flips to avoid wasting time.
            flipped = False
            for _ in range(100):
                if time.time() - time_start > DEADLINE - SAFETY_MARGIN:
                    break
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                
                bit_i = 1 << j
                bit_j = 1 << i
                adj[i] ^= bit_i
                adj[j] ^= bit_j
                
                if _is_55_free(adj, n):
                    flipped = True
                    break
                else:
                    adj[i] ^= bit_i
                    adj[j] ^= bit_j
            
            if not flipped:
                # If we can't even flip an edge to keep it valid, we might be at a local optimum
                # or the graph is very rigid.
                # Try a few more aggressive moves? Or just stop.
                # Given time constraints, stop if we can't flip.
                break

    # Final verification
    if _is_55_free(best_adj, best_n):
        edges = _to_edges(best_adj, best_n)
        return best_n, edges
    else:
        # Fallback to seed if something went wrong (should not happen)
        edges = _to_edges(_paley_graph(37)[1], 37)
        return 37, edges

if __name__ == "__main__":
    n, edges = construct()
    print(f"Constructed graph with {n} vertices and {len(edges)} edges")
