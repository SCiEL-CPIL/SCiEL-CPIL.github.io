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

def _has_k4_in_mask(mask, adj, n):
    """Check if the subgraph induced by vertices in 'mask' contains a K4."""
    # Extract vertices in mask
    verts = []
    m = mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        verts.append(v)
        m ^= lsb
    
    if len(verts) < 4:
        return False
        
    # Check for K4 among these vertices
    # For efficiency, only check if the induced subgraph is dense enough?
    # Simple O(k^4) check where k is popcount(mask)
    # If k is large, this is slow. But in the search, k might be around n/2.
    # For n=37, k~18, 18^4 = 100k ops. This is fast enough for a heuristic step.
    
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
                    if (adj[vi] & (1 << vl)) and (adj[vj] & (1 << vl)) and \
                       (adj[vk] & (1 << vl)):
                        return True
    return False

def _has_i4_in_mask(mask, adj, n):
    """Check if the subgraph induced by vertices in 'mask' contains an independent set of size 4 (I4)."""
    # This is equivalent to checking if the complement subgraph has a K4.
    # Construct complement adjacency for the vertices in mask?
    # Or just check directly: 4 vertices with no edges between them.
    
    verts = []
    m = mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        verts.append(v)
        m ^= lsb
        
    if len(verts) < 4:
        return False
        
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
                        continue
                    return True
    return False

def _count_violations(S_mask, adj, n):
    """Count K4s in S and I4s in complement of S."""
    # We just need to know if there are ANY, for the search.
    # But for annealing, a count is better.
    # However, counting all K4s is expensive.
    # Let's just return 1 if violation exists, 0 otherwise.
    # This makes it a boolean search, which might get stuck in local minima.
    # Better: return the number of violating cliques?
    # Let's stick to boolean for now, but use a randomized restart strategy.
    
    if _has_k4_in_mask(S_mask, adj, n):
        return 1
    comp_mask = ((1 << n) - 1) ^ S_mask
    if _has_i4_in_mask(comp_mask, adj, n):
        return 1
    return 0

def _try_add_vertex(adj, n, time_start):
    """Try to find a valid neighborhood S for a new vertex."""
    full_mask = (1 << n) - 1
    
    # Heuristic initial S: 
    # Maybe start with a random subset of size n/2?
    # Or start with a small set and grow?
    # Let's try a few random initializations.
    
    best_S = None
    
    for init_attempt in range(20):
        if time.time() - time_start > DEADLINE - SAFETY_MARGIN:
            return None
            
        # Random initialization
        # Bias towards balanced sets?
        S_mask = 0
        for i in range(n):
            if random.random() < 0.5:
                S_mask |= (1 << i)
                
        # Local search / Simulated Annealing
        T = 1.0
        for step in range(1000):
            if time.time() - time_start > DEADLINE - SAFETY_MARGIN:
                return None
                
            current_violations = _count_violations(S_mask, adj, n)
            
            if current_violations == 0:
                return S_mask
            
            # Pick a random vertex to flip
            v = random.randint(0, n - 1)
            bit = 1 << v
            new_S_mask = S_mask ^ bit
            
            new_violations = _count_violations(new_S_mask, adj, n)
            
            # Accept if better, or with prob exp(-delta/T)
            delta = new_violations - current_violations
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                S_mask = new_S_mask
                
            T *= 0.995 # Cool down
    
    return None

def _is_55_free(adj, n):
    """Check if graph is (5,5)-free using bitsets."""
    # Check K5
    for i in range(n):
        for j in range(i+1, n):
            if (adj[i] >> j) & 1:
                common = adj[i] & adj[j]
                # We need a K3 in common
                # Iterate k in common
                mask = common
                while mask:
                    lsb = mask & -mask
                    k = lsb.bit_length() - 1
                    mask ^= lsb
                    if k <= j: continue # Ensure i < j < k
                    # Need an edge in common & adj[k]
                    # Let's check if (adj[k] & common) contains an edge
                    # Actually, we need two distinct nodes u, v in (adj[k] & common) such that u-v is an edge.
                    # This is equivalent to: does the subgraph induced by (adj[k] & common) have any edge?
                    # Let C = adj[k] & common
                    # We need to check if any u in C has a neighbor in C.
                    # 
                    # Optimization: 
                    # If C has < 2 bits, no edge.
                    # If C has >= 2 bits, check if any adj[u] & C is non-zero.
                    C = adj[k] & common
                    if C & (C - 1): # Check if at least 2 bits are set
                        # Iterate u in C
                        mask_c = C
                        while mask_c:
                            lsb_c = mask_c & -mask_c
                            u = lsb_c.bit_length() - 1
                            mask_c ^= lsb_c
                            # Check if u has a neighbor in C
                            # adj[u] & C will include u? No, adj[u] doesn't have bit u.
                            # So if (adj[u] & C) is non-zero, there is a neighbor v != u in C.
                            if (adj[u] & C):
                                return False # Found K5
    # Check I5 (K5 in complement)
    # Construct complement adj
    comp_adj = [0] * n
    for i in range(n):
        # All bits 0..n-1 set, except i and adj[i]
        comp_adj[i] = ((1 << n) - 1) ^ (1 << i) ^ adj[i]
        
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
                                return False # Found I5
    return True

def _to_edges(adj, n):
    """Convert bitset adjacency to list of (i, j) edges."""
    edges = []
    for i in range(n):
        # Get bits set in adj[i] greater than i
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
    
    # Start with Paley graph on 37 vertices
    q, adj = _paley_graph(37)
    best_n = q
    best_adj = adj[:]
    
    # Verify seed
    if not _is_55_free(adj, q):
        raise ValueError("Seed graph is not (5,5)-free")
        
    random.seed(42)
    
    # Try to extend
    current_adj = best_adj[:]
    current_n = best_n
    
    while time.time() - time_start < DEADLINE - SAFETY_MARGIN:
        # Try to add a vertex
        S_mask = _try_add_vertex(current_adj, current_n, time_start)
        
        if S_mask is not None:
            # Add new vertex n
            new_n = current_n + 1
            new_adj = current_adj[:] + [0]
            
            # Connect new vertex to S_mask
            for i in range(current_n):
                if (S_mask >> i) & 1:
                    new_adj[i] |= (1 << new_n)
                    new_adj[new_n] |= (1 << i)
            
            current_adj = new_adj
            current_n = new_n
            
            # Update best if valid (it should be by construction, but verify just in case? 
            # The local search ensured no K4 in S and no I4 in comp(S).
            # This guarantees no K5 and no I5 involving the new vertex.
            # The old graph was already (5,5)-free.
            # So the new graph is (5,5)-free.
            if current_n > best_n:
                best_n = current_n
                best_adj = current_adj[:]
                
            # Print progress
            print(f"Extended to {current_n} vertices", file=sys.stderr)
        else:
            # Stuck. Try random edge flips to unblock?
            # Or just break and restart with a different seed for the subset search?
            # Let's try a few random edge flips.
            improved = False
            for _ in range(100):
                if time.time() - time_start > DEADLINE - SAFETY_MARGIN:
                    break
                    
                # Pick a random edge
                i = random.randint(0, current_n - 1)
                j = random.randint(0, current_n - 1)
                if i == j: continue
                
                # Flip edge
                bit_i = 1 << j
                bit_j = 1 << i
                current_adj[i] ^= bit_i
                current_adj[j] ^= bit_j
                
                # Check if still (5,5)-free
                if _is_55_free(current_adj, current_n):
                    improved = True
                    # Try to add vertex again immediately?
                    # For now, just accept the flip and loop back to try adding vertex.
                    break
                else:
                    # Revert
                    current_adj[i] ^= bit_i
                    current_adj[j] ^= bit_j
            
            if not improved:
                # If we can't flip edges, we are stuck in a local minimum for this graph structure.
                # We might need to restart the whole process or give up.
                # Since we started from a good seed, let's just break.
                break
                
    edges = _to_edges(best_adj, best_n)
    return best_n, edges

if __name__ == "__main__":
    n, edges = construct()
    print(f"Constructed graph with {n} vertices and {len(edges)} edges")
