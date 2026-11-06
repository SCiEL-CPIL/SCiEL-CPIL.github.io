import random
import time
import itertools

def construct():
    """
    Constructs a graph on n vertices with no regular induced subgraph on k >= 6 vertices.
    Uses a randomized greedy construction with backtracking, plus local search refinement.
    """
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill

    def is_regular_induced(adj, mask, n):
        """Check if the induced subgraph on vertices in mask is regular."""
        if mask == 0:
            return True
        m = mask
        first_deg = -1
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            deg = (adj[v] & mask).bit_count()
            if first_deg == -1:
                first_deg = deg
            elif deg != first_deg:
                return False
            m ^= lsb
        return True

    def count_violations_fast(adj, n):
        """
        Count regular induced subgraphs of size >= 6.
        Returns (count, list_of_violation_masks) - capped at 100 masks.
        """
        count = 0
        masks = []
        for k in range(6, n + 1):
            for combo in itertools.combinations(range(n), k):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                if is_regular_induced(adj, mask, n):
                    count += 1
                    if len(masks) < 100:
                        masks.append(mask)
                    if count > 500:
                        return count, masks
        return count, masks

    def greedy_construct(n, deadline, p_base=0.45):
        """
        Greedy vertex-by-vertex construction with backtracking.
        Returns (success, adj) where adj is a list of n ints (bitmasks).
        """
        adj = [0] * n
        # We build the graph by deciding edges as we add vertices.
        # Actually, let's do a full random initialization then local search,
        # but with a smarter initial distribution.
        
        # Try multiple random seeds with different densities
        for attempt in range(30):
            if time.time() > deadline:
                break
            
            # Vary density slightly
            p = p_base + random.uniform(-0.1, 0.1)
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Quick check: if already valid, return
            count, _ = count_violations_fast(adj, n)
            if count == 0:
                return True, adj
            
            # Local search to fix violations
            ok, final_adj = local_search(adj, n, deadline)
            if ok:
                return True, final_adj
        
        return False, adj

    def local_search(initial_adj, n, deadline):
        """
        Local search: minimize violation count by flipping edges.
        Returns (success, final_adj)
        """
        current_adj = initial_adj[:]
        best_adj = current_adj[:]
        
        best_count, _ = count_violations_fast(current_adj, n)
        if best_count == 0:
            return True, current_adj
        
        stuck_count = 0
        max_stuck = 100
        
        while time.time() < deadline:
            count, masks = count_violations_fast(current_adj, n)
            
            if count == 0:
                return True, current_adj
            
            if count < best_count:
                best_adj = current_adj[:]
                best_count = count
                stuck_count = 0
            else:
                stuck_count += 1
            
            if stuck_count > max_stuck:
                break
            
            if not masks:
                break
            
            # Pick a random violation mask
            mask = random.choice(masks)
            
            # Get vertices in this violation
            verts = []
            m = mask
            while m:
                lsb = m & -m
                verts.append(lsb.bit_length() - 1)
                m ^= lsb
            
            if len(verts) < 2:
                continue
            
            # Choose an edge to flip: prefer edges within the violation
            if random.random() < 0.8:
                u, v = random.sample(verts, 2)
            else:
                u = random.choice(verts)
                outside = [i for i in range(n) if i not in set(verts)]
                if outside:
                    v = random.choice(outside)
                else:
                    v = random.choice(verts)
                    if u == v:
                        continue
            
            # Flip edge u-v
            if (current_adj[u] >> v) & 1:
                current_adj[u] ^= (1 << v)
                current_adj[v] ^= (1 << u)
            else:
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
        
        if best_count == 0:
            return True, best_adj
        return False, best_adj

    best_n = 1
    best_edges = []
    
    # Try to find valid graphs for n = 22, 21, 20
    for target_n in [22, 21, 20]:
        if time.time() > time_limit - 5:
            break
        
        success = False
        # Multiple attempts with greedy + local search
        max_attempts = 20 if target_n <= 20 else 10
        for attempt in range(max_attempts):
            if time.time() > time_limit - 3:
                break
            
            ok, adj = greedy_construct(target_n, time_limit - 2)
            if ok:
                # Verify
                count, _ = count_violations_fast(adj, target_n)
                if count == 0:
                    edges = []
                    for i in range(target_n):
                        for j in range(i + 1, target_n):
                            if (adj[i] >> j) & 1:
                                edges.append([i, j])
                    best_n = target_n
                    best_edges = edges
                    success = True
                    break
        
        if success and target_n == 22:
            break
    
    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    construct()
