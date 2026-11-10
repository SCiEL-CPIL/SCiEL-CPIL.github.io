import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a graph with no regular induced subgraph on 6 or more vertices.
    Uses Simulated Annealing to minimize the number of regular induced subgraphs of size 6.
    Once 0 is reached, checks for larger regular induced subgraphs and repairs if found.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    n = 20
    
    # Precompute all subsets of size 6
    all_subsets_6 = list(combinations(range(n), 6))
    num_subsets = len(all_subsets_6)
    
    # Precompute masks for each subset
    subset_masks = [0] * num_subsets
    for i, S in enumerate(all_subsets_6):
        m = 0
        for v in S:
            m |= (1 << v)
        subset_masks[i] = m
        
    # Map vertex to list of subset indices containing it
    vertex_to_subsets = [[] for _ in range(n)]
    for i, S in enumerate(all_subsets_6):
        for v in S:
            vertex_to_subsets[v].append(i)

    # Helper: Check if subset index i is regular in adj
    def is_reg_idx(i, adj):
        m = subset_masks[i]
        S = all_subsets_6[i]
        first_deg = (adj[S[0]] & m).bit_count()
        for v in S[1:]:
            if (adj[v] & m).bit_count() != first_deg:
                return False
        return True

    def find_conflict_size_gt(k_start, n, adj):
        """
        Finds a regular induced subgraph of size >= k_start.
        Returns mask if found, else None.
        """
        for k in range(k_start, n + 1):
            for S in combinations(range(n), k):
                mask = 0
                for v in S:
                    mask |= (1 << v)
                
                first_deg = (adj[S[0]] & mask).bit_count()
                is_reg = True
                for v in S[1:]:
                    if (adj[v] & mask).bit_count() != first_deg:
                        is_reg = False
                        break
                
                if is_reg:
                    return mask
        return None

    # Initialize random graph
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    # Initial count of regular size-6 subgraphs
    current_count = sum(1 for i in range(num_subsets) if is_reg_idx(i, adj))

    # Simulated Annealing Parameters
    T = 1.0
    cooling = 0.999
    min_T = 0.01

    while time.time() < deadline and T > min_T:
        # Pick random edge
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue

        # Determine affected subsets
        affected = set(vertex_to_subsets[u])
        affected.update(vertex_to_subsets[v])

        # Check if edge exists
        edge_exists = bool(adj[u] & (1 << v))

        # 1. Revert flip (ensure we are in "old" state)
        # We assume we are in the state BEFORE this step's decision.
        # So we calculate old_affected_count based on current adj.
        old_affected_count = 0
        for idx in affected:
            if is_reg_idx(idx, adj):
                old_affected_count += 1

        # 2. Apply flip
        if edge_exists:
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

        # 3. Calculate new_affected_count
        new_affected_count = 0
        for idx in affected:
            if is_reg_idx(idx, adj):
                new_affected_count += 1

        delta = new_affected_count - old_affected_count

        # 4. Acceptance
        if delta < 0:
            current_count += delta
        else:
            # Accept with prob exp(-delta / T)
            if delta == 0:
                current_count += delta
            else:
                # Use math.exp for speed
                if random.random() < math.exp(-delta / T):
                    current_count += delta
                else:
                    # Revert
                    if edge_exists:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    else:
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)

        T *= cooling

        # If we reach 0 size-6 conflicts, check for larger ones
        if current_count == 0:
            conflict_mask = find_conflict_size_gt(7, n, adj)
            if conflict_mask is None:
                # Valid graph found
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
            else:
                # Found size 7+ conflict.
                # Heuristic repair: flip a random edge in the conflict.
                verts = []
                m = conflict_mask
                while m:
                    lsb = m & -m
                    verts.append(lsb.bit_length() - 1)
                    m ^= lsb
                if len(verts) >= 2:
                    u2, v2 = random.sample(verts, 2)
                    if adj[u2] & (1 << v2):
                        adj[u2] ^= (1 << v2)
                        adj[v2] ^= (1 << u2)
                    else:
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
                
                # Recompute current_count as it might have changed
                current_count = sum(1 for i in range(num_subsets) if is_reg_idx(i, adj))
                # Reheat
                T = 0.5

    # If loop finishes, do a final check
    if current_count == 0:
        if find_conflict_size_gt(7, n, adj) is None:
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i] & (1 << j):
                        edges.append([i, j])
            return {'n': n, 'edges': edges}

    # Fallback: try n=19 with a simpler search if n=20 failed
    # This is unlikely to be needed if SA is effective, but good for safety.
    # For now, return the best effort for n=20 (which is invalid) or a smaller valid one?
    # The task requires maximizing n.
    # If we can't find n=20, we should try n=19.
    
    # Quick attempt for n=19
    n_alt = 19
    adj_alt = [0] * n_alt
    for i in range(n_alt):
        for j in range(i + 1, n_alt):
            if random.random() < 0.5:
                adj_alt[i] |= (1 << j)
                adj_alt[j] |= (1 << i)
    
    # Simple repair for n=19
    for _ in range(100):
        if time.time() > deadline:
            break
        # Check validity
        valid = True
        # Check size 6
        for S in combinations(range(n_alt), 6):
            mask = 0
            for v in S:
                mask |= (1 << v)
            first_deg = (adj_alt[S[0]] & mask).bit_count()
            is_reg = True
            for v in S[1:]:
                if (adj_alt[v] & mask).bit_count() != first_deg:
                    is_reg = False
                    break
            if is_reg:
                valid = False
                # Flip random edge in S
                u, v = random.sample(S, 2)
                if adj_alt[u] & (1 << v):
                    adj_alt[u] ^= (1 << v)
                    adj_alt[v] ^= (1 << u)
                else:
                    adj_alt[u] |= (1 << v)
                    adj_alt[v] |= (1 << u)
                break
        if valid:
            # Check size 7+
            conflict = find_conflict_size_gt(7, n_alt, adj_alt)
            if conflict is None:
                edges = []
                for i in range(n_alt):
                    for j in range(i + 1, n_alt):
                        if adj_alt[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n_alt, 'edges': edges}
            else:
                # Repair
                verts = []
                m = conflict
                while m:
                    lsb = m & -m
                    verts.append(lsb.bit_length() - 1)
                    m ^= lsb
                u, v = random.sample(verts, 2)
                if adj_alt[u] & (1 << v):
                    adj_alt[u] ^= (1 << v)
                    adj_alt[v] ^= (1 << u)
                else:
                    adj_alt[u] |= (1 << v)
                    adj_alt[v] |= (1 << u)

    # Last resort
    return {'n': 5, 'edges': []}

# EVOLVE-BLOCK-END
