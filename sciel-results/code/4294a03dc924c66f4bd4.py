import time
import random
import math

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin for 90s limit

    n = P['n']
    edges = P['edges']

    # Build adjacency list and bitmask
    adj = [0] * n
    edge_set = set()
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        if u < v:
            edge_set.add((u, v))
        else:
            edge_set.add((v, u))

    # Seed Solution: One biclique per vertex covering its outgoing edges (u < v)
    seed_bicliques = []
    for u in range(n):
        right_side = [v for u2, v in edges if u2 == u]
        if right_side:
            seed_bicliques.append([[u], right_side])

    best_bicliques = seed_bicliques
    best_count = len(seed_bicliques)

    # Helper: Expand a partial biclique (L_mask, R_mask) to maximal
    # L_mask: bitmask of vertices in Left
    # R_mask: bitmask of vertices in Right
    # adj: adjacency bitmasks
    def expand_biclique(L_mask, R_mask, adj):
        changed = True
        while changed:
            changed = False
            # Expand L: find u not in L such that u connected to all in R
            # For all v in R, adj[u] has bit v set.
            # i.e., (adj[u] & R_mask) == R_mask
            for u in range(n):
                if (L_mask >> u) & 1:
                    continue
                if (adj[u] & R_mask) == R_mask and R_mask != 0:
                    L_mask |= (1 << u)
                    changed = True
            # Expand R: find v not in R such that v connected to all in L
            # For all u in L, adj[v] has bit u set.
            # i.e., (adj[v] & L_mask) == L_mask
            for v in range(n):
                if (R_mask >> v) & 1:
                    continue
                if (adj[v] & L_mask) == L_mask and L_mask != 0:
                    R_mask |= (1 << v)
                    changed = True
        return L_mask, R_mask

    def mask_to_list(mask):
        return [i for i in range(n) if (mask >> i) & 1]

    def get_random_edge(rem_mask):
        # Find a random remaining edge (u, v) with u < v
        # We iterate u from 0 to n-1, if rem_mask[u] has bits set for v > u
        # Actually, rem_mask is symmetric. We just need any pair u, v where rem_mask[u] has v.
        # To ensure u < v for consistency, we can just pick any and sort.
        # But for efficiency, let's pick a random u with non-zero rem_mask[u]
        # and a random v from rem_mask[u].
        candidates = [u for u in range(n) if rem_mask[u] != 0]
        if not candidates:
            return None
        u = random.choice(candidates)
        # Get v from rem_mask[u]
        mask = rem_mask[u]
        # Pick random set bit
        # Convert to list of set bits
        vs = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            vs.append(v)
            m ^= lsb
        if not vs:
            return None
        v = random.choice(vs)
        return (u, v)

    def get_deterministic_edge(rem_mask, edge_idx):
        # Get the edge_idx-th remaining edge in a deterministic order
        # This helps with coverage in case random sampling misses some structures
        count = -1
        for u in range(n):
            mask = rem_mask[u]
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                if u < v:
                    count += 1
                    if count == edge_idx:
                        return (u, v)
        return None

    def try_greedy(restarts=100):
        nonlocal best_bicliques, best_count
        for r in range(restarts):
            if time.time() > deadline:
                return

            # Initialize remaining masks
            rem_mask = adj[:]
            edges_left = len(edge_set)
            current_bicliques = []

            # Use a mix of random and deterministic strategies
            # For the first few restarts, use more deterministic choices
            use_deterministic = (r < 10)
            edge_counter = 0

            while edges_left > 0:
                if time.time() > deadline:
                    return

                best_local = None
                best_size = 0

                # Try a few starting edges
                # Mix random and deterministic
                num_trials = 8 if use_deterministic else 5
                for trial in range(num_trials):
                    if time.time() > deadline:
                        return
                    
                    if use_deterministic and trial < 3:
                        edge = get_deterministic_edge(rem_mask, edge_counter)
                        if edge is None:
                            break
                    else:
                        edge = get_random_edge(rem_mask)
                        if edge is None:
                            break
                    
                    u, v = edge
                    # Ensure u < v for canonical representation? Not strictly necessary for expansion
                    # but good for consistency. Expansion is symmetric.
                    L_mask = (1 << u)
                    R_mask = (1 << v)

                    # Expand
                    L_mask, R_mask = expand_biclique(L_mask, R_mask, rem_mask)

                    size = L_mask.bit_count() * R_mask.bit_count()
                    if size > best_size:
                        best_size = size
                        best_local = (L_mask, R_mask)

                if not best_local:
                    # Fallback: pick single edge
                    edge = get_random_edge(rem_mask)
                    if edge is None:
                        break
                    u, v = edge
                    best_local = ((1 << u), (1 << v))
                    best_size = 1

                L_mask, R_mask = best_local
                current_bicliques.append([mask_to_list(L_mask), mask_to_list(R_mask)])

                # Update rem_mask: remove all edges between L and R
                # For u in L, clear bits in R
                # For v in R, clear bits in L
                # Iterate L
                l_list = mask_to_list(L_mask)
                r_list = mask_to_list(R_mask)
                for u in l_list:
                    rem_mask[u] &= ~R_mask
                for v in r_list:
                    rem_mask[v] &= ~L_mask

                edges_left -= best_size
                edge_counter += 1

                # Safety check: if edges_left < 0, something went wrong
                if edges_left < 0:
                    break

            if len(current_bicliques) < best_count:
                best_count = len(current_bicliques)
                best_bicliques = current_bicliques
                if best_count == 1:
                    return

    try_greedy(restarts=200)

    # Local search: try to merge bicliques
    def try_local_search():
        nonlocal best_bicliques, best_count
        
        if best_count <= 1:
            return
            
        # Try to improve by merging adjacent bicliques
        # This is a heuristic improvement
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current = best_bicliques[:]
            
            for i in range(len(current)):
                for j in range(i + 1, len(current)):
                    if time.time() > deadline:
                        return
                    
                    # Try to merge bicliques i and j
                    L1, R1 = current[i][0], current[i][1]
                    L2, R2 = current[j][0], current[j][1]
                    
                    # Check if we can merge: 
                    # All edges between L1 and R2, and between L2 and R1 must exist
                    # This is complex, so we'll use a simpler heuristic
                    # Just try to see if there's a common structure
                    
                    # For now, skip complex merging as it's computationally expensive
                    pass
        
        # Alternative: try re-solving with different random seeds
        # This is already handled by try_greedy with multiple restarts

    # Return best found
    result = []
    for b in best_bicliques:
        left = [int(x) for x in b[0]]
        right = [int(x) for x in b[1]]
        result.append([left, right])

    return result
