import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n = 20
    # adj[u] is a bitmask of neighbors of u
    adj = [0] * n
    # cc[u][v] is the number of common neighbors of u and v
    cc = [[0] * n for _ in range(n)]

    def popcount(x):
        return bin(x).count('1')

    def can_add(u, v):
        if (adj[u] >> v) & 1:
            return False
        if cc[u][v] > 2:
            return False
        # Check neighbors of v
        mask_v = adj[v]
        while mask_v:
            lsb = mask_v & -mask_v
            w = lsb.bit_length() - 1
            if cc[u][w] > 1:
                return False
            mask_v ^= lsb
        # Check neighbors of u
        mask_u = adj[u]
        while mask_u:
            lsb = mask_u & -mask_u
            w = lsb.bit_length() - 1
            if cc[v][w] > 1:
                return False
            mask_u ^= lsb
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

        # Update cc
        old_mask_v = adj[v] ^ (1 << u)
        m = old_mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            cc[u][w] += 1
            cc[w][u] += 1
            m ^= lsb

        old_mask_u = adj[u] ^ (1 << v)
        m = old_mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            cc[v][w] += 1
            cc[w][v] += 1
            m ^= lsb

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

        # Update cc
        m = adj[v]
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            cc[u][w] -= 1
            cc[w][u] -= 1
            m ^= lsb

        m = adj[u]
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            cc[v][w] -= 1
            cc[w][v] -= 1
            m ^= lsb

    def get_edges():
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def count_edges():
        count = 0
        for u in range(n):
            count += popcount(adj[u])
        return count // 2

    def find_blocking_edges(u, v):
        """Find which existing edges are blocking the addition of (u, v).
        An edge (a, b) blocks (u, v) if removing it would make (u, v) addable.
        We approximate: edges incident to u or v that contribute to cc[u][v] or to the
        'cc[u][w] > 1' or 'cc[v][w] > 1' conditions."""
        blockers = set()
        
        # If cc[u][v] > 2, then edges from u and v to common neighbors are blockers
        if cc[u][v] > 2:
            common = adj[u] & adj[v]
            m = common
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                blockers.add((min(u, w), max(u, w)))
                blockers.add((min(v, w), max(v, w)))
                m ^= lsb
        
        # Check neighbors of v where cc[u][w] > 1
        mask_v = adj[v]
        while mask_v:
            lsb = mask_v & -mask_v
            w = lsb.bit_length() - 1
            if cc[u][w] > 1:
                # Edge (u, w) or edges contributing to cc[u][w]
                # The edge (v, w) is a potential blocker since it creates common neighbor with u
                blockers.add((min(v, w), max(v, w)))
                # Also edges from u to common neighbors of u and w
                common_uw = adj[u] & adj[w]
                m2 = common_uw
                while m2:
                    lsb2 = m2 & -m2
                    x = lsb2.bit_length() - 1
                    blockers.add((min(u, x), max(u, x)))
                    blockers.add((min(w, x), max(w, x)))
                    m2 ^= lsb2
            mask_v ^= lsb
        
        # Check neighbors of u where cc[v][w] > 1
        mask_u = adj[u]
        while mask_u:
            lsb = mask_u & -mask_u
            w = lsb.bit_length() - 1
            if cc[v][w] > 1:
                blockers.add((min(u, w), max(u, w)))
                common_vw = adj[v] & adj[w]
                m2 = common_vw
                while m2:
                    lsb2 = m2 & -m2
                    x = lsb2.bit_length() - 1
                    blockers.add((min(v, x), max(v, x)))
                    blockers.add((min(w, x), max(w, x)))
                    m2 ^= lsb2
            mask_u ^= lsb
        
        return blockers

    def greedy_construct():
        """Greedy: try to add all possible edges in a smart order."""
        nonlocal adj, cc
        adj = [0] * n
        cc = [[0] * n for _ in range(n)]
        
        # Try adding edges in order of "potential" - edges that are less constrained first
        # Actually, let's just try all edges multiple times until no progress
        progress = True
        while progress and time.time() < deadline - 10.0:
            progress = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not ((adj[u] >> v) & 1) and can_add(u, v):
                        add_edge(u, v)
                        progress = True
        return count_edges()

    def local_search_greedy():
        """Local search: try to add edges, and when blocked, try removing blockers."""
        nonlocal adj, cc, best_edges, best_count
        
        # Work on copies
        adj_ls = adj[:]
        cc_ls = [row[:] for row in cc]
        count_ls = count_edges()
        
        def ls_can_add(u, v):
            if (adj_ls[u] >> v) & 1:
                return False
            if cc_ls[u][v] > 2:
                return False
            mask_v = adj_ls[v]
            while mask_v:
                lsb = mask_v & -mask_v
                w = lsb.bit_length() - 1
                if cc_ls[u][w] > 1:
                    return False
                mask_v ^= lsb
            mask_u = adj_ls[u]
            while mask_u:
                lsb = mask_u & -mask_u
                w = lsb.bit_length() - 1
                if cc_ls[v][w] > 1:
                    return False
                mask_u ^= lsb
            return True

        def ls_add_edge(u, v):
            adj_ls[u] |= (1 << v)
            adj_ls[v] |= (1 << u)
            
            old_mask_v = adj_ls[v] ^ (1 << u)
            m = old_mask_v
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                cc_ls[u][w] += 1
                cc_ls[w][u] += 1
                m ^= lsb
                
            old_mask_u = adj_ls[u] ^ (1 << v)
            m = old_mask_u
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                cc_ls[v][w] += 1
                cc_ls[w][v] += 1
                m ^= lsb

        def ls_remove_edge(u, v):
            adj_ls[u] &= ~(1 << v)
            adj_ls[v] &= ~(1 << u)
            
            m = adj_ls[v]
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                cc_ls[u][w] -= 1
                cc_ls[w][u] -= 1
                m ^= lsb
                
            m = adj_ls[u]
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                cc_ls[v][w] -= 1
                cc_ls[w][v] -= 1
                m ^= lsb

        def ls_count_edges():
            c = 0
            for u in range(n):
                c += popcount(adj_ls[u])
            return c // 2

        def ls_get_edges():
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj_ls[u] >> v) & 1:
                        edges.append([u, v])
            return edges

        def ls_find_blocking_edges(u, v):
            """Find which existing edges are blocking the addition of (u, v) in ls state."""
            blockers = set()
            
            if cc_ls[u][v] > 2:
                common = adj_ls[u] & adj_ls[v]
                m = common
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    blockers.add((min(u, w), max(u, w)))
                    blockers.add((min(v, w), max(v, w)))
                    m ^= lsb
            
            mask_v = adj_ls[v]
            while mask_v:
                lsb = mask_v & -mask_v
                w = lsb.bit_length() - 1
                if cc_ls[u][w] > 1:
                    blockers.add((min(v, w), max(v, w)))
                    common_uw = adj_ls[u] & adj_ls[w]
                    m2 = common_uw
                    while m2:
                        lsb2 = m2 & -m2
                        x = lsb2.bit_length() - 1
                        blockers.add((min(u, x), max(u, x)))
                        blockers.add((min(w, x), max(w, x)))
                        m2 ^= lsb2
                mask_v ^= lsb
            
            mask_u = adj_ls[u]
            while mask_u:
                lsb = mask_u & -mask_u
                w = lsb.bit_length() - 1
                if cc_ls[v][w] > 1:
                    blockers.add((min(u, w), max(u, w)))
                    common_vw = adj_ls[v] & adj_ls[w]
                    m2 = common_vw
                    while m2:
                        lsb2 = m2 & -m2
                        x = lsb2.bit_length() - 1
                        blockers.add((min(v, x), max(v, x)))
                        blockers.add((min(w, x), max(w, x)))
                        m2 ^= lsb2
                mask_u ^= lsb
            
            return blockers

        # Main local search loop
        no_progress = 0
        max_no_progress = 500
        
        while time.time() < deadline - 5.0 and no_progress < max_no_progress:
            # Try to add a random non-existing edge
            non_existing = [(u, v) for u in range(n) for v in range(u+1, n) if not ((adj_ls[u] >> v) & 1)]
            
            if not non_existing:
                break
            
            random.shuffle(non_existing)
            added_this_round = False
            
            for u, v in non_existing:
                if ls_can_add(u, v):
                    ls_add_edge(u, v)
                    count_ls += 1
                    added_this_round = True
                    if count_ls > best_count:
                        best_count = count_ls
                        best_edges = ls_get_edges()
                    break
            
            if added_this_round:
                no_progress = 0
                continue
            
            # No edge could be added. Try to remove a blocker and then add.
            # Pick a random non-existing edge
            u, v = random.choice(non_existing)
            blockers = ls_find_blocking_edges(u, v)
            
            if not blockers:
                no_progress += 1
                continue
            
            # Try each blocker: remove it, then try to add (u, v)
            for bu, bv in blockers:
                # Check if (bu, bv) is actually an edge
                if not ((adj_ls[bu] >> bv) & 1):
                    continue
                
                # Remove the blocker
                ls_remove_edge(bu, bv)
                count_ls -= 1
                
                # Try to add (u, v)
                if ls_can_add(u, v):
                    ls_add_edge(u, v)
                    count_ls += 1
                    # Net change: 0 (removed one, added one)
                    # But now we might be able to add more edges
                    added_this_round = True
                    break
                else:
                    # Revert
                    ls_add_edge(bu, bv)
                    count_ls += 1
                    continue
            
            if added_this_round:
                no_progress = 0
            else:
                no_progress += 1
        
        return count_ls

    random.seed(42)

    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]

    best_edges = []
    best_count = 0

    # Multiple restarts with different random seeds
    max_restarts = 20
    
    for restart in range(max_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Reset state
        adj = [0] * n
        cc = [[0] * n for _ in range(n)]
        
        # Greedy construction with random edge ordering
        random.shuffle(all_edges)
        for u, v in all_edges:
            if can_add(u, v):
                add_edge(u, v)
        
        # Then do a second pass in different order
        random.shuffle(all_edges)
        for u, v in all_edges:
            if can_add(u, v):
                add_edge(u, v)
        
        current_count = count_edges()
        
        if current_count > best_count:
            best_count = current_count
            best_edges = get_edges()
        
        # Local search
        local_search_greedy()
        
        # Update best if local search improved
        # (local_search_greedy updates best_edges/best_count directly)

    # Final polish on the best solution
    if best_edges and time.time() < deadline - 10.0:
        # Rebuild state from best_edges
        best_adj = [0] * n
        for u, v in best_edges:
            best_adj[u] |= (1 << v)
            best_adj[v] |= (1 << u)
        
        best_cc = [[0] * n for _ in range(n)]
        for u in range(n):
            for v in range(n):
                if u == v:
                    continue
                common = best_adj[u] & best_adj[v]
                best_cc[u][v] = popcount(common)
        
        # Run local search on the best solution
        adj = best_adj[:]
        cc = best_cc
        local_search_greedy()

    return best_edges

# Test
if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
    # Verify K2,3-free
    n = 20
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    valid = True
    for u in range(n):
        for v in range(u+1, n):
            common = adj[u] & adj[v]
            if bin(common).count('1') > 2:
                valid = False
                print(f"Invalid: vertices {u} and {v} have {bin(common).count('1')} common neighbors")
                break
        if not valid:
            break
    print(f"Valid: {valid}")
