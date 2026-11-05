import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin for 120s limit

    n = 20

    def build_paley_19():
        """Construct the Paley graph on 19 vertices (quadratic residues mod 19)."""
        adj = [0] * 19
        residues = set((x * x) % 19 for x in range(1, 10))
        for u in range(19):
            for v in range(u + 1, 19):
                if (v - u) % 19 in residues or (u - v) % 19 in residues:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            v = adj[u]
            while v:
                low = v & (-v)
                vv = low.bit_length() - 1
                if vv > u:
                    edges.append([u, vv])
                v ^= low
        return edges

    def is_valid(adj):
        """Check if no triple of vertices has 3 or more common neighbours."""
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                cij = ai & adj[j]
                if cij.bit_count() < 3:
                    continue
                for k in range(j + 1, n):
                    if (cij & adj[k]).bit_count() >= 3:
                        return False
        return True

    def find_violation(adj):
        """Find a violating triple and its common neighbors, or None."""
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                cij = ai & adj[j]
                if cij.bit_count() < 3:
                    continue
                for k in range(j + 1, n):
                    cijk = cij & adj[k]
                    if cijk.bit_count() >= 3:
                        return (i, j, k, cijk)
        return None

    def make_valid(adj):
        """Remove edges until the graph is valid."""
        attempts = 0
        while True:
            viol = find_violation(adj)
            if viol is None:
                return True
            i, j, k, cijk = viol
            # Remove edge (i, x) where x is a common neighbor
            x = -1
            v = cijk
            while v:
                low = v & (-v)
                x = low.bit_length() - 1
                break
            adj[i] &= ~(1 << x)
            adj[x] &= ~(1 << i)
            attempts += 1
            if attempts > 5000:
                return False

    def greedy_add(adj):
        """Try to add edges greedily. Returns number of edges added."""
        added = 0
        improved = True
        while improved:
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u] & (1 << v):
                        continue
                    old_u, old_v = adj[u], adj[v]
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    if is_valid(adj):
                        added += 1
                        improved = True
                        break
                    else:
                        adj[u] = old_u
                        adj[v] = old_v
                if improved:
                    break
        return added

    def local_search(adj, time_limit):
        """Local search: try swaps (remove one edge, add another) to unlock more additions."""
        def get_edges_and_non_edges(adj):
            edges_in = []
            non_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i] & (1 << j):
                        edges_in.append((i, j))
                    else:
                        non_edges.append((i, j))
            return edges_in, non_edges

        edges_in, non_edges = get_edges_and_adj(adj) if False else get_edges_and_non_edges(adj)
        best_count = count_edges(adj)
        
        # Use a deterministic order for reproducibility, but shuffle for diversity
        while time.time() < time_limit:
            if not non_edges:
                break
            
            # Try all pairs of (edge_to_remove, edge_to_add) in a systematic way
            # To keep it fast, limit the number of candidates per iteration
            found = False
            
            # We'll iterate through a subset of swaps
            # To make it deterministic but varied, we can rotate the starting index
            # or just try all if the graph is small enough. 190 edges max, so 190*190 ~ 36k checks.
            # is_valid is fast enough? 190*190 * (check cost).
            # Check cost: O(n^3) ~ 8000 ops. 36k * 8000 = 288M ops. Might be too slow in Python.
            # So we need to limit the candidates or optimize is_valid.
            
            # Let's optimize is_valid by pre-computing intersections?
            # Or just limit the swaps tried per "epoch".
            
            # Strategy: Try a random sample of swaps, or a systematic sweep.
            # Given the small size, a full sweep might take ~1-2 seconds.
            # Let's do a full sweep of swaps, but break early if we find an improvement.
            
            # Actually, let's just try to add edges first (greedy), then try swaps.
            # The current local_search tries swaps directly.
            
            # Let's try a limited number of random swaps to keep it fast, 
            # or a systematic sweep with a time check.
            
            # Let's do a systematic sweep of all possible swaps.
            # If it's too slow, we'll fall back to random.
            
            # For now, let's try a few random swaps per loop to be safe.
            # No, the prompt says "systematic". Let's try a full sweep but with a time limit inside.
            
            # Actually, the most effective local search for these small graphs is often just 
            # trying to add any missing edge, and if none can be added, trying a 1-edge swap.
            
            # Let's implement a full sweep of swaps.
            # To avoid O(N^4) explosion, we rely on the fact that n=20 is small.
            
            # Let's just try to add edges greedily first.
            greedy_add(adj)
            c = count_edges(adj)
            if c > best_count:
                best_count = c
                edges_in, non_edges = get_edges_and_non_edges(adj)
                if not non_edges:
                    break
            
            # Now try swaps
            # We'll try to remove one edge and add another.
            # To be systematic, we iterate over all pairs.
            # But 190*190 = 36100 pairs.
            # Each pair requires an is_valid check.
            # is_valid is O(n^3) = 8000 bit ops.
            # 36100 * 8000 = 288,800,000 bit ops.
            # In Python, this might take ~10-20 seconds.
            # So we can do one full sweep per second or so.
            
            # Let's do a full sweep.
            # If we find an improvement, we restart the sweep.
            
            # To make it faster, we can pre-compute the "cost" of adding an edge?
            # No, let's just do it.
            
            # Optimization: Only try swaps where the edge to add is "close" to the edge to remove?
            # No, let's just do a full sweep.
            
            # Actually, let's limit the number of swaps tried per iteration to avoid timeout.
            # We'll try 1000 random swaps.
            
            # Wait, the instruction says "systematic".
            # Let's do a full sweep, but break if time is running out.
            
            # Let's try a full sweep.
            improved = False
            for e_rm in edges_in:
                if time.time() > time_limit - 0.1:
                    break
                u1, v1 = e_rm
                for e_add in non_edges:
                    u2, v2 = e_add
                    if (u1 == u2 and v1 == v2) or (u1 == v2 and v1 == u2):
                        continue
                    # Check if e_add is already present (shouldn't be, but just in case)
                    if adj[u2] & (1 << v2):
                        continue
                    
                    old = (adj[u1], adj[v1], adj[u2], adj[v2])
                    # Remove
                    adj[u1] &= ~(1 << v1)
                    adj[v1] &= ~(1 << u1)
                    # Add
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    
                    if is_valid(adj):
                        # Try to add more edges after the swap
                        greedy_add(adj)
                        c = count_edges(adj)
                        if c > best_count:
                            best_count = c
                            edges_in, non_edges = get_edges_and_non_edges(adj)
                            improved = True
                            # Restart the sweep with the new sets
                            break
                        else:
                            # Revert if no improvement in edge count
                            adj[u1], adj[v1], adj[u2], adj[v2] = old
                    else:
                        adj[u1], adj[v1], adj[u2], adj[v2] = old
                
                if improved:
                    break
            
            if not improved:
                # No beneficial swap found in this sweep
                break

        return best_count

    # --- Main Logic ---

    # 1. Start with Paley graph on 19 vertices + isolated vertex 19
    adj = build_paley_19() + [0]
    
    # 2. Connect vertex 19 to a maximal independent set of the Paley graph
    # Find a maximal independent set (MIS) of G_19
    # Greedy MIS: sort vertices by degree (ascending), pick if no neighbor picked
    # Actually, to maximize edges, we want to connect 19 to as many vertices as possible.
    # So we want a large independent set.
    # Greedy MIS:
    picked = [False] * 19
    # Sort by degree ascending
    degrees = [adj[i].bit_count() for i in range(19)]
    vertices = list(range(19))
    vertices.sort(key=lambda x: degrees[x])
    
    for v in vertices:
        if picked[v]:
            continue
        # Check if v is adjacent to any picked vertex
        conflict = False
        for u in range(19):
            if picked[u] and (adj[v] & (1 << u)):
                conflict = True
                break
        if not conflict:
            picked[v] = True
            # Connect 19 to v
            adj[19] |= (1 << v)
            adj[v] |= (1 << 19)
    
    # 3. Make valid (should already be valid, but just in case)
    if not is_valid(adj):
        make_valid(adj)
    
    # 4. Greedy add
    greedy_add(adj)
    
    best_adj = adj[:]
    best_count = count_edges(adj)
    
    # 5. Local search
    remaining = deadline - time.time()
    if remaining > 5:
        local_search(adj, time.time() + min(remaining * 0.9, 100))
        
        c = count_edges(adj)
        if c > best_count:
            best_count = c
            best_adj = adj[:]
    
    # 6. Return edges
    return adj_to_edges(best_adj)
