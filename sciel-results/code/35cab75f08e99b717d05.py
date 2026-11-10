import random
import time
import itertools

def construct():
    """
    Constructs a graph with no induced regular subgraph on 6 or more vertices.
    Returns a dictionary with 'n' and 'edges'.
    """
    
    def is_regular(subset, adj_mask):
        """
        Checks if the induced subgraph on 'subset' (list of ints) is regular.
        Returns True if regular, False otherwise.
        """
        if len(subset) < 6:
            return False
        
        # Build mask for the subset
        mask = 0
        for v in subset:
            mask |= (1 << v)
        
        # Calculate degrees
        degs = []
        for v in subset:
            degs.append((adj_mask[v] & mask).bit_count())
            
        return len(set(degs)) == 1

    def find_bad_sets(n, adj_mask, size):
        """
        Finds all subsets of 'size' that induce a regular subgraph.
        """
        bad_sets = []
        vertices = range(n)
        for subset in itertools.combinations(vertices, size):
            if is_regular(subset, adj_mask):
                bad_sets.append(subset)
        return bad_sets

    def fix_bad_set(bad_set, adj_mask, n):
        """
        Attempts to break the regularity of 'bad_set' by flipping a random edge.
        Returns True if an edge was flipped, False otherwise.
        """
        # Pick a random vertex in the set and a random vertex outside (or inside)
        # Flipping an edge inside the set changes degrees of 2 vertices in the set.
        # Flipping an edge from set to outside changes degree of 1 vertex in the set.
        
        # Strategy: Flip a random edge involving a vertex in the set.
        u = random.choice(bad_set)
        v = random.randint(0, n - 1)
        
        if u == v:
            return False
            
        # Flip edge (u, v)
        if (adj_mask[u] >> v) & 1:
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
        else:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            
        return True

    def try_construct(n, time_limit=10.0):
        """
        Attempts to construct a valid graph of size n.
        """
        start_time = time.time()
        
        # Initialize random graph
        adj_mask = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        
        # Iteratively fix bad sets
        # We prioritize size 6, then 7, etc.
        # We loop until no bad sets of size 6 are found, then check 7, etc.
        
        while time.time() - start_time < time_limit:
            # Check size 6
            bad_6 = find_bad_sets(n, adj_mask, 6)
            if bad_6:
                # Pick a random bad set and fix it
                bad_set = random.choice(bad_6)
                if not fix_bad_set(bad_set, adj_mask, n):
                    # If fix failed (shouldn't happen), try another
                    continue
                # After fixing, we might create new bad sets or break others.
                # We restart the check loop.
                continue
            
            # If no bad 6-sets, check size 7
            bad_7 = find_bad_sets(n, adj_mask, 7)
            if bad_7:
                bad_set = random.choice(bad_7)
                if not fix_bad_set(bad_set, adj_mask, n):
                    continue
                continue
            
            # If no bad 7-sets, check size 8
            bad_8 = find_bad_sets(n, adj_mask, 8)
            if bad_8:
                bad_set = random.choice(bad_8)
                if not fix_bad_set(bad_set, adj_mask, n):
                    continue
                continue
            
            # Check size 9 to n
            valid = True
            for size in range(9, n + 1):
                bad_k = find_bad_sets(n, adj_mask, size)
                if bad_k:
                    bad_set = random.choice(bad_k)
                    if not fix_bad_set(bad_set, adj_mask, n):
                        continue
                    valid = False
                    break
                # If we found and fixed a bad set, we need to re-check from 6
                # because the fix might have created a bad 6-set.
                if not valid:
                    break
            
            if valid:
                # No bad sets found in 6..n
                # Final verification to be absolutely sure
                is_valid = True
                for size in range(6, n + 1):
                    if find_bad_sets(n, adj_mask, size):
                        is_valid = False
                        break
                
                if is_valid:
                    return adj_mask
            else:
                # We fixed a set, so the state changed. Loop continues to re-check 6.
                pass
        
        return None

    # Try to construct n=20
    # We allow multiple restarts if the first few fail
    max_restarts = 5
    for i in range(max_restarts):
        adj = try_construct(20, time_limit=20.0)
        if adj:
            edges = []
            for u in range(20):
                for v in range(u + 1, 20):
                    if (adj[u] >> v) & 1:
                        edges.append([u, v])
            return {'n': 20, 'edges': edges}
            
    # Fallback to n=19 if 20 fails
    for i in range(10):
        adj = try_construct(19, time_limit=10.0)
        if adj:
            edges = []
            for u in range(19):
                for v in range(u + 1, 19):
                    if (adj[u] >> v) & 1:
                        edges.append([u, v])
            return {'n': 19, 'edges': edges}

    # Fallback to n=18
    for i in range(20):
        adj = try_construct(18, time_limit=5.0)
        if adj:
            edges = []
            for u in range(18):
                for v in range(u + 1, 18):
                    if (adj[u] >> v) & 1:
                        edges.append([u, v])
            return {'n': 18, 'edges': edges}
            
    # Last resort: n=10
    adj = try_construct(10, time_limit=2.0)
    if adj:
        edges = []
        for u in range(10):
            for v in range(u + 1, 10):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return {'n': 10, 'edges': edges}

    # Absolute fallback
    return {'n': 5, 'edges': []}
