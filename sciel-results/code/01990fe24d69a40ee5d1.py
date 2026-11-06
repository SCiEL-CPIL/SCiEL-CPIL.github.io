import time
import random
import itertools

def construct():
    start_time = time.time()
    # 150s limit, keep 5s margin for safety
    deadline = start_time + 145.0
    
    def get_edges_from_masks(adj_masks, n):
        edges = []
        for i in range(n):
            mask = adj_masks[i]
            for j in range(i + 1, n):
                if (mask >> j) & 1:
                    edges.append([i, j])
        return edges

    def find_violation(adj_masks, n):
        """
        Finds a subset S of size >= 6 such that the induced subgraph is regular.
        Prioritizes size 6 as it is the most restrictive and likely to be found fast.
        Returns the subset as a tuple, or None.
        """
        # Check size 6
        # We can optimize by precomputing masks? No, n is small.
        # Just iterate combinations.
        # For n=22, C(22,6) = 74613. This is fast enough to run frequently.
        for S in itertools.combinations(range(n), 6):
            mask = 0
            for v in S:
                mask |= (1 << v)
            
            d0 = (adj_masks[S[0]] & mask).bit_count()
            is_regular = True
            for v in S[1:]:
                if (adj_masks[v] & mask).bit_count() != d0:
                    is_regular = False
                    break
            if is_regular:
                return S

        # Check larger sizes only if no size 6 violation found.
        # This is expensive, so we only do it if we are confident about smaller sizes.
        # Actually, if there is a regular subgraph of size 7, it contains 6 vertices.
        # Wait, a regular graph of size 7 does NOT necessarily contain a regular subgraph of size 6.
        # Example: 7-cycle C7 is regular (degree 2). Any 6 vertices induce a path or disconnected graph, not regular.
        # So we MUST check larger sizes.
        
        # However, for local search, finding a size 6 violation is sufficient to start fixing.
        # If we fix all size 6 violations, we might still have size 7+ violations.
        # But usually, size 6 are the "bottlenecks".
        
        for k in range(7, n + 1):
            # This can be very slow for n=22. C(22, 10) is huge.
            # We will rely on the fact that if we eliminate all size 6 violations, 
            # larger ones become much rarer, or we might miss them if we don't check.
            # To be safe, we should check. But for speed in local search steps, 
            # we might only check size 6 in the inner loop and do a full check at the end.
            
            # Let's stick to checking size 6 in the fast loop.
            # The final validation will check all sizes.
            pass
        
        return None

    def is_valid_graph(adj_masks, n):
        """
        Full validation: checks all subsets of size >= 6.
        """
        # Check size 6
        for S in itertools.combinations(range(n), 6):
            mask = 0
            for v in S:
                mask |= (1 << v)
            d0 = (adj_masks[S[0]] & mask).bit_count()
            regular = True
            for v in S[1:]:
                if (adj_masks[v] & mask).bit_count() != d0:
                    regular = False
                    break
            if regular:
                return False
        
        # Check larger sizes
        # This is the expensive part.
        for k in range(7, n + 1):
            for S in itertools.combinations(range(n), k):
                mask = 0
                for v in S:
                    mask |= (1 << v)
                d0 = (adj_masks[S[0]] & mask).bit_count()
                regular = True
                for v in S[1:]:
                    if (adj_masks[v] & mask).bit_count() != d0:
                        regular = False
                        break
                if regular:
                    return False
        return True

    def local_search(adj_masks, n, deadline, rng):
        """
        Attempts to repair the graph by flipping edges to eliminate violations.
        Returns (adj_masks, is_valid)
        """
        cur = adj_masks[:]
        max_iters = 5000 # Cap iterations to avoid infinite loops in bad local minima
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Find a size-6 violation quickly
            viol = find_violation(cur, n)
            
            if viol is None:
                # No size 6 violation found. 
                # We must check if the graph is fully valid (no size >= 7 violations).
                # This check is expensive, so we do it here only if we think we are close.
                # Actually, let's just do the full check. If it's valid, we're done.
                if is_valid_graph(cur, n):
                    return cur, True
                else:
                    # There is a violation of size > 6, but no size 6.
                    # This is tricky. We need to find the specific violation.
                    # For now, we can try random flips to break it.
                    # Or we can just return invalid and let the outer loop restart.
                    # Given the rarity of "no size 6 but size 7 violation", 
                    # a random restart is often better than a complex repair.
                    # But let's try a few random flips.
                    for _ in range(10):
                        u = rng.randint(0, n-1)
                        v = rng.randint(0, n-1)
                        if u == v: continue
                        cur[u] ^= (1 << v)
                        cur[v] ^= (1 << u)
                        if is_valid_graph(cur, n):
                            return cur, True
                    return cur, False

            # Found a size-6 violation S
            S_set = set(viol)
            
            # Strategy: Flip an edge (u, w) where u in S, w not in S.
            # This changes degree of u in S by 1, breaking regularity.
            # It does not change degrees of other vertices in S.
            # It might create new violations, but it breaks the current one.
            
            if len(S_set) < n:
                u = rng.choice(list(S_set))
                outside = [v for v in range(n) if v not in S_set]
                if outside:
                    w = rng.choice(outside)
                    # Flip edge (u, w)
                    cur[u] ^= (1 << w)
                    cur[w] ^= (1 << u)
            else:
                # Violation is the whole graph
                u = rng.choice(list(S_set))
                v = rng.choice([x for x in S_set if x != u])
                cur[u] ^= (1 << v)
                cur[v] ^= (1 << u)
                
        # Time limit reached or max iters
        # Final check
        if is_valid_graph(cur, n):
            return cur, True
        return cur, False

    def try_find_graph(n, time_budget, rng):
        """
        Attempts to find a valid graph on n vertices within time_budget.
        """
        # Try multiple random restarts
        # Number of restarts depends on time budget
        # Each local search step is fast (find_violation is C(n,6) ~ 75k ops for n=22)
        
        # We will run local search in a loop, restarting if it fails.
        # Or just run local search with many iterations.
        
        # Better: Run local search. If it fails, restart with new random graph.
        
        attempts = 0
        while time.time() < time_budget and attempts < 50:
            attempts += 1
            
            # Random initialization
            # Try different densities
            p = rng.choice([0.3, 0.4, 0.5, 0.6, 0.7])
            adj_masks = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < p:
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
            
            # Run local search
            # Give local search a portion of the remaining time
            # But local search returns quickly if it fails (max_iters or time)
            # Let's let local search run until it succeeds or the global deadline is close.
            # Actually, we should pass the remaining time for this n.
            
            adj_out, valid = local_search(adj_masks, n, time_budget, rng)
            
            if valid:
                return adj_out
            
            # If local search failed, try again with a new random graph
            # But we should check time.
            if time.time() > time_budget - 5:
                break
                
        return None

    # Allocate time for different n
    # n=22 is the prize. Give it a good chunk.
    # n=21 is next.
    # n=20 is the safe fallback.
    
    # Let's allocate 60s for 22, 30s for 21, 50s for 20.
    # Total 140s.
    
    # But we want to maximize n. So we try 22 first.
    # If 22 succeeds, we return 22.
    # If 22 fails, we try 21.
    # If 21 fails, we try 20.
    
    # We need to be careful with the deadline.
    
    # Phase 1: Try n=22
    time_for_22 = 60.0
    deadline_22 = start_time + time_for_22
    rng = random.Random(12345) # Fixed seed for reproducibility in this run
    
    adj_22 = try_find_graph(22, deadline_22, rng)
    if adj_22 is not None:
        return {'n': 22, 'edges': get_edges_from_masks(adj_22, 22)}
        
    # Phase 2: Try n=21
    time_for_21 = 30.0
    deadline_21 = start_time + time_for_22 + time_for_21
    rng = random.Random(54321)
    
    adj_21 = try_find_graph(21, deadline_21, rng)
    if adj_21 is not None:
        return {'n': 21, 'edges': get_edges_from_masks(adj_21, 21)}
        
    # Phase 3: Try n=20 (Known to be possible, should be fast)
    time_for_20 = 40.0
    deadline_20 = start_time + time_for_22 + time_for_21 + time_for_20
    rng = random.Random(99999)
    
    adj_20 = try_find_graph(20, deadline_20, rng)
    if adj_20 is not None:
        return {'n': 20, 'edges': get_edges_from_masks(adj_20, 20)}
        
    # Phase 4: Fallback to smaller n if something went wrong
    # n=19
    if time.time() < deadline:
        deadline_19 = min(deadline, time.time() + 10)
        rng = random.Random(11111)
        adj_19 = try_find_graph(19, deadline_19, rng)
        if adj_19 is not None:
            return {'n': 19, 'edges': get_edges_from_masks(adj_19, 19)}
            
    # Ultimate fallback: n=5 is always valid
    return {'n': 5, 'edges': []}

if __name__ == "__main__":
    result = construct()
    print(f"Result: n={result['n']}, edges={len(result['edges'])}")
