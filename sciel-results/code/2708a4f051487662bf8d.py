import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    # Precomputed: smallest known valid n is 67 (Paley tournament, p=67, 67%4==3)
    # We want to minimize n. Let's try to find a valid tournament for the smallest n.
    # But first, let's check if we can return 67 quickly (Paley construction).
    # If we have time, we could try smaller n, but known lower bound is 31.
    # The task says "The smallest known has 67 vertices" so we aim for 67.
    # But the evaluator wants to MINIMIZE n. So if we can find a valid tournament
    # on fewer than 67 vertices, that would be better.
    #
    # Strategy:
    # 1. First, construct the Paley tournament on p=67 (known valid).
    # 2. Then, try to find a valid tournament on smaller n using a search.
    #    But the search space is huge. Let's think about this differently.
    #
    # Actually, re-reading: "Minimize n. A valid tournament on n vertices certifies f(4) <= n.
    # The smallest known has 67 vertices; Erdős's lower bound gives f(4) >= 31."
    #
    # So the known best is 67. We should try to find something <= 67.
    # If we can't find better, return 67.
    #
    # Let's use the Paley construction for p=67 as our baseline.
    # Then, try a randomized/local search to see if we can do better.
    #
    # But given time constraints, let's be smart:
    # - Return 67 quickly if we can.
    # - Use remaining time for local search on smaller n.
    #
    # Actually, let's first verify that p=67 Paley works.
    
    def make_paley(p):
        """Create Paley tournament on p vertices. Returns orientation list."""
        squares = set((x * x) % p for x in range(1, p))
        orientation = []
        for i in range(p):
            for j in range(i + 1, p):
                # i -> j iff (j - i) % p in squares
                if ((j - i) % p) in squares:
                    orientation.append(0)
                else:
                    orientation.append(1)
        return orientation

    def check_validity(n, orientation):
        """Check if every 4-set has a common dominator.
        orientation is a list of bits for pairs (i,j) with i<j in lexicographic order.
        0 means i->j, 1 means j->i.
        """
        # Build adjacency: adj[i][j] = 1 if i -> j
        adj = [[0] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                bit = orientation[idx]
                idx += 1
                if bit == 0:
                    adj[i][j] = 1
                else:
                    adj[j][i] = 1

        # For each 4-set {a,b,c,d}, check if there's a vertex v such that
        # v->a, v->b, v->c, v->d.
        # v can be one of a,b,c,d or outside.
        # Actually, v just needs to be any vertex that dominates all 4.
        
        # Precompute for each vertex v, the set of vertices it dominates.
        # dom[v] = bitmask of vertices that v points to.
        dom = [0] * n
        for i in range(n):
            for j in range(n):
                if adj[i][j]:
                    dom[i] |= (1 << j)

        # Check all 4-sets
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    for d in range(c + 1, n):
                        # Check if any vertex dominates {a,b,c,d}
                        found = False
                        for v in range(n):
                            if (dom[v] >> a) & 1 and (dom[v] >> b) & 1 and \
                               (dom[v] >> c) & 1 and (dom[v] >> d) & 1:
                                found = True
                                break
                        if not found:
                            return False
        return True

    # First, get the baseline: Paley on 67
    baseline_n = 67
    baseline_orient = make_paley(baseline_n)
    
    # Verify baseline (should be fast for n=67? 67 choose 4 = ~700k, times 67 checks... might be slow)
    # Let's optimize the check.
    
    def check_validity_fast(n, orientation):
        """Faster check using bitmask intersection."""
        adj = [[0] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                bit = orientation[idx]
                idx += 1
                if bit == 0:
                    adj[i][j] = 1
                else:
                    adj[j][i] = 1

        # in_mask[v] = bitmask of vertices that point TO v
        in_mask = [0] * n
        for i in range(n):
            for j in range(n):
                if adj[i][j]:
                    in_mask[j] |= (1 << i)

        # For 4-set {a,b,c,d}, need in_mask[a] & in_mask[b] & in_mask[c] & in_mask[d] != 0
        # (i.e., there's some vertex pointing to all 4)
        
        for a in range(n):
            ma = in_mask[a]
            if ma == 0:
                continue
            for b in range(a + 1, n):
                mab = ma & in_mask[b]
                if mab == 0:
                    continue
                for c in range(b + 1, n):
                    mabc = mab & in_mask[c]
                    if mabc == 0:
                        continue
                    for d in range(c + 1, n):
                        if (mabc & in_mask[d]) == 0:
                            return False
        return True

    # Verify baseline
    baseline_valid = check_validity_fast(baseline_n, baseline_orient)
    
    # If baseline is valid, we have a fallback. Now try to find smaller n.
    best_n = baseline_n if baseline_valid else 90
    best_orient = baseline_orient
    
    # Try smaller n values. We need to search for valid tournaments.
    # For small n, we can try random tournaments and check.
    # But the probability of a random tournament being valid might be low.
    
    # Let's try a different approach: start from the Paley construction
    # and try to "delete" vertices? No, that changes n.
    
    # Actually, let's just try random tournaments for n from 31 up to best_n-1.
    # But this might not find anything quickly.
    
    # Given time constraints, let's be pragmatic:
    # - If baseline is valid, return it (n=67).
    # - Try a short search for smaller n if time permits.
    
    if baseline_valid:
        # Try to find smaller n
        for target_n in range(31, best_n):
            if time.time() - start_time > DEADLINE - 20:
                break
            # Try a few random tournaments
            for attempt in range(5):
                if time.time() - start_time > DEADLINE - 5:
                    break
                # Generate random tournament
                orient = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        orient.append(0 if (i * 31 + j * 7 + attempt * 13) % 2 == 0 else 1)
                if check_validity_fast(target_n, orient):
                    best_n = target_n
                    best_orient = orient
                    break
            if best_n <= target_n:
                break
    
    return {'n': best_n, 'orientation': best_orient}
