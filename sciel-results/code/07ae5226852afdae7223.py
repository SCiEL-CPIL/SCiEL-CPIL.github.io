"""Snake-in-the-box Q_9: multi-start local search with fast validity checks."""
import time
import random

N = 9
FULL_MASK = (1 << N) - 1
DEADLINE_FRACTION = 0.88  # safety margin for the 150s kill

# Precompute bit masks
BIT_MASKS = [1 << b for b in range(N)]


def _neighbors(v):
    """Return list of 9 neighbors of vertex v."""
    return [v ^ m for m in BIT_MASKS]


def _build_neighbor_sets(path):
    """For each vertex in path, build a set of its 9 neighbors for O(1) checks."""
    nsets = []
    for v in path:
        nsets.append(frozenset(v ^ m for m in BIT_MASKS))
    return nsets


def _is_valid_snake(path):
    """Check that path is a valid induced path: consecutive differ by 1 bit,
    non-consecutive differ by >= 2 bits."""
    n = len(path)
    if n < 2:
        return True
    used = set(path)
    if len(used) != n:
        return False
    # Check consecutive vertices differ by exactly one bit
    for i in range(n - 1):
        if (path[i] ^ path[i + 1]).bit_count() != 1:
            return False
    # Check non-consecutive vertices differ by at least 2 bits
    # Use neighbor sets: for each vertex, its neighbors must not appear non-adjacent
    for i in range(n):
        v = path[i]
        for nb in (v ^ m for m in BIT_MASKS):
            if nb in used:
                # nb must be path[i-1] or path[i+1] only
                if i > 0 and path[i - 1] == nb:
                    continue
                if i < n - 1 and path[i + 1] == nb:
                    continue
                return False
    return True


def _greedy_seed():
    """Greedy snake: always extend with the lowest bit flip that keeps the path induced."""
    path = [0]
    used = {0}
    # For fast checking, maintain neighbor sets of all path vertices
    while True:
        cur = path[-1]
        nxt = None
        for b in range(N):
            v = cur ^ BIT_MASKS[b]
            if v in used:
                continue
            # Check v is not a neighbor of any non-consecutive vertex
            # i.e., v is not adjacent to path[i] for i < len(path)-2
            safe = True
            for i in range(len(path) - 2):
                if (v ^ path[i]).bit_count() == 1:
                    safe = False
                    break
            if safe:
                nxt = v
                break
        if nxt is None:
            return path
        path.append(nxt)
        used.add(nxt)


def _local_search(path, deadline):
    """Improve a snake via 2-opt reversals and insertions. Returns best found."""
    best = list(path)
    best_len = len(best)
    n = len(best)
    
    # Build neighbor sets for fast validity
    nsets = _build_neighbor_sets(best)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(best)
        
        # --- 2-opt: reverse a segment [i+1, j] ---
        # Reversing segment [i+1..j] keeps edges valid (same vertices, reversed order).
        # Need to check: new non-adjacent pairs don't become adjacent.
        # The only new potential conflicts are at the boundaries:
        #   - best[i] and best[j] become adjacent (were non-consecutive)
        #   - best[i+1] and best[j+1] become adjacent (were non-consecutive)
        # Actually in a 2-opt on a path: we reverse [i+1..j], so:
        #   edge (best[i], best[i+1]) becomes (best[i], best[j])
        #   edge (best[j], best[j+1]) becomes (best[i+1], best[j+1])
        # We need: best[i] and best[j] differ by 1 bit (they were >= 2 apart)
        #          best[i+1] and best[j+1] differ by 1 bit (they were >= 2 apart)
        # And no new non-consecutive adjacency is created.
        
        for i in range(n - 1):
            if time.time() >= deadline:
                break
            vi = best[i]
            for j in range(i + 2, n):
                if time.time() >= deadline:
                    break
                vj = best[j]
                # Check the two new edges
                if (vi ^ vj).bit_count() != 1:
                    continue
                if j + 1 < n:
                    if (best[i+1] ^ best[j+1]).bit_count() != 1:
                        continue
                # Check no new non-consecutive adjacency
                # After reversal, the segment [i+1..j] is reversed.
                # The internal adjacencies are preserved (reversed path of a valid path is valid).
                # New potential conflicts:
                #   vi is now adjacent to vj (checked above as edge)
                #   vi must not be adjacent to any other vertex in reversed segment except vj
                #   vj must not be adjacent to any other vertex in reversed segment except vi
                #   If j+1 < n: best[j+1] must not be adjacent to any vertex in reversed segment except best[i+1]
                #   If i > 0: best[i-1] must not be adjacent to any vertex in reversed segment except best[i]... 
                #     wait, best[i-1] was adjacent to best[i] before, and best[i] is still there.
                #     Actually best[i] is NOT in the reversed segment; it stays.
                #     The reversed segment is [i+1..j]. best[i-1] was adjacent to best[i] (outside segment).
                #     After reversal, best[i-1] is still adjacent to best[i] (unchanged). OK.
                #     But could best[i-1] now be adjacent to some vertex in the reversed segment?
                #     Before: best[i-1] was not adjacent to any vertex in [i+1..j] (non-consecutive).
                #     After reversal: same set of vertices, so still not adjacent. OK.
                #   Similarly for best[j+1]: before, not adjacent to [i+1..j] except best[j].
                #     After reversal, best[j+1] is adjacent to best[i+1] (new edge, checked).
                #     Could best[j+1] be adjacent to other vertices in reversed [i+1..j]?
                #     Before: best[j+1] was not adjacent to any in [i+1..j] except best[j].
                #     After: same set, so only possible new adjacency is with best[i+1] (which we checked as edge).
                #     Wait, we need best[j+1] NOT adjacent to others. Since the set is the same,
                #     and before it wasn't adjacent to any except best[j], after reversal it's still
                #     not adjacent to any except... well best[j] is still in the segment.
                #     Hmm, this is getting complex. Let me just do a full validation for candidates.
                
                # Full validation is expensive but 2-opt candidates are rare enough
                candidate = best[:i+1] + best[i+1:j+1][::-1] + best[j+1:]
                if _is_valid_snake(candidate):
                    best = candidate
                    best_len = n
                    improved = True
                    break
            if improved:
                break
        if improved:
            nsets = _build_neighbor_sets(best)
            continue
        
        # --- Insertion: remove vertex at position k, insert at position m ---
        n = len(best)
        for k in range(1, n - 1):  # can't remove endpoints easily (would create disconnected)
            if time.time() >= deadline:
                break
            vk = best[k]
            # Remove vk: best[:k] + best[k+1:]
            # Need best[k-1] and best[k+1] to differ by 1 bit (new edge)
            if (best[k-1] ^ best[k+1]).bit_count() != 1:
                continue
            # Now try inserting vk at position m (0 <= m <= n-1, m != k, m != k-1)
            for m in range(n):
                if m == k or m == k - 1:
                    continue
                if time.time() >= deadline:
                    break
                # Insert vk at position m in the (n-1)-element list
                # New list: best[:k] + best[k+1:m] + [vk] + best[m:k+1] (if m > k+1)
                # or best[:k] + best[k+1:] with vk inserted at m
                # Let's construct:
                removed = best[:k] + best[k+1:]  # length n-1
                # Insert at position m (0-indexed in removed list)
                # m ranges 0..n-1, but in removed list positions are 0..n-2
                # If m > k, the position in removed list is m-1
                # Actually let's just build the candidate
                if m <= k:
                    candidate = best[:m] + [vk] + best[m:k] + best[k+1:]
                else:
                    candidate = best[:k] + best[k+1:m] + [vk] + best[m:]
                
                # Quick check: consecutive edges
                valid = True
                for idx in range(len(candidate) - 1):
                    if (candidate[idx] ^ candidate[idx+1]).bit_count() != 1:
                        valid = False
                        break
                if not valid:
                    continue
                # Full check
                if _is_valid_snake(candidate):
                    best = candidate
                    best_len = n
                    improved = True
                    break
            if improved:
                break
        if improved:
            nsets = _build_neighbor_sets(best)
    
    return best


def _multi_start(deadline):
    """Run multiple greedy seeds with different bit orderings, then local search."""
    best_overall = None
    best_len = 0
    
    # Different bit orderings for greedy
    orderings = []
    base = list(range(N))
    orderings.append(base)
    orderings.append(base[::-1])
    random.shuffle(base)
    orderings.append(base)
    
    # Also try starting from different vertices
    start_vertices = [0]
    for _ in range(5):
        start_vertices.append(random.randint(0, (1 << N) - 1))
    
    for sv in start_vertices:
        for ordering in orderings:
            if time.time() >= deadline:
                break
            
            # Greedy with specific start and bit ordering
            path = [sv]
            used = {sv}
            while True:
                cur = path[-1]
                nxt = None
                for b in ordering:
                    v = cur ^ BIT_MASKS[b]
                    if v in used:
                        continue
                    safe = True
                    for i in range(len(path) - 2):
                        if (v ^ path[i]).bit_count() == 1:
                            safe = False
                            break
                    if safe:
                        nxt = v
                        break
                if nxt is None:
                    break
                path.append(nxt)
                used.add(nxt)
            
            if len(path) > best_len:
                best_len = len(path)
                best_overall = list(path)
            
            # Local search on this seed
            if time.time() < deadline - 10:
                improved = _local_search(path, deadline)
                if len(improved) > best_len:
                    best_len = len(improved)
                    best_overall = improved
    
    return best_overall


def construct():
    """Construct the longest induced path (snake) in Q_9."""
    start_time = time.time()
    deadline = start_time + 150 * DEADLINE_FRACTION
    
    # Get initial greedy seed
    seed = _greedy_seed()
    best = list(seed)
    best_len = len(best)
    
    # Multi-start local search
    result = _multi_start(deadline)
    if result and len(result) > best_len:
        best = result
        best_len = len(result)
    
    # Final safety: validate
    if not _is_valid_snake(best):
        # Fallback to greedy seed
        best = _greedy_seed()
    
    return best
