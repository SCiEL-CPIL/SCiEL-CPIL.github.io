"""5-colouring of complete graph without monochromatic C4. Maximize n."""
import time
import random
import sys

# Deadline guard
_DEADLINE = time.time() + 140.0  # 10s safety margin before 150s kill


def _elapsed():
    return time.time() - _DEADLINE


def _is_valid(colors, n):
    """Check that no colour class contains a C4 (two vertices with two common neighbours in one colour)."""
    # Build adjacency for each colour
    adj = [[set() for _ in range(n)] for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            if c < 0 or c > 4:
                return False
            adj[c][i].add(j)
            adj[c][j].add(i)
            idx += 1
    for c in range(5):
        for i in range(n):
            for j in range(i + 1, n):
                # Check if i and j have two common neighbours in colour c
                common = adj[c][i] & adj[c][j]
                if len(common) >= 2:
                    return False
    return True


def _count_c4s(colors, n):
    """Count total C4 violations across all colours."""
    adj = [set() for _ in range(n)]
    # Actually we need per-colour
    adj_c = [[set() for _ in range(n)] for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj_c[c][i].add(j)
            adj_c[c][j].add(i)
            idx += 1
    total = 0
    for c in range(5):
        for i in range(n):
            for j in range(i + 1, n):
                common = adj_c[c][i] & adj_c[c][j]
                total += max(0, len(common) - 1)  # each pair with >=2 common neighbours contributes
    return total


def _lazy_woldar_construct(n):
    """
    Attempt a construction based on finite field / cyclic group ideas.
    For n = 11, use the circular distance method (known valid).
    For larger n, try structured assignments.
    Returns list of colours or None.
    """
    if n <= 1:
        return []
    if n == 2:
        return [0]
    if n == 11:
        # Known valid: circular distance 1..5 mod 11
        return [min(j - i, n - (j - i)) - 1 for i in range(n) for j in range(i + 1, n)]
    return None


def _random_greedy(n, seed=0):
    """
    Greedy random construction: assign colours one edge at a time in lexicographic order,
    choosing a random valid colour if possible.
    """
    rng = random.Random(seed)
    colors = []
    # For speed, track adj per colour
    adj = [[set() for _ in range(n)] for _ in range(5)]
    for i in range(n):
        for j in range(i + 1, n):
            valid_colors = []
            for c in range(5):
                # Check if adding (i,j) in colour c creates a C4
                # A C4 is created if there exist two common neighbours k,l of i and j in colour c
                # More precisely: if |adj[c][i] ∩ adj[c][j]| >= 1, then adding edge creates
                # at least one pair with 2 common neighbours? No:
                # After adding (i,j) to colour c, for any pair (a,b) that both are neighbours
                # of i and j in colour c, they now have i and j as two common neighbours.
                # So the condition is: |adj[c][i] ∩ adj[c][j]| >= 2 would mean violation.
                # Actually: we need that NO pair (a,b) has i and j as two common neighbours.
                # Pair (a,b) has common neighbours i and j in colour c iff:
                #   a-i, b-i, a-j, b-j all in colour c.
                # So we need: for all a, b: not (a in adj[c][i] and b in adj[c][i] and a in adj[c][j] and b in adj[c][j])
                # i.e., |adj[c][i] ∩ adj[c][j]| <= 1
                common = adj[c][i] & adj[c][j]
                if len(common) <= 1:
                    valid_colors.append(c)
            if not valid_colors:
                return None
            c = rng.choice(valid_colors)
            colors.append(c)
            adj[c][i].add(j)
            adj[c][j].add(i)
    return colors


def _local_search(n, colors, max_steps=50000, seed=0):
    """
    Local search: randomly pick an edge, try changing its colour, accept if it reduces
    total C4 count or keeps it the same with some probability.
    """
    rng = random.Random(seed)
    total_edges = n * (n - 1) // 2
    if total_edges == 0:
        return colors
    
    # Build adjacency structures
    adj = [[set() for _ in range(n)] for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i].add(j)
            adj[c][j].add(i)
            idx += 1
    
    def c4_count_for_colour(c):
        """Count C4 violations for a specific colour."""
        cnt = 0
        for i in range(n):
            for j in range(i + 1, n):
                common = adj[c][i] & adj[c][j]
                if len(common) >= 2:
                    cnt += len(common) - 1
        return cnt
    
    total_cost = sum(c4_count_for_colour(c) for c in range(5))
    
    for step in range(max_steps):
        if _elapsed() > 5:
            break
        # Pick random edge
        e = rng.randint(0, total_edges - 1)
        # Find (i, j) for edge index e
        # Lexicographic: for i in range(n), j in range(i+1, n)
        # Edge index for (i,j) = i*n - i*(i+1)//2 + (j - i - 1)
        # Inverse: find i such that tri(i) <= e < tri(i+1) where tri(i) = i*n - i*(i+1)//2
        # Actually simpler: precompute or binary search
        lo, hi = 0, n - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            cnt = mid * n - mid * (mid + 1) // 2
            if cnt <= e:
                lo = mid
            else:
                hi = mid - 1
        i = lo
        offset = e - (i * n - i * (i + 1) // 2)
        j = i + 1 + offset
        
        old_c = colors[e]
        
        # Try each other colour
        best_c = None
        best_cost = total_cost
        
        for new_c in range(5):
            if new_c == old_c:
                continue
            # Simulate: remove (i,j) from old_c, add to new_c
            adj[old_c][i].discard(j)
            adj[old_c][j].discard(i)
            
            # Check if new_c is valid: |adj[new_c][i] ∩ adj[new_c][j]| <= 1
            common = adj[new_c][i] & adj[new_c][j]
            if len(common) <= 1:
                # Compute delta cost
                # Removing (i,j) from old_c: for each pair (a,b) in adj[old_c][i] ∩ adj[old_c][j],
                # before removal they had i,j as common neighbours... 
                # This is complex. Just recompute locally.
                # Actually, let's just compute the cost change for affected pairs.
                # Pairs affected by removing (i,j) from old_c: any pair (a,b) where both a,b
                # were common neighbours of i and j in old_c.
                # This is getting complex. Let's just do a simpler acceptance criterion.
                pass
            adj[old_c][i].add(j)
            adj[old_c][j].add(i)
        
        # Simpler approach: just try random recolouring and accept if valid
        new_c = rng.randint(0, 4)
        if new_c == old_c:
            continue
        
        # Remove from old
        adj[old_c][i].discard(j)
        adj[old_c][j].discard(i)
        
        # Check if new_c works
        common = adj[new_c][i] & adj[new_c][j]
        if len(common) <= 1:
            # Add to new
            adj[new_c][i].add(j)
            adj[new_c][j].add(i)
            colors[e] = new_c
        else:
            # Revert
            adj[old_c][i].add(j)
            adj[old_c][j].add(i)
    
    return colors


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # First try known constructions
    # n=11 is known valid
    best_n = 11
    best_colors = [min(j - i, 11 - (j - i)) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # Try to find larger n via greedy + local search
    for n in range(12, 29):
        if time.time() > deadline - 20:
            break
        
        found = False
        for attempt in range(30):
            if time.time() > deadline - 10:
                break
            colors = _random_greedy(n, seed=attempt * 1000 + n)
            if colors is not None:
                # Verify
                if _is_valid(colors, n):
                    best_n = n
                    best_colors = colors
                    found = True
                    break
        
        if found:
            # Try to go even larger with local search improvements
            continue
        else:
            # If greedy failed, try local search from random init
            for attempt in range(5):
                if time.time() > deadline - 10:
                    break
                # Random init
                colors = [random.randint(0, 4) for _ in range(n * (n - 1) // 2)]
                # Local search to reduce C4s
                colors = _local_search(n, colors, max_steps=10000, seed=attempt)
                if _is_valid(colors, n):
                    best_n = n
                    best_colors = colors
                    found = True
                    break
    
    # Final verification
    if not _is_valid(best_colors, best_n):
        # Fallback to known good
        best_n = 11
        best_colors = [min(j - i, 11 - (j - i)) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    return {'n': best_n, 'colors': best_colors}
