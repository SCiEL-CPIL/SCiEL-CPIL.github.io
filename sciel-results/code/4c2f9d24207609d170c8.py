# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

def _build_adj(n, orientation):
    adj = np.zeros((n, n), dtype=np.uint8)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = orientation[idx]
            idx += 1
            if bit == 0:
                adj[i, j] = 1
            else:
                adj[j, i] = 1
    return adj

def _local_search(n, max_time=10.0):
    """
    Attempt to find a valid tournament on n vertices using local search.
    Returns orientation list if found, else None.
    """
    start_time = time.time()
    
    # Random initialization
    orientation = [0] * (n * (n - 1) // 2)
    rng = np.random.default_rng(42 + n)
    for i in range(len(orientation)):
        orientation[i] = rng.integers(0, 2)
        
    adj = _build_adj(n, orientation)
    
    # Precompute indices for pairs
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
            
    # Map pair to index in orientation list
    pair_to_idx = {}
    for idx, (i, j) in enumerate(pairs):
        pair_to_idx[(i, j)] = idx
        pair_to_idx[(j, i)] = idx
        
    # Initialize cover counts and bad count
    # We need to iterate over all 4-sets.
    # For efficiency, we store cover_count in a dictionary or array.
    # Number of 4-sets is C(n, 4).
    # For n=31, C(31,4) = 31465.
    # For n=40, C(40,4) = 91390.
    # We can map each 4-set to an index.
    
    # To speed up updates, we need to know which 4-sets are affected by a flip.
    # As derived:
    # Flip (u, v) where u -> v becomes v -> u.
    # Lost covers for u: S = {v, x, y, z} with {x,y,z} subset of D_u \ {v}
    # Gained covers for v: S = {u, x, y, z} with {x,y,z} subset of D_v
    
    # We maintain:
    # cover_count[S_index]: number of vertices dominating S.
    # bad_count: number of S with cover_count 0.
    
    # To get S_index, we can use a mapping from sorted tuple to index.
    # Or just use a dict: key=tuple(sorted(S)), val=cover_count.
    # Dict lookup is O(1) average.
    
    cover_counts = {}
    bad_count = 0
    
    # Initial setup
    # For each vertex v, D_v is the set of vertices it dominates.
    # For each v, for each 4-subset S of D_v, increment cover_count[S].
    
    D = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and adj[i, j] == 1:
                D[i].add(j)
                
    # Build cover counts
    for v in range(n):
        if len(D[v]) >= 4:
            for S in itertools.combinations(D[v], 4):
                key = S
                if key in cover_counts:
                    cover_counts[key] += 1
                else:
                    cover_counts[key] = 1
                    # If it was 0, it was bad. But we are building from scratch.
                    # We don't know if it was bad before.
                    # Better: Initialize all to 0, then increment.
                    # Then count how many are 0 at the end.
                    
    # Reset and do it properly
    cover_counts = {}
    for v in range(n):
        if len(D[v]) >= 4:
            for S in itertools.combinations(D[v], 4):
                if S in cover_counts:
                    cover_counts[S] += 1
                else:
                    cover_counts[S] = 1
                    
    # Count bads
    total_4sets = len(list(itertools.combinations(range(n), 4)))
    # This is inefficient. Let's just count zeros in cover_counts?
    # No, keys not in cover_counts have count 0.
    # We need the total number of 4-sets.
    # C(n, 4).
    # bad_count = total_4sets - len(cover_counts)
    
    from math import comb
    total_4sets = comb(n, 4)
    bad_count = total_4sets - len(cover_counts)
    
    if bad_count == 0:
        return orientation
        
    # Local Search Loop
    current_orientation = orientation[:]
    current_adj = adj.copy()
    current_D = [set(d) for d in D]
    
    while time.time() - start_time < max_time:
        # Pick random edge to flip
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
            
        # Ensure i < j for index lookup? No, pair_to_idx handles both.
        # But we need to know direction.
        # current_adj[i, j] is 1 if i -> j.
        
        if current_adj[i, j] == 1:
            # i -> j, flip to j -> i
            u, v = i, j
        else:
            # j -> i, flip to i -> j
            u, v = j, i
            
        # u currently dominates v. We are removing u's dominance over v.
        # v currently does NOT dominate u. We are adding v's dominance over u.
        
        # Update D sets
        # u loses v
        # v gains u
        
        # Identify affected 4-sets
        # 1. Sets that were covered by u and contain v (and not u)
        #    S = {v, x, y, z} where {x,y,z} subset of D[u] \ {v}
        lost_sets = []
        D_u_without_v = current_D[u] - {v}
        if len(D_u_without_v) >= 3:
            for combo in itertools.combinations(D_u_without_v, 3):
                S = (v,) + combo
                lost_sets.append(S)
                
        # 2. Sets that will be covered by v and contain u (and not v)
        #    S = {u, x, y, z} where {x,y,z} subset of D[v]
        gained_sets = []
        if len(current_D[v]) >= 3:
            for combo in itertools.combinations(current_D[v], 3):
                S = (u,) + combo
                gained_sets.append(S)
                
        # Calculate delta bad_count
        delta_bad = 0
        
        # Process lost sets
        for S in lost_sets:
            if S in cover_counts:
                old_count = cover_counts[S]
                if old_count == 1:
                    # Becomes 0
                    delta_bad += 1
                    del cover_counts[S]
                else:
                    cover_counts[S] = old_count - 1
            # If S not in cover_counts, it was already 0 (bad). Removing a cover?
            # Wait, if S is in lost_sets, it WAS covered by u.
            # So cover_counts[S] must have been >= 1.
            # So S must be in cover_counts.
            
        # Process gained sets
        for S in gained_sets:
            if S in cover_counts:
                old_count = cover_counts[S]
                cover_counts[S] = old_count + 1
                # If it was bad (count 0), it's not in cover_counts.
                # So if it's in cover_counts, it was good, and remains good.
                # No change to bad_count.
            else:
                # Was bad (count 0). Now covered by v.
                cover_counts[S] = 1
                delta_bad -= 1
                
        # If delta_bad < 0, we improved.
        # If delta_bad == 0, we are neutral.
        # If delta_bad > 0, we worsened.
        
        # Accept if delta_bad <= 0?
        # To avoid getting stuck, maybe accept small worsenings?
        # But we want to find a valid one. Greedy is safer.
        
        if delta_bad <= 0:
            # Accept flip
            # Update adj
            current_adj[u, v] = 0
            current_adj[v, u] = 1
            
            # Update D
            current_D[u].remove(v)
            current_D[v].add(u)
            
            # Update orientation list
            idx = pair_to_idx[(u, v)]
            # u -> v was 1 (bit 1 means j->i? No.)
            # orientation bit: 0 means i -> j, 1 means j -> i.
            # If u < v:
            #   u -> v means bit 0.
            #   v -> u means bit 1.
            # We are flipping u -> v to v -> u.
            # So bit changes from 0 to 1.
            # If v < u:
            #   u -> v means v < u is false.
            #   pair is (v, u).
            #   u -> v means v -> u is false?
            #   Let's stick to the definition.
            #   Pair (i, j) with i < j.
            #   Bit 0: i -> j.
            #   Bit 1: j -> i.
            
            if u < v:
                # Pair is (u, v).
                # Old: u -> v. Bit was 0.
                # New: v -> u. Bit becomes 1.
                current_orientation[idx] = 1
            else:
                # Pair is (v, u).
                # Old: u -> v. This means v < u is false? No, v < u is true.
                # Wait, if u > v, then pair is (v, u).
                # u -> v means the larger index u points to smaller v.
                # Definition: Bit 0 means i -> j (smaller to larger).
                # Bit 1 means j -> i (larger to smaller).
                # So if u > v, and u -> v, then it is "larger to smaller", so bit was 1.
                # New: v -> u. This is "smaller to larger", so bit becomes 0.
                current_orientation[idx] = 0
                
            if len(cover_counts) == total_4sets:
                return current_orientation
                
    return None

def _paleyn_tournament(p):
    """Paley tournament on p vertices: i -> j iff (j-i) mod p is a nonzero quadratic residue."""
    squares = set()
    for x in range(1, p):
        squares.add(x * x % p)
    
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                orientation.append(0)  # i -> j
            else:
                orientation.append(1)  # j -> i
    return p, orientation

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Strategy:
    # 1. Try local search for n from 31 to 66.
    # 2. If found, return.
    # 3. If not, return Paley 67 (or 83 if 67 fails verification, but 67 is known).
    
    # We know n=67 is valid.
    # Let's try to find smaller n.
    
    # Time budget per n:
    # We have ~110s.
    # If we try 36 values (31 to 66), we can spend ~3s each.
    # Local search on n=31 is very fast.
    
    for n in range(31, 67):
        if time.time() > deadline - 10:
            break
            
        # Allocate more time for smaller n?
        # Smaller n is more likely to be solvable if it is solvable.
        # But if 31 is impossible, we waste time.
        # Let's give 2 seconds to each n.
        max_time = 2.0
        
        result = _local_search(n, max_time=max_time)
        if result is not None:
            return {'n': n, 'orientation': result}
            
    # Fallback to known valid n=67
    # The current program found 67.
    # We should verify it or just trust the previous logic.
    # The previous program used Paley 83 and local search? No, it used Paley and checked.
    # It found 67.
    # Let's use the Paley 67 construction?
    # Paley requires p prime, p=3 mod 4.
    # 67 is prime. 67 % 4 = 3.
    # So Paley 67 is a valid tournament.
    # Is it valid for the 4-dominator property?
    # The previous program checked it and it was valid.
    
    n, orientation = _paleyn_tournament(67)
    
    return {'n': n, 'orientation': orientation}
# EVOLVE-BLOCK-END
