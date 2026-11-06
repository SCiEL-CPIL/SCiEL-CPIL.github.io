import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    limit = 55.0  # Safety margin
    
    N = 128
    # Elements are 1..128, index 0..127
    
    # 1. Precompute bad triples
    # A triple (a,b,c) is bad if lcm(a,b)==lcm(a,c)==lcm(b,c)
    bad_triples = []
    # To speed up, we can just iterate and check
    # 128 choose 3 is ~350k, lcm is fast
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            lab = a * b // __import__('math').gcd(a, b)
            for c in range(b+1, N+1):
                # Optimization: if lab > lcm(a,c) and lcm(a,c) > lab... no
                # Just compute
                lac = a * c // __import__('math').gcd(a, c)
                if lac != lab:
                    continue
                lbc = b * c // __import__('math').gcd(b, c)
                if lbc == lab:
                    bad_triples.append((a-1, b-1, c-1)) # 0-indexed
    
    # 2. Build data structures
    # neighbors[u]: list of (x, v) such that (u, x, v) is a bad triple
    # Note: The triple is unordered in the property, but we stored (a,b,c) with a<b<c.
    # We need to handle all permutations for neighbor lists.
    neighbors = [[] for _ in range(N)]
    # pair_mask[v][w]: bitmask of y such that (v, w, y) is a bad triple
    pair_mask = [[0]*N for _ in range(N)]
    
    for a, b, c in bad_triples:
        # Add to neighbors for a, b, c
        # For a: neighbors with (b, c), (c, b)
        neighbors[a].append((b, c))
        neighbors[a].append((c, b))
        
        neighbors[b].append((a, c))
        neighbors[b].append((c, a))
        
        neighbors[c].append((a, b))
        neighbors[c].append((b, a))
        
        # Update pair masks
        # For pair (a,b), c is a witness
        pair_mask[a][b] |= (1 << c)
        pair_mask[b][a] |= (1 << c)
        
        # For pair (a,c), b is a witness
        pair_mask[a][c] |= (1 << b)
        pair_mask[c][a] |= (1 << b)
        
        # For pair (b,c), a is a witness
        pair_mask[b][c] |= (1 << a)
        pair_mask[c][b] |= (1 << a)

    # 3. Greedy Hitting Set to get initial valid set
    # Start with all elements
    S_mask = (1 << N) - 1
    R_mask = 0
    
    # We need to remove elements until no bad triples remain.
    # We can maintain a list of active bad triples.
    # Or simpler: Just iterate, find a bad triple in S, remove one.
    # To be efficient, let's use the block_count idea?
    # Actually, for initialization, a simple loop is fine if N is small.
    
    # Let's count how many bad triples are fully in S.
    # If > 0, we have a conflict.
    
    # Greedy:
    # 1. Count for each element how many bad triples it is part of (within S).
    # 2. Remove the one with highest count.
    # 3. Update counts.
    
    # This is O(N * E) in worst case, but E is small enough.
    # Let's just do a simple removal loop.
    
    # To make it fast, we'll maintain a set of active bad triples.
    # But updating a set of 350k tuples is slow.
    
    # Alternative:
    # Just run the local search from the FULL set, but penalize conflicts?
    # No, we need a valid set.
    
    # Let's use the block_count method for initialization too?
    # Initialize block_count for all x in R (which is empty initially? No, S is full, R is empty).
    # Wait, if S is full, R is empty. We can't add anything.
    # We must remove.
    
    # Let's just remove elements greedily.
    # While True:
    #   Find any bad triple (a,b,c) where a,b,c in S.
    #   If none, break.
    #   Remove the element among a,b,c that appears in the most other bad triples in S.
    
    # To find "most other bad triples", we can precompute total degree in the hypergraph.
    # But it changes as S changes.
    # Approximation: Just remove the first one found? Or random one?
    # Random removal might be better to avoid bias.
    
    # Let's try:
    # 1. List all bad triples.
    # 2. Filter to those in S.
    # 3. If empty, done.
    # 4. Pick a random bad triple from the list.
    # 5. Pick a random element from the triple to remove.
    # 6. Update S, R.
    
    # This is O(E) per step. Number of steps is size of removed set.
    # Max removed ~ 50. So 50 * 350k = 17M ops. Fast.
    
    active_triples = [t for t in bad_triples if (S_mask >> t[0]) & 1 and (S_mask >> t[1]) & 1 and (S_mask >> t[2]) & 1]
    
    while active_triples:
        if time.time() - start_time > limit - 10: break
        # Pick a random active triple
        idx = random.randint(0, len(active_triples) - 1)
        a, b, c = active_triples[idx]
        # Remove one of them
        # Let's remove the one with highest "degree" in active_triples?
        # Or just random.
        rem = random.choice([a, b, c])
        
        # Remove rem from S
        S_mask &= ~(1 << rem)
        R_mask |= (1 << rem)
        
        # Update active_triples: remove all triples containing rem
        # This is O(E).
        new_active = []
        for t in active_triples:
            if rem not in t:
                new_active.append(t)
        active_triples = new_active
        
    # Now S is valid.
    
    # 4. Local Search
    # Maintain block_count for elements in R.
    # block_count[x] = number of pairs (y,z) in S such that (x,y,z) is bad.
    
    block_count = [0] * N
    
    # Initialize block_count
    for x in range(N):
        if (R_mask >> x) & 1:
            # Count pairs in S
            # Iterate neighbors[x]
            # neighbors[x] contains (y, z) for each bad triple (x,y,z)
            # We need to count how many (y,z) have y,z in S
            # Note: neighbors[x] has duplicates?
            # In our construction, for a triple (a,b,c), we added (b,c) and (c,b) to neighbors[a].
            # So each pair is listed twice?
            # Wait, if (a,b,c) is a triple, the pairs are (b,c).
            # In the loop:
            # neighbors[a].append((b,c))
            # neighbors[a].append((c,b))
            # This implies that if we just check "is y in S and z in S", we will count the same pair twice if we iterate through the list.
            # We should use a set of pairs or just be careful.
            # Actually, for a specific x, the set of "witness pairs" is unique per triple.
            # If there are multiple triples (x, y, z1) and (x, y, z2), the pairs are distinct.
            # If there is only one triple (x,y,z), we added (y,z) and (z,y).
            # So we will count 2 for the same pair.
            # We just need to know if count > 0.
            # So we can just check if ANY pair is in S.
            pass
            
    # Better initialization:
    for x in range(N):
        if (R_mask >> x) & 1:
            count = 0
            for y, z in neighbors[x]:
                if ((S_mask >> y) & 1) and ((S_mask >> z) & 1):
                    count += 1
            block_count[x] = count
            
    # Local Search Loop
    best_size = bin(S_mask).count('1')
    best_S = S_mask
    
    while time.time() - start_time < limit:
        # Try Add
        # Pick random x in R
        r_bits = [i for i in range(N) if (R_mask >> i) & 1]
        if r_bits:
            x = random.choice(r_bits)
            if block_count[x] == 0:
                # Add x
                S_mask |= (1 << x)
                R_mask &= ~(1 << x)
                # Update block_count for other elements in R
                # For each neighbor (y, z) of x:
                # If y in S and z in R (or vice versa), it might affect block_count of z (if z in R)
                # Wait, block_count[z] counts pairs in S.
                # Adding x to S increases the number of pairs in S.
                # Specifically, for any z in R, if there is a bad triple (z, x, w) with w in S (w != x),
                # then (x, w) is a new pair in S that blocks z.
                # So for each (y, w) in neighbors[x]:
                #   If y in S (and y != x) and w in R:
                #       block_count[w] += 1
                #   If w in S (and w != x) and y in R:
                #       block_count[y] += 1
                for y, w in neighbors[x]:
                    if y == x or w == x: continue
                    y_in_S = (S_mask >> y) & 1
                    w_in_S = (S_mask >> w) & 1
                    y_in_R = (R_mask >> y) & 1
                    w_in_R = (R_mask >> w) & 1
                    if y_in_S and w_in_R:
                        block_count[w] += 1
                    elif w_in_S and y_in_R:
                        block_count[y] += 1
                
                if bin(S_mask).count('1') > best_size:
                    best_size = bin(S_mask).count('1')
                    best_S = S_mask
        else:
            break # R is empty, max size reached
            
        # Try Expand (Remove 1, Add 2)
        # Pick random u in S
        s_bits = [i for i in range(N) if (S_mask >> i) & 1]
        if s_bits and len(r_bits) >= 2:
            u = random.choice(s_bits)
            # Pick two random from R
            # We need to check if they can be added
            # To avoid O(N^2) sampling, we can just pick random
            v, w = random.sample(r_bits, 2)
            
            # Check if (S \ {u}) U {v, w} is valid
            # 1. Simulate remove u
            #    Update block_count for R elements
            #    This is expensive to do and undo.
            #    Instead, let's just check validity directly for this specific move.
            
            S_new_mask = S_mask & ~(1 << u)
            
            # Check if v can be added to S_new
            # v is in R, so block_count[v] is w.r.t S.
            # We need block_count[v] w.r.t S_new.
            # block_count[v]_new = block_count[v] - (1 if (v, u, y) is bad for some y in S_new else 0 ... no)
            # Actually, block_count[v] counts pairs (y,z) in S.
            # Removing u removes pairs (u, z) for z in S.
            # So block_count[v]_new = block_count[v] - (number of z in S such that (v, u, z) is bad)
            
            # Let's compute this delta.
            # For each (y, z) in neighbors[v]:
            #   If y == u and z in S: delta += 1
            #   If z == u and y in S: delta += 1
            
            delta_v = 0
            for y, z in neighbors[v]:
                if y == u:
                    if (S_mask >> z) & 1: delta_v += 1
                elif z == u:
                    if (S_mask >> y) & 1: delta_v += 1
            bc_v_new = block_count[v] - delta_v
            
            # Similarly for w
            delta_w = 0
            for y, z in neighbors[w]:
                if y == u:
                    if (S_mask >> z) & 1: delta_w += 1
                elif z == u:
                    if (S_mask >> y) & 1: delta_w += 1
            bc_w_new = block_count[w] - delta_w
            
            if bc_v_new == 0 and bc_w_new == 0:
                # Check if v and w are compatible with each other in S_new
                # i.e. no y in S_new such that (v, w, y) is bad
                # pair_mask[v][w] & S_new_mask == 0
                if (pair_mask[v][w] & S_new_mask) == 0:
                    # Valid move!
                    # Commit
                    S_mask = S_new_mask | (1 << v) | (1 << w)
                    R_mask &= ~(1 << v) & ~(1 << w)
                    # Update block_count for all R elements
                    # This is O(N * deg).
                    # We can just recompute block_count for all R elements?
                    # N=128, R is small. Recomputing is O(N * E_deg).
                    # Might be faster than incremental update if bugs are hard to find.
                    for x in range(N):
                        if (R_mask >> x) & 1:
                            count = 0
                            for y, z in neighbors[x]:
                                if ((S_mask >> y) & 1) and ((S_mask >> z) & 1):
                                    count += 1
                            block_count[x] = count
                            
                    if bin(S_mask).count('1') > best_size:
                        best_size = bin(S_mask).count('1')
                        best_S = S_mask

    # Convert best_S to list
    result = [i+1 for i in range(N) if (best_S >> i) & 1]
    if not result:
        result = [1]
    return result
