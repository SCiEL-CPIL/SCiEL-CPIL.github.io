import time
import random
import math
import numpy as np

N = 64
TOTAL = N * N

# Precompute squared distances using numpy for speed
xs = np.repeat(np.arange(N), N)
ys = np.tile(np.arange(N), N)
dx = xs[:, None] - xs[None, :]
dy = ys[:, None] - ys[None, :]
dist2 = (dx * dx + dy * dy).astype(np.int32)

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    best_set = []
    best_size = 0
    
    # Helper to convert index to (x, y)
    def idx_to_xy(i):
        return (i // N, i % N)
        
    # Helper to check if a set of indices is valid
    # This is O(K^2) but K is small (<100), so it's fast enough for verification
    def is_valid(si):
        k = len(si)
        if k < 3:
            return True
        # Check all triples? No, that's O(K^3).
        # Better: Check for each pair (i,j), if dist(i,j) == dist(i,k) or dist(j,k) for any k.
        # Actually, the condition is: no three points form an isosceles triangle.
        # This means for any point p, all distances from p to other points in the set must be distinct.
        # AND for any two points p, q, the distance d(p,q) must be unique in the sense that
        # there is no other point r such that d(p,r) == d(p,q) or d(q,r) == d(p,q).
        # Wait, the condition "no three points forming an isosceles triangle" is equivalent to:
        # For any distinct i, j, k in the set, d(i,j) != d(i,k) and d(i,j) != d(j,k) and d(i,k) != d(j,k).
        # This is equivalent to: For every point p in the set, the set of distances from p to all other points in the set contains no duplicates.
        # Let's verify:
        # If d(a,b) == d(a,c) for b!=c, then triangle abc is isosceles.
        # If d(a,b) == d(b,c) for a!=c, then triangle abc is isosceles.
        # So yes, the condition is simply: For each point p in S, all distances d(p, q) for q in S\{p} are distinct.
        
        for i in range(k):
            pi = si[i]
            dists = []
            for j in range(k):
                if i == j:
                    continue
                pj = si[j]
                dists.append(dist2[pi, pj])
            if len(set(dists)) != len(dists):
                return False
        return True

    # Local Search: Remove-and-Refill
    # State: list of indices `si`
    # We maintain `ud` (used distances) for fast checking during the refill phase.
    
    def local_search(si, ud, time_limit):
        """
        si: list of indices
        ud: list of sets, ud[i] contains distances from si[i] to all other points in si
        Returns improved si, ud
        """
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            k = len(si)
            if k == 0:
                break
                
            # Try removing each point
            for i in range(k):
                if time.time() > time_limit:
                    return si, ud
                    
                # Remove si[i]
                removed_idx = si[i]
                new_si = si[:i] + si[i+1:]
                
                # Rebuild ud for new_si? 
                # It's expensive to rebuild fully. 
                # Instead, we can update ud by removing the distances to removed_idx.
                new_ud = []
                for j in range(len(new_si)):
                    # new_ud[j] corresponds to new_si[j]
                    # We need to remove the distance to removed_idx from the set
                    # But we don't have the original ud mapping easily if we just slice.
                    # Let's just rebuild ud from scratch for simplicity, since K is small.
                    # O(K^2) rebuild is ~100*100 = 10000 ops. Very fast.
                    pass
                
                # Rebuild ud for new_si
                new_k = len(new_si)
                new_ud = [set() for _ in range(new_k)]
                for a in range(new_k):
                    for b in range(a+1, new_k):
                        d = dist2[new_si[a], new_si[b]]
                        new_ud[a].add(d)
                        new_ud[b].add(d)
                        
                # Now try to add points not in new_si
                # Candidate points: all points not in new_si
                # Sort candidates by "potential" or just random?
                # Let's try all candidates, pick the one that allows the most additions?
                # Or just try to add as many as possible in a greedy way.
                
                added_count = 0
                # Create a set of indices in new_si for fast lookup
                new_si_set = set(new_si)
                candidates = [p for p in range(TOTAL) if p not in new_si_set]
                
                # To make it deterministic and fast, we can iterate candidates in a fixed order 
                # or shuffle. Shuffling is better for escaping local minima.
                random.shuffle(candidates)
                
                # Greedy refill
                for p in candidates:
                    # Check if p can be added to new_si
                    # 1. Distances from p to all in new_si must be distinct
                    # 2. For each q in new_si, dist(p,q) not in new_ud[q_idx]
                    
                    valid = True
                    dists_to_new = []
                    for j, q in enumerate(new_si):
                        d = dist2[p, q]
                        dists_to_new.append(d)
                        if d in new_ud[j]:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    if len(set(dists_to_new)) != len(dists_to_new):
                        continue
                    
                    # If valid, add it
                    new_si.append(p)
                    new_ud.append(set(dists_to_new))
                    for j in range(len(new_si) - 1):
                        new_ud[j].add(dists_to_new[j])
                    added_count += 1
                    
                    if time.time() > time_limit:
                        break
                        
                # If we added more points than we removed (1), it's an improvement
                if len(new_si) > k:
                    si = new_si
                    ud = new_ud
                    improved = True
                    break # Restart the removal loop with the new state
        
        return si, ud

    # Initial deterministic set: Diagonal
    # Points (i, i) for i in range(N)
    # Distances from (i,i) to (j,j) are (j-i)^2 + (j-i)^2 = 2(j-i)^2.
    # Are these distinct? Yes, for fixed i, j varies, so 2(j-i)^2 varies.
    # Wait, if j1 != j2, can 2(j1-i)^2 == 2(j2-i)^2?
    # (j1-i)^2 == (j2-i)^2 => j1-i = j2-i or j1-i = -(j2-i) => j1=j2 or j1+j2=2i.
    # So if j1 != j2 but j1+j2=2i, the distance is the same.
    # Example: i=0, j1=1, j2=-1 (not in grid). i=10, j1=9, j2=11.
    # dist(10,10) to (9,9) is 2(1)^2 = 2.
    # dist(10,10) to (11,11) is 2(1)^2 = 2.
    # So diagonal is NOT valid.
    
    # Let's try a modular lattice: (i, 2*i % N)
    # Or just random start is safer.
    
    # We will run multiple random restarts, but use local search on each.
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Random initial set
        perm = list(range(TOTAL))
        random.shuffle(perm)
        
        si = []
        ud = []
        
        # Greedy fill
        for p_idx in perm:
            if not si:
                si.append(p_idx)
                ud.append(set())
                continue
            
            valid = True
            dists = []
            for q_idx in si:
                d = dist2[p_idx, q_idx]
                dists.append(d)
            
            if len(set(dists)) != len(dists):
                continue
            
            for i in range(len(si)):
                if dists[i] in ud[i]:
                    valid = False
                    break
            
            if valid:
                si.append(p_idx)
                new_ud = set(dists)
                ud.append(new_ud)
                
                for i in range(len(si) - 1):
                    ud[i].add(dists[i])
        
        # Run local search
        time_limit = deadline - 5 # Leave 5s for final checks/return
        si, ud = local_search(si, ud, time_limit)
        
        if len(si) > best_size:
            best_size = len(si)
            best_set = [(idx // N, idx % N) for idx in si]
            
            # If we found a very good set, we can try to polish it further
            # but local search already did that.
            
    return best_set
