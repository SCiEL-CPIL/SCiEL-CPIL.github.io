<<<<<<< SEARCH
import time
import random
import numpy as np

N = 9
NUM_VERTS = 1 << N  # 512

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    best_path = []
    best_len = 0
    
    # Precompute neighbors for all vertices
    neighbors = [[v ^ (1 << b) for b in range(N)] for v in range(NUM_VERTS)]
    
    # Precompute bit_count for XOR differences (Hamming distance)
    # dist[a][b] = popcount(a ^ b)
    dist = [[(a ^ b).bit_count() for b in range(NUM_VERTS)] for a in range(NUM_VERTS)]
    
    def greedy_extend(start, use_random_tie_breaker=False):
        """
        Greedily extend a snake starting from `start`.
        Returns the path as a list of integers.
        """
        path = [start]
        used = {start}
        
        # blocked: vertices that cannot be visited next because they are adjacent to non-consecutive vertices
        # Specifically, if we are at path[-1], any vertex adjacent to path[-2] is blocked.
        # More generally, a vertex v is blocked if it is adjacent to any u in path where u is not path[-1] or path[-2].
        # Wait, the condition is: non-consecutive vertices must differ in at least 2 bits.
        # If we add v to path, v must differ from all path[:-2] by >= 2 bits.
        # So v is invalid if it is a neighbor of any u in path[:-2].
        
        # We can maintain a set of "forbidden" vertices.
        # Initially forbidden is empty.
        # When we move from path[-2] to path[-1], path[-2] becomes part of path[:-2] relative to the next step?
        # No.
        # Let's trace:
        # Path: [v0, v1]
        # Next vertex v2 must be neighbor of v1.
        # v2 must not be neighbor of v0.
        # So forbidden = neighbors(v0) \ {v1} (v1 is used, so it's excluded anyway, but logically it's the constraint).
        
        # Path: [v0, v1, v2]
        # Next vertex v3 must be neighbor of v2.
        # v3 must not be neighbor of v0 or v1.
        # So forbidden = neighbors(v0) U neighbors(v1) \ {v2, v1?}
        # Actually, v3 is neighbor of v2.
        # Constraint: dist(v3, v0) >= 2 and dist(v3, v1) >= 2.
        # So forbidden set should contain all neighbors of v0 and v1.
        
        # General rule:
        # When extending from current end `cur` (which is path[-1]),
        # the new vertex `nxt` must be a neighbor of `cur`.
        # `nxt` must not be a neighbor of any `u` in `path[:-2]`.
        # So we maintain a set `forbidden` which is the union of neighbors of all vertices in `path[:-2]`.
        
        # Update rule:
        # Start: path = [v0]. forbidden = empty set.
        # Step 1: Choose v1 from neighbors(v0).
        #   Constraint: v1 not in forbidden (empty).
        #   Update: path = [v0, v1].
        #   New forbidden: neighbors of path[:-2] = neighbors of [] = empty?
        #   Wait. For the NEXT step (choosing v2), the constraint is v2 not neighbor of v0.
        #   So after adding v1, we must add neighbors(v0) to forbidden?
        #   Let's check.
        #   If path is [v0, v1], and we want to add v2.
        #   v2 must be neighbor of v1.
        #   v2 must not be neighbor of v0.
        #   So forbidden should contain neighbors(v0).
        
        #   So, after adding `nxt` to path, making `cur` the new `path[-2]` (which is the old `path[-1]`),
        #   we should add `neighbors(cur)` to `forbidden`.
        
        # Let's verify:
        # path = [v0]. cur = v0. forbidden = {}.
        # Choose v1 from neighbors(v0) \ forbidden.
        # path = [v0, v1].
        # Add neighbors(v0) to forbidden. forbidden = neighbors(v0).
        # cur = v1.
        # Choose v2 from neighbors(v1) \ forbidden \ used.
        #   forbidden contains neighbors(v0). So v2 is not neighbor of v0. Correct.
        # path = [v0, v1, v2].
        # Add neighbors(v1) to forbidden. forbidden = neighbors(v0) U neighbors(v1).
        # cur = v2.
        # Choose v3 from neighbors(v2) \ forbidden \ used.
        #   forbidden contains neighbors(v0) and neighbors(v1). So v3 is not neighbor of v0 or v1. Correct.
        
        forbidden = set()
        
        while True:
            cur = path[-1]
            candidates = []
            
            for nxt in neighbors[cur]:
                if nxt in used:
                    continue
                if nxt in forbidden:
                    continue
                candidates.append(nxt)
                
            if not candidates:
                break
            
            # Score candidates
            # Heuristic: prefer candidates that have many free neighbors (degree in remaining graph)
            # Free neighbor of nxt: neighbor of nxt that is not in used and not in forbidden.
            # Note: When we move to nxt, `cur` will be added to the "path[:-2]" set for the next step.
            # So `neighbors(cur)` will be added to `forbidden`.
            # Thus, a neighbor `m` of `nxt` is "effectively free" for the next step if:
            # 1. m not in used
            # 2. m not in forbidden (current)
            # 3. m not in neighbors(cur) (since they will be added to forbidden)
            
            best_score = -1
            best_nxt = None
            best_rand = random.random() if use_random_tie_breaker else 0
            
            for nxt in candidates:
                score = 0
                for m in neighbors[nxt]:
                    if m in used:
                        continue
                    if m in forbidden:
                        continue
                    # Check if m is neighbor of cur
                    # m is neighbor of cur if m in neighbors[cur]
                    # We can precompute a set of neighbors[cur] for faster lookup?
                    # Or just check dist[nxt][m]==1 and dist[cur][m]==1? No, m is neighbor of nxt.
                    # We just need to know if m is neighbor of cur.
                    # (m ^ cur).bit_count() == 1
                    if (m ^ cur).bit_count() == 1:
                        continue
                    score += 1
                
                if score > best_score:
                    best_score = score
                    best_nxt = nxt
                    if use_random_tie_breaker:
                        best_rand = random.random()
                elif score == best_score:
                    if use_random_tie_breaker:
                        r = random.random()
                        if r > best_rand:
                            best_nxt = nxt
                            best_rand = r
                    # If not random tie break, keep first found (deterministic)
            
            if best_nxt is None:
                break
                
            path.append(best_nxt)
            used.add(best_nxt)
            
            # Update forbidden
            # The previous `cur` (which is now path[-2]) becomes part of the "forbidden history"
            for nb in neighbors[cur]:
                forbidden.add(nb)
                
            # Time check
            if time.time() > deadline:
                break
        
        return path

    # Strategy 1: Deterministic greedy from all starts
    for start in range(NUM_VERTS):
        if time.time() > deadline - 5:
            break
        p = greedy_extend(start, use_random_tie_breaker=False)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p[:]
            
    # Strategy 2: Randomized greedy restarts
    while time.time() < deadline - 2:
        start = random.randint(0, NUM_VERTS - 1)
        p = greedy_extend(start, use_random_tie_breaker=True)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p[:]
            
    return best_path
=======
import time
import random
import sys

N = 9
NUM_VERTS = 1 << N  # 512

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 150s kill

    best_path = []
    best_len = 0

    # Precompute neighbor lists
    neighbors = [[v ^ (1 << b) for b in range(N)] for v in range(NUM_VERTS)]
    
    # Precompute neighbor bitmask for each vertex
    # nbmask[v] has bit i set iff vertex i is a neighbor of vertex v
    nbmask = [0] * NUM_VERTS
    for v in range(NUM_VERTS):
        m = 0
        for nb in neighbors[v]:
            m |= (1 << nb)
        nbmask[v] = m

    def warnsdorff_dfs(path, used_mask, forbidden_mask, cur, depth):
        """
        Warnsdorff-style backtracking search.
        path: list of vertices in current snake
        used_mask: bitmask of used vertices
        forbidden_mask: bitmask of vertices that are neighbors of path[:-2] (cannot be next)
        cur: current end vertex (path[-1])
        depth: current depth
        Returns: (success, extended_path, extended_used, extended_forbidden)
        """
        if time.time() > deadline:
            return False, path, used_mask, forbidden_mask

        # Find candidates: neighbors of cur that are not used and not forbidden
        avail = nbmask[cur] & ~used_mask & ~forbidden_mask
        
        if avail == 0:
            # Dead end - return what we have
            return False, path, used_mask, forbidden_mask

        # Get list of candidate vertices
        candidates = []
        a = avail
        while a:
            lsb = a & (-a)
            v = lsb.bit_length() - 1
            candidates.append(v)
            a ^= lsb

        # Score each candidate using Warnsdorff's rule:
        # Prefer the candidate with fewest onward moves
        scored = []
        for nxt in candidates:
            # When we move to nxt, cur becomes part of forbidden history
            # New forbidden = forbidden_mask | nbmask[cur]
            new_forbidden = forbidden_mask | nbmask[cur]
            # Onward moves from nxt: neighbors of nxt that are not used and not in new_forbidden
            onward = nbmask[nxt] & ~(used_mask | (1 << nxt)) & ~new_forbidden
            score = bin(onward).count('1')
            scored.append((score, nxt))
        
        # Sort by score (fewest onward moves first) - Warnsdorff's rule
        scored.sort(key=lambda x: x[0])
        
        # Try candidates in order
        for score, nxt in scored:
            new_path = path + [nxt]
            new_used = used_mask | (1 << nxt)
            new_forbidden = forbidden_mask | nbmask[cur]
            
            # Check if this extends our best
            if len(new_path) > best_len:
                nonlocal best_len, best_path
                best_len = len(new_path)
                best_path = new_path[:]
            
            # Recurse
            success, ext_path, ext_used, ext_forb = warnsdorff_dfs(
                new_path, new_used, new_forbidden, nxt, depth + 1
            )
            if success:
                return True, ext_path, ext_used, ext_forb
            
            if time.time() > deadline:
                return False, path, used_mask, forbidden_mask

        return False, path, used_mask, forbidden_mask

    def greedy_extend(start, random_tie=False):
        """
        Fast greedy extension using bitmasks.
        """
        path = [start]
        used_mask = 1 << start
        forbidden_mask = 0
        cur = start

        while True:
            if time.time() > deadline:
                break

            avail = nbmask[cur] & ~used_mask & ~forbidden_mask
            if avail == 0:
                break

            # Get candidates
            candidates = []
            a = avail
            while a:
                lsb = a & (-a)
                v = lsb.bit_length() - 1
                candidates.append(v)
                a ^= lsb

            # Score candidates: prefer those with fewest onward moves (Warnsdorff)
            best_score = 999
            best_nxt = None
            best_rand = random.random() if random_tie else 0

            for nxt in candidates:
                # New forbidden after moving to nxt
                new_forb = forbidden_mask | nbmask[cur]
                # Onward from nxt
                onward = nbmask[nxt] & ~(used_mask | (1 << nxt)) & ~new_forb
                score = bin(onward).count('1')

                if score < best_score:
                    best_score = score
                    best_nxt = nxt
                    if random_tie:
                        best_rand = random.random()
                elif score == best_score and random_tie:
                    r = random.random()
                    if r > best_rand:
                        best_nxt = nxt
                        best_rand = r

            if best_nxt is None:
                break

            path.append(best_nxt)
            used_mask |= (1 << best_nxt)
            forbidden_mask |= nbmask[cur]
            cur = best_nxt

        return path

    # Strategy 1: Greedy from all 512 starts (fast with bitmasks)
    for start in range(NUM_VERTS):
        if time.time() > deadline - 10:
            break
        p = greedy_extend(start, random_tie=False)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p[:]

    # Strategy 2: Randomized greedy restarts
    while time.time() < deadline - 15:
        start = random.randint(0, NUM_VERTS - 1)
        p = greedy_extend(start, random_tie=True)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p[:]

    # Strategy 3: Warnsdorff backtracking from promising starts
    # Pick top few greedy results and try to extend them with backtracking
    if time.time() < deadline - 5:
        # Get top greedy starts
        greedy_results = []
        for start in range(NUM_VERTS):
            if time.time() > deadline - 20:
                break
            p = greedy_extend(start, random_tie=False)
            greedy_results.append((len(p), start, p))
        
        greedy_results.sort(key=lambda x: -x[0])
        
        # Try backtracking from the best few greedy endpoints
        tried = set()
        for glen, gstart, gpath in greedy_results[:20]:
            if time.time() > deadline - 3:
                break
            if gstart in tried:
                continue
            tried.add(gstart)
            
            if glen <= best_len:
                continue
            
            # Use the greedy path as initial state for backtracking
            used_mask = 0
            for v in gpath:
                used_mask |= (1 << v)
            cur = gpath[-1]
            # forbidden = neighbors of gpath[:-2]
            forbidden_mask = 0
            for v in gpath[:-2]:
                forbidden_mask |= nbmask[v]
            
            warnsdorff_dfs(gpath, used_mask, forbidden_mask, cur, len(gpath))
    
    return best_path
>>>>>>> REPLACE

