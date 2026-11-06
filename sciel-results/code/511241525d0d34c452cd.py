# EVOLVE-BLOCK-START
import numpy as np

def _is_valid_snake(path):
    """Check if path is a valid induced path in Q_N."""
    n = len(path)
    if n == 0:
        return True
    used = set(path)
    if len(used) != n:
        return False
    for i in range(n - 1):
        diff = path[i] ^ path[i + 1]
        if diff & (diff - 1) != 0:
            return False
    for i in range(n):
        for j in range(i + 2, n):
            if (path[i] ^ path[j]).bit_count() == 1:
                return False
    return True

def _beam_search_construct(deadline, beam_width=200, seed=42):
    """Beam search to construct a long snake in Q_9 using incremental Bad set."""
    rng = random.Random(seed)
    
    # Precompute neighbors for all vertices as lists for faster iteration
    neighbors = [[] for _ in range(TOTAL_VERTICES)]
    for v in range(TOTAL_VERTICES):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
            
    # Beam states: (path_list, last_vertex, Bad_set)
    # Bad_set contains all vertices that are in path OR adjacent to any vertex in path[:-1]
    # Actually, for the current last vertex, Bad contains {u | u in path OR dist(u, path[i])=1 for i < len(path)-1}
    # Valid nexts are neighbors(last) not in Bad.
    
    beam = []
    # Initialize with several random starts
    for _ in range(beam_width):
        sv = rng.randint(0, TOTAL_VERTICES - 1)
        # Bad for start: {sv} U neighbors(sv)
        bad = {sv}
        for nb in neighbors[sv]:
            bad.add(nb)
        beam.append(([sv], sv, bad))
    
    best_path = []
    best_len = 0
    
    while time.time() < deadline:
        if time.time() > deadline - 0.5:
            break
        
        new_beam = []
        
        for path, last, bad in beam:
            if time.time() > deadline - 0.5:
                break
            
            # Find valid nexts
            valid_nexts = []
            for nb in neighbors[last]:
                if nb not in bad:
                    valid_nexts.append(nb)
            
            if not valid_nexts:
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path[:]
                continue
            
            # Score candidates
            # Heuristic: number of valid nexts for the candidate
            scored = []
            for v in valid_nexts:
                # If we pick v, new_bad = bad U {v} U neighbors(last)
                # Valid nexts for v are neighbors(v) not in new_bad
                # Note: 'last' is in bad? No, bad contains neighbors of path[:-1].
                # Wait, let's re-verify the Bad set definition.
                # State: Path P=[v0...vk], last=vk.
                # Bad = {u | u in P OR dist(u, vi)=1 for i < k}.
                # Valid next v_{k+1} must be in N(vk) and NOT in Bad.
                # If we pick v_{k+1}, new path P' = P + [v_{k+1}]. New last = v_{k+1}.
                # New Bad' = {u | u in P' OR dist(u, vi)=1 for i <= k}.
                # Bad' = Bad U {v_{k+1}} U N(vk).
                
                # We can pre-compute N(last) set for speed? 
                # neighbors[last] is a list.
                # new_bad = bad | {v} | set(neighbors[last])
                # This is expensive if we do it for every candidate.
                
                # Optimization:
                # base_bad_for_next = bad | set(neighbors[last])
                # For each candidate v:
                #   candidate_bad = base_bad_for_next | {v}
                #   count = 0
                #   for nb2 in neighbors[v]:
                #       if nb2 not in candidate_bad:
                #           count += 1
                #   scored.append((count, v))
                
                # Since neighbors[last] is fixed for this state, we can compute the union once?
                # But we need to check membership for each v.
                # set(neighbors[last]) creation is O(N).
                
                pass
            
            # Let's optimize the scoring loop
            # Precompute the set of neighbors of last to add to bad
            last_neighbors_set = set(neighbors[last])
            base_new_bad = bad | last_neighbors_set
            
            for v in valid_nexts:
                # candidate_bad = base_new_bad | {v}
                # We don't want to create a new set every time if possible, but N is small (512 vertices max, degree 9).
                # Checking membership in a set is O(1).
                # Creating a set of size ~100 is fast.
                
                # Alternative: 
                # count = 0
                # for nb2 in neighbors[v]:
                #    if nb2 == v: continue # not in neighbors
                #    if nb2 in base_new_bad: continue
                #    if nb2 == v: continue
                #    # Actually v is not in base_new_bad?
                #    # v is in N(last). last is in P. 
                #    # bad contains P. So last is in bad?
                #    # Bad = {u | u in P OR ...}. Yes, last is in P, so last is in bad.
                #    # neighbors[last] are added to base_new_bad.
                #    # v is in neighbors[last]. So v is in base_new_bad?
                #    # If v is in base_new_bad, then v is "blocked" for the *next* step?
                #    # No, v is the vertex we are adding. 
                #    # The condition for the NEXT vertex w (neighbor of v) is:
                #    # w not in P' and dist(w, Pi) >= 2 for i <= k.
                #    # P' = P U {v}.
                #    # So w cannot be in P U {v}.
                #    # And w cannot be in N(Pi) for i <= k.
                #    # N(Pi) for i < k is contained in bad (specifically the "dist=1" part).
                #    # N(vk) is added to base_new_bad.
                #    # So base_new_bad = {u | u in P} U N(v0..vk-1) U N(vk).
                #    # Wait, bad = {u | u in P} U N(v0..vk-1).
                #    # So base_new_bad = bad U N(vk).
                #    # v is in N(vk). So v is in base_new_bad.
                #    # This means if we check `if nb2 in base_new_bad`, it correctly blocks v itself?
                #    # Yes, w cannot be v.
                #    # So we just need to count neighbors of v not in base_new_bad.
                
                count = 0
                for nb2 in neighbors[v]:
                    if nb2 not in base_new_bad:
                        count += 1
                scored.append((count, v))
            
            scored.sort(key=lambda x: -x[0])
            top = scored[:5] # Keep top 5
            
            for score, v in top:
                new_path = path + [v]
                # new_bad = base_new_bad | {v}
                # Since v is in base_new_bad (as v in N(last) and N(last) subset base_new_bad), 
                # new_bad is actually just base_new_bad.
                # Let's verify: 
                # base_new_bad = bad U N(last).
                # v is in N(last). So v is in base_new_bad.
                # So new_bad = base_new_bad.
                # This is very efficient!
                
                new_beam.append((new_path, v, base_new_bad))
        
        if not new_beam:
            break
        
        # Sort and prune beam
        # Sort by length descending, then by heuristic score of the last step?
        # We don't have the score stored in the state. 
        # We can re-evaluate or just sort by length and random tie-breaker.
        # Sorting by length is primary.
        new_beam.sort(key=lambda x: (-len(x[0]), x[1]))
        
        # If beam is too large, keep top beam_width
        if len(new_beam) > beam_width:
            # To keep diversity, we might want to sample or keep top by some other metric.
            # For now, top by length. If many have same length, we keep the first ones.
            # To improve diversity, we can shuffle those with same length.
            # Simple approach: just take first beam_width after sort.
            beam = new_beam[:beam_width]
        else:
            beam = new_beam
        
        # Update best
        for path, last, bad in beam:
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]
        
        # Check if all dead
        all_dead = True
        for path, last, bad in beam:
            for nb in neighbors[last]:
                if nb not in bad:
                    all_dead = False
                    break
            if not all_dead:
                break
        if all_dead:
            break
    
    return best_path

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s limit
    
    best_path = []
    best_len = 0
    
    # Multiple beam search restarts with different seeds
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
    
    for seed in seeds:
        if time.time() > deadline - 5:
            break
        
        path = _beam_search_construct(deadline, beam_width=200, seed=seed)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
    
    # Final validation
    if best_path and _is_valid_snake(best_path):
        return best_path
    
    # Fallback: simple greedy from vertex 0
    path, used = [0], {0}
    while True:
        cur, nxt = path[-1], None
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used or any((v ^ p).bit_count() == 1 for p in path[:-1]):
                continue
            nxt = v
            break
        if nxt is None:
            return path
        path.append(nxt)
        used.add(nxt)
# EVOLVE-BLOCK-END
