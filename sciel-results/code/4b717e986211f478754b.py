import random
import time
import itertools
import math
import numpy as np

N = 16
ALL_POINTS = list(itertools.product(range(N), repeat=2))
N_PTS = len(ALL_POINTS)

# Precompute squared distances
coords = np.array(ALL_POINTS)
diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
dist2 = np.sum(diff**2, axis=2)

# Precompute Iso[u][v] as a bitset (Python int) of points w such that {u, v, w} is isosceles.
# Iso[u][v] includes u and v? No, w is distinct from u, v usually, but isosceles triangle implies distinct vertices.
# However, if u==v, it's not a triangle. We only consider u != v.
# The condition for {u, v, w} to be isosceles:
# dist2[u][w] == dist2[v][w] OR dist2[u][v] == dist2[u][w] OR dist2[u][v] == dist2[v][w]
# Note: w must be distinct from u and v.

ISO_BITSETS = [[0] * N_PTS for _ in range(N_PTS)]

def _precompute_iso_bitsets():
    for u in range(N_PTS):
        for v in range(u + 1, N_PTS):
            mask = 0
            # Iterate over all w
            # Optimization: Vectorize check for all w at once?
            # dist2[u, :] == dist2[v, :]
            # dist2[u, v] == dist2[u, :]
            # dist2[u, v] == dist2[v, :]
            
            d_uv = dist2[u, v]
            
            # w such that d_uw == d_vw
            mask1 = np.where(dist2[u, :] == dist2[v, :])[0]
            # w such that d_uv == d_uw
            mask2 = np.where(dist2[u, :] == d_uv)[0]
            # w such that d_uv == d_vw
            mask3 = np.where(dist2[v, :] == d_uv)[0]
            
            candidates = np.union1d(np.union1d(mask1, mask2), mask3)
            
            # Exclude u and v themselves
            candidates = candidates[candidates != u]
            candidates = candidates[candidates != v]
            
            for w in candidates:
                mask |= (1 << w)
            
            ISO_BITSETS[u][v] = mask
            ISO_BITSETS[v][u] = mask

_start_time = time.time()
_precompute_iso_bitsets()
FULL_MASK = (1 << N_PTS) - 1

def _get_blocked(S_set):
    """
    Calculate the bitset of points blocked by pairs in S_set.
    Blocked points are those w such that there exist u, v in S_set with w in ISO_BITSETS[u][v].
    """
    blocked = 0
    S_list = list(S_set)
    for i in range(len(S_list)):
        u = S_list[i]
        for j in range(i + 1, len(S_list)):
            v = S_list[j]
            blocked |= ISO_BITSETS[u][v]
    return blocked

def _is_maximal(S_set):
    """
    Check if S_set is a maximal no-isosceles set.
    Condition: (S_set union Blocked(S_set)) == FULL_MASK
    """
    if not S_set:
        return False # Empty set is not maximal unless N_PTS=0
    
    blocked = _get_blocked(S_set)
    # Points in S_set are "covered" by being in the set.
    # Points in blocked are "covered" by being blocked.
    # All points must be covered.
    
    s_mask = 0
    for p in S_set:
        s_mask |= (1 << p)
    
    return (s_mask | blocked) == FULL_MASK

def _greedy_construct(deadline):
    """
    Greedy construction of a maximal no-isosceles set.
    Prioritizes points that block the most currently uncovered points.
    """
    S_set = set()
    covered_mask = 0 # Bitset of points in S_set or blocked by S_set
    
    while covered_mask != FULL_MASK:
        if time.time() > deadline:
            break
        
        best_p = -1
        best_gain = -1
        
        # Candidates: all points not in S_set
        # We can optimize by only considering points that are not yet covered?
        # If a point is already covered, adding it to S_set might still block new points.
        # But usually, we want to cover uncovered points.
        
        candidates = [p for p in range(N_PTS) if p not in S_set]
        random.shuffle(candidates) # Add randomness to break ties and explore different paths
        
        for p in candidates:
            # Calculate gain: how many NEW points does adding p cover?
            # New blocked points: union of ISO_BITSETS[p][u] for u in S_set
            # Plus p itself.
            
            new_blocked = 0
            for u in S_set:
                new_blocked |= ISO_BITSETS[p][u]
            
            # The point p itself is covered
            new_covered = new_blocked | (1 << p)
            
            # Gain is the number of bits in (new_covered & ~covered_mask)
            gain_mask = new_covered & ~covered_mask
            gain = bin(gain_mask).count('1')
            
            if gain > best_gain:
                best_gain = gain
                best_p = p
        
        if best_p == -1:
            break
            
        S_set.add(best_p)
        # Update covered_mask
        # We need to recompute blocked for the new S_set?
        # Or just add the new contributions.
        # new_blocked was calculated relative to old S_set.
        # So new covered = old covered | new_blocked | (1 << best_p)
        for u in S_set:
            if u != best_p:
                covered_mask |= ISO_BITSETS[best_p][u]
        covered_mask |= (1 << best_p)
        
    return S_set

def _minimize(S_set, deadline):
    """
    Try to remove points from S_set while maintaining maximality.
    """
    S_list = list(S_set)
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        random.shuffle(S_list)
        for p in S_list:
            if p not in S_set:
                continue
            new_S = set(S_set)
            new_S.remove(p)
            if _is_maximal(new_S):
                S_set = new_S
                changed = True
                S_list = list(S_set)
                break # Restart loop with new S_list
    return S_set

def _local_search_swap(S_set, deadline):
    """
    Perform swaps to try to find a smaller maximal set.
    This is a heuristic.
    """
    best_S = set(S_set)
    best_size = len(best_S)
    
    temp = 1.0
    cooling = 0.95
    
    while time.time() < deadline:
        if temp < 0.01:
            temp = 0.01
            
        p = random.choice(list(S_set))
        q = random.choice([x for x in range(N_PTS) if x not in S_set])
        
        new_S = set(S_set)
        new_S.remove(p)
        new_S.add(q)
        
        if _is_maximal(new_S):
            # Same size, but different composition
            # Accept with probability based on "potential" (e.g. how many removals are possible?)
            # For now, accept if it allows immediate removal?
            # Or just accept to explore.
            if random.random() < 0.5: # 50% chance to accept neutral move
                S_set = new_S
                # Try to minimize immediately
                S_set = _minimize(S_set, deadline)
                if len(S_set) < best_size:
                    best_size = len(S_set)
                    best_S = set(S_set)
        
        temp *= cooling
        
    return best_S

def construct():
    start_time = time.time()
    deadline = start_time + 290.0 # Safety margin
    
    best_set = None
    best_size = float('inf')
    
    num_restarts = 0
    
    while time.time() < deadline:
        num_restarts += 1
        
        # 1. Greedy Init
        S_init = _greedy_construct(deadline)
        
        # 2. Minimize (Remove redundant points)
        S_min = _minimize(S_init, deadline)
        
        # 3. Local Search (Swaps)
        S_opt = _local_search_swap(S_min, deadline)
        
        if len(S_opt) < best_size:
            best_size = len(S_opt)
            best_set = S_opt
            
        # If we found a very small set, we might want to stop early?
        # Theoretical minimum is likely around N/4 or similar.
        # Let's just keep searching until time runs out.
        
    if best_set is None:
        # Fallback
        best_set = _greedy_construct(deadline)
        best_set = _minimize(best_set, deadline)
        
    return [ALL_POINTS[i] for i in best_set]
