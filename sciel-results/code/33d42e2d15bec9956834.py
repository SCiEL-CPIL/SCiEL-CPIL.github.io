"""
Snake-in-the-Box solver for Q_9 using bit-parallel representation and splicing.
"""
import time
import random
import sys

N = 9
SIZE = 1 << N  # 512

# Precompute neighbors for each vertex
_NEIGHBORS = []
for v in range(SIZE):
    nbrs = [v ^ (1 << b) for b in range(N)]
    _NEIGHBORS.append(nbrs)

# Precompute adjacency masks: _ADJ_MASK[v] is a 512-bit integer where bit u is 1 iff u is adjacent to v.
_ADJ_MASK = [0] * SIZE
for v in range(SIZE):
    m = 0
    for u in _NEIGHBORS[v]:
        m |= (1 << u)
    _ADJ_MASK[v] = m

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 135.0  # Safety margin before 150s kill
    best_path = None
    best_len = 0
    
    def _validate_path(path):
        """Check if path is a valid induced path (snake)."""
        if not path:
            return True
        seen = set()
        for i, v in enumerate(path):
            if v in seen:
                return False
            seen.add(v)
            if i > 0:
                if (v ^ path[i-1]).bit_count() != 1:
                    return False
                # Check non-consecutive
                for j in range(i-1):
                    if (v ^ path[j]).bit_count() == 1:
                        return False
        return True

    def _greedy_extend(start_v, rng, time_limit):
        """
        Greedy extension using bit-parallel validity checks.
        Returns the path list.
        """
        path = [start_v]
        used_mask = 1 << start_v
        # forbidden_mask: bits set for all vertices that are adjacent to any vertex in path[:-1]
        # Actually, let's maintain a 'blocked' mask.
        # When we add a vertex v to the path, its neighbors become 'blocked' for future steps?
        # No. The condition is: new vertex u must not be adjacent to any vertex in path except the last one.
        # Let path = [v0, v1, ..., vk]. We want to add v_{k+1}.
        # v_{k+1} must not be adjacent to v0, v1, ..., v_{k-1}.
        # v_{k+1} MUST be adjacent to vk.
        
        # So, we maintain a mask 'forbidden' which is the union of neighbors of v0, ..., v_{k-1}.
        # When we add vk+1, the new forbidden set becomes forbidden | neighbors(vk).
        # Because vk becomes part of the "non-consecutive" set for the next step.
        
        forbidden = 0  # Initially no vertices in path[:-1], so nothing forbidden
        # Note: start_v is path[0]. For the next step, path[:-1] is empty, so forbidden is 0.
        
        while time.time() < time_limit:
            cur = path[-1]
            # Candidates are neighbors of cur.
            # A candidate u is valid if:
            # 1. u is not used (bit u in used_mask is 0)
            # 2. u is not in forbidden (bit u in forbidden is 0)
            
            # Get neighbors of cur as a bitmask
            nbrs_mask = _ADJ_MASK[cur]
            
            # Valid candidates: neighbors of cur, not used, not forbidden
            valid_mask = nbrs_mask & ~used_mask & ~forbidden
            
            if valid_mask == 0:
                break
            
            # Get list of valid candidates
            # We can extract bits from valid_mask
            candidates = []
            vm = valid_mask
            while vm:
                # Get least significant set bit
                lsb = vm & -vm
                idx = lsb.bit_length() - 1
                candidates.append(idx)
                vm ^= lsb
            
            if not candidates:
                break
            
            # Warnsdorff-like heuristic: choose candidate with fewest onward options
            # Onward options for candidate u:
            #   neighbors of u that are not used and not in (forbidden | neighbors(cur))
            #   Because after adding u, cur becomes part of the forbidden set for the NEXT step.
            #   So new_forbidden = forbidden | _ADJ_MASK[cur]
            #   And u is added to used.
            #   So for next step from u, candidates v must satisfy:
            #   v in neighbors(u), v not in used | (1<<u), v not in (forbidden | _ADJ_MASK[cur])
            
            new_forbidden = forbidden | _ADJ_MASK[cur]
            
            scored_candidates = []
            for u in candidates:
                # Count onward options from u
                # u is now used, so we check neighbors of u against (used_mask | (1<<u)) and new_forbidden
                # used_mask | (1<<u) is the new used mask
                new_used = used_mask | (1 << u)
                nbrs_u = _ADJ_MASK[u]
                # Valid next steps from u: neighbors of u, not in new_used, not in new_forbidden
                valid_next = nbrs_u & ~new_used & ~new_forbidden
                # Count bits
                count = valid_next.bit_count()
                scored_candidates.append((count, u))
            
            # Sort by onward degree (Warnsdorff)
            scored_candidates.sort(key=lambda x: x[0])
            
            # Add some randomness to avoid getting stuck
            if rng.random() < 0.3 and len(scored_candidates) > 1:
                k = min(3, len(scored_candidates))
                _, nxt = rng.choice(scored_candidates[:k])
            else:
                _, nxt = scored_candidates[0]
            
            # Commit
            path.append(nxt)
            used_mask |= (1 << nxt)
            forbidden = new_forbidden  # Update forbidden for next iteration
            
        return path

    def _splice(path1, path2):
        """
        Try to splice two paths together.
        Find an index i in path1 and j in path2 such that path1[i] is adjacent to path2[j],
        and the resulting path is valid.
        This is a simplified splice: just try to connect the end of path1 to the start of path2,
        or find any valid connection.
        For efficiency, we'll try to connect the end of path1 to any vertex in path2,
        or the start of path2 to any vertex in path1.
        Actually, a common technique is to find a pair (u, v) where u is in path1, v is in path2,
        u and v are adjacent, and u is not the last element of path1 (or v is not the last of path2),
        then reverse one path and concatenate.
        Let's try a simple version:
        1. Reverse path2.
        2. Check if path1[-1] is adjacent to path2_rev[0] (which is path2[-1]).
           If so, and the combined path is valid, return it.
        3. More generally, try to find any valid splice point.
        """
        # Reverse path2
        rev_path2 = path2[::-1]
        
        # Try to connect path1 end to rev_path2 start
        if path1[-1] != rev_path2[0]:
            if (path1[-1] ^ rev_path2[0]).bit_count() == 1:
                # Check if the combined path is valid
                combined = path1 + rev_path2[1:]
                # We need to check that no new adjacencies are created.
                # Specifically, for each i in path1, j in rev_path2[1:], dist(path1[i], rev_path2[j]) >= 2
                # This is expensive. Let's just validate the whole thing.
                if _validate_path(combined):
                    return combined
        
        # Try other splice points? This is expensive. Let's just return None for now
        # or try a few random splice points.
        return None

    # Main search loop
    rng = random.Random(42)
    attempts = 0
    
    while time.time() < deadline:
        seed = rng.randint(0, SIZE - 1)
        path = _greedy_extend(seed, rng, deadline)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
        
        attempts += 1
        
        # Occasionally try to splice with a previously found path?
        # For now, just do more restarts.
        
        if attempts > 10000:
            break
    
    if best_path is None:
        # Fallback
        return [0]
    
    return best_path
# EVOLVE-BLOCK-END
