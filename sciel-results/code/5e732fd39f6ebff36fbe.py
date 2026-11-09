"""Bidirectional Warnsdorff snake construction for Q_9."""
import time
import random
import numpy as np

N = 9
MAX_VERTS = 2 ** N

# EVOLVE-BLOCK-START

def _elapsed(start_time):
    return time.time() - start_time

def _deadline_hit(start_time, margin=10.0):
    return time.time() - start_time > 120.0 - margin

def _is_valid_extension(v, path, used_set):
    """Check if v can be added to the end of path (adjacent to last, non-adjacent to all others)."""
    if v in used_set:
        return False
    for p in path[:-1]:
        if (v ^ p).bit_count() == 1:
            return False
    return True

def _is_valid_extension_head(v, path, used_set):
    """Check if v can be prepended to path (adjacent to first, non-adjacent to all others)."""
    if v in used_set:
        return False
    for p in path[1:]:
        if (v ^ p).bit_count() == 1:
            return False
    return True

def _count_safe_options(cur, path, used_set, end='tail'):
    """Count how many valid next moves are available from cur at the given end."""
    count = 0
    for b in range(N):
        v = cur ^ (1 << b)
        if end == 'tail':
            if _is_valid_extension(v, path, used_set):
                count += 1
        else:
            if _is_valid_extension_head(v, path, used_set):
                count += 1
    return count

def _bidirectional_grow(path, used_set, rng, max_iters=None):
    """
    Bidirectional growth with Warnsdorff's rule.
    At each step, consider both ends of the path. Choose the end with fewer
    safe options (more constrained), then pick the neighbor with fewest
    subsequent options (Warnsdorff). This avoids the tail-trap of forward greedy.
    """
    if max_iters is None:
        max_iters = 1000
    
    for _ in range(max_iters):
        if _deadline_hit(_bidirectional_grow._start_time):
            break
        
        # Evaluate both ends
        tail_options = []
        for b in range(N):
            v = path[-1] ^ (1 << b)
            if _is_valid_extension(v, path, used_set):
                # Count subsequent options for this candidate
                sub = 0
                for b2 in range(N):
                    v2 = v ^ (1 << b2)
                    if v2 == path[-2] if len(path) >= 2 else False:
                        pass
                    if v2 in used_set:
                        continue
                    valid = True
                    for p in path[:-1]:
                        if (v2 ^ p).bit_count() == 1:
                            valid = False
                            break
                    if valid:
                        sub += 1
                tail_options.append((sub, b, v))
        
        head_options = []
        for b in range(N):
            v = path[0] ^ (1 << b)
            if _is_valid_extension_head(v, path, used_set):
                sub = 0
                for b2 in range(N):
                    v2 = v ^ (1 << b2)
                    if v2 == path[1] if len(path) >= 2 else False:
                        pass
                    if v2 in used_set:
                        continue
                    valid = True
                    for p in path[1:]:
                        if (v2 ^ p).bit_count() == 1:
                            valid = False
                            break
                    if valid:
                        sub += 1
                head_options.append((sub, b, v))
        
        # Choose the more constrained end
        if not tail_options and not head_options:
            break
        
        if not tail_options:
            choose_head = True
        elif not head_options:
            choose_head = False
        else:
            # Pick the end with fewer options (more constrained)
            tail_min = min(o[0] for o in tail_options)
            head_min = min(o[0] for o in head_options)
            if head_min < tail_min:
                choose_head = True
            elif tail_min < head_min:
                choose_head = False
            else:
                choose_head = rng.random() < 0.5
        
        if choose_head:
            # Warnsdorff: pick the candidate with fewest subsequent options
            best = min(head_options, key=lambda x: x[0])
            # Add small random tie-breaking
            candidates = [o for o in head_options if o[0] == best[0]]
            chosen = rng.choice(candidates)
            v = chosen[2]
            path.insert(0, v)
            used_set.add(v)
        else:
            best = min(tail_options, key=lambda x: x[0])
            candidates = [o for o in tail_options if o[0] == best[0]]
            chosen = rng.choice(candidates)
            v = chosen[2]
            path.append(v)
            used_set.add(v)
    
    return path

def _validate(path):
    """Verify the path is a valid induced path in Q_N."""
    used = set(path)
    if len(used) != len(path):
        return False
    for i in range(len(path) - 1):
        if (path[i] ^ path[i+1]).bit_count() != 1:
            return False
    for i in range(len(path)):
        for j in range(i + 2, len(path)):
            if (path[i] ^ path[j]).bit_count() == 1:
                return False
    return True

def construct():
    """
    Construct the longest induced path (snake) in Q_9 within the time budget.
    Uses bidirectional growth with Warnsdorff's rule, multiple restarts,
    and local improvement via segment re-routing.
    """
    start_time = time.time()
    _bidirectional_grow._start_time = start_time
    
    best_path = None
    best_len = 0
    
    def _try_one(start_v, rng, seed_variant=0):
        """Run one bidirectional growth attempt from start_v."""
        path = [start_v]
        used = {start_v}
        
        # Initial greedy seed: a few forward steps to give the path some structure
        for _ in range(3):
            candidates = []
            for b in range(N):
                v = path[-1] ^ (1 << b)
                if _is_valid_extension(v, path, used):
                    candidates.append(v)
            if not candidates:
                break
            v = rng.choice(candidates)
            path.append(v)
            used.add(v)
        
        # Now bidirectional growth
        path = _bidirectional_grow(path, used, rng)
        return path
    
    def _local_improve(path, rng, max_iters=200):
        """
        Local improvement: try to insert a vertex at a random gap,
        or swap segments. Focus on extending the path.
        """
        current = path[:]
        used = set(current)
        
        for _ in range(max_iters):
            if _deadline_hit(start_time):
                break
            
            # Strategy 1: Try to insert a vertex at a random internal gap
            if len(current) < 2:
                break
            
            pos = rng.randint(1, len(current) - 1)
            a, c = current[pos - 1], current[pos]
            
            # Find vertices v such that v adjacent to a, v adjacent to c,
            # v not in used, and v non-adjacent to all other path vertices
            candidates = []
            for b in range(N):
                v = a ^ (1 << b)
                if v == c or v in used:
                    continue
                if (v ^ c).bit_count() != 1:
                    continue
                valid = True
                for p in current:
                    if p == a or p == c:
                        continue
                    if (v ^ p).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    candidates.append(v)
            
            if candidates:
                v = rng.choice(candidates)
                new_path = current[:pos] + [v] + current[pos:]
                # Quick validation
                ok = True
                for i in range(len(new_path) - 1):
                    if (new_path[i] ^ new_path[i+1]).bit_count() != 1:
                        ok = False
                        break
                if ok:
                    new_used = set(new_path)
                    for i in range(len(new_path)):
                        for j in range(i + 2, len(new_path)):
                            if (new_path[i] ^ new_path[j]).bit_count() == 1:
                                ok = False
                                break
                        if not ok:
                            break
                if ok:
                    current = new_path
                    used = new_path_set = set(new_path)
            
            # Strategy 2: Try to extend at either end
            end = rng.choice(['head', 'tail'])
            if end == 'tail':
                cur = current[-1]
                cands = []
                for b in range(N):
                    v = cur ^ (1 << b)
                    if _is_valid_extension(v, current, used):
                        cands.append(v)
            else:
                cur = current[0]
                cands = []
                for b in range(N):
                    v = cur ^ (1 << b)
                    if _is_valid_extension_head(v, current, used):
                        cands.append(v)
            
            if cands:
                v = rng.choice(cands)
                if end == 'tail':
                    new_path = current + [v]
                else:
                    new_path = [v] + current
                current = new_path
                used = set(current)
        
        return current
    
    # Phase 1: Multiple bidirectional growth restarts from vertex 0
    for attempt in range(200):
        if _deadline_hit(start_time):
            break
        
        rng = random.Random(attempt * 7919 + 13)
        path = _try_one(0, rng)
        
        if len(path) > best_len:
            best_path = path[:]
            best_len = len(path)
        
        # Periodically try local improvement on the best
        if attempt % 10 == 9 and best_path and not _deadline_hit(start_time):
            improved = _local_improve(best_path, random.Random(attempt * 31 + 7), max_iters=50)
            if len(improved) > best_len:
                best_path = improved[:]
                best_len = len(improved)
    
    # Phase 2: Try different starting vertices
    if not _deadline_hit(start_time):
        start_vertices = [0, 1, 2, 4, 8, 16, 32, 64, 128,
                          255, 511, 1023, 2047, 4095, 8191,
                          1234, 4321, 7777, 3333, 9999]
        for sv in start_vertices:
            if _deadline_hit(start_time):
                break
            rng = random.Random(sv * 104729 + 42)
            path = _try_one(sv, rng)
            if len(path) > best_len:
                best_path = path[:]
                best_len = len(path)
    
    # Phase 3: Final local improvement on the best path
    if best_path and not _deadline_hit(start_time):
        improved = _local_improve(best_path, random.Random(99999), max_iters=300)
        if len(improved) > best_len:
            best_path = improved[:]
            best_len = len(improved)
    
    # Phase 4: More restarts if time remains
    if not _deadline_hit(start_time):
        for attempt in range(500, 2000):
            if _deadline_hit(start_time):
                break
            rng = random.Random(attempt * 104729 + 7)
            sv = rng.randint(0, MAX_VERTS - 1)
            path = _try_one(sv, rng)
            if len(path) > best_len:
                best_path = path[:]
                best_len = len(path)
    
    if best_path is None:
        # Fallback: simple forward greedy
        path, used = [0], {0}
        while True:
            cur = path[-1]
            nxt = None
            for b in range(N):
                v = cur ^ (1 << b)
                if _is_valid_extension(v, path, used):
                    nxt = v
                    break
            if nxt is None:
                break
            path.append(nxt)
            used.add(nxt)
        return path
    
    return best_path
# EVOLVE-BLOCK-END
