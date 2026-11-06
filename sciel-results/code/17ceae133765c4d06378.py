"""Snake-in-the-box Q9: beam search with bitmask conflict detection."""
import time
import random

N = 9
NUM_VERTS = 1 << N
SEED_LEN = 104  # greedy baseline
DEADLINE_MARGIN = 8.0  # seconds before 150s wall-clock limit


def _build_neighbors():
    """Precompute neighbors for all vertices."""
    neighbors = []
    for v in range(NUM_VERTS):
        nb = [v ^ (1 << b) for b in range(N)]
        neighbors.append(nb)
    return neighbors


def _bit_count(x):
    """Popcount."""
    return bin(x).count('1')


def _validate(path):
    """Validate that path is a valid induced path (snake)."""
    if not path:
        return False
    used = set(path)
    if len(used) != len(path):
        return False
    for i in range(len(path)):
        for j in range(i + 2, len(path)):
            if (path[i] ^ path[j]).bit_count() == 1:
                return False
    for i in range(len(path) - 1):
        if (path[i] ^ path[i + 1]).bit_count() != 1:
            return False
    return True


def _greedy_seed():
    """Generate the greedy seed snake."""
    path = [0]
    used = {0}
    while True:
        cur = path[-1]
        nxt = None
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            valid = True
            for p in path[:-1]:
                if (v ^ p).bit_count() == 1:
                    valid = False
                    break
            if valid:
                nxt = v
                break
        if nxt is None:
            break
        path.append(nxt)
        used.add(nxt)
    return path


def _beam_search(neighbors, time_limit, rng, beam_width=200, max_steps=400):
    """
    Beam search for longest induced path.
    
    Key optimization: maintain a 'forbidden' bitmask per beam entry.
    forbidden[v] = 1 means vertex v is at Hamming distance 1 from some path vertex,
    so it cannot be added as the next vertex.
    
    Actually, we maintain the set of all vertices that are at distance <= 1 from any
    path vertex. A candidate next vertex must NOT be in this set (except the current
    last vertex's neighbors, but those are the only candidates anyway).
    
    More precisely: forbidden_mask has bit v set if v is within Hamming distance 1
    of any vertex in the path. When we add vertex w to the path, we set bits for
    w and all its neighbors in forbidden_mask.
    
    A candidate next vertex v (neighbor of current last) is valid iff
    forbidden_mask has bit v cleared (i.e., v is not within distance 1 of any
    earlier path vertex) AND v is not the current last vertex.
    """
    deadline = time.time() + time_limit
    
    # Initialize beam with the greedy seed
    seed = _greedy_seed()
    if not seed:
        return [0]
    
    # Build forbidden mask for the seed
    # forbidden_mask: bit v set if v is within Hamming distance 1 of any path vertex
    # For a candidate next vertex v (which is a neighbor of path[-1]),
    # v is valid iff it's not in forbidden_mask (meaning it's not within dist 1 of path[0..-2])
    # AND v != path[-1] (which is guaranteed since v is a neighbor)
    
    def build_forbidden_mask(path):
        """Build bitmask: bit v set if v is within Hamming distance 1 of any path vertex."""
        mask = 0
        for p in path:
            # Set bit for p and all its neighbors
            mask |= (1 << p)
            for nb in neighbors[p]:
                mask |= (1 << nb)
        return mask
    
    # For the seed, compute forbidden mask
    # But we only need to check: is candidate v within dist 1 of path[0:-1]?
    # So forbidden should be: all vertices within dist 1 of path[0:-1]
    # Then candidate v (neighbor of path[-1]) is valid iff v not in forbidden AND v not in path
    
    def build_forbidden_excluding_last(path):
        """Bitmask of all vertices within Hamming distance 1 of path[0:-1]."""
        mask = 0
        for p in path[:-1]:
            mask |= (1 << p)
            for nb in neighbors[p]:
                mask |= (1 << nb)
        return mask
    
    # Each beam entry: (path_list, forbidden_mask, used_mask)
    # used_mask: bit v set if v is in the path
    # forbidden_mask: bit v set if v is within dist 1 of path[0:-1]
    # Candidate v (neighbor of path[-1]) is valid iff:
    #   - (forbidden_mask >> v) & 1 == 0  (not within dist 1 of earlier vertices)
    #   - (used_mask >> v) & 1 == 0       (not already in path)
    
    seed_forbidden = build_forbidden_excluding_last(seed)
    seed_used = 0
    for v in seed:
        seed_used |= (1 << v)
    
    # Beam: list of (path, forbidden_mask, used_mask)
    # Start with the seed
    beam = [(seed, seed_forbidden, seed_used)]
    
    best_path = seed
    best_len = len(seed)
    
    # Expand the beam
    while time.time() < deadline:
        new_beam = []
        
        for path, forbidden, used in beam:
            cur = path[-1]
            # Try each neighbor of cur
            for nb in neighbors[cur]:
                # Check if nb is valid
                if (forbidden >> nb) & 1:
                    continue  # nb is within dist 1 of some earlier vertex
                if (used >> nb) & 1:
                    continue  # nb already in path
                
                new_path = path + [nb]
                new_len = len(new_path)
                
                if new_len > best_len:
                    best_len = new_len
                    best_path = new_path
                
                # Update forbidden and used
                # New forbidden: old forbidden + (nb and its neighbors)
                # But wait: forbidden is "within dist 1 of path[0:-1]"
                # New path is path + [nb], so new forbidden = "within dist 1 of path[0:] (i.e., path + [nb] excluding last)"
                # = "within dist 1 of path[0:-1] + [nb]"
                # = old_forbidden | (nb and its neighbors)
                new_forbidden = forbidden
                new_forbidden |= (1 << nb)
                for n2 in neighbors[nb]:
                    new_forbidden |= (1 << n2)
                
                new_used = used | (1 << nb)
                
                new_beam.append((new_path, new_forbidden, new_used))
        
        if not new_beam:
            break
        
        # Keep top beam_width entries by length (and maybe some diversity)
        new_beam.sort(key=lambda x: len(x[0]), reverse=True)
        
        # Keep top beam_width, but also ensure some diversity
        # Simple approach: just keep top beam_width
        if len(new_beam) > beam_width:
            new_beam = new_beam[:beam_width]
        
        beam = new_beam
    
    return best_path


def _beam_search_v2(neighbors, time_limit, rng, beam_width=500, max_depth=200):
    """
    Alternative beam search: instead of storing full paths, store them as tuples
    for memory efficiency. Use a different strategy for beam pruning.
    """
    deadline = time.time() + time_limit
    
    seed = _greedy_seed()
    if not seed:
        return [0]
    
    best_path = seed
    best_len = len(seed)
    
    def build_forbidden_excluding_last(path):
        mask = 0
        for p in path[:-1]:
            mask |= (1 << p)
            for nb in neighbors[p]:
                mask |= (1 << nb)
        return mask
    
    seed_forbidden = build_forbidden_excluding_last(seed)
    seed_used = 0
    for v in seed:
        seed_used |= (1 << v)
    
    # Beam entries: (path_tuple, forbidden_mask, used_mask)
    beam = [(tuple(seed), seed_forbidden, seed_used)]
    
    while time.time() < deadline:
        new_beam = []
        
        for path, forbidden, used in beam:
            cur = path[-1]
            for nb in neighbors[cur]:
                if (forbidden >> nb) & 1:
                    continue
                if (used >> nb) & 1:
                    continue
                
                new_path = path + (nb,)
                new_len = len(new_path)
                
                if new_len > best_len:
                    best_len = new_len
                    best_path = list(new_path)
                
                new_forbidden = forbidden | (1 << nb)
                for n2 in neighbors[nb]:
                    new_forbidden |= (1 << n2)
                
                new_used = used | (1 << nb)
                
                new_beam.append((new_path, new_forbidden, new_used))
        
        if not new_beam:
            break
        
        # Sort by length descending, keep top beam_width
        new_beam.sort(key=lambda x: len(x[0]), reverse=True)
        
        if len(new_beam) > beam_width:
            new_beam = new_beam[:beam_width]
        
        beam = new_beam
    
    return best_path


def _random_walk_extend(path, neighbors, rng, max_steps=50):
    """Try to extend a path using randomized choices."""
    path = list(path)
    used = set(path)
    
    def build_forbidden_set(path):
        """Set of all vertices within Hamming distance 1 of path[0:-1]."""
        s = set()
        for p in path[:-1]:
            s.add(p)
            for nb in neighbors[p]:
                s.add(nb)
        return s
    
    forbidden = build_forbidden_set(path)
    
    for _ in range(max_steps):
        cur = path[-1]
        candidates = [nb for nb in neighbors[cur] if nb not in forbidden and nb not in used]
        if not candidates:
            break
        # Random choice
        nb = rng.choice(candidates)
        path.append(nb)
        used.add(nb)
        # Update forbidden
        forbidden.add(nb)
        for n2 in neighbors[nb]:
            forbidden.add(n2)
    
    return path


def _local_improve(path, neighbors, rng, time_limit):
    """Try to improve a path by removing a middle vertex and re-extending."""
    deadline = time.time() + time_limit
    best = list(path)
    best_len = len(best)
    
    for _ in range(1000):
        if time.time() > deadline:
            break
        
        p = list(best)
        if len(p) < 3:
            break
        
        # Pick a random position to remove (not first or last)
        if len(p) <= 3:
            continue
        idx = rng.randint(1, len(p) - 2)
        removed = p[idx]
        new_p = p[:idx] + p[idx+1:]
        
        # Check if new_p is still valid
        valid = True
        for i in range(len(new_p) - 1):
            if (new_p[i] ^ new_p[i+1]).bit_count() != 1:
                valid = False
                break
        if valid:
            for i in range(len(new_p)):
                for j in range(i+2, len(new_p)):
                    if (new_p[i] ^ new_p[j]).bit_count() == 1:
                        valid = False
                        break
                if not valid:
                    break
        
        if not valid:
            continue
        
        # Try to extend
        extended = _random_walk_extend(new_p, neighbors, rng, max_steps=50)
        if len(extended) > best_len:
            best = extended
            best_len = len(extended)
    
    return best


def construct():
    """
    Construct the longest induced path (snake) in Q9.
    
    Strategy:
    1. Generate greedy seed
    2. Beam search with bitmask conflict detection
    3. Local improvement attempts
    """
    start_time = time.time()
    deadline = start_time + (150.0 - DEADLINE_MARGIN)
    
    rng = random.Random(42)
    
    # Precompute neighbors
    neighbors = _build_neighbors()
    
    # Phase 1: Greedy seed
    seed = _greedy_seed()
    best_path = seed
    best_len = len(seed)
    
    # Phase 2: Beam search with multiple beam widths
    time_spent = time.time() - start_time
    remaining = deadline - time.time()
    
    if remaining > 5:
        # Try beam search with moderate width
        t1 = time.time()
        result = _beam_search_v2(neighbors, min(remaining * 0.5, 60), rng, beam_width=300)
        if len(result) > best_len:
            best_path = result
            best_len = len(result)
        time_spent = time.time() - start_time
        remaining = deadline - time.time()
    
    if remaining > 5:
        # Try wider beam
        result = _beam_search_v2(neighbors, min(remaining * 0.4, 40), rng, beam_width=500)
        if len(result) > best_len:
            best_path = result
            best_len = len(result)
        remaining = deadline - time.time()
    
    # Phase 3: Local improvement
    if remaining > 3:
        improved = _local_improve(best_path, neighbors, rng, min(remaining * 0.5, 20))
        if len(improved) > best_len:
            best_path = improved
            best_len = len(improved)
        remaining = deadline - time.time()
    
    # Phase 4: More beam search attempts with different random seeds
    if remaining > 5:
        for seed_val in [1, 2, 3, 4, 5]:
            if time.time() > deadline:
                break
            rng2 = random.Random(seed_val)
            result = _beam_search_v2(neighbors, min(remaining * 0.15, 15), rng2, beam_width=200)
            if len(result) > best_len:
                best_path = result
                best_len = len(result)
            remaining = deadline - time.time()
    
    # Final validation
    if not _validate(best_path):
        # Fallback to seed
        best_path = seed
    
    return best_path


# Quick self-test
if __name__ == "__main__":
    import time
    t0 = time.time()
    path = construct()
    t1 = time.time()
    print(f"Path length: {len(path)}")
    print(f"Edges: {len(path) - 1}")
    print(f"Time: {t1 - t0:.2f}s")
    print(f"Valid: {_validate(path)}")
