"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

_N = P['n']
_EDGES = P['edges']
_MINV = P['minimum_vertices']

# Build adjacency bitsets (full graph) for fast induced-degree computation.
_ADJ = [0] * _N
for _a, _b in _EDGES:
    _ADJ[_a] |= (1 << _b)
    _ADJ[_b] |= (1 << _a)


def _induced_degree(v, mask):
    # Degree of v within the vertex set represented by mask.
    return (_ADJ[v] & mask).bit_count()


def _score(mask):
    # Number of distinct degrees among selected vertices in the induced subgraph.
    degs = set()
    m = mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        degs.add(_induced_degree(v, mask))
        m ^= lsb
    return len(degs)


def _score_with_cache(mask, deg_cache):
    """Score with degree caching for faster repeated evaluations."""
    degs = set()
    m = mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        degs.add(_induced_degree(v, mask))
        m ^= lsb
    return len(degs)


def _mask_to_list(mask):
    result = []
    m = mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        result.append(int(v))
        m ^= lsb
    result.sort()
    return list(result)


def _exhaustive_search_k8(deadline):
    """Exhaustively search all 8-vertex subsets with early termination."""
    best_mask = None
    best_score = 0
    # Precompute all 8-combinations
    for combo in combinations(range(_N), 8):
        if time.time() > deadline:
            break
        mask = 0
        for v in combo:
            mask |= (1 << v)
        s = _score(mask)
        if s > best_score:
            best_score = s
            best_mask = mask
            if best_score == 8:  # Maximum possible for k=8
                return best_mask, best_score
    return best_mask, best_score


def _exhaustive_search_k9(deadline):
    """Exhaustively search all 9-vertex subsets."""
    best_mask = None
    best_score = 0
    for combo in combinations(range(_N), 9):
        if time.time() > deadline:
            break
        mask = 0
        for v in combo:
            mask |= (1 << v)
        s = _score(mask)
        if s > best_score:
            best_score = s
            best_mask = mask
            if best_score == 9:  # Maximum possible for k=9
                return best_mask, best_score
    return best_mask, best_score


def _exhaustive_search_k10(deadline):
    """Exhaustively search all 10-vertex subsets with pruning."""
    best_mask = None
    best_score = 0
    for combo in combinations(range(_N), 10):
        if time.time() > deadline:
            break
        mask = 0
        for v in combo:
            mask |= (1 << v)
        s = _score(mask)
        if s > best_score:
            best_score = s
            best_mask = mask
            if best_score == 10:  # Maximum possible for k=10
                return best_mask, best_score
    return best_mask, best_score


def _local_search(start_mask, deadline, rng):
    """Local search with simulated annealing to escape local optima."""
    current_mask = start_mask
    current_score = _score(current_mask) if bin(current_mask).count('1') >= _MINV else 0
    best_mask = current_mask
    best_score = current_score
    
    # Simulated annealing parameters
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    while time.time() < deadline and T > min_T:
        # Generate a neighbor: toggle one vertex
        v = rng.randint(0, _N - 1)
        bit = 1 << v
        new_mask = current_mask ^ bit
        if bin(new_mask).count('1') < _MINV:
            T *= cooling
            continue
        
        new_score = _score(new_mask)
        delta = new_score - current_score
        
        if delta > 0 or (T > 0 and rng.random() < min(1.0, delta / T)):
            current_mask = new_mask
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask
        
        T *= cooling
    
    return best_mask, best_score


def _hill_climb(mask, deadline):
    """Standard hill climbing."""
    current_mask = mask
    current_score = _score(mask) if bin(mask).count('1') >= _MINV else 0
    best_mask = current_mask
    best_score = current_score
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for v in range(_N):
            bit = 1 << v
            new_mask = current_mask ^ bit
            if bin(new_mask).count('1') < _MINV:
                continue
            s = _score(new_mask)
            if s > current_score:
                current_mask = new_mask
                current_score = s
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
                break
    return best_mask, best_score


def construct():
    deadline = time.time() + 58.0  # safety margin under 90s limit
    rng = random.Random(42)

    best_mask = None
    best_score = 0

    def consider(mask):
        nonlocal best_mask, best_score
        if bin(mask).count('1') < _MINV:
            return
        s = _score(mask)
        if s > best_score:
            best_score = s
            best_mask = mask

    # Phase 1: Exhaustive search for k=8 (most likely to find high diversity)
    if time.time() < deadline:
        bm, bs = _exhaustive_search_k8(deadline)
        if bm is not None and bs > best_score:
            best_mask = bm
            best_score = bs

    # Phase 2: Exhaustive search for k=9
    if time.time() < deadline:
        bm, bs = _exhaustive_search_k9(deadline)
        if bm is not None and bs > best_score:
            best_mask = bm
            best_score = bs

    # Phase 3: Exhaustive search for k=10 if time permits
    if time.time() < deadline:
        bm, bs = _exhaustive_search_k10(deadline)
        if bm is not None and bs > best_score:
            best_mask = bm
            best_score = bs

    # Phase 4: Local search for various sizes using best found as seed
    if time.time() < deadline:
        # Start from the best mask found and try to improve
        if best_mask is not None:
            bm, bs = _hill_climb(best_mask, deadline)
            if bs > best_score:
                best_mask = bm
                best_score = bs
        
        # Try multiple random restarts with simulated annealing
        for trial in range(200):
            if time.time() > deadline:
                break
            # Random size between 8 and 17
            k = rng.randint(_MINV, _N)
            verts = rng.sample(range(_N), k)
            mask = 0
            for v in verts:
                mask |= (1 << v)
            consider(mask)
            
            # Simulated annealing from this start
            bm, bs = _local_search(mask, deadline, rng)
            if bs > best_score:
                best_mask = bm
                best_score = bs
            
            # Also try hill climbing
            bm2, bs2 = _hill_climb(mask, deadline)
            if bs2 > best_score:
                best_mask = bm2
                best_score = bs2

    # Phase 5: Targeted search for specific degree patterns
    # Try to find subsets with degrees 0,1,2,...,k-1
    if time.time() < deadline:
        for k in range(_MINV, _N + 1):
            if time.time() > deadline:
                break
            # For each k, try to find a subset with k distinct degrees
            # Use greedy approach: start with a vertex of degree 0, then add vertices with new degrees
            for trial in range(100):
                if time.time() > deadline:
                    break
                # Random start
                verts = rng.sample(range(_N), k)
                mask = 0
                for v in verts:
                    mask |= (1 << v)
                
                # Try to optimize for distinct degrees
                current_mask = mask
                current_score = _score(current_mask)
                for _ in range(50):  # Iterations
                    if time.time() > deadline:
                        break
                    improved = False
                    # Try swapping vertices to increase distinct degrees
                    for i in range(len(verts)):
                        for j in range(_N):
                            if j in verts:
                                continue
                            new_verts = verts.copy()
                            new_verts[i] = j
                            new_mask = 0
                            for v in new_verts:
                                new_mask |= (1 << v)
                            s = _score(new_mask)
                            if s > current_score:
                                current_mask = new_mask
                                current_score = s
                                verts = new_verts
                                improved = True
                                break
                        if improved:
                            break
                    if not improved:
                        break
                
                if current_score > best_score:
                    best_mask = current_mask
                    best_score = current_score

    # Fallback: ensure we always have a valid result.
    if best_mask is None:
        seed_mask = 0
        for v in range(_MINV):
            seed_mask |= (1 << v)
        best_mask = seed_mask
        best_score = _score(best_mask)

    # Convert mask to a list of plain Python ints.
    return _mask_to_list(best_mask)
# EVOLVE-BLOCK-END
