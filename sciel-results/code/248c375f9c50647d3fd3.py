# EVOLVE-BLOCK-START
import time

_P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

_N = _P['n']
_EDGES = _P['edges']

# Build adjacency bitmasks
_ADJ = [0] * _N
for _a, _b in _EDGES:
    _ADJ[_a] |= (1 << _b)
    _ADJ[_b] |= (1 << _a)

# Precompute degree
_DEG = [bin(_ADJ[v]).count('1') for v in range(_N)]

# Precompute all clique and independent set masks
def _is_clique_mask(mask):
    while mask:
        v = (mask & -mask).bit_length() - 1
        rest = mask ^ (1 << v)
        if (_ADJ[v] & rest) != rest:
            return False
        mask = rest
    return True

def _is_indep_mask(mask):
    while mask:
        v = (mask & -mask).bit_length() - 1
        rest = mask ^ (1 << v)
        if (_ADJ[v] & rest) != 0:
            return False
        mask = rest
    return True

_CLIQ = [False] * (1 << _N)
_INDEP = [False] * (1 << _N)
_VALID = [False] * (1 << _N)
for _m in range(1 << _N):
    if _m == 0:
        _CLIQ[_m] = True
        _INDEP[_m] = True
        _VALID[_m] = True
    else:
        _CLIQ[_m] = _is_clique_mask(_m)
        _INDEP[_m] = _is_indep_mask(_m)
        _VALID[_m] = _CLIQ[_m] or _INDEP[_m]

# Build list of valid part masks for each size
# valid_parts_by_size[k] = list of masks that are valid parts of size k
_MAX_SIZE = _N
_valid_parts_by_size = [[] for _ in range(_MAX_SIZE + 1)]
for _m in range(1, 1 << _N):
    if _VALID[_m]:
        sz = bin(_m).count('1')
        _valid_parts_by_size[sz].append(_m)

# For each valid part, precompute which vertices it contains
def _mask_to_list(mask):
    return [v for v in range(_N) if (mask >> v) & 1]

# Precompute: for each valid part mask, the set of vertices as a bitmask
# This is already the mask itself

# DP: minimum number of parts to cover each subset
# Use subset DP: _MIN_PARTS[m] = min over valid submasks s of m: 1 + _MIN_PARTS[m^s]
_MIN_PARTS = [0] * (1 << _N)
for _m in range(1, 1 << _N):
    _MIN_PARTS[_m] = 10 ** 9

# Iterate over masks in increasing order of popcount
for _m in range(1, 1 << _N):
    pc = bin(_m).count('1')
    best = 10 ** 9
    # Try all valid submasks of _m
    sub = _m
    while sub:
        if _VALID[sub]:
            rest = _m ^ sub
            if rest == 0:
                candidate = 1
            else:
                candidate = _MIN_PARTS[rest] + 1
            if candidate < best:
                best = candidate
        sub = (sub - 1) & _m
    _MIN_PARTS[_m] = best

# Get the minimum number of parts needed for the full graph
_FULL_MASK = (1 << _N) - 1
_MIN_PARTS_FULL = _MIN_PARTS[_FULL_MASK]

# Now use backtracking search to find a valid partition with exactly _MIN_PARTS_FULL parts
# This is a constraint satisfaction / exact cover problem

# State: (remaining_mask, list_of_parts)
# We want to find a partition of _FULL_MASK into _MIN_PARTS_FULL valid parts

_deadline = None
_best_partition = None

def _search(remaining_mask, parts, depth, target_depth):
    global _best_partition, _deadline
    if remaining_mask == 0:
        if depth == target_depth:
            _best_partition = [list(parts)]
            return True
        return False
    if depth >= target_depth:
        return False
    if time.time() > _deadline:
        return False
    if depth + _MIN_PARTS[remaining_mask] > target_depth:
        return False
    
    # Choose the vertex with highest degree among remaining
    rem_list = _mask_to_list(remaining_mask)
    v = max(rem_list, key=lambda x: _DEG[x])
    vbit = 1 << v
    
    # Try all valid parts that contain v and are subsets of remaining_mask
    # To reduce symmetry, we only consider parts where v is the highest-degree vertex
    # or the first vertex in some canonical ordering
    
    # Get all valid parts that contain v and are subsets of remaining_mask
    # Filter by: the part must contain v, and v must be the "representative" (highest degree)
    candidates = []
    for sz in range(1, min(len(rem_list), _N) + 1):
        for pmask in _valid_parts_by_size[sz]:
            if not (pmask & remaining_mask) == pmask:
                continue
            if not (pmask & vbit):
                continue
            # Symmetry breaking: v must be the highest-degree vertex in the part
            # or if there's a tie, v must be the smallest index
            is_rep = True
            for w in _mask_to_list(pmask):
                if w != v:
                    if _DEG[w] > _DEG[v]:
                        is_rep = False
                        break
                    elif _DEG[w] == _DEG[v] and w < v:
                        is_rep = False
                        break
            if is_rep:
                candidates.append(pmask)
    
    # Sort candidates by size descending (prefer larger parts to reduce search space)
    candidates.sort(key=lambda m: -bin(m).count('1'))
    
    for pmask in candidates:
        new_rem = remaining_mask ^ pmask
        parts.append(pmask)
        if _search(new_rem, parts, depth + 1, target_depth):
            return True
        parts.pop()
    
    return False

def _greedy_partition():
    """Greedy construction to get an initial upper bound."""
    parts = []
    remaining = _FULL_MASK
    while remaining:
        # Choose vertex with highest degree in remaining
        rem_list = _mask_to_list(remaining)
        v = max(rem_list, key=lambda x: _DEG[x])
        
        # Find the largest valid part containing v that is a subset of remaining
        best_part = 1 << v
        best_sz = 1
        for sz in range(len(rem_list), 0, -1):
            found = False
            for pmask in _valid_parts_by_size[sz]:
                if not (pmask & remaining) == pmask:
                    continue
                if not (pmask & (1 << v)):
                    continue
                # Symmetry: v should be representative
                is_rep = True
                for w in _mask_to_list(pmask):
                    if w != v:
                        if _DEG[w] > _DEG[v]:
                            is_rep = False
                            break
                        elif _DEG[w] == _DEG[v] and w < v:
                            is_rep = False
                            break
                if is_rep:
                    best_part = pmask
                    best_sz = sz
                    found = True
                    break
            if found:
                break
        parts.append(best_part)
        remaining ^= best_part
    return parts

def construct():
    global _best_partition, _deadline
    _deadline = time.time() + 50.0  # 50s safety margin
    
    # Start with greedy to get an upper bound
    greedy_parts = _greedy_partition()
    current_best_len = len(greedy_parts)
    _best_partition = [list(greedy_parts)]
    
    # Try to find a partition with exactly _MIN_PARTS_FULL parts
    if _MIN_PARTS_FULL < current_best_len:
        if _search(_FULL_MASK, [], 0, _MIN_PARTS_FULL):
            current_best_len = _MIN_PARTS_FULL
    
    # If we couldn't find the optimal, try _MIN_PARTS_FULL + 1, etc.
    # But since _MIN_PARTS_FULL is the DP optimum, we should find it
    # If search times out, fall back to greedy
    
    if _best_partition is None:
        # Fallback: singleton partition
        _best_partition = [[(1 << v)] for v in range(_N)]
    
    # Convert to list of vertex lists
    result = []
    for pmask in _best_partition[0]:
        result.append(_mask_to_list(pmask))
    
    # Ensure each part is nonempty and all vertices appear exactly once
    # (Should be guaranteed by construction, but verify)
    all_vertices = []
    for part in result:
        for v in part:
            all_vertices.append(int(v))
    all_vertices.sort()
    if all_vertices != list(range(_N)):
        # Fallback to safe partition
        result = [[v] for v in range(_N)]
    
    # Ensure types are correct
    final_result = []
    for part in result:
        final_result.append([int(x) for x in part])
    
    return final_result
# EVOLVE-BLOCK-END
