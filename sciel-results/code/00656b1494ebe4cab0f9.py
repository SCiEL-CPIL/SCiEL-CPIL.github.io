"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 9
TOTAL_VERTICES = 1 << N  # 512

# EVOLVE-BLOCK-START

def _build_bitsets():
    """Precompute bitsets for fast validity checks.
    For each vertex v, neighbor_mask[v] is a bitset of all vertices at Hamming distance 1 from v.
    """
    neighbor_mask = [0] * TOTAL_VERTICES
    for v in range(TOTAL_VERTICES):
        mask = 0
        for b in range(N):
            u = v ^ (1 << b)
            mask |= (1 << u)
        neighbor_mask[v] = mask
    return neighbor_mask

NEIGHBOR_MASK = _build_bitsets()

def _is_valid_extension(path, used_mask, candidate):
    """Check if candidate can be appended to path.
    - candidate not in used_mask
    - candidate differs in at least 2 bits from all non-last vertices in path
    (it already differs by 1 bit from path[-1] by construction)
    """
    if (used_mask >> candidate) & 1:
        return False
    # Check against all path vertices except the last one
    # For each p in path[:-1], we need (candidate ^ p).bit_count() >= 2
    # Equivalent: candidate is NOT a neighbor of p
    # i.e., (NEIGHBOR_MASK[p] >> candidate) & 1 == 0
    for i in range(len(path) - 1):
        p = path[i]
        if (NEIGHBOR_MASK[p] >> candidate) & 1:
            return False
    return True

def _greedy_extend(path, used_mask, heuristic='lowest'):
    """Try to extend path greedily. Returns (extended_path, extended_used_mask, extended)."""
    cur = path[-1]
    if heuristic == 'lowest':
        bits = list(range(N))
    elif heuristic == 'random':
        bits = list(range(N))
        random.shuffle(bits)
    elif heuristic == 'most_options':
        # Choose the flip that leaves the most future options
        best_bit = None
        best_score = -1
        for b in range(N):
            v = cur ^ (1 << b)
            if (used_mask >> v) & 1:
                continue
            # Check validity
            valid = True
            for i in range(len(path) - 1):
                p = path[i]
                if (NEIGHBOR_MASK[p] >> v) & 1:
                    valid = False
                    break
            if not valid:
                continue
            # Score: number of unused neighbors of v that are not adjacent to any path vertex (except cur)
            # Approximate: count unused neighbors
            score = 0
            for nb_b in range(N):
                u = v ^ (1 << nb_b)
                if u == cur:
                    continue
                if (used_mask >> u) & 1:
                    continue
                # Check if u is adjacent to any path vertex (would make it invalid later)
                # Actually for scoring, just count available neighbors
                score += 1
            if score > best_score:
                best_score = score
                best_bit = b
        if best_bit is None:
            return path, used_mask, False
        nxt = cur ^ (1 << best_bit)
    else:
        bits = list(range(N))
        random.shuffle(bits)
    
    for b in bits:
        v = cur ^ (1 << b)
        if _is_valid_extension(path, used_mask, v):
            return path + [v], used_mask | (1 << v), True
    return path, used_mask, False

def _backtrack_extend(path, used_mask, depth, max_depth=500):
    """Backtracking search to extend path by up to max_depth more vertices.
    Returns the longest path found."""
    best_path = list(path)
    best_used = used_mask
    
    def _search(cur_path, cur_used, remaining_depth):
        nonlocal best_path, best_used
        if len(cur_path) > len(best_path):
            best_path = list(cur_path)
            best_used = cur_used
        if remaining_depth == 0:
            return
        cur = cur_path[-1]
        # Try all bit flips
        for b in range(N):
            v = cur ^ (1 << b)
            if _is_valid_extension(cur_path, cur_used, v):
                new_path = cur_path + [v]
                new_used = cur_used | (1 << v)
                _search(new_path, new_used, remaining_depth - 1)
    
    _search(path, used_mask, max_depth)
    return best_path, best_used

def _local_search(path, used_mask, iterations=10000):
    """Local search: try to swap/insert to extend the path.
    Simple approach: try to find a vertex that can be inserted between two consecutive vertices,
    or try to extend from the end with a different choice."""
    best_path = list(path)
    best_used = used_mask
    
    # Try extending from the end with random restarts of the last few steps
    for _ in range(iterations):
        # Remove last k vertices and try to re-extend
        k = random.randint(1, min(5, len(best_path) - 1))
        truncated = best_path[:-k]
        truncated_used = 0
        for v in truncated:
            truncated_used |= (1 << v)
        
        # Greedily extend with random heuristic
        cur_path = list(truncated)
        cur_used = truncated_used
        for _ in range(k + random.randint(0, 20)):
            cur_path, cur_used, ext = _greedy_extend(cur_path, cur_used, 'random')
            if not ext:
                break
        
        if len(cur_path) > len(best_path):
            best_path = cur_path
            best_used = cur_used
    
    return best_path, best_used

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    best_path = []
    best_used = 0
    
    # Multiple strategies with random restarts
    strategies = ['lowest', 'random', 'most_options']
    
    for restart in range(100000):  # Upper bound, will be cut by deadline
        if time.time() > deadline - 5:
            break
        
        strategy = strategies[restart % len(strategies)]
        
        # Start from a random seed vertex
        start = random.randint(0, TOTAL_VERTICES - 1)
        path = [start]
        used_mask = 1 << start
        
        # Greedy extension
        extended = True
        while extended:
            path, used_mask, extended = _greedy_extend(path, used_mask, strategy)
        
        # Try to improve with local search if time permits
        if time.time() < deadline - 10:
            path, used_mask = _local_search(path, used_mask, iterations=100)
        
        # Try short backtracking from the end
        if time.time() < deadline - 5 and len(path) < 150:
            # Only do backtracking if path is not too long (to keep it fast)
            path, used_mask = _backtrack_extend(path, used_mask, depth=20, max_depth=20)
        
        if len(path) > len(best_path):
            best_path = path
            best_used = used_mask
    
    # Final polish: try to extend the best path found
    if time.time() < deadline - 2:
        extended = True
        while extended and time.time() < deadline - 1:
            best_path, best_used, extended = _greedy_extend(best_path, best_used, 'lowest')
            if not extended:
                # Try random
                best_path, best_used, extended = _greedy_extend(best_path, best_used, 'random')
    
    return best_path

# EVOLVE-BLOCK-END
