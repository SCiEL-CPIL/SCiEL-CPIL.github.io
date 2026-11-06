"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

N = 9
TOTAL_VERTICES = 1 << N

# EVOLVE-BLOCK-START
import numpy as np

# Precompute neighbors
neighbors_list = [[] for _ in range(TOTAL_VERTICES)]
for v in range(TOTAL_VERTICES):
    for b in range(N):
        u = v ^ (1 << b)
        neighbors_list[v].append(u)

# Bit masks for quick adjacency check
# adj_mask[v] is an integer with bits set for neighbors of v
adj_mask = [0] * TOTAL_VERTICES
for v in range(TOTAL_VERTICES):
    mask = 0
    for u in neighbors_list[v]:
        mask |= (1 << u)
    adj_mask[v] = mask

def _is_valid_snake(path):
    """Check if path is a valid induced path in Q_N."""
    n = len(path)
    if n == 0:
        return True
    if len(set(path)) != n:
        return False
    # Check consecutive adjacency
    for i in range(n - 1):
        if (path[i] ^ path[i + 1]).bit_count() != 1:
            return False
    # Check non-consecutive non-adjacency
    # Optimize: for each i, check j > i+1
    for i in range(n):
        vi = path[i]
        # If vi is adjacent to any path[j] for j > i+1, invalid
        # We can check if adj_mask[vi] intersects with the set of path[i+2:]
        # But constructing the set is O(n).
        # Simpler: loop j
        for j in range(i + 2, n):
            if (vi ^ path[j]).bit_count() == 1:
                return False
    return True

def _greedy_construct(rng, start_vertex=None):
    """Construct a valid snake using a greedy strategy with randomization."""
    if start_vertex is None:
        start_vertex = rng.randint(0, TOTAL_VERTICES - 1)
    
    path = [start_vertex]
    used = {start_vertex}
    # Forbidden for the next step: neighbors of start_vertex are candidates, 
    # but we must ensure that the new vertex's neighbors (except the one we came from) 
    # don't include any previously used vertex that is not the immediate predecessor.
    # Actually, the constraint is: new vertex w must not be adjacent to any v in path[:-1].
    # Because if w is adjacent to v (where v is in path[:-1]), then v and w are non-consecutive 
    # (since w is new, it's at the end) and adjacent.
    # Wait, if w is adjacent to path[-1], that's the connection.
    # If w is adjacent to path[-2], that's a triangle (impossible in hypercube).
    # If w is adjacent to path[-k] for k>1, that's invalid.
    
    current = start_vertex
    while True:
        candidates = []
        for v in neighbors_list[current]:
            if v in used:
                continue
            # Check if v is adjacent to any node in path[:-1]
            # If yes, adding v would create an edge (v, path[j]) where j < len(path)-1.
            # Since v is new, it is at index len(path).
            # path[j] and v are non-consecutive. So they must NOT be adjacent.
            valid = True
            for j in range(len(path) - 1): # path[0] to path[-2]
                if (v ^ path[j]).bit_count() == 1:
                    valid = False
                    break
            if valid:
                candidates.append(v)
        
        if not candidates:
            break
        
        # Choose randomly from candidates
        next_v = rng.choice(candidates)
        path.append(next_v)
        used.add(next_v)
        current = next_v
        
    return path

def _local_search(path, rng, time_limit=1.0):
    """
    Try to extend the path by inserting vertices from the unused set.
    Returns the improved path.
    """
    start_time = time.time()
    used = set(path)
    unused = [v for v in range(TOTAL_VERTICES) if v not in used]
    
    improved = True
    while improved and time.time() < start_time + time_limit:
        improved = False
        # Shuffle unused to avoid bias
        rng.shuffle(unused)
        
        for v in unused:
            # Try to insert v at any position i in path
            # Inserting v at position i means:
            # path[i-1] -- v -- path[i]
            # Constraints:
            # 1. v adjacent to path[i-1] (if i>0)
            # 2. v adjacent to path[i] (if i<len(path))
            # 3. v NOT adjacent to any path[j] where j != i-1, i
            # 4. path[i-1] NOT adjacent to path[i] (already true if path is valid, but they become non-consecutive if we insert between them? No, they are separated by v. So path[i-1] and path[i] become non-consecutive. They must NOT be adjacent.
            #    Wait, if path[i-1] and path[i] were adjacent (consecutive in original), inserting v between them breaks that edge.
            #    But path[i-1] and path[i] are consecutive in the original path, so they ARE adjacent.
            #    If we insert v between them, they are no longer consecutive.
            #    So we require that path[i-1] and path[i] are NOT adjacent?
            #    But they ARE adjacent in the original valid path.
            #    So we CANNOT insert a vertex between two consecutive vertices of a valid induced path, because that would make them non-consecutive and adjacent, which is forbidden.
            #    Therefore, we can only insert v at the ends? Or replace?
            #    Actually, if we insert v between u and w, u and w must not be adjacent.
            #    But in a path, u and w are adjacent.
            #    So insertion in the middle is impossible for a simple path extension?
            #    Yes, you cannot "subdivide" an edge in an induced path without violating the induced property (unless the original edge didn't exist, but it does).
            #    So we can only append to the ends?
            #    Or we can "move" a vertex?
            
            # Let's try appending to the end.
            # Check if v can be appended to the end.
            last = path[-1]
            if (v ^ last).bit_count() != 1:
                continue
            # Check if v is adjacent to any path[j] for j < len(path)-1
            valid = True
            for j in range(len(path) - 1):
                if (v ^ path[j]).bit_count() == 1:
                    valid = False
                    break
            if valid:
                path.append(v)
                used.add(v)
                unused.remove(v)
                improved = True
                break # Restart loop with new path state? Or continue?
                # Restarting is safer to re-evaluate other unused vertices against the new last vertex.
            
            # Check if v can be prepended to the start.
            first = path[0]
            if (v ^ first).bit_count() != 1:
                continue
            valid = True
            for j in range(1, len(path)):
                if (v ^ path[j]).bit_count() == 1:
                    valid = False
                    break
            if valid:
                path.insert(0, v)
                used.add(v)
                unused.remove(v)
                improved = True
                break
        
        # If no insertion worked, try "swaps" or "moves"?
        # For now, just append/prepend.
        
    return path

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    best_path = []
    best_len = 0
    
    # Initial best from a few greedy runs
    for i in range(10):
        rng = random.Random(i)
        p = _greedy_construct(rng)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p[:]
            
    # Iterated Local Search
    rng = random.Random(123)
    
    while time.time() < deadline - 5:
        # Perturb the current best path
        # Strategy: Remove a random segment of length k and re-greedy fill?
        # Or just take a random prefix/suffix?
        # Let's try: Take a random subpath of best_path and use it as seed for greedy?
        # Or: Take best_path, remove a random vertex, and re-greedy?
        
        # Better: Take best_path, pick a random vertex v in it, remove it, 
        # and then try to re-insert it or extend from the ends.
        
        # Simpler Perturbation: 
        # 1. Pick a random start vertex from the best_path (or random entirely)
        # 2. Run greedy with that start?
        # This might just regenerate similar snakes.
        
        # Let's try "2-opt" like reversal?
        # Reversing a segment of an induced path preserves validity?
        # Path: u0, u1, ..., uk.
        # Reverse u_i...u_j.
        # New: ..., u_{i-1}, u_j, u_{j-1}, ..., u_i, u_{j+1}, ...
        # Check edges:
        # (u_{i-1}, u_j) must be adjacent.
        # (u_j, u_{j-1}) adjacent (was u_{j-1}, u_j).
        # ...
        # (u_i, u_{j+1}) must be adjacent.
        # Check induced:
        # Any new non-consecutive adjacencies?
        # The internal structure is reversed, so adjacencies among u_i..u_j are preserved (they were consecutive, now consecutive in reverse).
        # The boundaries are the risk.
        # If (u_{i-1}, u_j) is adjacent and (u_i, u_{j+1}) is adjacent, is the path still induced?
        # We must ensure u_{i-1} is not adjacent to u_{j-1}, u_{j-2}...
        # And u_{j+1} is not adjacent to u_{i+1}, u_{i+2}...
        # This is a local check.
        
        # Let's implement a simple perturbation: Random Reversal
        if len(best_path) > 5:
            i = rng.randint(0, len(best_path) - 2)
            j = rng.randint(i + 1, len(best_path) - 1)
            
            new_path = best_path[:i] + best_path[i:j+1][::-1] + best_path[j+1:]
            
            if _is_valid_snake(new_path):
                # Try to extend
                new_path = _local_search(new_path, rng, time_limit=0.5)
                if len(new_path) > best_len:
                    best_len = len(new_path)
                    best_path = new_path
            else:
                # If reversal failed, try a different perturbation
                # Just re-greedy from a random vertex in the path
                start_v = rng.choice(best_path)
                p = _greedy_construct(rng, start_v)
                p = _local_search(p, rng, time_limit=0.5)
                if len(p) > best_len:
                    best_len = len(p)
                    best_path = p
        else:
            # If path is short, just re-greedy
            p = _greedy_construct(rng)
            p = _local_search(p, rng, time_limit=0.5)
            if len(p) > best_len:
                best_len = len(p)
                best_path = p

    return best_path

# EVOLVE-BLOCK-END
