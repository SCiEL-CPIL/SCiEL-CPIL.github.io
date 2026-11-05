"""HP lattice protein folding using Hamiltonian path skeletons with fast pivot search."""
import random
import time
import sys

# EVOLVE-BLOCK-START


def _generate_skeleton(n, dimension=2):
    """Generate a space-filling Hamiltonian path skeleton of length n on square lattice.
    
    Returns a list of (x, y) tuples forming a self-avoiding walk of length n.
    Uses a snake-like pattern in a bounding box.
    """
    # Find smallest bounding box that can hold n residues in a snake pattern
    # For a snake pattern in a w x h box: capacity = w * h
    # We want w * h >= n, and the snake fills it row by row
    
    # Try different aspect ratios
    best_skel = None
    best_score = -1
    
    # The snake pattern: go right along row 0, down, left along row 1, down, right along row 2, etc.
    # This gives many adjacent pairs that are non-consecutive in the chain.
    
    for w in range(1, n + 1):
        h = (n + w - 1) // w  # ceiling division
        if w * h < n:
            continue
        # Generate snake path of length n in w x h box
        path = []
        for row in range(h):
            if row % 2 == 0:
                for col in range(w):
                    if len(path) < n:
                        path.append((col, row))
            else:
                for col in range(w - 1, -1, -1):
                    if len(path) < n:
                        path.append((col, row))
        if len(path) != n:
            continue
        
        # Score: count non-consecutive adjacent pairs
        pos_set = {p: i for i, p in enumerate(path)}
        contacts = 0
        for i, (x, y) in enumerate(path):
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                j = pos_set.get((x + dx, y + dy))
                if j is not None and abs(j - i) > 1:
                    contacts += 1
        contacts //= 2  # each pair counted twice
        
        if contacts > best_score:
            best_score = contacts
            best_skel = path
    
    if best_skel is None:
        # Fallback: straight line
        best_skel = [(i, 0) for i in range(n)]
    
    return best_skel


def _energy_fast(chain, is_h):
    """Compute energy using a position-to-index map. Returns negative contact count."""
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i, (x, y) in enumerate(chain):
        if not is_h[i]:
            continue
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            j = where.get((x + dx, y + dy))
            if j is not None and j > i + 1 and is_h[j]:
                contacts += 1
    return -contacts


def _pivot_search(chain, is_h, n, max_iters, deadline):
    """Local search using pivot moves. Returns best chain found."""
    best_chain = list(chain)
    best_energy = _energy_fast(best_chain, is_h)
    
    # Precompute where map
    where = {p: i for i, p in enumerate(best_chain)}
    
    rng = random.Random(42)
    iters = 0
    
    while iters < max_iters:
        if time.time() > deadline:
            break
        iters += 1
        
        # Pick a pivot point
        i = rng.randint(0, n - 1)
        if i == 0 or i == n - 1:
            continue
        
        # Pick a direction for the pivot
        dx, dy = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        
        # The pivot move: rotate the tail (i+1 to end) by 90 degrees around best_chain[i]
        # In 2D, rotating (x, y) by 90 deg CCW around origin gives (-y, x)
        # We need to rotate the vector from pivot to each tail point
        
        pivot = best_chain[i]
        px, py = pivot
        
        # Compute new tail positions
        new_tail = []
        valid = True
        for j in range(i + 1, n):
            ox, oy = best_chain[j]
            # Vector from pivot
            vx, vy = ox - px, oy - py
            # Rotate 90 deg CCW: (vx, vy) -> (-vy, vx)
            nx = px - vy
            ny = py + vx
            new_tail.append((nx, ny))
        
        # Check self-avoidance: new tail positions must not collide with head (0..i) or each other
        head_set = set(best_chain[:i + 1])
        tail_set = set(new_tail)
        
        if len(tail_set) != n - i - 1:
            continue  # self-intersection in tail
        if head_set & tail_set:
            continue  # collision with head
        
        # Also need to check that the connection at i is valid (unit step)
        # The first point of new_tail must be adjacent to pivot
        fx, fy = new_tail[0]
        if abs(fx - px) + abs(fy - py) != 1:
            continue
        
        # Compute energy delta
        # Remove old tail contacts, add new tail contacts
        # For efficiency, just compute full energy of new chain
        new_chain = best_chain[:i + 1] + new_tail
        new_energy = _energy_fast(new_chain, is_h)
        
        if new_energy < best_energy:
            best_chain = new_chain
            best_energy = new_energy
            where = {p: i for i, p in enumerate(best_chain)}
    
    return best_chain, best_energy


def _multi_start_fold(sequence, dimension, deadline):
    """Try multiple skeleton variants and pivot search each."""
    n = len(sequence)
    is_h = [c == 'H' for c in sequence]
    
    best_chain = None
    best_energy = 0  # 0 means no contacts (unfolded)
    
    # Generate several different skeletons
    skeletons = []
    
    # Primary snake skeleton
    skeletons.append(_generate_skeleton(n, dimension))
    
    # Transposed snake (swap x and y)
    if n > 1:
        transposed = [(y, x) for x, y in skeletons[0]]
        # Verify it's valid (should be, just rotated)
        skeletons.append(transposed)
    
    # Try different bounding box sizes by padding
    for extra_w in range(1, 4):
        w = 0
        h = 0
        # Find the dimensions of the first skeleton
        xs = [p[0] for p in skeletons[0]]
        ys = [p[1] for p in skeletons[0]]
        base_w = max(xs) + 1
        base_h = max(ys) + 1
        
        new_w = base_w + extra_w
        new_h = (n + new_w - 1) // new_w
        if new_w * new_h < n:
            continue
        
        path = []
        for row in range(new_h):
            if row % 2 == 0:
                for col in range(new_w):
                    if len(path) < n:
                        path.append((col, row))
            else:
                for col in range(new_w - 1, -1, -1):
                    if len(path) < n:
                        path.append((col, row))
        if len(path) == n:
            skeletons.append(path)
    
    # Also try a spiral-like pattern
    if n >= 4:
        # Simple spiral: go right, down, left, up, right...
        spiral = []
        x, y = 0, 0
        dx, dy = 1, 0
        steps_in_dir = 1
        dir_count = 0
        for _ in range(n):
            spiral.append((x, y))
            x += dx
            y += dy
            dir_count += 1
            if dir_count == steps_in_dir:
                dir_count = 0
                # Turn left
                dx, dy = -dy, dx
                if (dx, dy) == (1, 0) or (dx, dy) == (-1, 0):
                    steps_in_dir += 1
        # Verify spiral is self-avoiding
        if len(set(spiral)) == n:
            skeletons.append(spiral)
    
    # Deduplicate skeletons
    seen = set()
    unique_skeletons = []
    for skel in skeletons:
        key = tuple(skel)
        if key not in seen:
            seen.add(key)
            unique_skeletons.append(skel)
    
    # Time budget per skeleton
    num_skeletons = len(unique_skeletons)
    time_per_skel = max(1.0, (deadline - time.time()) / max(1, num_skeletons) * 0.8)
    
    for skel in unique_skeletons:
        if time.time() > deadline:
            break
        
        skel_deadline = min(deadline, time.time() + time_per_skel)
        
        # Initial energy
        init_energy = _energy_fast(skel, is_h)
        
        # Pivot search
        result_chain, result_energy = _pivot_search(skel, is_h, n, 10000, skel_deadline)
        
        if best_chain is None or result_energy < best_energy:
            best_chain = result_chain
            best_energy = result_energy
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
        best_energy = 0
    
    return [list(p) for p in best_chain]


def fold(sequence, dimension):
    """Best conformation using skeleton-based approach with pivot local search."""
    n = len(sequence)
    
    # Global deadline: leave safety margin
    global_deadline = time.time() + 190  # 190s out of 210s limit
    
    if n <= 2:
        return [[i, 0] for i in range(n)]
    
    result = _multi_start_fold(sequence, dimension, global_deadline)
    
    # Validate self-avoidance
    seen = set()
    for p in result:
        tp = tuple(p)
        if tp in seen:
            # Fallback to straight line
            return [[i, 0] for i in range(n)]
        seen.add(tp)
    
    # Validate unit steps
    for i in range(1, n):
        dx = abs(result[i][0] - result[i-1][0])
        dy = abs(result[i][1] - result[i-1][1])
        if dx + dy != 1:
            return [[i, 0] for i in range(n)]
    
    return result

# EVOLVE-BLOCK-END
