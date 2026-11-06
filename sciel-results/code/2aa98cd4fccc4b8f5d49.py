import time
import math
import random

def construct():
    start_time = time.time()
    DEADLINE = 85

    N = 256

    squares = set()
    k = 1
    while k * k <= 512:
        squares.add(k * k)
        k += 1

    diagonal_forbidden = set()
    m = 1
    while 2 * m * m <= N:
        diagonal_forbidden.add(2 * m * m)
        m += 1

    # Build conflict adjacency using bitsets (Python ints as bitsets)
    # Vertex i (1-indexed) -> bit i in integer
    # For vertex i, conflict_mask[i] has bits set for all j where i+j is a perfect square
    conflict_mask = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if i + j in squares:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)

    # Allowed vertices: not diagonal-forbidden
    allowed_mask = 0
    for i in range(1, N + 1):
        if i not in diagonal_forbidden:
            allowed_mask |= (1 << i)

    def greedy_coloring(vertices_mask):
        """
        Greedy coloring to get an upper bound on the independence number.
        Returns a list of color classes (each a bitmask).
        The number of color classes is an upper bound on the max independent set size
        within vertices_mask (since independent set can take at most one per color class).
        """
        color_classes = []
        remaining = vertices_mask
        while remaining:
            # Greedy: pick a maximal independent set from remaining
            color_class = 0
            r = remaining
            while r:
                # Pick vertex with lowest index in r
                v = (r & (-r)).bit_length() - 1  # position of lowest set bit
                # Actually, let's pick the vertex with minimum degree in remaining for better coloring
                # For simplicity, pick lowest index
                bit = 1 << v
                if not (color_class & bit):
                    # Check v is not in color_class already (it isn't since we remove from r)
                    color_class |= bit
                    # Remove v and its neighbors from r
                    r &= ~(bit | conflict_mask[v])
            color_classes.append(color_class)
            remaining &= ~color_class
        return color_classes

    def branch_and_bound(available_mask, current_set_mask, current_size, best_size, best_set_mask, deadline):
        """
        Branch and bound for maximum independent set.
        available_mask: vertices that can still be added
        current_set_mask: vertices already in the independent set
        current_size: number of vertices in current set
        Returns (best_size, best_set_mask)
        """
        if time.time() > deadline:
            return best_size, best_set_mask

        if available_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set_mask = current_set_mask
            return best_size, best_set_mask

        # Upper bound: current_size + number of vertices in available_mask
        # Better bound: use greedy coloring
        # For a tighter bound, color the available vertices
        color_classes = greedy_coloring(available_mask)
        upper_bound = current_size + len(color_classes)

        if upper_bound <= best_size:
            return best_size, best_set_mask

        # Select a vertex to branch on: pick one from the first color class with highest degree
        # For better performance, pick vertex with maximum degree in available_mask
        # Find vertex with max conflicts within available_mask
        best_v = -1
        best_degree = -1
        temp = available_mask
        while temp:
            v = (temp & (-temp)).bit_length() - 1
            deg = bin(conflict_mask[v] & available_mask).count('1')
            if deg > best_degree:
                best_degree = deg
                best_v = v
            temp &= temp - 1

        if best_v < 0:
            return best_size, best_set_mask

        bit_v = 1 << best_v

        # Branch 1: include best_v
        new_available = available_mask & ~bit_v & ~conflict_mask[best_v]
        new_set = current_set_mask | bit_v
        best_size, best_set_mask = branch_and_bound(
            new_available, new_set, current_size + 1,
            best_size, best_set_mask, deadline
        )

        # Branch 2: exclude best_v
        best_size, best_set_mask = branch_and_bound(
            available_mask & ~bit_v, current_set_mask, current_size,
            best_size, best_set_mask, deadline
        )

        return best_size, best_set_mask

    def greedy_independent_set(vertices_mask):
        """Greedy: repeatedly pick vertex with minimum degree in remaining."""
        result = 0
        remaining = vertices_mask
        while remaining:
            # Find vertex with minimum degree in remaining
            best_v = -1
            best_deg = N + 1
            temp = remaining
            while temp:
                v = (temp & (-temp)).bit_length() - 1
                deg = bin(conflict_mask[v] & remaining).count('1')
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                temp &= temp - 1
            if best_v < 0:
                break
            bit = 1 << best_v
            result |= bit
            remaining &= ~bit & ~conflict_mask[best_v]
        return result

    # Phase 1: Greedy initialization with multiple random orderings
    best_size = 0
    best_set_mask = 0

    # Natural greedy
    gs = greedy_independent_set(allowed_mask)
    best_size = bin(gs).count('1')
    best_set_mask = gs

    # Try several random perturbations
    for trial in range(30):
        if time.time() - start_time > DEADLINE:
            break
        # Randomize by picking random subset to start
        # Or: greedy with random tie-breaking
        remaining = allowed_mask
        result = 0
        while remaining:
            # Pick random vertex from remaining with min-degree preference
            # Get all vertices in remaining
            verts = []
            temp = remaining
            while temp:
                v = (temp & (-temp)).bit_length() - 1
                verts.append(v)
                temp &= temp - 1
            if not verts:
                break
            # Score each by degree + random noise
            best_v = None
            best_score = float('inf')
            for v in verts:
                deg = bin(conflict_mask[v] & remaining).count('1')
                score = deg + random.random() * 2
                if score < best_score:
                    best_score = score
                    best_v = v
            bit = 1 << best_v
            result |= bit
            remaining &= ~bit & ~conflict_mask[best_v]
        sz = bin(result).count('1')
        if sz > best_size:
            best_size = sz
            best_set_mask = result

    # Phase 2: Branch and bound to try to improve
    # Use the greedy solution as initial lower bound
    deadline = DEADLINE - 5
    if time.time() < deadline:
        # Start B&B from the full allowed set
        best_size_bb, best_set_mask_bb = branch_and_bound(
            allowed_mask, 0, 0, best_size, best_set_mask, deadline
        )
        if best_size_bb > best_size:
            best_size = best_size_bb
            best_set_mask = best_set_mask_bb

    # Phase 3: Local search on the best found set to try to improve further
    if time.time() < DEADLINE - 3:
        # Convert mask to set
        current_set = set()
        temp = best_set_mask
        while temp:
            v = (temp & (-temp)).bit_length() - 1
            current_set.add(v)
            temp &= temp - 1

        for ls_iter in range(100):
            if time.time() > DEADLINE - 3:
                break
            improved = False

            # Try adding a vertex
            candidates = [v for v in range(1, N + 1) if v not in current_set and v not in diagonal_forbidden]
            random.shuffle(candidates)
            for v in candidates:
                can_add = True
                for u in current_set:
                    if u in conflict_mask[v] or v in conflict_mask[u]:
                        can_add = False
                        break
                if can_add:
                    current_set.add(v)
                    improved = True
                    break

            if not improved:
                # Try swap: remove one, add one
                s_list = list(current_set)
                random.shuffle(s_list)
                for rv in s_list:
                    if len(current_set) <= 1:
                        break
                    current_set.remove(rv)
                    candidates2 = [v for v in range(1, N + 1) if v not in current_set and v not in diagonal_forbidden]
                    random.shuffle(candidates2)
                    for v in candidates2:
                        can_add = True
                        for u in current_set:
                            if (1 << v) & conflict_mask[u] or (1 << u) & conflict_mask[v]:
                                can_add = False
                                break
                        if can_add:
                            current_set.add(v)
                            improved = True
                            break
                        else:
                            current_set.add(rv)
                            break
                    if improved:
                        break

            if not improved:
                break

        # Update best
        sz = len(current_set)
        if sz > best_size:
            best_size = sz
            best_set_mask = 0
            for v in current_set:
                best_set_mask |= (1 << v)

    # Convert to list
    result = []
    temp = best_set_mask
    while temp:
        v = (temp & (-temp)).bit_length() - 1
        result.append(v)
        temp &= temp - 1
    result.sort()

    # Final verification
    valid = True
    for a in result:
        if 2 * a in squares:
            valid = False
            break
    if valid:
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                if result[i] + result[j] in squares:
                    valid = False
                    break
            if not valid:
                break

    if not valid or not result:
        # Fallback
        result = []
        for v in range(1, N + 1):
            if v in diagonal_forbidden:
                continue
            can_add = True
            for u in result:
                if v + u in squares:
                    can_add = False
                    break
            if can_add:
                result.append(v)
        if not result:
            result = [1]

    return result
