import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Strategy 1: Algebraic initialization + incremental local search for n=26
    if time.time() < deadline - 10:
        r26 = algebraic_init_local_search(26, time_limit=120.0)
        if r26 is not None:
            # Try to extend to 27
            if time.time() < deadline - 15:
                r27 = try_extend_focused(r26, time_limit=12.0)
                if r27 is not None:
                    return r27
            return r26

    # Strategy 2: Pure local search for n=26
    if time.time() < deadline - 10:
        r26 = focused_local_search(26, time_limit=120.0)
        if r26 is not None:
            if time.time() < deadline - 15:
                r27 = try_extend_focused(r26, time_limit=12.0)
                if r27 is not None:
                    return r27
            return r26

    # Fallback: smaller n
    for n in [25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]:
        if time.time() > deadline - 5:
            break
        r = focused_local_search(n, time_limit=8.0)
        if r is not None:
            return r

    # Absolute fallback
    return {'n': 11, 'colors': [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]}


def algebraic_init_local_search(n, time_limit=120.0):
    """
    Use a deterministic algebraic initialization (Z_7 x Z_7 modular construction)
    followed by incremental local search to find a valid 5-coloring of K_n.
    """
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # Algebraic initialization: map vertices to Z_7 x Z_7, color by modular difference
    def algebraic_color(i, j):
        # Map vertex index to (x, y) in Z_7 x Z_7
        xi, yi = i % 7, i // 7
        xj, yj = j % 7, j // 7
        dx = (xj - xi) % 7
        dy = (yj - yi) % 7
        if dx == 0:
            # Vertical: use dy to determine color (5 colors for 6 possible dy values)
            return dy % 5
        else:
            # Slope: dy/dx mod 7, map to 5 colors
            slope = (dy * pow(dx, 5, 7)) % 7  # dx^{-1} mod 7 = dx^5 mod 7
            return slope % 5

    colors = [algebraic_color(i, j) for (i, j) in edges]

    # Build adjacency bitmasks
    adj_mask = [[0] * n for _ in range(5)]
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        adj_mask[c][i] |= (1 << j)
        adj_mask[c][j] |= (1 << i)

    # Compute initial violations using incremental method
    total_violations = compute_total_violations(adj_mask, n)

    if total_violations == 0:
        return {'n': n, 'colors': colors}

    # Edge index lookup: (i, j) -> index
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    # Build set of violating edges for targeted selection
    def get_violating_edges(adj_mask):
        violating = set()
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj_mask[c][u] & adj_mask[c][v]
                    if common.bit_count() >= 2:
                        violating.add(edge_idx(u, v))
                        temp = common
                        while temp:
                            w = (temp & -temp).bit_length() - 1
                            violating.add(edge_idx(min(u, w), max(u, w)))
                            violating.add(edge_idx(min(v, w), max(v, w)))
                            temp &= temp - 1
        return violating

    violating_edges = get_violating_edges(adj_mask)

    best_violations = total_violations
    best_colors = colors[:]
    start_time = time.time()
    temperature = 1.5
    no_improve = 0
    max_no_improve = 5000

    while time.time() - start_time < time_limit:
        if no_improve > max_no_improve:
            # Restart with algebraic init + random perturbation
            colors = [algebraic_color(i, j) for (i, j) in edges]
            for _ in range(num_edges // 4):
                colors[random.randint(0, num_edges - 1)] = random.randint(0, 4)
            adj_mask = [[0] * n for _ in range(5)]
            for idx, (i, j) in enumerate(edges):
                c = colors[idx]
                adj_mask[c][i] |= (1 << j)
                adj_mask[c][j] |= (1 << i)
            total_violations = compute_total_violations(adj_mask, n)
            violating_edges = get_violating_edges(adj_mask)
            temperature = 1.5
            no_improve = 0
            continue

        # Select edge: 70% from violating, 30% random
        if violating_edges and random.random() < 0.7:
            e_idx = random.choice(list(violating_edges))
        else:
            e_idx = random.randint(0, num_edges - 1)

        i, j = edges[e_idx]
        old_color = colors[e_idx]

        # Try all 5 colors with incremental violation update
        best_new_color = old_color
        best_new_violations = total_violations

        for new_color in range(5):
            if new_color == old_color:
                continue

            # Compute delta in violations
            delta = compute_delta_violations(adj_mask, i, j, old_color, new_color, n)

            if total_violations + delta < best_new_violations:
                best_new_violations = total_violations + delta
                best_new_color = new_color

        if best_new_color != old_color:
            # Apply best change
            adj_mask[old_color][i] &= ~(1 << j)
            adj_mask[old_color][j] &= ~(1 << i)
            adj_mask[best_new_color][i] |= (1 << j)
            adj_mask[best_new_color][j] |= (1 << i)
            colors[e_idx] = best_new_color

            delta = best_new_violations - total_violations
            if delta < 0 or random.random() < math.exp(-delta / max(temperature, 0.01)):
                total_violations = best_new_violations
                if total_violations < best_violations:
                    best_violations = total_violations
                    best_colors = colors[:]
                    no_improve = 0
                    if best_violations == 0:
                        return {'n': n, 'colors': colors}
                else:
                    no_improve += 1
                # Update violating edges (only if we made progress or periodically)
                if best_violations < 100 or no_improve % 200 == 0:
                    violating_edges = get_violating_edges(adj_mask)
            else:
                # Revert
                adj_mask[best_new_color][i] &= ~(1 << j)
                adj_mask[best_new_color][j] &= ~(1 << i)
                adj_mask[old_color][i] |= (1 << j)
                adj_mask[old_color][j] |= (1 << i)
                colors[e_idx] = old_color
                no_improve += 1
        else:
            no_improve += 1

        temperature *= 0.9997
        if temperature < 0.05:
            temperature = 0.05

    return None


def compute_total_violations(adj_mask, n):
    """Compute total number of C4 violations across all colors."""
    total = 0
    for c in range(5):
        for u in range(n):
            for v in range(u + 1, n):
                common = adj_mask[c][u] & adj_mask[c][v]
                k = common.bit_count()
                if k >= 2:
                    total += k * (k - 1) // 2
    return total


def compute_delta_violations(adj_mask, i, j, old_color, new_color, n):
    """
    Compute the change in total C4 violations when edge (i,j) changes from old_color to new_color.
    Only C4s involving edge (i,j) in either old_color or new_color are affected.
    """
    delta = 0

    # Remove edge from old_color: C4s in old_color that used (i,j) disappear
    # A C4 in old_color uses (i,j) if there exist u,v such that
    # (i,u), (j,u), (i,v), (j,v) are all in old_color, and {u,v} is the other pair
    # Actually, the C4s affected are those where (i,j) is one of the 4 edges.
    # In a C4 with edges (i,j), (i,u), (j,u), and one more... 
    # Better: count C4s in old_color subgraph that include edge (i,j).
    # These are C4s of form i-a-j-b-i where a,b are common neighbors of i,j in old_color.
    # Plus C4s where (i,j) is a diagonal? No, in a C4 all 4 edges are present.
    # C4s containing edge (i,j): vertices i, j, a, b where (i,a),(j,a),(i,b),(j,b) in old_color
    # and (a,b) in old_color? No! A C4 is i-a-j-b-i, edges are (i,a),(a,j),(j,b),(b,i).
    # So (i,j) is NOT an edge of this C4. 
    # A C4 containing edge (i,j): the C4 must have (i,j) as one of its 4 edges.
    # C4: i-j-a-b-i. Edges: (i,j),(j,a),(a,b),(b,i).
    # So we need a,b such that (j,a) in old_color, (a,b) in old_color, (b,i) in old_color.
    # And the C4 is i-j-a-b.
    # Count of such C4s = for each a in N_old(j) \ {i}, for each b in N_old(i) \ {j,a},
    #   if (a,b) in old_color: count 1
    # But this is O(n^2) per color. Let's think differently.
    
    # Alternative: the number of C4s in a graph containing a specific edge (i,j)
    # equals the number of paths of length 2 from i to j that don't use edge (i,j),
    # where both intermediate vertices are connected.
    # Path i-a-j: a is common neighbor of i and j.
    # For each common neighbor a of i,j in old_color, the C4 is i-a-j-b-i where b is another
    # common neighbor... no that's not right either.
    
    # Let me reconsider. In the old_color subgraph, the number of C4s containing edge (i,j):
    # A C4 is a 4-cycle. If (i,j) is an edge of the C4, the other two vertices a,b must satisfy:
    # (i,a), (a,j), (j,b), (b,i) are all edges. Wait no: C4 is i-j-a-b-i, so edges are
    # (i,j), (j,a), (a,b), (b,i).
    # So: a must be neighbor of j (not i necessarily, but a != i), b must be neighbor of i (b != j),
    # and (a,b) must be an edge.
    # Count = sum over a in N(j)\{i} of |N(a) ∩ N(i) \ {j, a}|
    # This is still O(n^2) worst case.
    
    # Simpler approach: just compute violations for the two affected colors fully.
    # Since we're changing one edge, only old_color and new_color subgraphs change.
    # Compute C4 count in old_color before, after removing (i,j).
    # Compute C4 count in new_color before, after adding (i,j).
    
    # For efficiency, compute the contribution of edge (i,j) to C4 count in a color c:
    # C4s in color c containing edge (i,j):
    #   = number of pairs (a,b) such that (j,a) in c, (a,b) in c, (b,i) in c, a!=i, b!=j, a!=b
    #   = for each a in N_c(j) \ {i}: count of b in N_c(i) \ {j} with (a,b) in c
    #     and b != a
    # This can be computed as: for each a in N_c(j)\{i}, |N_c(a) ∩ N_c(i) \ {j, a}|
    
    # Let's just do the full recount for the two colors - it's O(n^2) but only for 2 colors
    # and we save the other 3 colors' computation.
    
    old_c4 = 0
    new_c4 = 0
    
    # Count C4s in old_color subgraph (before change) involving vertices i or j
    # Actually, only C4s that include edge (i,j) will change.
    # Let me just count all C4s in old_color that include at least one of edges incident to i or j... 
    # No, the cleanest is: count C4s in old_color that use edge (i,j).
    
    # C4s using edge (i,j) in color c:
    # Pattern: i-j-a-b-i where (j,a),(a,b),(b,i) in c
    # = sum_{a in N_c(j)\{i}} |{b in N_c(i)\{j} : (a,b) in c and b != a}|
    
    ni_old = adj_mask[old_color][i]
    nj_old = adj_mask[old_color][j]
    
    # For old_color: C4s containing (i,j)
    # a must be in nj_old \ {i}, b must be in ni_old \ {j}, and (a,b) in old_color, a != b
    c4_in_old = 0
    for a in range(n):
        if a == i or a == j:
            continue
        if not (nj_old >> a) & 1:
            continue
        # a is neighbor of j in old_color
        # Count b in ni_old \ {j, a} such that (a,b) in old_color
        na_old = adj_mask[old_color][a]
        candidates = ni_old & ~((1 << j) | (1 << a))
        c4_in_old += (na_old & candidates).bit_count()
    
    # For new_color: C4s containing (i,j) after adding it
    ni_new = adj_mask[new_color][i]
    nj_new = adj_mask[new_color][j]
    
    c4_in_new = 0
    for a in range(n):
        if a == i or a == j:
            continue
        if not (nj_new >> a) & 1:
            continue
        na_new = adj_mask[new_color][a]
        candidates = ni_new & ~((1 << j) | (1 << a))
        c4_in_new += (na_new & candidates).bit_count()
    
    # Also need to count C4s where (i,j) is NOT an edge but i or j is a vertex
    # that becomes part of a new C4 in new_color or loses a C4 in old_color.
    # When we remove (i,j) from old_color, we lose C4s that had (i,j) as an edge: c4_in_old
    # When we add (i,j) to new_color, we gain C4s that have (i,j) as an edge: c4_in_new
    # But also: removing (i,j) from old_color might break C4s where (i,j) was a "chord"?
    # No, in a simple graph C4, all 4 edges are sides. (i,j) is either a side or not involved.
    # If (i,j) is not a side of any C4 in old_color, removing it doesn't affect C4 count.
    # If (i,j) IS a side of some C4s, those C4s disappear.
    # Similarly for new_color.
    
    # But wait: there's another type of C4 affected. Consider C4 i-a-j-b-i.
    # Here (i,j) is a diagonal, not an edge. Removing (i,j) doesn't affect this C4.
    # So only C4s where (i,j) is an actual edge are affected.
    
    delta = -c4_in_old + c4_in_new
    return delta


def focused_local_search(n, time_limit=25.0):
    """Local search with incremental violation updates."""
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

    colors = [random.randint(0, 4) for _ in range(num_edges)]

    adj_mask = [[0] * n for _ in range(5)]
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        adj_mask[c][i] |= (1 << j)
        adj_mask[c][j] |= (1 << i)

    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def get_violating_edges(adj_mask):
        violating = set()
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj_mask[c][u] & adj_mask[c][v]
                    if common.bit_count() >= 2:
                        violating.add(edge_idx(u, v))
                        temp = common
                        while temp:
                            w = (temp & -temp).bit_length() - 1
                            violating.add(edge_idx(min(u, w), max(u, w)))
                            violating.add(edge_idx(min(v, w), max(v, w)))
                            temp &= temp - 1
        return violating

    total_violations = compute_total_violations(adj_mask, n)

    if total_violations == 0:
        return {'n': n, 'colors': colors}

    violating_edges = get_violating_edges(adj_mask)
    best_violations = total_violations
    best_colors = colors[:]
    start_time = time.time()
    temperature = 1.5
    no_improve = 0
    max_no_improve = 5000

    while time.time() - start_time < time_limit:
        if no_improve > max_no_improve:
            colors = [random.randint(0, 4) for _ in range(num_edges)]
            adj_mask = [[0] * n for _ in range(5)]
            for idx, (i, j) in enumerate(edges):
                c = colors[idx]
                adj_mask[c][i] |= (1 << j)
                adj_mask[c][j] |= (1 << i)
            total_violations = compute_total_violations(adj_mask, n)
            violating_edges = get_violating_edges(adj_mask)
            temperature = 1.5
            no_improve = 0
            continue

        if violating_edges and random.random() < 0.7:
            e_idx = random.choice(list(violating_edges))
        else:
            e_idx = random.randint(0, num_edges - 1)

        i, j = edges[e_idx]
        old_color = colors[e_idx]

        best_new_color = old_color
        best_new_violations = total_violations

        for new_color in range(5):
            if new_color == old_color:
                continue
            delta = compute_delta_violations(adj_mask, i, j, old_color, new_color, n)
            if total_violations + delta < best_new_violations:
                best_new_violations = total_violations + delta
                best_new_color = new_color

        if best_new_color != old_color:
            adj_mask[old_color][i] &= ~(1 << j)
            adj_mask[old_color][j] &= ~(1 << i)
            adj_mask[best_new_color][i] |= (1 << j)
            adj_mask[best_new_color][j] |= (1 << i)
            colors[e_idx] = best_new_color

            delta = best_new_violations - total_violations
            if delta < 0 or random.random() < math.exp(-delta / max(temperature, 0.01)):
                total_violations = best_new_violations
                if total_violations < best_violations:
                    best_violations = total_violations
                    best_colors = colors[:]
                    no_improve = 0
                    if best_violations == 0:
                        return {'n': n, 'colors': colors}
                else:
                    no_improve += 1
                if best_violations < 100 or no_improve % 200 == 0:
                    violating_edges = get_violating_edges(adj_mask)
            else:
                adj_mask[best_new_color][i] &= ~(1 << j)
                adj_mask[best_new_color][j] &= ~(1 << i)
                adj_mask[old_color][i] |= (1 << j)
                adj_mask[old_color][j] |= (1 << i)
                colors[e_idx] = old_color
                no_improve += 1
        else:
            no_improve += 1

        temperature *= 0.9997
        if temperature < 0.05:
            temperature = 0.05

    return None


def try_extend_focused(base_result, time_limit=40.0):
    """Extend a valid n-vertex coloring to n+1 vertices using focused search."""
    base_colors = base_result['colors']
    n_base = base_result['n']
    n_new = n_base + 1

    adj_mask = [[0] * n_base for _ in range(5)]
    idx = 0
    for i in range(n_base):
        for j in range(i + 1, n_base):
            c = base_colors[idx]
            adj_mask[c][i] |= (1 << j)
            adj_mask[c][j] |= (1 << i)
            idx += 1

    def count_violations_new(new_colors):
        violations = 0
        for c in range(5):
            neighbors = [i for i in range(n_base) if new_colors[i] == c]
            for a in range(len(neighbors)):
                u = neighbors[a]
                for b in range(a + 1, len(neighbors)):
                    v = neighbors[b]
                    common = adj_mask[c][u] & adj_mask[c][v]
                    k = common.bit_count()
                    if k >= 1:
                        violations += k
        return violations

    new_colors = [random.randint(0, 4) for _ in range(n_base)]
    current_violations = count_violations_new(new_colors)

    if current_violations == 0:
        return build_full_coloring(base_colors, new_colors, n_base)

    best_violations = current_violations
    best_new_colors = new_colors[:]

    start_time = time.time()
    temperature = 1.0
    no_improve = 0

    while time.time() - start_time < time_limit:
        if best_violations == 0:
            break
        if no_improve > 2000:
            new_colors = [random.randint(0, 4) for _ in range(n_base)]
            current_violations = count_violations_new(new_colors)
            temperature = 1.0
            no_improve = 0
            continue

        e_idx = random.randint(0, n_base - 1)
        old_c = new_colors[e_idx]
        new_c = random.randint(0, 4)
        if new_c == old_c:
            continue

        new_colors[e_idx] = new_c
        new_violations = count_violations_new(new_colors)

        delta = new_violations - current_violations
        if delta < 0 or random.random() < math.exp(-delta / max(temperature, 0.01)):
            current_violations = new_violations
            if current_violations < best_violations:
                best_violations = current_violations
                best_new_colors = new_colors[:]
                no_improve = 0
                if best_violations == 0:
                    break
            else:
                no_improve += 1
        else:
            new_colors[e_idx] = old_c
            no_improve += 1

        temperature *= 0.999
        if temperature < 0.05:
            temperature = 0.05

    if best_violations == 0:
        return build_full_coloring(base_colors, best_new_colors, n_base)

    return None


def build_full_coloring(base_colors, new_colors, n_base):
    """Build the full (n_base+1)-vertex coloring from base and new vertex colors."""
    n_new = n_base + 1
    final_colors = [0] * (n_new * (n_new - 1) // 2)

    idx = 0
    for i in range(n_new):
        for j in range(i + 1, n_new):
            if j < n_base:
                base_idx = i * (n_base - 1) - i * (i - 1) // 2 + (j - i - 1)
                final_colors[idx] = base_colors[base_idx]
            else:
                final_colors[idx] = new_colors[i]
            idx += 1

    return {'n': n_new, 'colors': final_colors}
