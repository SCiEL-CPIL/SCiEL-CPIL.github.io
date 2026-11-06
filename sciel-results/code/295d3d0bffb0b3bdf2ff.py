import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    N = 512
    LIMIT = 512

    # 1. Precompute LCM compatibility using bitsets
    # neighbors[i] is a bitmask where bit j is set if lcm(i, j+1) <= LIMIT
    neighbors = [0] * (N + 1)

    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            g = gcd(i, j)
            if (i // g) * j <= LIMIT:
                mask |= (1 << (j - 1))
        neighbors[i] = mask

    initial_P = (1 << N) - 1

    # 2. Greedy coloring for upper bound (Tomita's algorithm style)
    # We need a function that colors the subgraph induced by P.
    # Colors are assigned such that vertices of the same color form an independent set.
    # The number of colors is an upper bound on the clique size.
    # We also want the vertices sorted by color for branching.

    def color_sort(P):
        """
        Returns a list of vertices in P, sorted by greedy color.
        The color of a vertex is the number of colors used so far.
        Vertices with lower colors come first.
        This allows us to prune: if len(R) + color(v) <= best_size, we can stop.
        """
        if P == 0:
            return []

        colors = [0] * (N + 1)  # 1-indexed
        color_classes = []  # List of lists of vertices
        uncolored = P

        while uncolored:
            color_classes.append([])
            independent = 0
            temp = uncolored
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length()  # 1-based index
                # Check if v is compatible with any vertex in current independent set
                # If not, it can join the current color class
                if (independent & neighbors[v]) == 0:
                    independent |= lsb
                    uncolored ^= lsb
                    color_classes[-1].append(v)
                temp ^= lsb

        # Flatten and reverse so that higher colors (fewer vertices remaining) are processed first?
        # Actually, standard Tomita processes vertices in order of increasing color.
        # The bound is: if |R| + color_of_last_vertex_in_P <= best, prune.
        # Since we assign colors 0, 1, ..., the max color is len(color_classes)-1.
        # We return the list of vertices in the order they were colored (increasing color).
        ordered = []
        for cls in color_classes:
            ordered.extend(cls)
        return ordered

    # 3. Maximum Clique Algorithm (Bron-Kerbosch with Tomita's coloring bound)

    best_clique = []

    def bron_kerbosch(R, P, X):
        nonlocal best_clique

        if time.time() > deadline:
            return

        if P == 0 and X == 0:
            if len(R) > len(best_clique):
                best_clique = R[:]
            return

        # Pruning: if R + all of P cannot beat best, return
        if len(R) + P.bit_count() <= len(best_clique):
            return

        # Get ordered vertices and their colors
        # We need to know the color of each vertex to prune effectively.
        # Let's modify color_sort to return (vertex, color) pairs or just vertices and we track colors.
        # For simplicity and speed, let's re-implement the coloring to return the max color bound directly
        # and the list of vertices sorted by color.

        if P == 0:
            return

        # Perform greedy coloring on P
        # We want to assign colors such that we can prune.
        # Standard approach:
        # 1. Color vertices in P greedily.
        # 2. Sort vertices by color.
        # 3. Iterate through sorted vertices. If |R| + color(v) <= best, break.

        # Efficient coloring
        uncolored = P
        color_list = []  # List of (vertex, color)
        current_color = 0
        independent = 0

        # To speed up, we can use a simple greedy coloring that produces a valid coloring.
        # The number of colors is an upper bound on clique size.

        # Let's use a slightly different approach for speed:
        # Iterate through vertices in P.
        # Assign the smallest available color.
        # This is O(N^2) per call, which might be slow.
        # Tomita's algorithm uses a specific coloring that is efficient.

        # Alternative: Just use the number of colors as the bound for the whole P.
        # But we need per-vertex pruning.

        # Let's try a fast approximation of coloring for pruning.
        # We'll compute the greedy coloring of P.

        colors = {}
        uncolored = P
        while uncolored:
            independent = 0
            temp = uncolored
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length()
                if (independent & neighbors[v]) == 0:
                    independent |= lsb
                    uncolored ^= lsb
                    colors[v] = current_color
                temp ^= lsb
            current_color += 1

        if len(R) + current_color <= len(best_clique):
            return

        # Sort vertices by color
        sorted_vertices = sorted(colors.keys(), key=lambda x: colors[x])

        # Pivot selection
        # Choose pivot u in P U X that maximizes |P ∩ N(u)|
        PX = P | X
        max_overlap = -1
        pivot = -1
        temp = PX
        while temp:
            lsb = temp & -temp
            u = lsb.bit_length()
            overlap = P & neighbors[u]
            count = overlap.bit_count()
            if count > max_overlap:
                max_overlap = count
                pivot = u
            temp ^= lsb

        if pivot == -1:
            return

        candidates = P & ~neighbors[pivot]

        temp = candidates
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length()
            temp ^= lsb

            # Prune based on color of v
            if len(R) + 1 + colors[v] <= len(best_clique):
                # Since vertices are processed in increasing color order?
                # Wait, the loop above iterates over candidates in arbitrary order (bit order).
                # We should iterate in sorted order to use the color bound effectively.
                pass

            # Actually, the standard Tomita algorithm iterates through the sorted list.
            # Let's restart the loop with the sorted list.

        # Restart loop with sorted vertices
        for v in sorted_vertices:
            if len(R) + 1 + colors[v] <= len(best_clique):
                break

            if v not in candidates:
                continue

            bron_kerbosch(R + [v], P & neighbors[v], X & neighbors[v])

            P &= ~(1 << (v - 1))
            X |= (1 << (v - 1))

            if time.time() > deadline:
                return

    # Initial lower bound via greedy
    def greedy_clique():
        P = initial_P
        R = []
        while P:
            best_v = -1
            max_deg = -1
            temp = P
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length()
                deg = (P & neighbors[v]).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                temp ^= lsb
            if best_v == -1:
                break
            R.append(best_v)
            P &= ~(1 << (best_v - 1))
            P &= neighbors[best_v]
        return R

    best_clique = greedy_clique()

    # Run Maximum Clique
    bron_kerbosch([], initial_P, 0)

    return best_clique
