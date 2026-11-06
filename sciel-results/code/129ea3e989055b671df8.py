"""Fresh design: start from a valid 32-vertex 4-colouring (Clebsch-based),
then grow the graph via local search (simulated annealing) to reach n=50.
Uses bitsets for fast monochromatic-triangle counting."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    deadline = time.time() + 140  # safety margin under 150s kill

    # ---- bitset helpers ----
    # For each colour c and vertex i, bitset[c][i] has bit j set iff edge (i,j) has colour c.
    # A monochromatic triangle of colour c through vertex i is counted by popcount(AND over j in bitset[c][i]).

    def init_bitsets(colors, n):
        bitsets = [0] * (4 * n)  # index c*n + i
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[i * n - i * (i + 1) // 2 + j - i - 1]
                bitsets[c * n + i] |= (1 << j)
                bitsets[c * n + j] |= (1 << i)
        return bitsets

    def popcount(x):
        return bin(x).count('1')

    # ---- cost: total number of monochromatic triangles ----
    def count_mono_triangles(bitsets, n):
        total = 0
        for c in range(4):
            base = c * n
            for i in range(n):
                bits = bitsets[base + i]
                # count triangles: for each pair j<k both in bits, check if edge (j,k) is colour c
                # faster: popcount of intersection of bitsets[base+j] and bitsets[base+k] summed, but that's O(n^2)
                # Use: number of triangles through i = sum over j in bits of popcount(bits & bitsets[base+j]) / 2
                # Actually simpler: iterate j in bits, count common neighbors
                # For speed, use: for each j where bit is set, count bits & bitsets[base+j]
                # But that's expensive. Let's use a different approach.
                # Count = (1/3) * sum over all vertices of (number of mono triangles through that vertex)
                # Number of mono triangles of colour c through vertex i:
                # = for each j where edge(i,j) is colour c, count k where both (i,k) and (j,k) are colour c
                # = sum_{j in N_c(i)} |N_c(i) & N_c(j)|  ... but this double counts
                # Actually: triangles through i of colour c = (1/2) * sum_{j in N_c(i)} popcount(N_c(i) & N_c(j))
                # No wait, that's not right either. Let me think again.
                # A triangle (i,j,k) all colour c is counted once for each vertex.
                # Through vertex i: count pairs (j,k) in N_c(i) such that edge(j,k) is also colour c.
                # That's: sum over j in N_c(i) of |N_c(i) & N_c(j)| but this counts each triangle twice (once for j, once for k).
                # So triangles through i = (1/2) * sum_{j in N_c(i)} popcount(N_c(i) & N_c(j))
                # But computing this is O(degree * word_ops) which is fine for n<=62.
                pass
        # Simpler but still fast: brute force over triples for small n, or use the per-vertex formula.
        # For n up to 50-62, brute force over all triples is O(n^3) = ~125k-238k, which is fast.
        # But we call this many times in SA, so we need incremental updates.
        # Let's use the per-vertex formula with bitsets.
        total = 0
        for c in range(4):
            base = c * n
            for i in range(n):
                ni = bitsets[base + i]
                # count pairs (j,k) in ni such that edge(j,k) is colour c
                # = (1/2) * sum_{j: bit j in ni} popcount(ni & bitsets[base+j])
                # We only need j < k implicitly handled by /2
                s = 0
                temp = ni
                while temp:
                    lsb = temp & (-temp)
                    j = lsb.bit_length() - 1
                    s += popcount(ni & bitsets[base + j])
                    temp ^= lsb
                total += s // 2
        return total

    # Better: incremental cost update when changing edge (i,j) from old_c to new_c
    # Removing edge (i,j) of colour old_c: decreases mono triangles that use this edge.
    # A mono triangle using edge (i,j) with colour old_c is a vertex k such that (i,k) and (j,k) are both old_c.
    # Count = popcount(bitsets[old_c*n+i] & bitsets[old_c*n+j])
    # Adding edge (i,j) of colour new_c: increases by popcount(bitsets[new_c*n+i] & bitsets[new_c*n+j])

    def edge_triangle_count(bitsets, i, j, c):
        """Number of mono triangles of colour c that include edge (i,j)."""
        return popcount(bitsets[c * n + i] & bitsets[c * n + j])

    def remove_edge(bitsets, i, j, c):
        bitsets[c * n + i] &= ~(1 << j)
        bitsets[c * n + j] &= ~(1 << i)

    def add_edge(bitsets, i, j, c):
        bitsets[c * n + i] |= (1 << j)
        bitsets[c * n + j] |= (1 << i)

    # ---- initial 32-vertex construction (Clebsch-based) ----
    def make_initial():
        def mul(a, b):
            r = 0
            for i in range(4):
                if b >> i & 1:
                    r ^= a << i
            for i in range(7, 3, -1):
                if r >> i & 1:
                    r ^= 0b10011 << (i - 4)
            return r
        log, x = {}, 1
        for e in range(15):
            log[x] = e
            x = mul(x, 2)
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return colors

    colors = make_initial()
    n = 32
    bitsets = init_bitsets(colors, n)
    cost = count_mono_triangles(bitsets, n)
    assert cost == 0, f"Initial construction has {cost} mono triangles"

    best_n = n
    best_colors = list(colors)
    best_bitsets = [b for b in bitsets]
    best_cost = cost

    # ---- local search: try to add vertices and recolor edges ----
    # Strategy: 
    # 1. Try adding a new vertex connected to all existing vertices with some colour assignment.
    # 2. If that creates mono triangles, use SA to fix them by recoloring edges.
    # 3. Repeat until we can't grow further or time runs out.

    def try_add_vertex(n_current, colors_current, bitsets_current, cost_current):
        """Try to add one vertex. Returns (success, new_n, new_colors, new_bitsets, new_cost)."""
        # For the new vertex (index n_current), assign colours to edges (i, n_current) for i in 0..n_current-1.
        # We want to choose colours to minimize new mono triangles.
        # A new mono triangle involving the new vertex: for each pair (i,j) where i,j < n_current,
        # if edge(i,j) has colour c, and we assign colour c to both (i, new) and (j, new), that's a mono triangle.
        # So for each colour c, the number of new mono triangles = number of pairs (i,j) with edge(i,j)=c
        #   such that we assign c to both (i,new) and (j,new).
        # This is like: for each colour c, if S_c = {i : edge(i,new) = c}, then triangles += number of edges of colour c within S_c.
        
        # Greedy: try all 4^n_current assignments? Too many for n=32.
        # Instead: random assignment, then local optimize.
        # Or: for each vertex i, pick the colour that minimizes marginal triangle increase.
        
        # Let's do: random initial assignment, then hill-climb on the new vertex's edge colours.
        best_assign = None
        best_new_cost = float('inf')
        
        for trial in range(50):
            assign = [random.randint(0, 3) for _ in range(n_current)]
            # Compute new cost
            new_cost = cost_current
            for c in range(4):
                S = [i for i in range(n_current) if assign[i] == c]
                # count edges of colour c within S
                for idx_a in range(len(S)):
                    for idx_b in range(idx_a + 1, len(S)):
                        a, b = S[idx_a], S[idx_b]
                        # check if edge (a,b) is colour c
                        if bitsets_current[c * n_current + a] & (1 << b):
                            new_cost += 1
            # Also need to check: existing mono triangles are unchanged (they don't involve new vertex)
            # So new_cost = cost_current + (mono triangles involving new vertex)
            
            if new_cost < best_new_cost:
                best_new_cost = new_cost
                best_assign = list(assign)
        
        if best_new_cost > cost_current + 5:  # too many new triangles, skip
            return False, n_current, colors_current, bitsets_current, cost_current
        
        # Now build new bitsets and colors
        new_n = n_current + 1
        new_bitsets = [0] * (4 * new_n)
        # Copy old edges
        for c in range(4):
            for i in range(n_current):
                new_bitsets[c * new_n + i] = bitsets_current[c * n_current + i]
        # Add new vertex edges
        for i in range(n_current):
            c = best_assign[i]
            new_bitsets[c * new_n + i] |= (1 << n_current)
            new_bitsets[c * new_n + n_current] |= (1 << i)
        
        # Build new colors list
        new_colors = list(colors_current)
        for i in range(n_current):
            new_colors.append(best_assign[i])
        
        new_cost = best_new_cost
        return True, new_n, new_colors, new_bitsets, new_cost

    def sa_fix(colors, bitsets, n, cost, max_iters, deadline):
        """Simulated annealing to reduce mono triangles. Returns (colors, bitsets, cost)."""
        T = 5.0
        T_min = 0.01
        cooling = 0.999
        current_cost = cost
        current_colors = list(colors)
        current_bitsets = list(bitsets)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if current_cost == 0:
                break
            # Pick random edge (i,j)
            i = random.randint(0, n - 1)
            j = random.randint(i + 1, n - 1)
            old_c = current_colors[i * n - i * (i + 1) // 2 + j - i - 1]
            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue
            
            # Compute cost delta
            # Remove edge (i,j) of colour old_c: decreases triangles by edge_triangle_count
            delta = -edge_triangle_count(current_bitsets, i, j, old_c)
            # Add edge (i,j) of colour new_c: increases triangles by edge_triangle_count
            delta += edge_triangle_count(current_bitsets, i, j, new_c)
            
            if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                # Accept
                remove_edge(current_bitsets, i, j, old_c)
                add_edge(current_bitsets, i, j, new_c)
                current_colors[i * n - i * (i + 1) // 2 + j - i - 1] = new_c
                current_cost += delta
            T *= cooling
            if T < T_min:
                T = T_min
        
        return current_colors, current_bitsets, current_cost

    # Main loop: grow and fix
    attempts = 0
    while time.time() < deadline:
        # Try to add a vertex
        success, n2, colors2, bitsets2, cost2 = try_add_vertex(n, colors, bitsets, cost)
        if success and n2 > best_n:
            n, colors, bitsets, cost = n2, colors2, bitsets2, cost2
            # Now try to fix triangles with SA
            colors, bitsets, cost = sa_fix(colors, bitsets, n, cost, 20000, deadline)
            if cost < best_cost or (cost == 0 and n > best_n):
                best_n = n
                best_colors = list(colors)
                best_bitsets = list(bitsets)
                best_cost = cost
            if cost == 0:
                # Great! We have a valid n-vertex colouring. Try to grow more.
                continue
            else:
                # Try more SA
                colors, bitsets, cost = sa_fix(colors, bitsets, n, cost, 50000, deadline)
                if cost < best_cost or (cost == 0 and n > best_n):
                    best_n = n
                    best_colors = list(colors)
                    best_bitsets = list(bitsets)
                    best_cost = cost
        else:
            # Couldn't add vertex easily, try SA on current to reduce cost
            if cost > 0:
                colors, bitsets, cost = sa_fix(colors, bitsets, n, cost, 50000, deadline)
                if cost < best_cost or (cost == 0 and n > best_n):
                    best_n = n
                    best_colors = list(colors)
                    best_bitsets = list(bitsets)
                    best_cost = cost
            # If we have a valid colouring, try adding again
            if cost == 0 and n >= best_n:
                attempts += 1
                if attempts > 20:
                    break
            else:
                # Stuck, try a different approach: random restart with smaller n
                if n > 32:
                    # Roll back to best
                    n, colors, bitsets, cost = best_n, list(best_colors), list(best_bitsets), best_cost
                    attempts += 1
                    if attempts > 30:
                        break
                else:
                    attempts += 1
                    if attempts > 10:
                        break

    # Return best result
    if best_cost == 0:
        return {'n': best_n, 'colors': best_colors}
    else:
        # Fallback: return the 32-vertex construction
        return {'n': 32, 'colors': make_initial()}
# EVOLVE-BLOCK-END
