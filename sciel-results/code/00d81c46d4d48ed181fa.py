import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s kill

    # --- Paley coloring for prime p ---
    def generate_paley(n, prime):
        res = set()
        for x in range(1, prime):
            res.add((x * x) % prime)
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % prime
                colors.append(0 if diff in res else 1)
        return colors

    # --- Build adjacency bitmasks ---
    def build_adj(colors, n):
        adj = [[0] * n for _ in range(2)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        return adj

    # --- Fast K2,2,2 violation count using bitmasks ---
    def count_violations(adj, n):
        total = 0
        for c in (0, 1):
            A = adj[c]
            for u1 in range(n):
                for u2 in range(u1 + 1, n):
                    S = A[u1] & A[u2]
                    # Need at least 4 common neighbors to have K2,2,2
                    # Count bits quickly
                    if S == 0:
                        continue
                    # Check if S has at least 4 bits
                    s1 = S
                    s2 = S & (S - 1)
                    s3 = s2 & (s2 - 1)
                    s4 = s3 & (s3 - 1)
                    if s4 == 0:
                        continue
                    # Now check: among common neighbors, do 2 pairs share >=2 common neighbors in S?
                    # Extract up to first 20 bits for speed
                    bits = []
                    temp = S
                    while temp and len(bits) < 25:
                        lsb = temp & (-temp)
                        bits.append(lsb.bit_length() - 1)
                        temp ^= lsb
                    k = len(bits)
                    if k < 4:
                        continue
                    # For each pair (v1,v2) in bits, check if they share >=2 neighbors in S
                    found = False
                    for a in range(k):
                        v1 = bits[a]
                        C1 = A[v1] & S
                        for b in range(a + 1, k):
                            v2 = bits[b]
                            C2 = A[v2] & S
                            common = C1 & C2
                            # Check if common has at least 2 bits
                            if common and (common & (common - 1)):
                                found = True
                                break
                        if found:
                            break
                    if found:
                        total += 1
        return total

    def is_valid(colors, n):
        adj = build_adj(colors, n)
        return count_violations(adj, n) == 0

    # --- Simulated Annealing with incremental updates ---
    def simulated_annealing(colors, n, time_budget):
        adj = build_adj(colors, n)
        current_viol = count_violations(adj, n)
        best_colors = colors[:]
        best_viol = current_viol

        num_edges = n * (n - 1) // 2
        edge_map = [(i, j) for i in range(n) for j in range(i + 1, n)]

        temp = 2.0
        cooling = 0.99995
        iters = 0
        stall = 0
        no_improve = 0

        while time.time() < time_budget:
            iters += 1
            ei = random.randint(0, num_edges - 1)
            i, j = edge_map[ei]
            old = colors[ei]
            new = 1 - old

            # Flip
            adj[old][i] &= ~(1 << j)
            adj[old][j] &= ~(1 << i)
            adj[new][i] |= (1 << j)
            adj[new][j] |= (1 << i)
            colors[ei] = new

            nv = count_violations(adj, n)
            delta = nv - current_viol

            if delta <= 0 or random.random() < pow(2.718281828, -delta / temp):
                current_viol = nv
                if nv < best_viol:
                    best_viol = nv
                    best_colors = colors[:]
                    stall = 0
                    no_improve = 0
                    if nv == 0:
                        return best_colors
                else:
                    no_improve += 1
            else:
                # Revert
                adj[new][i] &= ~(1 << j)
                adj[new][j] &= ~(1 << i)
                adj[old][i] |= (1 << j)
                adj[old][j] |= (1 << i)
                colors[ei] = old
                stall += 1
                no_improve += 1

            temp *= cooling

            # Perturbation on stall
            if stall > 5000:
                nflip = random.randint(3, 8)
                saved = []
                for _ in range(nflip):
                    e1 = random.randint(0, num_edges - 1)
                    i1, j1 = edge_map[e1]
                    o1 = colors[e1]
                    n1 = 1 - o1
                    adj[o1][i1] &= ~(1 << j1)
                    adj[o1][j1] &= ~(1 << i1)
                    adj[n1][i1] |= (1 << j1)
                    adj[n1][j1] |= (1 << i1)
                    colors[e1] = n1
                    saved.append((e1, i1, j1, o1, n1))
                nv2 = count_violations(adj, n)
                if nv2 < current_viol:
                    current_viol = nv2
                    if nv2 < best_viol:
                        best_viol = nv2
                        best_colors = colors[:]
                        no_improve = 0
                else:
                    for (e1, i1, j1, o1, n1) in saved:
                        adj[n1][i1] &= ~(1 << j1)
                        adj[n1][j1] &= ~(1 << i1)
                        adj[o1][i1] |= (1 << j1)
                        adj[o1][j1] |= (1 << i1)
                        colors[e1] = o1
                stall = 0
                temp = 3.0

            if no_improve > 30000:
                # Larger perturbation
                nflip = random.randint(8, 15)
                saved = []
                for _ in range(nflip):
                    e1 = random.randint(0, num_edges - 1)
                    i1, j1 = edge_map[e1]
                    o1 = colors[e1]
                    n1 = 1 - o1
                    adj[o1][i1] &= ~(1 << j1)
                    adj[o1][j1] &= ~(1 << i1)
                    adj[n1][i1] |= (1 << j1)
                    adj[n1][j1] |= (1 << i1)
                    colors[e1] = n1
                    saved.append((e1, i1, j1, o1, n1))
                nv2 = count_violations(adj, n)
                if nv2 < current_viol:
                    current_viol = nv2
                    if nv2 < best_viol:
                        best_viol = nv2
                        best_colors = colors[:]
                        no_improve = 0
                else:
                    for (e1, i1, j1, o1, n1) in saved:
                        adj[n1][i1] &= ~(1 << j1)
                        adj[n1][j1] &= ~(1 << i1)
                        adj[o1][i1] |= (1 << j1)
                        adj[o1][j1] |= (1 << i1)
                        colors[e1] = o1
                no_improve = 0
                temp = 4.0

        return best_colors

    # --- Generate n=29 Paley coloring ---
    n29 = 29
    colors29 = generate_paley(n29, 29)

    # Verify n=29
    if not is_valid(colors29, n29):
        # Fallback: smaller valid coloring
        for pn in (17, 13, 11):
            c = generate_paley(pn, pn)
            if is_valid(c, pn):
                return {'n': pn, 'colors': c}
        return {'n': 1, 'colors': []}

    # --- Try to extend to n=30 using structural approach ---
    # Strategy: For each vertex v in 0..28, the edges from v to new vertex 29
    # must be chosen so that neither color class gets a K2,2,2.
    # 
    # Key structural insight: A K2,2,2 in color c involving vertex 29 requires:
    #   - 29 has at least 2 neighbors in color c (call them a, b)
    #   - a and b share at least 2 common neighbors in color c among {0..28}
    #   - those common neighbors share at least 2 more common neighbors in color c
    #
    # So for each pair (a,b) that are both color-c neighbors of 29,
    # we need |N_c(a) ∩ N_c(b) ∩ {0..28}| to be such that no K2,2,2 forms.
    #
    # Better approach: enumerate the 2^29 possible colorings of edges from 29,
    # but that's infeasible. Instead, use constraint propagation + SA.

    # Precompute for the n=29 graph: for each color c and each pair (a,b),
    # the set of common neighbors in color c.
    adj29 = build_adj(colors29, 29)

    # For each color c, for each vertex v, degree in color c
    deg = [[0]*29 for _ in range(2)]
    for c in (0,1):
        for v in range(29):
            deg[c][v] = bin(adj29[c][v]).count('1')

    # Try multiple initializations for vertex 29's edges
    best_30 = None
    best_30_viol = float('inf')

    # Time allocation: use remaining time for multiple SA runs
    remaining = deadline - time.time() - 3
    if remaining > 5:
        # Several diverse initializations
        inits = []
        
        # Init 0: all 0s
        inits.append([0]*29)
        # Init 1: all 1s
        inits.append([1]*29)
        # Init 2: based on degree - connect to high-degree vertices with 0
        sorted_by_deg0 = sorted(range(29), key=lambda v: -deg[0][v])
        inits.append([0 if v in sorted_by_deg0[:14] else 1 for v in range(29)])
        # Init 3: opposite
        inits.append([1 if v in sorted_by_deg0[:14] else 0 for v in range(29)])
        # Init 4: random
        inits.append([random.randint(0,1) for _ in range(29)])
        # Init 5: based on adjacency to vertex 0
        inits.append([1 if (adj29[0][0] >> v) & 1 else 0 for v in range(29)])
        # Init 6: based on adjacency to vertex 1
        inits.append([1 if (adj29[1][1] >> v) & 1 else 0 for v in range(29)])
        # Init 7: alternating
        inits.append([v % 2 for v in range(29)])
        # Init 8: mod 3
        inits.append([v % 3 < 2 for v in range(29)])
        # Init 9: quadratic residues mod 29
        qr = set((x*x) % 29 for x in range(1, 29))
        inits.append([0 if v in qr else 1 for v in range(29)])
        # Init 10-14: random
        for _ in range(5):
            inits.append([random.randint(0,1) for _ in range(29)])

        per_init_time = min(remaining / len(inits), 15)
        
        for init in inits:
            if time.time() > deadline - 3:
                break
            colors30 = colors29[:] + init
            result = simulated_annealing(colors30, 30, min(time.time() + per_init_time, deadline - 3))
            adj30 = build_adj(result, 30)
            viol = count_violations(adj30, 30)
            if viol < best_30_viol:
                best_30_viol = viol
                best_30 = result
            if viol == 0:
                return {'n': 30, 'colors': result}

        # If we found a low-violation coloring but not zero, try one more focused SA
        if best_30 is not None and best_30_viol > 0 and time.time() < deadline - 5:
            result = simulated_annealing(best_30[:], 30, deadline - 3)
            adj30 = build_adj(result, 30)
            viol = count_violations(adj30, 30)
            if viol < best_30_viol:
                best_30_viol = viol
                best_30 = result
            if viol == 0:
                return {'n': 30, 'colors': result}

    # Return n=29
    return {'n': 29, 'colors': colors29}
