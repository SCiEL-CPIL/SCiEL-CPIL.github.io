import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit

    def count_triangles_vec(mat):
        """Vectorized triangle count."""
        n = mat.shape[0]
        total = 0
        for c in range(4):
            M = (mat == c).astype(np.int32)
            np.fill_diagonal(M, 0)
            M2 = M @ M
            total += int(np.trace(M2 @ M)) // 6
        return total

    def find_mono_triangle_fast(mat):
        """Find a monochromatic triangle using vectorized approach."""
        n = mat.shape[0]
        for _ in range(30):
            i, j, k = random.sample(range(n), 3)
            if mat[i][j] == mat[i][k] == mat[j][k]:
                return (i, j, k)
        for c in range(4):
            M = (mat == c)
            for i in range(n):
                for j in range(i+1, n):
                    if M[i][j]:
                        common = np.where(M[i] & M[j])[0]
                        for k in common:
                            if k > j:
                                return (i, j, k)
        return None

    def local_search_sa(mat, n, time_budget):
        """Simulated annealing to eliminate monochromatic triangles."""
        current_cost = count_triangles_vec(mat)
        if current_cost == 0:
            return mat, True

        best_mat = mat.copy()
        best_cost = current_cost

        T = 2.0
        cooling = 0.99995
        start = time.time()

        while time.time() - start < time_budget:
            if current_cost == 0:
                return mat, True

            tri = find_mono_triangle_fast(mat)
            if tri:
                i, j, k = tri
                edge_idx = random.randint(0, 2)
                if edge_idx == 0:
                    u, v = i, j
                elif edge_idx == 1:
                    u, v = i, k
                else:
                    u, v = j, k
            else:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v:
                    continue

            old_color = mat[u][v]
            new_color = random.randint(0, 3)
            if new_color == old_color:
                continue

            delta = 0
            for w in range(n):
                if w == u or w == v:
                    continue
                if mat[u][w] == old_color and mat[v][w] == old_color:
                    delta -= 1
                if mat[u][w] == new_color and mat[v][w] == new_color:
                    delta += 1

            new_cost = current_cost + delta

            if new_cost < current_cost:
                mat[u][v] = new_color
                mat[v][u] = new_color
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_mat = mat.copy()
            elif delta == 0:
                mat[u][v] = new_color
                mat[v][u] = new_color
                current_cost = new_cost
            else:
                if random.random() < np.exp(-delta / max(T, 1e-10)):
                    mat[u][v] = new_color
                    mat[v][u] = new_color
                    current_cost = new_cost

            T *= cooling
            T = max(T, 0.01)

            if time.time() > deadline:
                break

        return best_mat, best_cost == 0

    def gf2_construct():
        """
        Construct a 4-coloring of K_32 using (Z_2)^5.
        
        Vertices are elements of V = (Z_2)^5.
        For a pair (u, v), let d = u ^ v (XOR difference).
        For non-zero d, we map to one of 4 colors.
        
        Key insight: In (Z_2)^5, a monochromatic triangle (a,b,c) requires
        a^b = b^c = c^a = d (same difference). This is impossible since
        a^b ^ b^c ^ c^a = 0, so if all equal d, then d^d^d = d = 0, contradiction.
        
        Wait, that's not right. Let me reconsider.
        In (Z_2)^5, if we color edge (u,v) by f(u^v) where f maps non-zero vectors to {0,1,2,3},
        then a triangle (a,b,c) is monochromatic color c iff f(a^b) = f(b^c) = f(c^a) = c.
        Since a^b ^ b^c ^ c^a = 0, we need three non-zero vectors x,y,z with x^y^z = 0,
        i.e., x^y = z, all mapped to the same color.
        
        So we need to partition the 31 non-zero vectors into 4 sets such that no set
        contains x, y, x^y (for distinct non-zero x, y with x^y also non-zero).
        
        This is equivalent to: no color class contains a "sum-free" triple.
        
        One approach: Use a linear functional. Let h: (Z_2)^5 -> (Z_2)^3 be a linear map.
        Then for non-zero d, color = h(d) interpreted as 0..3. But we need 4 colors and
        h maps to 8 values. We can split: use first 2 bits for 4 colors.
        
        Actually, let's use: color = (d[0] * 2 + d[1]) for d != 0.
        But this might create monochromatic triangles.
        
        Better: use a known construction. The (Z_2)^5 coloring can be done by
        identifying with GF(2^5) and using the trace to GF(2) or similar.
        
        Let me try: for d != 0, let k be the position of the leftmost 1-bit in d.
        color = k % 4. Since d has exactly one leftmost 1, and for a triangle x^y^z=0,
        the leftmost 1 positions of x, y, z must satisfy: the max of the three positions
        appears in exactly one or three of them. If x^y=z, then pos(z) = min(pos(x), pos(y))
        if pos(x) != pos(y), or pos(z) < pos(x) if pos(x) = pos(y).
        
        This is getting complex. Let me just use random + local search from n=32.
        """
        n = 32
        # Use a structured initialization
        mat = np.zeros((n, n), dtype=np.int8)
        np.fill_diagonal(mat, -1)
        
        # Try: color based on (i^j) mapped via a fixed permutation
        # Use the fact that (Z_2)^5 has a natural structure
        for i in range(n):
            for j in range(i+1, n):
                d = i ^ j
                # d is in 1..31
                # Map d to color using a hash-like function
                # Use: color = (d * 7 + (d >> 3) * 3) % 4
                c = (d * 7 + ((d >> 3) & 1) * 3 + ((d >> 1) & 1)) % 4
                mat[i][j] = c
                mat[j][i] = c
        
        return mat

    def try_extend(mat, n, time_budget):
        """Try to extend the coloring by adding vertices one at a time."""
        current_n = n
        current_mat = mat.copy()
        
        while time.time() < start_time + time_budget:
            new_n = current_n + 1
            if new_n > 62:
                break
            
            new_mat = np.zeros((new_n, new_n), dtype=np.int8)
            new_mat[:current_n, :current_n] = current_mat
            
            success = True
            for i in range(current_n):
                forbidden = set()
                for j in range(i):
                    c = current_mat[i][j]
                    if current_mat[j, new_n-1] >= 0:
                        forbidden.add(c)
                
                available = [c for c in range(4) if c not in forbidden]
                if not available:
                    success = False
                    break
                c = random.choice(available)
                new_mat[i, new_n-1] = c
                new_mat[new_n-1, i] = c
            
            if success:
                current_n = new_n
                current_mat = new_mat
            else:
                break
        
        return current_n, current_mat

    def get_seed_solution():
        """Known 32-vertex construction from (Z_2)^5 with GF(16) substructure."""
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
        return {'n': 32, 'colors': colors}

    # Strategy:
    # 1. Try to build a good n=49 coloring with local search
    # 2. If that fails, try n=32 with extension
    # 3. Fall back to known 32-vertex construction

    best_result = get_seed_solution()
    best_n = best_result['n']

    # Try n=49 with local search
    n49 = 49
    mat49 = np.random.randint(0, 4, (n49, n49), dtype=np.int8)
    np.fill_diagonal(mat49, -1)
    for i in range(n49):
        for j in range(i+1, n49):
            mat49[j][i] = mat49[i][j]

    time_budget49 = min(50.0, deadline - 10 - (time.time() - start_time))
    if time_budget49 > 5:
        res_mat, success = local_search_sa(mat49, n49, time_budget49)
        if success:
            best_n = n49
            best_result = {'n': n49, 'colors': [int(res_mat[i][j]) for i in range(n49) for j in range(i+1, n49)]}
            # Try to extend beyond 49
            extended_n, extended_mat = try_extend(res_mat, n49, min(30.0, deadline - 5 - (time.time() - start_time)))
            if extended_n > best_n:
                best_n = extended_n
                best_result = {'n': extended_n, 'colors': [int(extended_mat[i][j]) for i in range(extended_n) for j in range(i+1, extended_n)]}

    # If we didn't get 49, try to improve 32
    if best_n < 49:
        remaining = deadline - 10 - (time.time() - start_time)
        if remaining > 10:
            # Try structured 32-vertex construction
            mat32 = gf2_construct()
            tri_count = count_triangles_vec(mat32)
            
            if tri_count == 0:
                best_n = 32
                best_result = {'n': 32, 'colors': [int(mat32[i][j]) for i in range(32) for j in range(i+1, 32)]}
                # Try to extend
                extended_n, extended_mat = try_extend(mat32, 32, remaining - 5)
                if extended_n > best_n:
                    best_n = extended_n
                    best_result = {'n': extended_n, 'colors': [int(extended_mat[i][j]) for i in range(extended_n) for j in range(i+1, extended_n)]}
            else:
                # Local search on 32
                res_mat32, success32 = local_search_sa(mat32, 32, remaining - 5)
                if success32:
                    best_n = 32
                    best_result = {'n': 32, 'colors': [int(res_mat32[i][j]) for i in range(32) for j in range(i+1, 32)]}
                    extended_n, extended_mat = try_extend(res_mat32, 32, min(15.0, deadline - 5 - (time.time() - start_time)))
                    if extended_n > best_n:
                        best_n = extended_n
                        best_result = {'n': extended_n, 'colors': [int(extended_mat[i][j]) for i in range(extended_n) for j in range(i+1, extended_n)]}

    # Final fallback: use the seed solution
    if best_result['n'] < 32:
        best_result = get_seed_solution()
        best_n = 32

    return best_result
