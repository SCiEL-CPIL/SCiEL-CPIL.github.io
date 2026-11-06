# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115
    n = 18

    def edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def has_k4_fast(adj):
        """Fast K4 detection using bitmask operations."""
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b] & ~((1 << (b + 1)) - 1)
                for c in range(b + 1, n):
                    if (common >> c) & 1:
                        if common & adj[c] & ~((1 << (c + 1)) - 1):
                            return True
        return False

    def find_all_triangles(adj):
        """Find all triangles in the graph and return them as a list of bitmasks."""
        triangles = []
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b] & ~((1 << (b + 1)) - 1)
                tmp = common
                while tmp:
                    bit = tmp & (-tmp)
                    c = bit.bit_length() - 1
                    mask = (1 << a) | (1 << b) | (1 << c)
                    triangles.append(mask)
                    tmp &= tmp - 1
        return triangles

    def greedy_triangle_packing(triangles, mask):
        """
        Greedy packing of triangles from the list 'triangles' that are fully contained in 'mask'.
        Returns the maximum number of disjoint triangles found.
        """
        # Filter triangles that are fully in mask
        candidates = [t for t in triangles if (t & mask) == t]
        if not candidates:
            return 0
        
        # Sort by something? Smallest vertex id?
        # To be fast, we just iterate.
        # A better heuristic: pick triangle with min vertex id to break ties deterministically.
        candidates.sort()
        
        selected = []
        used_mask = 0
        for t in candidates:
            if (t & used_mask) == 0:
                selected.append(t)
                used_mask |= t
        
        return len(selected)

    def max_triangle_free_size(adj, time_limit=3.0, upper_bound=None):
        """Exact branch and bound for maximum triangle-free induced subgraph size."""
        best = [0]
        deg = [bin(adj[v]).count('1') for v in range(n)]
        # Order by degree descending, then by vertex index for determinism
        order = sorted(range(n), key=lambda v: (-deg[v], v))
        t0 = time.time()
        timeout = [False]
        
        # Precompute triangles for packing bound
        all_triangles = find_all_triangles(adj)
        
        # Precompute remaining masks for each index in order
        # remaining_mask[idx] contains vertices order[idx], order[idx+1], ...
        remaining_masks = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            remaining_masks[i] = remaining_masks[i + 1] | (1 << order[i])

        def has_triangle_with_v(v, mask):
            """Check if adding v to mask creates a triangle."""
            neighbors_in_mask = mask & adj[v]
            tmp = neighbors_in_mask
            while tmp:
                bit = tmp & (-tmp)
                u = bit.bit_length() - 1
                rest = tmp & ~bit
                if (adj[u] & rest) != 0:
                    return True
                tmp = rest
            return False

        def backtrack(idx, current_mask, current_size):
            if timeout[0]:
                return
            if time.time() - t0 > time_limit:
                timeout[0] = True
                return
            if idx == n:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Simple pruning
            if current_size + (n - idx) <= best[0]:
                return
            
            # Stronger pruning using triangle packing
            # Only compute if simple bound is not enough
            if current_size + (n - idx) > best[0]:
                rem_mask = remaining_masks[idx]
                # We can add at most popcount(rem_mask) - packing_bound(rem_mask)
                # But we are building a set, so we are picking vertices.
                # If there are M disjoint triangles in rem_mask, we can pick at most 2 from each.
                # So max pick <= popcount(rem_mask) - M.
                if all_triangles:
                    m = greedy_triangle_packing(all_triangles, rem_mask)
                    if current_size + (bin(rem_mask).count('1') - m) <= best[0]:
                        return

            v = order[idx]

            # Try including v first
            if not has_triangle_with_v(v, current_mask):
                backtrack(idx + 1, current_mask | (1 << v), current_size + 1)
                if timeout[0]:
                    return

            # Try excluding v
            if current_size + (n - idx - 1) > best[0]:
                backtrack(idx + 1, current_mask, current_size)

        backtrack(0, 0, 0)
        return best[0]

    def make_paley17_plus_iso():
        """Paley(17) graph on vertices 0..16, vertex 17 isolated."""
        adj = [0] * n
        residues = {1, 2, 4, 8, 9, 13, 15, 16}
        for i in range(17):
            for j in range(i + 1, 17):
                diff = (j - i) % 17
                if diff in residues:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def make_paley17_connected():
        """Paley(17) with vertex 17 connected to 1-3 vertices to break large triangle-free sets."""
        adj = make_paley17_plus_iso()
        # Connect 17 to some vertices
        # Try connecting to 0, 1, 2 etc.
        # We need to ensure no K4.
        # If 17 is connected to v, and v is in a triangle in Paley(17) with u, w,
        # and 17 is also connected to u or w, we might form a K4?
        # K4 requires 17-v-u-w to be a clique.
        # 17-v is edge. v-u, v-w, u-w are edges in Paley(17) (triangle).
        # If 17-u is edge, then {17, v, u, w} is K4 if 17-w is also edge.
        # So if 17 is connected to v and u, and v-u-w is a triangle, we must NOT connect 17 to w.
        
        # Strategy: Connect 17 to a set S of vertices.
        # S must be such that for any triangle v-u-w in Paley(17), if v, u in S, then w not in S?
        # Actually, if v, u in S, and v-u edge, and v-w, u-w edges (triangle),
        # then if w in S, {17, v, u, w} is K4 (since 17 connected to all).
        # So S must be an independent set in the "triangle graph"?
        # Simpler: Just try random connections and check K4.
        
        best_adj = None
        for _ in range(20):
            new_adj = make_paley17_plus_iso()
            # Connect 17 to 1 to 4 random vertices
            k = random.randint(1, 4)
            targets = random.sample(range(17), k)
            for t in targets:
                new_adj[17] |= (1 << t)
                new_adj[t] |= (1 << 17)
            
            if not has_k4_fast(new_adj):
                # Check score
                # We don't want to run full B&B here for every candidate, but we can estimate
                # Or just return the first valid one?
                # Let's collect a few.
                if best_adj is None:
                    best_adj = new_adj
        return best_adj

    def make_complete_tripartite():
        """Complete tripartite graph K(6,6,6) - known to be K4-free."""
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                gi = i // 6
                gj = j // 6
                if gi != gj:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def remove_k4(adj):
        """Remove edges to make graph K4-free."""
        for _ in range(500):
            found = False
            for a in range(n):
                for b in range(a + 1, n):
                    if not ((adj[a] >> b) & 1):
                        continue
                    common = adj[a] & adj[b] & ~((1 << (b + 1)) - 1)
                    for c in range(b + 1, n):
                        if (common >> c) & 1:
                            remaining = common & adj[c] & ~((1 << (c + 1)) - 1)
                            if remaining:
                                d_bit = remaining & (-remaining)
                                d = d_bit.bit_length() - 1
                                pair = random.choice([(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)])
                                x, y = pair
                                adj[x] &= ~(1 << y)
                                adj[y] &= ~(1 << x)
                                found = True
                                break
                    if found:
                        break
                if found:
                    break
            if not found:
                break
        return not has_k4_fast(adj)

    random.seed(42)
    
    candidates = []
    
    # Primary: Paley(17) + isolated vertex
    candidates.append(make_paley17_plus_iso())
    
    # Paley(17) + connected vertex
    p17c = make_paley17_connected()
    if p17c:
        candidates.append(p17c)
        
    # Complete tripartite
    candidates.append(make_complete_tripartite())

    # Random dense graphs
    for _ in range(20):
        if time.time() > deadline - 50:
            break
        density = 0.4 + random.random() * 0.4
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        if remove_k4(adj):
            candidates.append(adj)

    # Evaluate candidates
    best_adj = None
    best_score = 18
    for adj in candidates:
        if time.time() > deadline - 30:
            break
        score = max_triangle_free_size(adj, time_limit=2.0)
        if score < best_score:
            best_score = score
            best_adj = adj[:]

    if best_adj is None:
        best_adj = candidates[0]
        best_score = max_triangle_free_size(best_adj, time_limit=1.5)

    # Local Search
    adj = best_adj[:]
    current_score = best_score
    best_adj_sa = adj[:]
    best_score_sa = current_score
    
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() < deadline - 8 and no_improve_count < max_no_improve:
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
            
        old_adj = adj[:]
        if (adj[i] >> j) & 1:
            adj[i] &= ~(1 << j)
            adj[j] &= ~(1 << i)
        else:
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            
        if has_k4_fast(adj):
            adj = old_adj
            no_improve_count += 1
            continue
            
        # Quick bound check
        # We skip the packing bound here for speed, rely on exact score
        # But if we want to prune, we can check if it's likely to be worse
        # For now, just evaluate
        
        new_score = max_triangle_free_size(adj, time_limit=0.5)
        
        if new_score < current_score:
            current_score = new_score
            if current_score < best_score_sa:
                best_score_sa = current_score
                best_adj_sa = adj[:]
            no_improve_count = 0
        else:
            adj = old_adj
            no_improve_count += 1
            
        # Perturbation
        if no_improve_count > 400 and random.random() < 0.2:
            for _ in range(random.randint(2, 4)):
                i2 = random.randint(0, n - 1)
                j2 = random.randint(0, n - 1)
                if i2 == j2:
                    continue
                if i2 > j2:
                    i2, j2 = j2, i2
                if (adj[i2] >> j2) & 1:
                    adj[i2] &= ~(1 << j2)
                    adj[j2] &= ~(1 << i2)
                else:
                    adj[i2] |= (1 << j2)
                    adj[j2] |= (1 << i2)
            if has_k4_fast(adj):
                adj = best_adj_sa[:]
                current_score = best_score_sa
            else:
                new_score = max_triangle_free_size(adj, time_limit=0.5)
                if new_score < current_score:
                    current_score = new_score
                    if current_score < best_score_sa:
                        best_score_sa = current_score
                        best_adj_sa = adj[:]
                    no_improve_count = 0
                else:
                    adj = best_adj_sa[:]
                    current_score = best_score_sa
                    no_improve_count += 1

    # Final Polish
    adj = best_adj_sa[:]
    for i in range(n):
        if time.time() > deadline - 3:
            break
        for j in range(i + 1, n):
            if time.time() > deadline - 3:
                break
            old_adj = adj[:]
            if (adj[i] >> j) & 1:
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)
            else:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
            if has_k4_fast(adj):
                adj = old_adj
                continue
                
            score = max_triangle_free_size(adj, time_limit=0.3)
            if score < best_score_sa:
                best_score_sa = score
                best_adj_sa = adj[:]
            else:
                adj = old_adj

    return edges_from_adj(best_adj_sa)
# EVOLVE-BLOCK-END
