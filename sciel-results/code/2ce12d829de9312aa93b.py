"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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

    def max_triangle_free_size(adj, time_limit=3.0):
        """Exact branch and bound for maximum triangle-free induced subgraph size."""
        best = [0]
        deg = [bin(adj[v]).count('1') for v in range(n)]
        # Order by degree descending, then by vertex index for determinism
        order = sorted(range(n), key=lambda v: (-deg[v], v))
        t0 = time.time()
        timeout = [False]
        
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
            # Pruning: even if we include all remaining vertices, can't beat best
            if current_size + (n - idx) <= best[0]:
                return

            v = order[idx]

            # Try including v first (finds large solutions faster for pruning)
            if not has_triangle_with_v(v, current_mask):
                backtrack(idx + 1, current_mask | (1 << v), current_size + 1)
                if timeout[0]:
                    return

            # Try excluding v
            # Additional pruning: if excluding v means we can't beat best
            if current_size + (n - idx - 1) > best[0]:
                backtrack(idx + 1, current_mask, current_size)

        backtrack(0, 0, 0)
        return best[0]

    def make_paley17_plus_iso():
        """Paley(17) graph on vertices 0..16, vertex 17 isolated."""
        adj = [0] * n
        # Paley graph: edge (i,j) if i-j is a quadratic residue mod 17
        # Quadratic residues mod 17: 1^2=1, 2^2=4, 3^2=9, 4^2=16, 5^2=8, 6^2=2, 7^2=15, 8^2=13
        residues = {1, 2, 4, 8, 9, 13, 15, 16}
        for i in range(17):
            for j in range(i + 1, 17):
                diff = (j - i) % 17
                if diff in residues:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        # Vertex 17 is isolated
        return adj

    def make_complete_tripartite():
        """Complete tripartite graph K(6,6,6) - known to be K4-free."""
        adj = [0] * n
        # Partition vertices into 3 groups of 6
        # Group 0: 0-5, Group 1: 6-11, Group 2: 12-17
        for i in range(n):
            for j in range(i + 1, n):
                gi = i // 6
                gj = j // 6
                if gi != gj:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def make_paley17_shuffled():
        """Paley(17) graph with a random permutation of vertices 0..16, vertex 17 isolated."""
        adj_base = make_paley17_plus_iso()
        perm = list(range(17))
        random.shuffle(perm)
        new_adj = [0] * n
        for i in range(17):
            for j in range(17):
                if i == j:
                    continue
                if (adj_base[i] >> j) & 1:
                    new_adj[perm[i]] |= (1 << perm[j])
                    new_adj[perm[j]] |= (1 << perm[i])
        return new_adj

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
    
    # Generate diverse initial candidates
    candidates = []
    
    # Primary: Complete tripartite K(6,6,6) - strong baseline
    candidates.append(make_complete_tripartite())
    
    # Paley(17) + isolated vertex
    candidates.append(make_paley17_plus_iso())
    
    # Shuffled Paley(17) variants
    for _ in range(15):
        if time.time() > deadline - 50:
            break
        candidates.append(make_paley17_shuffled())
        
    # Random dense graphs as backup
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

    # Evaluate candidates with time limit to find best starting point
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

    # Structured local search: only accept moves that reduce the exact score
    adj = best_adj[:]
    current_score = best_score
    best_adj_sa = adj[:]
    best_score_sa = current_score
    
    no_improve_count = 0
    max_no_improve = 800
    
    while time.time() < deadline - 8 and no_improve_count < max_no_improve:
        # Pick a random edge to flip
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
            
        new_score = max_triangle_free_size(adj, time_limit=0.7)
        
        if new_score < current_score:
            current_score = new_score
            if current_score < best_score_sa:
                best_score_sa = current_score
                best_adj_sa = adj[:]
            no_improve_count = 0
        else:
            adj = old_adj
            no_improve_count += 1
            
        # If we've been stuck for a while, try a more aggressive perturbation
        if no_improve_count > 300 and random.random() < 0.15:
            # Flip 2-4 random edges
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
                new_score = max_triangle_free_size(adj, time_limit=0.7)
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

    # Final polish: try all single-edge flips from best solution
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
            score = max_triangle_free_size(adj, time_limit=0.5)
            if score < best_score_sa:
                best_score_sa = score
                best_adj_sa = adj[:]
            else:
                adj = old_adj
    
    # Second polish pass if time allows
    if time.time() < deadline - 5:
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
                score = max_triangle_free_size(adj, time_limit=0.4)
                if score < best_score_sa:
                    best_score_sa = score
                    best_adj_sa = adj[:]
                else:
                    adj = old_adj

    return edges_from_adj(best_adj_sa)
# EVOLVE-BLOCK-END
