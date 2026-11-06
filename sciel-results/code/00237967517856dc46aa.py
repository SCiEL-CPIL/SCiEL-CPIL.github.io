import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    def has_k4(adj_mask, n):
        """Return True if graph has K4."""
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_mask[u] >> v) & 1:
                    common = adj_mask[u] & adj_mask[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        c ^= lsb
                        rest = common & ~(1 << w)
                        if (adj_mask[w] & rest) != 0:
                            return True
        return False

    def find_independent_set_6(adj_mask, n):
        """Return a list of 6 independent vertices or None."""
        best = [None]

        def dfs(cand, need, path):
            if need == 0:
                best[0] = list(path)
                return True
            if cand == 0 or need > bin(cand).count('1'):
                return False
            # Pick vertex with minimum degree in candidate set
            min_deg = n + 1
            min_v = -1
            c = cand
            while c:
                lsb = c & -c
                v = lsb.bit_length() - 1
                c ^= lsb
                deg = bin(cand & adj_mask[v]).count('1')
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
                    if deg == 0:
                        break
            if min_v < 0:
                return False
            v = min_v
            lsb = 1 << v
            cand_rest = cand ^ lsb

            # Include v
            new_cand = cand_rest & ~adj_mask[v]
            path.append(v)
            if dfs(new_cand, need - 1, path):
                return True
            path.pop()

            # Exclude v
            if dfs(cand_rest, need, path):
                return True
            return False

        if dfs((1 << n) - 1, 6, []):
            return best[0]
        return None

    def is_valid(adj_mask, n):
        if has_k4(adj_mask, n):
            return False
        if find_independent_set_6(adj_mask, n) is not None:
            return False
        return True

    def find_violation(adj_mask, n):
        """Returns ('K4', [u,v,w,x]) or ('IND', [6 vertices]) or None."""
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_mask[u] >> v) & 1:
                    common = adj_mask[u] & adj_mask[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        c ^= lsb
                        rest = common & ~(1 << w)
                        if (adj_mask[w] & rest) != 0:
                            x_lsb = (adj_mask[w] & rest) & -((adj_mask[w] & rest))
                            x = x_lsb.bit_length() - 1
                            return ('K4', [u, v, w, x])
        ind = find_independent_set_6(adj_mask, n)
        if ind is not None:
            return ('IND', ind)
        return None

    def make_edges(adj_mask, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_mask[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def count_k4s(adj_mask, n):
        """Count all K4 subgraphs."""
        count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_mask[u] >> v) & 1:
                    common = adj_mask[u] & adj_mask[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        c ^= lsb
                        rest = common & ~(1 << w)
                        count += bin(adj_mask[w] & rest).count('1')
        return count

    def max_independent_set_size(adj_mask, n, target=6):
        """Find size of maximum independent set, or at least whether >= target."""
        best_size = [0]

        def dfs(cand, size, path):
            if size > best_size[0]:
                best_size[0] = size
            if best_size[0] >= target:
                return
            if cand == 0:
                return
            # Upper bound: size + popcount(cand)
            if size + bin(cand).count('1') <= best_size[0]:
                return
            # Pick vertex with minimum degree in candidate set
            min_deg = n + 1
            min_v = -1
            c = cand
            while c:
                lsb = c & -c
                v = lsb.bit_length() - 1
                c ^= lsb
                deg = bin(cand & adj_mask[v]).count('1')
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
                    if deg == 0:
                        break
            if min_v < 0:
                return
            v = min_v
            lsb = 1 << v
            cand_rest = cand ^ lsb

            # Include v
            new_cand = cand_rest & ~adj_mask[v]
            path.append(v)
            dfs(new_cand, size + 1, path)
            path.pop()

            # Exclude v
            dfs(cand_rest, size, path)

        dfs((1 << n) - 1, 0, [])
        return best_size[0]

    def violation_score(adj_mask, n):
        """Score: higher is worse. 0 means valid."""
        k4_count = count_k4s(adj_mask, n)
        if k4_count > 0:
            return k4_count * 1000  # Heavy penalty for K4s
        ind_size = max_independent_set_size(adj_mask, n, 6)
        if ind_size >= 6:
            return ind_size - 5
        return 0

    def local_search_sa(adj_mask, n, max_iters, rng, start_temp=1.0, cool=0.999):
        """Simulated annealing local search."""
        current_score = violation_score(adj_mask, n)
        if current_score == 0:
            return {'n': n, 'edges': make_edges(adj_mask, n)}
        
        best_adj = adj_mask[:]
        best_score = current_score
        temp = start_temp
        
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            if current_score == 0:
                return {'n': n, 'edges': make_edges(adj_mask, n)}
            
            # Pick a random edge to flip
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u == v:
                continue
            
            # Flip edge (u,v)
            adj_mask[u] ^= (1 << v)
            adj_mask[v] ^= (1 << u)
            
            new_score = violation_score(adj_mask, n)
            delta = new_score - current_score
            
            if delta <= 0 or rng.random() < np.exp(-delta / max(temp, 0.01)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = adj_mask[:]
            else:
                # Revert
                adj_mask[u] ^= (1 << v)
                adj_mask[v] ^= (1 << u)
            
            temp *= cool
        
        # Return best found
        if best_score == 0:
            return {'n': n, 'edges': make_edges(best_adj, n)}
        return None

    def random_graph(n, p, rng):
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def construct_35_systematic():
        """
        Construct a 35-vertex K4-free graph with alpha <= 5 using a
        structured approach based on the Paley graph of order 7.
        
        Partition into 5 groups of 7. Within each group, use Paley(7).
        Between groups, use a carefully designed bipartite graph.
        """
        n = 35
        adj = [0] * n
        
        # Paley(7): vertices 0-6, edges if (j-i) mod 7 in {1,2,4}
        paley7 = {(1,2), (1,4), (2,4), (0,1), (0,3), (0,5), (1,3), (1,5), (2,3), (2,5), (3,5), (0,2), (0,4), (0,6), (1,6), (2,6), (3,6), (4,6), (5,6)}
        # Actually let me compute Paley(7) properly
        # Quadratic residues mod 7: 1^2=1, 2^2=4, 3^2=2, 4^2=2, 5^2=4, 6^2=1
        # So QR = {1, 2, 4}
        # Edge between i,j if (j-i) mod 7 in {1,2,4}
        qr7 = {1, 2, 4}
        for i in range(7):
            for j in range(i+1, 7):
                if (j - i) % 7 in qr7:
                    paley7.add((i, j))
        
        # Within each group of 7, use Paley(7)
        for g in range(5):
            offset = g * 7
            for i in range(7):
                for j in range(i+1, 7):
                    if (j - i) % 7 in qr7:
                        u = offset + i
                        v = offset + j
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        
        # Between groups: use a bipartite construction
        # For groups g1 and g2 (g1 < g2), connect vertex (g1, a) to (g2, b)
        # if (b - a) mod 7 in some set depending on (g2-g1) mod 5
        # We need to avoid K4 and large independent sets
        # Let's try: connect if (b - a) mod 7 in S_{(g2-g1) mod 5}
        # where S is a set of size ~3.5 (to get ~50% density)
        
        # Try different sets for each difference
        # S0 = {0,1,2,3,4} (5 elements - too many?)
        # S1 = {0,1,2,5,6}
        # S2 = {0,1,3,4,6}
        # S3 = {1,2,3,5,6}
        # S4 = {0,2,3,4,5}
        
        # Actually, let's use a simpler approach: for each pair of groups,
        # use a random bipartite graph with appropriate density, then fix
        # violations with local search.
        
        # For now, let's just return the within-group structure and let
        # local search add the between-group edges.
        return adj

    def local_search_edge_flip(adj_mask, n, max_iters, rng):
        """Local search that flips random edges and accepts if score doesn't increase."""
        current_score = violation_score(adj_mask, n)
        if current_score == 0:
            return {'n': n, 'edges': make_edges(adj_mask, n)}
        
        best_adj = adj_mask[:]
        best_score = current_score
        no_improve = 0
        
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            if current_score == 0:
                return {'n': n, 'edges': make_edges(adj_mask, n)}
            
            # Pick a random edge to flip
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u == v:
                continue
            
            # Flip edge (u,v)
            adj_mask[u] ^= (1 << v)
            adj_mask[v] ^= (1 << u)
            
            new_score = violation_score(adj_mask, n)
            
            if new_score <= current_score:
                current_score = new_score
                no_improve = 0
                if new_score < best_score:
                    best_score = new_score
                    best_adj = adj_mask[:]
            else:
                # Revert
                adj_mask[u] ^= (1 << v)
                adj_mask[v] ^= (1 << u)
                no_improve += 1
                if no_improve > 500:
                    # Perturbation: flip a random set of edges
                    for _ in range(rng.randint(5, 15)):
                        u2 = rng.randint(0, n - 1)
                        v2 = rng.randint(0, n - 1)
                        if u2 != v2:
                            adj_mask[u2] ^= (1 << v2)
                            adj_mask[v2] ^= (1 << u2)
                    current_score = violation_score(adj_mask, n)
                    no_improve = 0
        
        if best_score == 0:
            return {'n': n, 'edges': make_edges(best_adj, n)}
        return None

    def paley17():
        squares = {x * x % 17 for x in range(1, 17)}
        edges = []
        for i in range(17):
            for j in range(i + 1, 17):
                if (j - i) % 17 in squares:
                    edges.append([i, j])
        return {'n': 17, 'edges': edges}

    def paley_17_adj():
        squares = {x * x % 17 for x in range(1, 17)}
        adj = [0] * 17
        for i in range(17):
            for j in range(i + 1, 17):
                if (j - i) % 17 in squares:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    rng = random.Random(42)
    best_result = None

    # Strategy 1: Start with structured 35-vertex graph and refine
    if time.time() < deadline - 60:
        adj35 = construct_35_systematic()
        result = local_search_edge_flip(adj35, 35, 20000, rng)
        if result is not None:
            best_result = result
    
    # Strategy 2: Random 35-vertex graphs with SA
    if best_result is None or best_result['n'] < 35:
        for attempt in range(20):
            if time.time() > deadline - 30:
                break
            adj = random_graph(35, 0.5, rng)
            result = local_search_sa(adj, 35, 15000, rng, start_temp=2.0, cool=0.9995)
            if result is not None:
                if best_result is None or result['n'] > best_result['n']:
                    best_result = result
                break
    
    # Strategy 3: Try to extend best 35 to 36
    if best_result is not None and best_result['n'] == 35 and time.time() < deadline - 15:
        adj35 = [0] * 35
        for u, v in best_result['edges']:
            adj35[u] |= (1 << v)
            adj35[v] |= (1 << u)
        
        for trial in range(30):
            if time.time() > deadline - 5:
                break
            adj36 = adj35[:] + [0]
            # Add vertex 35 with random connections
            for i in range(35):
                if rng.random() < 0.5:
                    adj36[i] |= (1 << 35)
                    adj36[35] |= (1 << i)
            
            result = local_search_sa(adj36, 36, 10000, rng, start_temp=1.5, cool=0.9995)
            if result is not None:
                best_result = result
                break
    
    # Strategy 4: Fallback to smaller graphs
    if best_result is None or best_result['n'] < 35:
        for target_n in [34, 33, 32, 30]:
            if time.time() > deadline - 10:
                break
            for attempt in range(10):
                if time.time() > deadline - 5:
                    break
                adj = random_graph(target_n, 0.5, rng)
                result = local_search_edge_flip(adj, target_n, 10000, rng)
                if result is not None:
                    if best_result is None or result['n'] > best_result['n']:
                        best_result = result
                    break
            if best_result is not None and best_result['n'] >= 34:
                break

    # Final fallback: Paley(17)
    if best_result is None:
        best_result = paley17()

    # Final validation
    n_final = best_result['n']
    adj_final = [0] * n_final
    for u, v in best_result['edges']:
        adj_final[u] |= (1 << v)
        adj_final[v] |= (1 << u)

    if not is_valid(adj_final, n_final):
        best_result = paley17()

    return best_result
