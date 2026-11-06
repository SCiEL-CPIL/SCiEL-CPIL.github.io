"""Optimized K4-free graph construction with fast K4 check and triangle-aware fitness."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    deadline = time.time() + 115.0
    n = 18
    random.seed(42)
    np.random.seed(42)
    
    def is_k4_free_fast(masks):
        """Fast K4-free check: for each edge (a,b), check if any pair in N(a)∩N(b) is an edge."""
        for a in range(n):
            for b in range(a+1, n):
                if not (masks[a] >> b) & 1:
                    continue
                common = masks[a] & masks[b] & ~(1 << a) & ~(1 << b)
                # Check if common induces any edge (i.e., is not an independent set)
                c_bits = common
                while c_bits:
                    c = c_bits & (-c_bits)
                    c_idx = c.bit_length() - 1
                    c_bits ^= c
                    if masks[c_idx] & common & ~c:
                        return False
        return True
    
    def max_triangle_free_induced(masks):
        """Exact branch-and-bound for largest triangle-free induced subgraph."""
        best = 0
        all_vertices = (1 << n) - 1
        
        # Precompute for each vertex: which other vertices form a triangle with it
        # tri_neighbors[v] = set of vertices u such that v-u is an edge and they share a common neighbor
        # Actually, for the search we need: given current set S, can we add v?
        # v can be added iff no two neighbors of v in S are adjacent to each other.
        
        def can_add(current_mask, v):
            """Check if adding v to current_mask keeps it triangle-free."""
            v_neigh = masks[v] & current_mask
            # Check if v_neigh is an independent set in the graph
            bits = v_neigh
            while bits:
                b = bits & (-bits)
                u = b.bit_length() - 1
                bits ^= b
                if masks[u] & v_neigh & ~b:
                    return False
            return True
        
        # Branch and bound with better vertex ordering
        # Use a greedy upper bound: color the remaining candidates
        def greedy_bound(candidates_mask):
            """Upper bound: maximum independent set size in complement of triangle-conflict graph."""
            # Simple bound: just count vertices (loose but fast)
            return candidates_mask.bit_count()
        
        def search(current_mask, candidates_mask):
            nonlocal best
            if time.time() > deadline - 3:
                return
            cur_size = current_mask.bit_count()
            if cur_size + candidates_mask.bit_count() <= best:
                return
            if candidates_mask == 0:
                if cur_size > best:
                    best = cur_size
                return
            
            # Pick vertex with most constraints (most neighbors in current)
            # to prune earlier
            v_bit = candidates_mask & (-candidates_mask)
            v = v_bit.bit_length() - 1
            remaining = candidates_mask ^ v_bit
            
            # Try including v first (more likely to hit good solutions)
            if can_add(current_mask, v):
                new_current = current_mask | v_bit
                # Filter candidates: only those compatible with new_current
                new_cands = 0
                temp = remaining
                while temp:
                    b = temp & (-temp)
                    idx = b.bit_length() - 1
                    temp ^= b
                    if can_add(new_current, idx):
                        new_cands |= b
                search(new_current, new_cands)
            
            # Try excluding v
            search(current_mask, remaining)
        
        search(0, all_vertices)
        return best
    
    def triangle_density_fitness(masks):
        """
        Fast proxy fitness: count triangles and measure how well they overlap.
        More overlapping triangles = harder to make triangle-free = better score.
        Returns a value where LOWER is better (inverse of triangle coverage).
        """
        # Count all triangles
        tri_count = 0
        tri_vertices = [set() for _ in range(n)]
        for a in range(n):
            for b in range(a+1, n):
                if not (masks[a] >> b) & 1:
                    continue
                common = masks[a] & masks[b] & ~(1 << a) & ~(1 << b)
                c_bits = common
                while c_bits:
                    c = c_bits & (-c_bits)
                    c_idx = c.bit_length() - 1
                    c_bits ^= c
                    tri_count += 1
                    tri_vertices[a].add(c_idx)
                    tri_vertices[b].add(c_idx)
                    tri_vertices[c_idx].add(a)
                    tri_vertices[c_idx].add(b)
        
        if tri_count == 0:
            return n  # No triangles, worst case
        
        # Measure: how many vertices are in at least one triangle
        covered = sum(1 for v in range(n) if len(tri_vertices[v]) >= 2)
        
        # Higher coverage and more triangles per vertex is better
        # Fitness = n - covered - bonus for density
        density_bonus = min(tri_count / n, 2.0)
        fitness = n - covered - density_bonus
        return fitness
    
    def adj_to_mask(adj_mat):
        masks = [0] * n
        for i in range(n):
            for j in range(n):
                if adj_mat[i][j] > 0.5:
                    masks[i] |= (1 << j)
        return masks
    
    def fix_k4_violations(adj, masks):
        """Remove edges to eliminate K4s."""
        changed = True
        while changed:
            changed = False
            for a in range(n):
                for b in range(a+1, n):
                    if (masks[a] >> b) & 1:
                        common = masks[a] & masks[b] & ~(1 << a) & ~(1 << b)
                        c_bits = common
                        while c_bits:
                            c = c_bits & (-c_bits)
                            c_idx = c.bit_length() - 1
                            c_bits ^= c
                            if masks[c_idx] & common & ~c:
                                # K4 found: a, b, c, and the neighbor in common
                                adj[a][b] = 0.0
                                adj[b][a] = 0.0
                                masks[a] &= ~(1 << b)
                                masks[b] &= ~(1 << a)
                                changed = True
                                break
                        if changed:
                            break
                if changed:
                    break
        return masks
    
    def simulated_annealing(init_prob=0.5):
        """SA with fast fitness for exploration, exact for final evaluation."""
        adj = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < init_prob:
                    adj[i][j] = 1.0
                    adj[j][i] = 1.0
        
        masks = adj_to_mask(adj)
        masks = fix_k4_violations(adj, masks)
        
        if not is_k4_free_fast(masks):
            return None, float('inf')
        
        # Initial exact score
        current_exact = max_triangle_free_induced(masks)
        best_exact = current_exact
        best_masks = masks[:]
        
        # SA loop with fast fitness
        temp = 2.0
        cooling = 0.997
        no_improve = 0
        
        while time.time() < deadline - 20 and temp > 0.05:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            # Flip edge
            adj[u][v] = 1.0 - adj[u][v]
            adj[v][u] = adj[u][v]
            masks[u] ^= (1 << v)
            masks[v] ^= (1 << u)
            
            if not is_k4_free_fast(masks):
                # Revert
                adj[u][v] = 1.0 - adj[u][v]
                adj[v][u] = adj[u][v]
                masks[u] ^= (1 << v)
                masks[v] ^= (1 << u)
                temp *= cooling
                continue
            
            # Fast fitness evaluation
            fast_fit = triangle_density_fitness(masks)
            
            # Periodically evaluate exactly
            if no_improve % 50 == 0 or fast_fit < 8:
                exact_score = max_triangle_free_induced(masks)
                delta_exact = exact_score - current_exact
                if delta_exact <= 0 or random.random() < np.exp(-delta_exact / temp):
                    current_exact = exact_score
                    if exact_score < best_exact:
                        best_exact = exact_score
                        best_masks = masks[:]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # Revert
                    adj[u][v] = 1.0 - adj[u][v]
                    adj[v][u] = adj[u][v]
                    masks[u] ^= (1 << v)
                    masks[v] ^= (1 << u)
                    no_improve += 1
            else:
                # Accept based on fast fitness (roughly)
                if fast_fit < 10:
                    no_improve = max(0, no_improve - 1)
                else:
                    no_improve += 1
                    # Revert if fast fitness is bad
                    if fast_fit > 12:
                        adj[u][v] = 1.0 - adj[u][v]
                        adj[v][u] = adj[u][v]
                        masks[u] ^= (1 << v)
                        masks[v] ^= (1 << u)
            
            temp *= cooling
        
        return best_masks, best_exact
    
    def local_search(masks, current_best):
        """Focused local search: try all single-edge changes."""
        all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        current_edges = set()
        for u in range(n):
            for v in range(u+1, n):
                if (masks[u] >> v) & 1:
                    current_edges.add((u, v))
        
        no_improve = 0
        max_no_improve = 200
        
        while time.time() < deadline - 10 and no_improve < max_no_improve:
            improved = False
            no_improve += 1
            
            # Try adding random missing edges
            missing = [(u, v) for u, v in all_edges if (u, v) not in current_edges]
            if missing:
                u, v = random.choice(missing)
                new_masks = masks[:]
                new_masks[u] |= (1 << v)
                new_masks[v] |= (1 << u)
                if is_k4_free_fast(new_masks):
                    score = max_triangle_free_induced(new_masks)
                    if score < current_best:
                        masks = new_masks
                        current_edges.add((u, v))
                        current_best = score
                        improved = True
                        no_improve = 0
            
            # Try removing random existing edges
            if current_edges and not improved:
                u, v = random.choice(list(current_edges))
                new_masks = masks[:]
                new_masks[u] &= ~(1 << v)
                new_masks[v] &= ~(1 << u)
                score = max_triangle_free_induced(new_masks)
                if score < current_best:
                    masks = new_masks
                    current_edges.discard((u, v))
                    current_best = score
                    improved = True
                    no_improve = 0
        
        return masks, current_best
    
    # Main optimization loop
    best_masks = None
    best_score = n  # Upper bound: empty graph has score n
    
    # Try multiple SA runs with different initialization densities
    densities = [0.5, 0.4, 0.6, 0.35, 0.45]
    for d in densities:
        if time.time() > deadline - 15:
            break
        sa_masks, sa_score = simulated_annealing(init_prob=d)
        if sa_masks is not None:
            if sa_score < best_score:
                best_score = sa_score
                best_masks = sa_masks[:]
            
            # Local search refinement
            if time.time() < deadline - 12:
                ls_masks, ls_score = local_search(sa_masks, sa_score)
                if ls_score < best_score:
                    best_score = ls_score
                    best_masks = ls_masks
    
    # Final polish: try all single-edge changes on best solution
    if best_masks is not None and time.time() < deadline - 5:
        all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        for u, v in all_edges:
            if time.time() > deadline - 3:
                break
            # Try adding
            if not (best_masks[u] >> v) & 1:
                new_masks = best_masks[:]
                new_masks[u] |= (1 << v)
                new_masks[v] |= (1 << u)
                if is_k4_free_fast(new_masks):
                    score = max_triangle_free_induced(new_masks)
                    if score < best_score:
                        best_score = score
                        best_masks = new_masks
            # Try removing
            else:
                new_masks = best_masks[:]
                new_masks[u] &= ~(1 << v)
                new_masks[v] &= ~(1 << u)
                score = max_triangle_free_induced(new_masks)
                if score < best_score:
                    best_score = score
                    best_masks = new_masks
    
    # Fallback: if no valid graph found, return empty graph
    if best_masks is None:
        best_masks = [0] * n
    
    # Convert to edge list
    result = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_masks[u] >> v) & 1:
                result.append([u, v])
    return result
# EVOLVE-BLOCK-END
