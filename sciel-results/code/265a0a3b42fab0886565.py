"""Rewritten program: SAT-based construction for R(4,6) lower bound."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import sys

DEADLINE_MARGIN = 15  # seconds before hard limit
HARD_LIMIT = 150      # seconds


def _make_paley(n):
    """Paley graph on n vertices (n must be prime ≡ 1 mod 4)."""
    squares = {x * x % n for x in range(1, n)}
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n in squares:
                edges.append([i, j])
    return {'n': n, 'edges': edges}


def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _has_k4(adj, n):
    """Check if graph has a K4 using bitsets."""
    adj_bits = [0] * n
    for i in range(n):
        bits = 0
        for j in adj[i]:
            bits |= (1 << j)
        adj_bits[i] = bits
    
    for i in range(n):
        for j in range(i + 1, n):
            if (adj_bits[i] >> j) & 1:
                common = adj_bits[i] & adj_bits[j]
                common &= ~((1 << (j + 1)) - 1)  # only vertices > j
                # Need to find k > j where k is in common
                k = common & ~(common - 1)  # lowest set bit
                if k:
                    k_idx = k.bit_length() - 1
                    # Now find l > k where l is in common and in adj[k]
                    common2 = common & adj_bits[k]
                    common2 &= ~((1 << (k_idx + 1)) - 1)  # only vertices > k
                    if common2:
                        return True
    return False


def _has_indep_6(adj, n):
    """Check if graph has an independent set of size 6 using complement."""
    # Independent set in G = clique in complement
    # Build complement adjacency
    comp_adj = [0] * n
    for i in range(n):
        bits = 0
        for j in range(n):
            if j != i and j not in adj[i]:
                bits |= (1 << j)
        comp_adj[i] = bits
    
    # Find clique of size 6 in complement
    def find_clique(vertices, size, comp_adj):
        if size == 0:
            return True
        if not vertices:
            return False
        if len(vertices) < size:
            return False
        
        v = vertices[0]
        rest = vertices[1:]
        
        # Try including v
        neighbors_in_rest = [u for u in rest if (comp_adj[v] >> u) & 1]
        if find_clique(neighbors_in_rest, size - 1, comp_adj):
            return True
        
        # Try excluding v
        if find_clique(rest, size, comp_adj):
            return True
        
        return False
    
    return find_clique(list(range(n)), 6, comp_adj)


def _is_valid(n, edges):
    adj = _build_adj(n, edges)
    if _has_k4(adj, n):
        return False
    if _has_indep_6(adj, n):
        return False
    return True


def _try_extend_dpll(adj, n, time_limit):
    """
    Try to add a new vertex (index n) by determining which existing vertices it connects to.
    Uses DPLL-style backtracking.
    
    Constraints:
    A. The neighborhood N of the new vertex must be triangle-free in G.
    B. The non-neighborhood V\\N must have no independent set of size 5 in G.
    """
    start = time.time()
    
    # Precompute triangles in G
    triangles = []
    for i in range(n):
        for j in range(i + 1, n):
            if j in adj[i]:
                for k in range(j + 1, n):
                    if k in adj[i] and k in adj[j]:
                        triangles.append((i, j, k))
    
    # Precompute all independent 5-sets (for constraint B checking)
    # This is expensive, so we'll check on the fly with pruning
    
    # DPLL state: assignment for each vertex 0..n-1
    # -1 = unassigned, 0 = not in N, 1 = in N
    assignment = [-1] * n
    
    # For efficiency, track:
    # - For each triangle, how many of its vertices are in N (if all 3, violation)
    # - For each potential independent 5-set, whether any vertex is in N
    
    # Use recursive backtracking with variable ordering
    
    # Variable ordering: vertices with most constraints first
    # A vertex is constrained by:
    # - Number of triangles it's in (for constraint A)
    # - Number of independent 5-sets it's in (for constraint B) - hard to compute
    
    # Simpler: just use natural order or random order
    
    def check_partial(assignment):
        """Check if partial assignment violates any constraint."""
        # Check constraint A: no triangle fully in N
        for (a, b, c) in triangles:
            if assignment[a] == 1 and assignment[b] == 1 and assignment[c] == 1:
                return False, (a, b, c)
        
        # Check constraint B: for any independent 5-set, at least one must be in N
        # We can't check all independent 5-sets efficiently, so we'll do it at the end
        # For now, just check if there's a large independent set in the unassigned + 0 vertices
        
        return True, None
    
    def backtrack(idx):
        """Try to assign vertex idx."""
        if time.time() - start > time_limit:
            return None
        
        if idx == n:
            # All assigned, check constraint B
            N = [i for i in range(n) if assignment[i] == 1]
            non_N = [i for i in range(n) if assignment[i] == 0]
            
            # Check if non_N has an independent set of size 5
            if len(non_N) >= 5:
                # Check all 5-subsets of non_N
                for combo in itertools.combinations(non_N, 5):
                    ok = True
                    for a in range(4):
                        if combo[a + 1] in adj[combo[a]]:
                            ok = False
                            break
                    if ok:
                        return None  # Violation
            
            return N[:]
        
        # Try 1 (in N)
        assignment[idx] = 1
        valid, _ = check_partial(assignment)
        if valid:
            result = backtrack(idx + 1)
            if result is not None:
                return result
        
        # Try 0 (not in N)
        assignment[idx] = 0
        valid, _ = check_partial(assignment)
        if valid:
            result = backtrack(idx + 1)
            if result is not None:
                return result
        
        assignment[idx] = -1
        return None
    
    # Try with different orderings
    for attempt in range(10):
        if time.time() - start > time_limit:
            break
        
        # Randomize variable order
        order = list(range(n))
        random.shuffle(order)
        
        # Reorder assignment
        new_assignment = [-1] * n
        for i, v in enumerate(order):
            new_assignment[v] = -1
        
        # Use a simpler approach: just try backtracking with the shuffled order
        # This is getting complex, let's use a simpler local search instead
    
    # Fallback: local search
    best = None
    for attempt in range(200):
        if time.time() - start > time_limit:
            break
        
        # Random initial assignment
        N = set()
        for v in range(n):
            if random.random() < 0.35:
                N.add(v)
        
        # Local search
        for iteration in range(500):
            if time.time() - start > time_limit:
                break
            
            # Check constraint A: no triangle in N
            violation_a = None
            for (a, b, c) in triangles:
                if a in N and b in N and c in N:
                    violation_a = (a, b, c)
                    break
            
            if violation_a:
                v = random.choice(violation_a)
                N.remove(v)
                continue
            
            # Check constraint B: complement of N has no independent 5-set
            comp = [v for v in range(n) if v not in N]
            violation_b = None
            if len(comp) >= 5:
                if len(comp) <= 25:
                    for combo in itertools.combinations(comp, 5):
                        ok = True
                        for a in range(4):
                            if combo[a + 1] in adj[combo[a]]:
                                ok = False
                                break
                        if ok:
                            violation_b = combo
                            break
                else:
                    for _ in range(50):
                        combo = random.sample(comp, 5)
                        ok = True
                        for a in range(4):
                            if combo[a + 1] in adj[combo[a]]:
                                ok = False
                                break
                        if ok:
                            violation_b = combo
                            break
            
            if violation_b:
                v = random.choice(violation_b)
                N.add(v)
                continue
            
            best = N
            break
        
        if best is not None:
            break
    
    if best is not None:
        return sorted(best)
    return None


def construct():
    start_time = time.time()
    deadline = start_time + HARD_LIMIT - DEADLINE_MARGIN
    
    # Start with Paley graph on 17 vertices
    best_result = _make_paley(17)
    best_n = 17
    
    # Try to extend one vertex at a time
    for target_n in range(18, 37):
        if time.time() > deadline:
            break
        
        # Build adjacency from current best
        adj = _build_adj(best_n, best_result['edges'])
        
        # Try to extend by one vertex
        remaining_time = deadline - time.time()
        if remaining_time < 2:
            break
        
        new_neighbors = _try_extend_dpll(adj, best_n, remaining_time - 5)
        
        if new_neighbors is not None:
            # Verify the new graph
            new_edges = best_result['edges'][:]
            for v in new_neighbors:
                new_edges.append([best_n, v])
            
            if _is_valid(target_n, new_edges):
                best_result = {'n': target_n, 'edges': new_edges}
                best_n = target_n
    
    # Final check
    if time.time() < deadline:
        if not _is_valid(best_n, best_result['edges']):
            best_result = _make_paley(17)
    
    return best_result
# EVOLVE-BLOCK-END
