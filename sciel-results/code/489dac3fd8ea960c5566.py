"""Rewritten program: structural change to graph-theoretic maximum clique search
with fast adjacency via prime factorization, greedy coloring + branch-and-bound,
multiple random restarts, and a final polish pass with single-node swaps."""
import time
import random
from math import gcd

N = 512
DEADLINE = 85.0  # safety margin before 90s kill


def _max_prime_exponent(n, p):
    """Return the exponent of prime p in n."""
    e = 0
    while n % p == 0:
        e += 1
        n //= p
    return e


def _build_adjacency():
    """
    Build adjacency structure for the graph where vertices are 1..512 and an edge
    exists between a and b iff lcm(a,b) <= 512.
    
    We use the fact that lcm(a,b) = a*b / gcd(a,b).
    For speed, precompute prime factorizations and use them to check lcm quickly.
    
    Actually, the fastest approach: for each pair (a,b), compute gcd(a,b) and check
    if a*b//gcd(a,b) <= 512. Since 512 is small, we can precompute all pairs.
    
    Even faster: note that lcm(a,b) <= 512 iff for every prime p, max(v_p(a), v_p(b)) <= v_p(512).
    Since 512 = 2^9, the only prime that matters is 2! Any odd prime factor in a or b
    would make lcm >= that odd prime > 1, but we need lcm <= 512. Wait, no: lcm can
    contain odd primes as long as the product stays <= 512.
    
    Let me reconsider. 512 = 2^9. So lcm(a,b) <= 512 means lcm(a,b) is a divisor of 512...
    No, that's not right either. lcm(a,b) <= 512 just means the integer lcm(a,b) is at most 512.
    
    For example, a=3, b=3: lcm=3 <= 512. a=3, b=5: lcm=15 <= 512. a=3, b=171: lcm=513 > 512.
    
    So we need to check all pairs. Let's just do it directly with gcd.
    """
    adj = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g = gcd(a, b)
            if a * b // g <= N:
                adj[a].add(b)
                adj[b].add(a)
    return adj


def _greedy_color_order(adj, candidates):
    """
    Order candidates using greedy coloring for the branch-and-bound.
    Returns list of (vertex, color) sorted by color.
    """
    # Greedy coloring of the candidate subgraph
    uncolored = list(candidates)
    color_class = {}  # vertex -> color index
    colors = []  # list of sets of vertices with same color
    
    # Sort by degree descending for better coloring
    uncolored.sort(key=lambda v: -len(adj[v] & set(candidates)))
    
    while uncolored:
        color_set = []
        available = set(uncolored)
        for v in uncolored:
            if v not in available:
                continue
            # Check if v can be added to current color class (no edge to any in color_set)
            can_add = True
            for u in color_set:
                if u in adj[v]:
                    can_add = False
                    break
            if can_add:
                color_set.append(v)
                available.discard(v)
        if not color_set:
            # Shouldn't happen, but safety
            color_set = [uncolored[0]]
            available.discard(uncolored[0])
        colors.append(set(color_set))
        uncolored = [v for v in uncolored if v not in set(color_set)]
    
    # Assign color numbers
    for i, cset in enumerate(colors):
        for v in cset:
            color_class[v] = i
    
    # Order: vertices sorted by their color (lower color first for pruning)
    ordered = sorted(candidates, key=lambda v: color_class[v])
    return ordered, color_class, len(colors)


def _max_clique_bb(adj, candidates, current_clique, best_clique):
    """
    Branch and bound maximum clique using greedy coloring upper bound.
    candidates: list of vertices that can be added (all adjacent to all in current_clique)
    current_clique: list of vertices in the current clique
    best_clique: list of vertices in the best clique found so far
    Returns best_clique.
    """
    if not candidates:
        if len(current_clique) > len(best_clique):
            best_clique = list(current_clique)
        return best_clique
    
    # Greedy coloring to get upper bound
    ordered, color_class, num_colors = _greedy_color_order(adj, candidates)
    
    # If even with all remaining colors we can't beat best, prune
    if len(current_clique) + num_colors <= len(best_clique):
        return best_clique
    
    # Process in reverse color order (higher colors first for better pruning)
    # Actually, standard approach: process from last to first
    for i in range(len(ordered) - 1, -1, -1):
        v = ordered[i]
        # Upper bound: current size + number of colors from i onwards
        if len(current_clique) + i + 1 <= len(best_clique):
            break
        
        # Add v to clique
        current_clique.append(v)
        
        # New candidates: vertices in ordered[:i] that are adjacent to v
        new_candidates = [u for u in ordered[:i] if u in adj[v]]
        
        best_clique = _max_clique_bb(adj, new_candidates, current_clique, best_clique)
        
        current_clique.pop()
    
    return best_clique


def _greedy_clique(adj, start_set):
    """
    Greedy heuristic: repeatedly pick the vertex with highest degree within remaining candidates.
    """
    clique = []
    candidates = set(start_set)
    
    while candidates:
        # Pick vertex with max degree within candidates
        best_v = max(candidates, key=lambda v: len(adj[v] & candidates))
        clique.append(best_v)
        candidates = candidates & adj[best_v]
    
    return clique


def _try_add_vertex(adj, clique, v):
    """Check if v can be added to clique (adjacent to all members)."""
    for u in clique:
        if u != v and v not in adj[u]:
            return False
    return True


def _polish_swap(adj, clique):
    """
    Try to improve clique by removing one vertex and adding two others,
    or removing one and adding one different vertex.
    """
    clique_set = set(clique)
    improved = False
    n_clique = len(clique)
    
    # Try single swap: remove one, try to add a non-member that's adjacent to all remaining
    for i in range(n_clique):
        removed = clique[i]
        remaining = [clique[j] for j in range(n_clique) if j != i]
        remaining_set = set(remaining)
        
        # Find all vertices adjacent to all in remaining
        addable = set()
        for v in range(1, N + 1):
            if v in remaining_set:
                continue
            if all(v in adj[u] for u in remaining):
                addable.add(v)
        
        # If we can add more than 1 vertex that weren't in original clique
        new_adds = addable - clique_set
        if len(new_adds) >= 1:
            # Try adding the best one (highest degree)
            best_new = max(new_adds, key=lambda v: len(adj[v]))
            new_clique = remaining + [best_new]
            if len(new_clique) > n_clique:
                return new_clique, True
    
    # Try double swap: remove two, add three
    # This is expensive, so only do it if we have time
    for i in range(n_clique):
        for j in range(i + 1, n_clique):
            removed = {clique[i], clique[j]}
            remaining = [clique[k] for k in range(n_clique) if k != i and k != j]
            remaining_set = set(remaining)
            
            addable = set()
            for v in range(1, N + 1):
                if v in remaining_set or v in removed:
                    continue
                if all(v in adj[u] for u in remaining):
                    addable.add(v)
            
            new_adds = addable - clique_set
            if len(new_adds) >= 2:
                # Pick best two
                sorted_adds = sorted(new_adds, key=lambda v: -len(adj[v]))
                # Check if the two are adjacent to each other
                for a in range(len(sorted_adds)):
                    for b in range(a + 1, len(sorted_adds)):
                        va, vb = sorted_adds[a], sorted_adds[b]
                        if vb in adj[va]:
                            new_clique = remaining + [va, vb]
                            if len(new_clique) > n_clique:
                                return new_clique, True
    
    return clique, False


def construct():
    start_time = time.time()
    
    # Build adjacency
    adj = _build_adjacency()
    
    # Initial greedy clique from all vertices
    best_clique = _greedy_clique(adj, set(range(1, N + 1)))
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return list(best_clique)
    
    # Multiple random restarts with greedy + branch-and-bound
    rng = random.Random(42)
    num_restarts = 0
    max_restarts = 100
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:  # leave 5s for polish
            break
        if num_restarts >= max_restarts:
            break
        num_restarts += 1
        
        # Random subset for greedy
        sample_size = min(N, rng.randint(100, 300))
        sample = rng.sample(range(1, N + 1), sample_size)
        greedy_result = _greedy_clique(adj, set(sample))
        
        if len(greedy_result) > len(best_clique):
            best_clique = greedy_result
        
        # Try branch-and-bound on candidates adjacent to all in best_clique
        if len(best_clique) > 0:
            candidates = set(range(1, N + 1))
            for v in best_clique:
                candidates = candidates & adj[v]
            candidates = candidates - set(best_clique)
            
            if candidates:
                # Use B&B with a time check
                bb_start = time.time()
                # Limit B&B depth/time
                def _bb_with_timeout(adj, candidates, current_clique, best_clique, timeout):
                    """B&B with timeout."""
                    if time.time() - timeout > 2.0:
                        return best_clique
                    if not candidates:
                        if len(current_clique) > len(best_clique):
                            best_clique = list(current_clique)
                        return best_clique
                    
                    ordered, color_class, num_colors = _greedy_color_order(adj, candidates)
                    
                    if len(current_clique) + num_colors <= len(best_clique):
                        return best_clique
                    
                    for i in range(len(ordered) - 1, -1, -1):
                        v = ordered[i]
                        if len(current_clique) + i + 1 <= len(best_clique):
                            break
                        
                        current_clique.append(v)
                        new_candidates = [u for u in ordered[:i] if u in adj[v]]
                        best_clique = _bb_with_timeout(adj, new_candidates, current_clique, best_clique, timeout)
                        current_clique.pop()
                    
                    return best_clique
                
                bb_result = _bb_with_timeout(adj, list(candidates), list(best_clique), list(best_clique), time.time())
                if len(bb_result) > len(best_clique):
                    best_clique = bb_result
    
    # Polish pass
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 2:
        for _ in range(10):
            if time.time() - start_time > DEADLINE - 1:
                break
            best_clique, improved = _polish_swap(adj, best_clique)
            if not improved:
                break
    
    return list(best_clique)
