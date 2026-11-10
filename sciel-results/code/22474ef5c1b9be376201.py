"""Design: greedy + local search for a linear 3-uniform hypergraph on 16 points
minimizing the independence number (largest set with no triple fully inside it).

Strategy:
- Build an initial hypergraph by greedy random placement of triples, enforcing
  the linearity constraint (any two triples share at most one point).
- Evaluate independence number via a branch-and-bound that tries to find an
  independent set of size k+1; if it fails, the independence number is <= k.
- Local search: try to add a new triple (chosen to cover as many currently
  uncovered "dangerous" points as possible) or swap an existing triple for a
  better one, always preserving linearity.
- Use a deadline guard to return the best valid system found.
"""
import time
import random
import itertools

_DEADLINE = 140.0  # seconds (safety margin below 150s kill)


def _check_linearity(triples):
    """Return True if any two triples share at most one point."""
    n = len(triples)
    for i in range(n):
        si = set(triples[i])
        for j in range(i + 1, n):
            if len(si & set(triples[j])) > 1:
                return False
    return True


def _independence_number(triples, timeout=5.0):
    """Compute (or upper-bound) the independence number of the hypergraph.
    
    Returns the smallest k such that no independent set of size k+1 exists,
    i.e., the independence number. Uses a recursive search with pruning.
    """
    n_pts = 16
    # For each point, store which triples contain it
    point_to_triples = [[] for _ in range(n_pts)]
    for idx, t in enumerate(triples):
        for p in t:
            point_to_triples[p].append(idx)
    
    # Precompute: for each triple, the set of points
    triple_sets = [set(t) for t in triples]
    
    best = n_pts  # upper bound
    
    def can_extend(available, current_size, target_size):
        """Check if we can build an independent set of size target_size from available points."""
        if current_size + len(available) < target_size:
            return False
        if current_size == target_size:
            return True
        if not available:
            return False
        
        # Try adding each available point
        # Prune: if adding a point would complete any triple, skip
        for i, p in enumerate(available):
            # Check if p can be added without completing a triple
            # A triple is completed if all 3 of its points are in the current set
            # But we're building incrementally, so we need to track which triples are "threatened"
            # Simpler: just try and check
            new_avail = list(available[i+1:])
            # Remove points that would form a triple with p and any two already chosen
            # Actually, let's just do a simpler check:
            # We maintain that no triple is fully contained in chosen set.
            # When adding p, we must ensure no triple {p, x, y} where x,y are already chosen.
            # Since we don't track chosen set explicitly here, let's restructure.
            pass
        
        # Simpler approach: just do DFS with memoization-like pruning
        return _dfs_indep(available, target_size, current_size, triple_sets, point_to_triples)
    
    def _dfs_indep(available, target_size, current_size, triple_sets, point_to_triples):
        if current_size == target_size:
            return True
        if current_size + len(available) < target_size:
            return False
        if not available:
            return False
        
        # Try each point in available
        for i, p in enumerate(available):
            # Check: is there a triple containing p where the other two points are both in 'chosen'?
            # We don't have 'chosen' here. Let's pass it.
            pass
        
        # This approach is getting complicated. Let's use a cleaner formulation.
        # We'll use a different strategy: iterate k from high to low, check if there's an
        # independent set of size k. Use backtracking.
        return False
    
    # Cleaner approach: for k from n_pts down to 1, check if there exists an independent set of size k
    # Return the largest such k.
    
    def has_independent_set_of_size(k):
        """Check if there exists a set of k points containing no triple."""
        if k <= 0:
            return True
        if k > n_pts:
            return False
        
        # Backtracking search
        chosen = []
        # Precompute for each pair of points, which triples contain both
        # A triple is forbidden if all 3 points are in chosen
        
        def backtrack(remaining, chosen_size):
            if chosen_size == k:
                return True
            if chosen_size + len(remaining) < k:
                return False
            if not remaining:
                return False
            
            for i, p in enumerate(remaining):
                # Check if adding p creates a complete triple
                # A triple is complete if it contains p and two other points in chosen
                valid = True
                for t in point_to_triples[p]:
                    ts = triple_sets[t]
                    # Count how many points of ts are in chosen
                    count_in_chosen = 0
                    for q in ts:
                        if q == p:
                            continue
                        if q in chosen_set:
                            count_in_chosen += 1
                    if count_in_chosen == 2:
                        valid = False
                        break
                
                if valid:
                    chosen.append(p)
                    chosen_set.add(p)
                    if backtrack(remaining[i+1:], chosen_size + 1):
                        return True
                    chosen.pop()
                    chosen_set.discard(p)
            
            return False
        
        chosen_set = set()
        all_points = list(range(n_pts))
        return backtrack(all_points, 0)
    
    # Binary search or linear search for the independence number
    # Try k from high to low
    for k in range(n_pts, 0, -1):
        if has_independent_set_of_size(k):
            return k
    return 1


def _score(triples):
    """Return the independence number (lower is better)."""
    if not _check_linearity(triples):
        return 100  # invalid
    return _independence_number(triples)


def _greedy_build(random_state):
    """Build a random linear hypergraph using greedy placement."""
    n_pts = 16
    triples = []
    # Precompute all possible triples
    all_triples = list(itertools.combinations(range(n_pts), 3))
    random_state.shuffle(all_triples)
    
    for t in all_triples:
        t_set = set(t)
        # Check linearity with existing triples
        ok = True
        for existing in triples:
            if len(t_set & set(existing)) > 1:
                ok = False
                break
        if ok:
            triples.append(list(t))
            # Stop when we have a reasonable number of triples
            # More triples = smaller independence number, but harder to maintain linearity
            if len(triples) >= 20:
                break
    
    return triples


def _local_search(triples, random_state, deadline):
    """Try to improve the hypergraph by adding or swapping triples."""
    n_pts = 16
    current_score = _score(triples)
    if current_score >= 100:
        return triples, current_score
    
    best_triples = triples
    best_score = current_score
    
    all_triples = list(itertools.combinations(range(n_pts), 3))
    
    while time.time() < deadline:
        improved = False
        
        # Try adding a new triple
        candidates = [t for t in all_triples if t not in [tuple(x) for x in triples]]
        random_state.shuffle(candidates)
        
        for t in candidates:
            if time.time() >= deadline:
                break
            t_set = set(t)
            ok = True
            for existing in triples:
                if len(t_set & set(existing)) > 1:
                    ok = False
                    break
            if ok:
                new_triples = triples + [list(t)]
                new_score = _score(new_triples)
                if new_score < best_score:
                    best_score = new_score
                    best_triples = new_triples
                    improved = True
                    break
        
        if improved:
            triples = best_triples
            continue
        
        # Try swapping: remove one triple, add another
        for i in range(len(triples)):
            if time.time() >= deadline:
                break
            without_i = triples[:i] + triples[i+1:]
            candidates = [t for t in all_triples if t not in [tuple(x) for x in without_i]]
            random_state.shuffle(candidates)
            
            for t in candidates[:50]:  # limit candidates for speed
                t_set = set(t)
                ok = True
                for existing in without_i:
                    if len(t_set & set(existing)) > 1:
                        ok = False
                        break
                if ok:
                    new_triples = without_i + [list(t)]
                    new_score = _score(new_triples)
                    if new_score < best_score:
                        best_score = new_score
                        best_triples = new_triples
                        improved = True
                        break
            if improved:
                triples = best_triples
                break
        
        if not improved:
            # No improvement found; try a different starting point
            break
    
    return best_triples, best_score


def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 140.0  # 140 second deadline
    
    random_state = random.Random(42)
    
    best_triples = None
    best_score = 100
    
    # Try multiple random starts
    for attempt in range(50):
        if time.time() >= _DEADLINE - 10:
            break
        
        triples = _greedy_build(random_state)
        if not triples:
            continue
        
        if not _check_linearity(triples):
            continue
        
        triples, score = _local_search(triples, random_state, _DEADLINE - 5)
        
        if score < best_score:
            best_score = score
            best_triples = triples
        
        if best_score <= 4:
            break  # Very good result, stop
    
    if best_triples is None:
        # Fallback: return the initial seed
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_triples
