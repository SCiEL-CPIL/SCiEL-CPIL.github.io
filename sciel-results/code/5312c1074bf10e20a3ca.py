"""Linear triple systems on 16 points - minimize independence number."""
import itertools
import random
import time
import numpy as np
from collections import defaultdict

def construct():
    deadline = time.time() + 140
    n = 16
    rng = random.Random(42)
    
    # Generate all possible triples
    all_triples = list(itertools.combinations(range(n), 3))
    num_triples = len(all_triples)
    
    # Precompute conflict graph: two triples conflict if they share 2 or more points
    # For each triple, store the set of triples it conflicts with
    triple_index = {t: i for i, t in enumerate(all_triples)}
    conflicts = [set() for _ in range(num_triples)]
    
    # Two triples conflict if they share exactly 2 points
    # (sharing 3 points means they're the same triple, sharing 0 or 1 is fine)
    for i in range(num_triples):
        ti = set(all_triples[i])
        for j in range(i + 1, num_triples):
            tj = set(all_triples[j])
            if len(ti & tj) == 2:
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    def get_pairs(triple):
        a, b, c = triple
        return frozenset({(a, b), (a, c), (b, c)})
    
    def is_linear(triples):
        """Check if any two triples share at most one point."""
        seen_pairs = set()
        for t in triples:
            ps = get_pairs(t)
            if seen_pairs & ps:
                return False
            seen_pairs |= ps
        return True
    
    def independence_number(triples):
        """Find the largest set of points containing no triple (independence number)."""
        if not triples:
            return n
        
        # Build incidence: for each point, which triples contain it
        point_to_triples = defaultdict(set)
        for idx, t in enumerate(triples):
            for p in t:
                point_to_triples[p].add(idx)
        
        # The independence number is the largest set S of points such that no triple is fully contained in S
        # Equivalently, S is independent if for every triple t, t is not a subset of S
        
        # Use a branch-and-bound / backtracking approach
        # We want to find the maximum size subset S of {0,...,15} such that no triple is contained in S
        
        best = 0
        # Sort points by some heuristic
        points = list(range(n))
        
        def backtrack(remaining, current_size, current_set):
            nonlocal best
            if current_size + len(remaining) <= best:
                return
            if not remaining:
                best = max(best, current_size)
                return
            
            # Try excluding the first point
            backtrack(remaining[1:], current_size, current_set)
            
            # Try including the first point if it doesn't complete any triple
            p = remaining[0]
            new_set = current_set | {p}
            # Check if adding p completes any triple
            valid = True
            for t in triples:
                if p in t and t[0] in new_set and t[1] in new_set and t[2] in new_set:
                    valid = False
                    break
            if valid:
                backtrack(remaining[1:], current_size + 1, new_set)
        
        backtrack(points, 0, set())
        return best
    
    def independence_number_fast(triples):
        """Faster independence number using bitmasks."""
        if not triples:
            return n
        
        # Represent each point set as a bitmask
        # For each triple, compute its bitmask
        triple_masks = [0] * len(triples)
        for i, t in enumerate(triples):
            mask = 0
            for p in t:
                mask |= (1 << p)
            triple_masks[i] = mask
        
        # A set S (as bitmask) is independent if no triple_mask is a subset of S
        # i.e., for all triple_masks tm: (tm & S) != tm
        
        best = 0
        all_points_mask = (1 << n) - 1
        
        # Enumerate subsets in decreasing order of size, stop when we find one
        # Actually, let's do branch and bound with bitmasks for speed
        
        def is_independent(mask):
            for tm in triple_masks:
                if (tm & mask) == tm:
                    return False
            return True
        
        # Try all subsets from largest to smallest
        # This is exponential but n=16 so 2^16 = 65536 subsets
        # We can be smarter: iterate by size from n down to 1
        
        for size in range(n, 0, -1):
            found = False
            # Generate all subsets of this size
            for subset in itertools.combinations(range(n), size):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                if is_independent(mask):
                    return size
            # If we didn't find any independent set of this size, continue
        return 0
    
    def greedy_construct():
        """Greedy construction of a linear triple system."""
        # Start with all triples available
        available = set(range(num_triples))
        selected = []
        
        # Sort triples by some heuristic - prefer those that cover rare pairs
        # For now, just use random order
        order = list(range(num_triples))
        rng.shuffle(order)
        
        for idx in order:
            t = all_triples[idx]
            # Check if this triple conflicts with any already selected
            can_add = True
            for sel_idx in selected:
                if sel_idx in conflicts[idx]:
                    can_add = False
                    break
            if can_add:
                selected.append(idx)
                # Remove all conflicting triples
                for c in conflicts[idx]:
                    available.discard(c)
                available.discard(idx)
        
        return [list(all_triples[i]) for i in selected]
    
    def local_search(initial_triples, max_iters=10000):
        """Improve a linear triple system via local search."""
        current = list(initial_triples)
        
        def evaluate(triples):
            return independence_number_fast(triples)
        
        current_score = evaluate(current)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try removing a random triple and adding a different one
            if current:
                remove_idx = rng.randrange(len(current))
                removed = current.pop(remove_idx)
                
                # Try to add a different triple
                # Find triples that don't conflict with the remaining ones
                candidates = []
                for i, t in enumerate(all_triples):
                    if t == removed:
                        continue
                    # Check if t conflicts with any in current
                    ok = True
                    for sel in current:
                        sel_set = set(all_triples[sel])
                        if len(set(t) & sel_set) >= 2:
                            ok = False
                            break
                    if ok:
                        candidates.append(i)
                
                if candidates:
                    add_idx = rng.choice(candidates)
                    current.append(add_idx)
                    new_score = evaluate(current)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        # Revert
                        current.pop()
                        current.insert(remove_idx, removed)
                else:
                    # Can't add anything, put back
                    current.insert(remove_idx, removed)
            
            if not improved and rng.random() < 0.1:
                # Perturbation: remove a random triple
                if current:
                    remove_idx = rng.randrange(len(current))
                    removed = current.pop(remove_idx)
                    # Don't put it back, try to find a replacement
                    candidates = []
                    for i, t in enumerate(all_triples):
                        ok = True
                        for sel in current:
                            sel_set = set(all_triples[sel])
                            if len(set(t) & sel_set) >= 2:
                                ok = False
                                break
                        if ok:
                            candidates.append(i)
                    if candidates:
                        add_idx = rng.choice(candidates)
                        current.append(add_idx)
                        current_score = evaluate(current)
        
        return current, current_score
    
    # Main search loop
    best_triples = None
    best_score = n + 1
    
    # Start with greedy construction
    initial = greedy_construct()
    if initial:
        score = independence_number_fast(initial)
        best_triples = initial
        best_score = score
    
    # Local search improvements
    for iteration in range(50):
        if time.time() > deadline - 5:
            break
        
        # Perturb the current best
        perturbed = list(best_triples) if best_triples else []
        # Remove 2-3 random triples
        num_remove = min(len(perturbed), rng.randint(2, 3))
        for _ in range(num_remove):
            if perturbed:
                idx = rng.randrange(len(perturbed))
                perturbed.pop(idx)
        
        # Run local search from perturbed state
        improved, score = local_search(perturbed, max_iters=2000)
        
        if score < best_score:
            best_score = score
            best_triples = improved
    
    # Final check
    if best_triples is None:
        return [[0, 1, 2]]
    
    return best_triples
