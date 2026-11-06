"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import defaultdict

def _is_linear(triples):
    """Check if any two triples share at most one point."""
    n = len(triples)
    for i in range(n):
        for j in range(i + 1, n):
            common = set(triples[i]) & set(triples[j])
            if len(common) > 1:
                return False
    return True

def _calc_alpha(triples, n_points=16):
    """Calculate the independence number (largest set with no triple)."""
    from itertools import combinations
    
    # A set S is independent if no triple is fully contained in S
    # We want the largest such set
    # Use the fact that alpha = n - tau where tau is minimum hitting set
    # But for small n, we can try subsets
    
    # Actually, let's compute it properly:
    # A subset S contains no triple if for every triple t, S does not contain all of t
    # Equivalently, S is independent if it doesn't contain any triple as a subset
    
    # For n=16, checking all 2^16 subsets is feasible
    # A subset is independent if no triple is a subset of it
    
    triples_set = [tuple(sorted(t)) for t in triples]
    triples_set = list(set(triples_set))
    
    # Check subsets from largest to smallest
    for size in range(n_points, 0, -1):
        found = False
        for subset in itertools.combinations(range(n_points), size):
            subset_set = set(subset)
            independent = True
            for t in triples_set:
                if all(p in subset_set for p in t):
                    independent = False
                    break
            if independent:
                return size
    return 0

def _greedy_build(n_points, deadline):
    """Greedy construction of a linear triple system."""
    triples = []
    # Track how many times each pair appears (should be at most 1 for linear)
    pair_count = defaultdict(int)
    point_count = defaultdict(int)
    
    # Generate candidate triples in a good order
    # Try to balance coverage
    candidates = []
    for a in range(n_points):
        for b in range(a + 1, n_points):
            for c in range(b + 1, n_points):
                candidates.append((a, b, c))
    
    # Shuffle with a deterministic seed for reproducibility, but try multiple orders
    import random
    rng = random.Random(42)
    rng.shuffle(candidates)
    
    for triple in candidates:
        if time.time() > deadline:
            break
        a, b, c = triple
        # Check linearity: no pair already used
        if pair_count[(a, b)] == 0 and pair_count[(a, c)] == 0 and pair_count[(b, c)] == 0:
            triples.append(list(triple))
            pair_count[(a, b)] += 1
            pair_count[(a, c)] += 1
            pair_count[(b, c)] += 1
            point_count[a] += 1
            point_count[b] += 1
            point_count[c] += 1
    
    return triples

def _backtrack_build(n_points, max_triples, deadline):
    """Backtracking construction to find a good linear triple system."""
    best_triples = []
    best_score = -1
    
    def backtrack(triples, pair_count, point_count, depth):
        nonlocal best_triples, best_score
        if time.time() > deadline:
            return
        
        # Evaluate current solution
        if len(triples) > len(best_triples):
            best_triples = [list(t) for t in triples]
            best_score = len(triples)
        
        if len(triples) >= max_triples:
            return
        
        # Try adding a new triple
        for a in range(n_points):
            for b in range(a + 1, n_points):
                if pair_count[(a, b)] > 0:
                    continue
                for c in range(b + 1, n_points):
                    if pair_count[(a, c)] > 0 or pair_count[(b, c)] > 0:
                        continue
                    
                    # Add triple
                    triples.append((a, b, c))
                    pair_count[(a, b)] += 1
                    pair_count[(a, c)] += 1
                    pair_count[(b, c)] += 1
                    point_count[a] += 1
                    point_count[b] += 1
                    point_count[c] += 1
                    
                    backtrack(triples, pair_count, point_count, depth + 1)
                    
                    # Remove triple
                    triples.pop()
                    pair_count[(a, b)] -= 1
                    pair_count[(a, c)] -= 1
                    pair_count[(b, c)] -= 1
                    point_count[a] -= 1
                    point_count[b] -= 1
                    point_count[c] -= 1
                    
                    if time.time() > deadline:
                        return
    
    backtrack([], defaultdict(int), defaultdict(int), 0)
    return best_triples

def _optimize_point15(triples_15, deadline):
    """Given a linear triple system on 15 points (0-14), find best way to add point 15."""
    # Option 1: Don't add any triples with point 15
    base_triples = [list(t) for t in triples_15]
    base_alpha = _calc_alpha(base_triples, 16)
    
    best_triples = base_triples
    best_alpha = base_alpha
    
    # Try adding triples that include point 15
    # A triple with 15 is (15, a, b) where a, b are from 0-14
    # Must maintain linearity: (a,b) pair not already used, and (15,a), (15,b) not used
    
    # First, check which pairs are already used in triples_15
    used_pairs = set()
    for t in triples_15:
        a, b, c = t
        used_pairs.add((min(a,b), max(a,b)))
        used_pairs.add((min(a,c), max(a,c)))
        used_pairs.add((min(b,c), max(b,c)))
    
    # Find all valid (a,b) pairs for triples with 15
    valid_pairs_15 = []
    for a in range(15):
        for b in range(a + 1, 15):
            if (a, b) not in used_pairs:
                valid_pairs_15.append((a, b))
    
    # Try adding subsets of these triples (up to a few)
    # For small number of valid pairs, try all subsets up to size 3
    if len(valid_pairs_15) <= 10:
        for r in range(4):
            if time.time() > deadline:
                break
            for subset in itertools.combinations(valid_pairs_15, r):
                new_triples = base_triples + [[15, a, b] for a, b in subset]
                if _is_linear(new_triples):
                    alpha = _calc_alpha(new_triples, 16)
                    if alpha < best_alpha:
                        best_alpha = alpha
                        best_triples = new_triples
    
    return best_triples, best_alpha

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Strategy: Build a good linear triple system
    # Approach: Use greedy with multiple random seeds, then optimize
    
    best_triples = []
    best_alpha = 16  # Worst case
    
    # First, try a structured approach based on Steiner triple system ideas
    # For 16 points, we can't have a perfect STS, but we can get close
    
    # Try greedy construction with different orderings
    import random
    
    for seed in range(50):
        if time.time() > deadline:
            break
        
        rng = random.Random(seed)
        triples = []
        pair_count = defaultdict(int)
        
        # Generate all candidate triples
        candidates = []
        for a in range(16):
            for b in range(a + 1, 16):
                for c in range(b + 1, 16):
                    candidates.append((a, b, c))
        
        rng.shuffle(candidates)
        
        for triple in candidates:
            a, b, c = triple
            if pair_count[(a, b)] == 0 and pair_count[(a, c)] == 0 and pair_count[(b, c)] == 0:
                triples.append(list(triple))
                pair_count[(a, b)] += 1
                pair_count[(a, c)] += 1
                pair_count[(b, c)] += 1
        
        if len(triples) > len(best_triples):
            best_triples = triples
            best_alpha = _calc_alpha(triples, 16)
    
    # Now try to improve by optimizing point 15 placement
    # Extract triples not involving 15
    triples_15_only = [t for t in best_triples if 15 not in t]
    triples_with_15 = [t for t in best_triples if 15 in t]
    
    # Try rebuilding with point 15 optimization
    if time.time() < deadline - 10:
        optimized_triples, opt_alpha = _optimize_point15(triples_15_only, deadline)
        if opt_alpha < best_alpha:
            best_triples = optimized_triples
            best_alpha = opt_alpha
    
    # Final check
    if not _is_linear(best_triples):
        # Fallback to a known valid system
        best_triples = [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_triples
# EVOLVE-BLOCK-END
