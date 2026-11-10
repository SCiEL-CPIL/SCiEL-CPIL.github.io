"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
from functools import lru_cache

P = {'n': 12, 'm': 40}

def _count_comparable(family):
    """Count unordered pairs where one set is a strict subset of the other."""
    n = len(family)
    # Convert to frozensets for fast subset checks
    fs = [frozenset(s) for s in family]
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if fs[i] < fs[j] or fs[j] < fs[i]:
                count += 1
    return count

def _greedy_construct():
    """
    Greedy construction: build a family of 40 distinct subsets of {0..11}
    maximizing comparable pairs.
    
    Strategy:
    - Work within a small universe (e.g., 6 or 7 elements) to maximize density.
    - Use a greedy approach: start with empty set, then add sets that maximize
      the number of new comparable pairs with existing sets.
    - Prefer smaller sets early (they can be subsets of many larger sets).
    """
    n_univ = 6  # Use 6-element universe for maximum density
    vertices = list(range(n_univ))
    
    # Precompute all subsets
    all_subsets = []
    for r in range(n_univ + 1):
        for combo in itertools.combinations(vertices, r):
            all_subsets.append(list(combo))
    
    all_fs = [frozenset(s) for s in all_subsets]
    
    # We want to select 40 sets greedily
    # Start with the empty set (it's a subset of everything)
    selected = [frozenset()]
    selected_indices = {0}  # index in all_fs
    
    # For each step, add the set that maximizes new comparable pairs
    # A set S is comparable with existing set T if S < T or T < S
    
    m = P['m']
    for step in range(1, m):
        best_score = -1
        best_idx = -1
        
        for i, s in enumerate(all_fs):
            if i in selected_indices:
                continue
            
            # Count comparable pairs with existing selected sets
            score = 0
            for t in selected:
                if s < t or t < s:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_idx = i
        
        if best_idx >= 0:
            selected.append(all_fs[best_idx])
            selected_indices.add(best_idx)
        else:
            break
    
    # Convert back to lists
    result = [list(s) for s in selected]
    return result

def _local_search_improve(family, max_iter=5000):
    """
    Local search: try to swap one set for another to increase comparable count.
    """
    n = len(family)
    if n < 2:
        return family
    
    fs = [frozenset(s) for s in family]
    current_count = _count_comparable(fs)
    
    # Generate candidate replacements: all subsets of a 6-element universe
    n_univ = 6
    vertices = list(range(n_univ))
    all_subsets = []
    for r in range(n_univ + 1):
        for combo in itertools.combinations(vertices, r):
            all_subsets.append(frozenset(combo))
    
    existing = set(fs)
    
    improved = True
    iteration = 0
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        
        for i in range(n):
            for candidate in all_subsets:
                if candidate in existing:
                    continue
                
                # Try swapping fs[i] with candidate
                new_fs = fs[:]
                new_fs[i] = candidate
                new_count = _count_comparable(new_fs)
                
                if new_count > current_count:
                    # Accept the swap
                    existing.discard(fs[i])
                    existing.add(candidate)
                    fs[i] = candidate
                    current_count = new_count
                    improved = True
                    break
            if improved:
                break
    
    return [list(s) for s in fs]

def _multi_start_greedy():
    """
    Multiple greedy starts with different initial seeds, then local search.
    """
    results = []
    
    # Start 1: Standard greedy
    fam1 = _greedy_construct()
    results.append(fam1)
    
    # Start 2: Greedy but start with a few small sets first
    n_univ = 6
    vertices = list(range(n_univ))
    all_subsets = []
    for r in range(n_univ + 1):
        for combo in itertools.combinations(vertices, r):
            all_subsets.append(list(combo))
    
    # Sort by size, take smallest 40
    all_subsets.sort(key=len)
    fam2 = all_subsets[:P['m']]
    results.append(fam2)
    
    # Start 3: Random-ish greedy with different tie-breaking
    import random
    random.seed(42)
    
    all_fs = [frozenset(s) for s in all_subsets]
    selected = [frozenset()]
    selected_indices = {0}
    
    m = P['m']
    for step in range(1, m):
        best_score = -1
        best_candidates = []
        
        for i, s in enumerate(all_fs):
            if i in selected_indices:
                continue
            
            score = 0
            for t in selected:
                if s < t or t < s:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_candidates = [i]
            elif score == best_score:
                best_candidates.append(i)
        
        if best_candidates:
            # Pick randomly among best candidates
            idx = random.choice(best_candidates)
            selected.append(all_fs[idx])
            selected_indices.add(idx)
    
    fam3 = [list(s) for s in selected]
    results.append(fam3)
    
    # Apply local search to each
    best_fam = None
    best_count = -1
    for fam in results:
        improved = _local_search_improve(fam, max_iter=2000)
        count = _count_comparable([frozenset(s) for s in improved])
        if count > best_count:
            best_count = count
            best_fam = improved
    
    return best_fam

def construct():
    """
    Construct a family of 40 distinct subsets of {0..11} maximizing comparable pairs.
    """
    deadline = time.time() + 80  # Safety margin
    
    # Try multi-start greedy with local search
    best_fam = _multi_start_greedy()
    
    # Verify constraints
    assert len(best_fam) == 40
    seen = set()
    for s in best_fam:
        assert isinstance(s, list)
        assert all(isinstance(x, int) and not isinstance(x, bool) for x in s)
        assert len(s) == len(set(s))
        assert all(0 <= x <= 11 for x in s)
        fs = frozenset(s)
        assert fs not in seen
        seen.add(fs)
    
    return best_fam
# EVOLVE-BLOCK-END
