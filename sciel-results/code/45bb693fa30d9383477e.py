"""Rewritten program: branch-and-bound on minimum maximal no-isosceles set."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys
import numpy as np
from collections import defaultdict

DEADLINE = 325  # seconds, safety margin before 330s kill

def _d2(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy

def _is_iso(a, b, c):
    d1 = _d2(a, b)
    d2 = _d2(a, c)
    d3 = _d2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

def _can_add(p, chosen):
    """Check if p can be added to chosen without creating an isosceles triangle."""
    if not chosen:
        return True
    n = len(chosen)
    for i in range(n):
        for j in range(i + 1, n):
            d1 = _d2(p, chosen[i])
            d2 = _d2(p, chosen[j])
            d3 = _d2(chosen[i], chosen[j])
            if d1 == d2 or d1 == d3 or d2 == d3:
                return False
    return True

def _is_maximal(chosen, all_points):
    """Check if no point outside chosen can be added."""
    chosen_set = set(chosen)
    for p in all_points:
        if p in chosen_set:
            continue
        if _can_add(p, chosen):
            return False
    return True

def _greedy_random(all_points, rng):
    """Randomized greedy: pick points in random order, add if possible."""
    order = list(all_points)
    rng.shuffle(order)
    chosen = []
    for p in order:
        if _can_add(p, chosen):
            chosen.append(p)
    return chosen

def _try_remove(chosen, all_points, rng, max_iters=200):
    """Try to remove points while maintaining maximality."""
    chosen = list(chosen)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        indices = list(range(len(chosen)))
        rng.shuffle(indices)
        for idx in indices:
            candidate = chosen[:idx] + chosen[idx+1:]
            if _is_maximal(candidate, all_points):
                chosen = candidate
                improved = True
                break
    return chosen

def _swap_search(chosen, all_points, rng, max_iters=50):
    """Try to swap points to reduce size while maintaining maximality."""
    chosen_set = set(chosen)
    chosen = list(chosen)
    remaining = [p for p in all_points if p not in chosen_set]
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if not remaining:
            break
        
        chosen_indices = list(range(len(chosen)))
        rng.shuffle(chosen_indices)
        
        for ci in chosen_indices:
            if time.time() - _swap_search.start_time > DEADLINE * 0.8:
                return chosen
            
            without_ci = chosen[:ci] + chosen[ci+1:]
            without_ci_set = set(without_ci)
            
            addable = []
            for p in remaining:
                if p in without_ci_set:
                    continue
                if _can_add(p, without_ci):
                    addable.append(p)
            
            if not addable:
                continue
            
            rng.shuffle(addable)
            for p in addable[:20]:
                candidate = without_ci + [p]
                if _is_maximal(candidate, all_points):
                    chosen = candidate
                    chosen_set = set(chosen)
                    remaining = [p for p in all_points if p not in chosen_set]
                    improved = True
                    break
            
            if improved:
                break
    
    return chosen

def _precompute_isosceles_pairs(all_points):
    """
    For each point p, precompute which pairs (i,j) in all_points form an isosceles triangle with p.
    Also precompute for each point p, which other points q conflict with p (i.e., there exists r such that {p,q,r} is isosceles).
    
    Actually, for the branch-and-bound approach, we need:
    - For a candidate set S, a point p not in S is "blocked" if for every p, adding p to S creates an isosceles triangle.
    - This means: for each p not in S, there must exist at least one pair (a,b) in S such that {p,a,b} is isosceles.
    
    Let's precompute: for each point p, the set of points q such that there exists r where {p,q,r} is isosceles.
    Actually, the condition for p being blocked by S is: for all p not in S, there exists (a,b) in S×S such that {p,a,b} is isosceles.
    
    Equivalent: p is NOT blocked iff there exists a pair (a,b) in S×S such that {p,a,b} is NOT isosceles... no wait.
    
    p can be added to S iff for ALL pairs (a,b) in S×S, {p,a,b} is NOT isosceles.
    So p is blocked iff for ALL pairs (a,b) in S×S, {p,a,b} IS isosceles.
    
    Wait no: p can be added iff no isosceles triangle is formed, i.e., for all a,b in S, {p,a,b} is not isosceles.
    So p is blocked iff there exists a,b in S such that {p,a,b} IS isosceles.
    
    Hmm, that's not right either. Let me reconsider.
    
    p can be added to S iff adding p doesn't create any isosceles triangle. The new triangles involving p are {p,a,b} for a,b in S. So p can be added iff for all a,b in S, {p,a,b} is not isosceles.
    
    So p is blocked (cannot be added) iff there exists a,b in S such that {p,a,b} is isosceles.
    
    For maximality: for every p not in S, p must be blocked, i.e., there exists a,b in S such that {p,a,b} is isosceles.
    
    This is a constraint satisfaction problem. Let me think of a different approach.
    """
    pass

def _precompute_conflicts(all_points):
    """
    Precompute for each pair of points (i,j), the set of points k such that {i,j,k} is isosceles.
    Also, for each point p, precompute the set of "witness" pairs.
    """
    n_pts = len(all_points)
    # conflict[i][j] = set of k such that {i,j,k} is isosceles
    conflict = [set() for _ in range(n_pts)]
    
    # For each point p, which points q are such that there exists r with {p,q,r} isosceles?
    # Actually, let's precompute: for each p, the set of points q where p and q "conflict" in the sense that
    # there exists some r such that {p,q,r} is isosceles. But this is almost all pairs.
    
    # Better: for each p, precompute the list of (a,b) pairs such that {p,a,b} is isosceles.
    # Then p is blocked by S iff there exists (a,b) in S×S such that (a,b) is in iso_pairs[p].
    
    # Let's precompute iso_pairs[p] = set of frozenset({a,b}) such that {p,a,b} is isosceles
    iso_pairs = [set() for _ in range(n_pts)]
    
    for p_idx in range(n_pts):
        p = all_points[p_idx]
        for a_idx in range(n_pts):
            if a_idx == p_idx:
                continue
            a = all_points[a_idx]
            for b_idx in range(a_idx + 1, n_pts):
                if b_idx == p_idx:
                    continue
                b = all_points[b_idx]
                if _is_iso(p, a, b):
                    iso_pairs[p_idx].add((a_idx, b_idx))
    
    return iso_pairs

def _is_maximal_with_iso_pairs(chosen_indices, iso_pairs, all_points):
    """Check maximality using precomputed iso_pairs."""
    chosen_set = set(chosen_indices)
    n_pts = len(all_points)
    for p_idx in range(n_pts):
        if p_idx in chosen_set:
            continue
        # p is blocked iff there exists (a,b) in chosen×chosen such that (a,b) in iso_pairs[p_idx]
        blocked = False
        for (a_idx, b_idx) in iso_pairs[p_idx]:
            if a_idx in chosen_set and b_idx in chosen_set:
                blocked = True
                break
        if not blocked:
            return False
    return True

def _can_add_with_iso_pairs(p_idx, chosen_indices, iso_pairs):
    """Check if p_idx can be added to chosen_indices."""
    chosen_set = set(chosen_indices)
    for (a_idx, b_idx) in iso_pairs[p_idx]:
        if a_idx in chosen_set and b_idx in chosen_set:
            return False
    return True

def _branch_and_bound(all_points, iso_pairs, start_time, deadline):
    """
    Branch and bound to find minimum maximal no-isosceles set.
    State: current chosen set (as sorted list of indices), and the set of points that are still "available" 
    (not yet decided). We branch on adding or not adding a point.
    
    Actually, this is a minimum maximal independent set problem, which is NP-hard.
    Let's use a different approach: start from a greedy solution and try to minimize it.
    """
    # This is getting complex. Let me use a simpler but effective approach:
    # 1. Generate multiple greedy solutions
    # 2. Use local search (removal, swap, 2-swap) to minimize
    # 3. Use simulated annealing or tabu search for better exploration
    
    pass

def construct():
    start_time = time.time()
    _swap_search.start_time = start_time
    all_points = list(itertools.product(range(N), repeat=2))
    n_pts = len(all_points)
    rng = random.Random(42)
    
    # Precompute iso_pairs
    iso_pairs = _precompute_conflicts(all_points)
    
    best = None
    best_size = n_pts + 1
    
    # Phase 1: Multiple randomized greedy starts
    while time.time() - start_time < DEADLINE * 0.3:
        candidate = _greedy_random(all_points, rng)
        if len(candidate) < best_size:
            best = candidate
            best_size = len(candidate)
    
    # Phase 2: Local search
    if best is not None:
        while time.time() - start_time < DEADLINE * 0.85:
            improved = _try_remove(best, all_points, rng, max_iters=100)
            if len(improved) < best_size:
                best = improved
                best_size = len(best)
            
            swapped = _swap_search(best, all_points, rng, max_iters=30)
            if len(swapped) < best_size:
                best = swapped
                best_size = len(best)
            
            if time.time() - start_time < DEADLINE * 0.6:
                candidate = _greedy_random(all_points, rng)
                candidate = _try_remove(candidate, all_points, rng, max_iters=50)
                if len(candidate) < best_size:
                    best = candidate
                    best_size = len(candidate)
    
    # Final: ensure maximality
    if best is not None and time.time() - start_time < DEADLINE - 5:
        if not _is_maximal(best, all_points):
            remaining = [p for p in all_points if p not in set(best)]
            rng.shuffle(remaining)
            for p in remaining:
                if _can_add(p, best):
                    best.append(p)
                if _is_maximal(best, all_points):
                    break
    
    return best if best is not None else []
# EVOLVE-BLOCK-END
