"""Optimized solver for disjoint residue classes with distinct moduli."""
import time
import math
from math import gcd

# EVOLVE-BLOCK-START

def _factorize(n):
    """Return list of (prime, exponent) for n."""
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            e = 0
            while n % d == 0:
                n //= d
                e += 1
            factors.append((d, e))
        d += 1 if d == 2 else 2
    if n > 1:
        factors.append((n, 1))
    return factors

def _disjoint(r1, m1, r2, m2):
    """Two residue classes r1 mod m1 and r2 mod m2 are disjoint iff
    gcd(m1,m2) does not divide (r1 - r2)."""
    g = gcd(m1, m2)
    return (r1 - r2) % g != 0

def _try_greedy(candidates, deadline):
    """Greedy: repeatedly pick the candidate with fewest conflicts among remaining."""
    # candidates: list of (residue, modulus)
    # Build conflict graph
    n = len(candidates)
    # For each candidate, count how many others it conflicts with
    # Conflict = NOT disjoint
    # We want to pick candidates that are disjoint from each other.
    # Greedy: pick candidate that is disjoint from the most already-picked? No.
    # Better: pick candidate with minimum degree in the "non-disjoint" graph (i.e., conflicts with fewest others).
    
    # Actually, let's think differently. We want a set of mutually disjoint classes.
    # Two classes are "compatible" if they are disjoint.
    # So we want a clique in the compatibility graph.
    # Greedy clique: pick vertex with highest degree in compatibility graph, then restrict.
    
    # Let's use a simple greedy: at each step, pick the candidate that is disjoint from all currently selected.
    # To maximize count, pick the one that leaves the most options.
    
    # Simpler greedy: sort candidates by modulus (smaller first tends to be more restrictive)
    # and greedily add if disjoint from all selected.
    
    # Better: try multiple orderings and take the best.
    
    selected = []
    remaining = list(range(n))
    
    while remaining:
        if time.time() > deadline:
            break
        # Find the candidate in remaining that is disjoint from all in selected
        # and has the "best" property (e.g., smallest modulus, or fewest conflicts with remaining)
        best_idx = -1
        best_score = -1
        for ri in remaining:
            r, m = candidates[ri]
            # Check if disjoint from all selected
            ok = True
            for (sr, sm) in selected:
                if not _disjoint(r, m, sr, sm):
                    ok = False
                    break
            if not ok:
                continue
            # Score: prefer candidates that conflict with fewer remaining candidates
            # (i.e., are disjoint from more remaining candidates)
            # Actually, we want to pick the one that is most "restrictive" to prune others,
            # OR the one that is most "flexible". Let's try: pick the one with smallest modulus.
            # Or: pick the one that is disjoint from the most remaining candidates (most flexible).
            flex = 0
            for rj in remaining:
                if rj == ri:
                    continue
                rj_val, mj_val = candidates[rj]
                if _disjoint(r, m, rj_val, mj_val):
                    flex += 1
            # Prefer higher flexibility (disjoint from more)
            if flex > best_score:
                best_score = flex
                best_idx = ri
        
        if best_idx == -1:
            break
        
        selected.append(candidates[best_idx])
        remaining.remove(best_idx)
    
    return selected

def _backtracking(candidates, deadline, max_time=5.0):
    """Backtracking search for maximum clique in compatibility graph."""
    n = len(candidates)
    if n == 0:
        return []
    
    # Precompute compatibility matrix
    compat = [[False] * n for _ in range(n)]
    for i in range(n):
        ri, mi = candidates[i]
        for j in range(i + 1, n):
            rj, mj = candidates[j]
            c = _disjoint(ri, mi, rj, mj)
            compat[i][j] = c
            compat[j][i] = c
    
    # Use a greedy upper bound: for a set of candidates, the max clique size is at most
    # the number of candidates (trivial) or we can use a coloring-based bound.
    # For simplicity, use a branch and bound with the bound being the number of remaining candidates.
    
    best = []
    current = []
    
    def bound(candidates_idx):
        """Upper bound on clique size from candidates_idx."""
        return len(candidates_idx)
    
    def backtrack(available, current_set):
        nonlocal best
        if time.time() > deadline:
            return
        if len(current_set) >= len(best):
            if not available:
                if len(current_set) > len(best):
                    best = current_set[:]
                return
            # Prune: if even adding all available can't beat best
            if len(current_set) + len(available) <= len(best):
                return
        if not available:
            if len(current_set) > len(best):
                best = current_set[:]
            return
        
        # Pick the next candidate to try
        # Try candidates in order of... let's try all
        for i, idx in enumerate(available):
            r, m = candidates[idx]
            # Check if disjoint from all in current_set
            ok = True
            for (sr, sm) in current_set:
                if not _disjoint(r, m, sr, sm):
                    ok = False
                    break
            if not ok:
                continue
            
            current_set.append((r, m))
            # New available: those in available[i+1:] that are compatible with idx
            new_available = [j for j in available[i+1:] if compat[idx][j]]
            backtrack(new_available, current_set)
            current_set.pop()
            
            if time.time() > deadline:
                return
        
        # Also try not picking any more (this is handled by the base case when available is empty
        # or when we've tried all candidates)
    
    # Order candidates by some heuristic to improve pruning
    # Sort by modulus
    candidates_sorted = sorted(range(n), key=lambda i: candidates[i][1])
    backtrack(candidates_sorted, [])
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    # Generate all candidate (residue, modulus) pairs for moduli 2..128
    # For each modulus m, residues 0..m-1
    # But that's too many. Let's be smarter.
    
    # Key insight: For a given modulus m, the residue class r mod m.
    # Two classes (r1,m1) and (r2,m2) are disjoint iff gcd(m1,m2) ∤ (r1-r2).
    
    # To maximize the number of classes, we want to pick one residue for each modulus we use.
    # We don't have to use all moduli. We want to select a subset of moduli and assign residues
    # such that all pairs are disjoint.
    
    # Strategy: For each modulus m in 2..128, we can pick any residue r in 0..m-1.
    # The constraint is pairwise disjointness.
    
    # Let's try a greedy approach first: process moduli in some order, and for each modulus,
    # pick a residue that is disjoint from all previously chosen residues.
    
    # To maximize the count, we want to pick residues that are "flexible" (disjoint from many options).
    
    # Let's try: for each modulus, try all residues and pick the one that maximizes the number
    # of future moduli that can still be satisfied. This is expensive, so let's use a simpler heuristic.
    
    # Simple greedy: for each modulus m (in increasing order), pick the smallest residue r such that
    # (r, m) is disjoint from all previously chosen (r', m').
    
    moduli = list(range(2, 129))
    
    # Try multiple orderings of moduli
    best_result = []
    
    orderings = [
        moduli,  # increasing
        sorted(moduli, reverse=True),  # decreasing
        sorted(moduli, key=lambda m: len(_factorize(m))),  # fewer prime factors first
        sorted(moduli, key=lambda m: -len(_factorize(m))),  # more prime factors first
    ]
    
    for ordering in orderings:
        if time.time() > deadline:
            break
        selected = []
        for m in ordering:
            found = False
            for r in range(m):
                ok = True
                for (sr, sm) in selected:
                    if not _disjoint(r, m, sr, sm):
                        ok = False
                        break
                if ok:
                    selected.append((r, m))
                    found = True
                    break
            # If no residue works for this modulus, skip it
        if len(selected) > len(best_result):
            best_result = selected[:]
    
    # Now try to improve with local search: try to swap residues
    # For each selected (r, m), try changing r to r' and see if we can add a new modulus
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current = list(best_result)
        # Try adding each unused modulus
        used_moduli = set(m for (r, m) in current)
        for m in range(2, 129):
            if m in used_moduli:
                continue
            # Try to find a residue r for m that is disjoint from all current
            for r in range(m):
                ok = True
                for (sr, sm) in current:
                    if not _disjoint(r, m, sr, sm):
                        ok = False
                        break
                if ok:
                    current.append((r, m))
                    improved = True
                    break
        if improved:
            best_result = current
    
    # Convert to list of [residue, modulus]
    result = [[r, m] for (r, m) in best_result]
    
    # Ensure it's nonempty
    if not result:
        result = [[1, 2]]
    
    return result

# EVOLVE-BLOCK-END
