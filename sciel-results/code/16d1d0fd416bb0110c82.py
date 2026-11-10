"""Rewritten program: greedy maximization of pairwise disjoint residue classes."""
# EVOLVE-BLOCK-START
import time
import math

def _is_disjoint(r1, m1, r2, m2):
    """Check if two residue classes are disjoint using exact gcd criterion.
    Classes {x: x ≡ r1 mod m1} and {x: x ≡ r2 mod m2} are disjoint iff
    gcd(|r1-r2|, m1, m2) > 1.
    """
    d = abs(r1 - r2)
    return math.gcd(math.gcd(d, m1), m2) > 1

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    MAX_MOD = 128
    
    # The seed solution: 7 classes with moduli 4,8,16,32,64,128 and residues 1,3,7,15,31,63,127
    # Wait, the seed is [(1<<k)-1, 1<<(k+1)] for k in range(7)
    # k=0: [1, 2]
    # k=1: [3, 4]
    # k=2: [7, 8]
    # k=3: [15, 16]
    # k=4: [31, 32]
    # k=5: [63, 64]
    # k=6: [127, 128]
    # So moduli are 2,4,8,16,32,64,128. Residues are 1,3,7,15,31,63,127.
    # These are all disjoint.
    
    # Strategy: Greedy search with multiple heuristics.
    # We want to maximize the number of classes.
    # Key insight: Moduli that are powers of 2 are "hard" to pack because they share factors.
    # Moduli that are coprime to each other are easier.
    # However, we can mix.
    
    # Let's try a backtracking search with a time limit, or a very aggressive greedy.
    # Given the small max modulus (128), we can afford some search.
    
    # Precompute all possible (r, m) pairs
    all_pairs = []
    for m in range(2, MAX_MOD + 1):
        for r in range(m):
            all_pairs.append((r, m))
    
    # Heuristic ordering of pairs to try first in greedy:
    # 1. Prefer smaller moduli? No, smaller moduli are more restrictive in some sense, but leave more room?
    #    Actually, larger moduli cover more numbers, so they are "bigger" obstacles.
    #    But we want to pack as many as possible.
    #    Let's try ordering by modulus descending, then residue.
    #    Or maybe by "difficulty".
    
    # Let's try a few different greedy orderings and keep the best.
    
    best_count = 0
    best_result = None
    
    # Ordering 1: Descending modulus
    order1 = sorted(all_pairs, key=lambda x: (-x[1], x[0]))
    # Ordering 2: Ascending modulus
    order2 = sorted(all_pairs, key=lambda x: (x[1], x[0]))
    # Ordering 3: Randomized restarts (with fixed seed for reproducibility if needed, but we have time)
    # Let's stick to deterministic first.
    
    # Additional heuristic: When choosing a pair, check against all chosen.
    # To speed up, we can maintain a set of "blocked" residues for each modulus?
    # No, the constraint is global.
    
    def greedy(order):
        chosen = []
        # For each candidate in order, if it's disjoint from all chosen, add it.
        # But we must ensure distinct moduli!
        # The problem states "distinct moduli".
        # So we can only pick ONE pair per modulus.
        
        used_moduli = set()
        for r, m in order:
            if m in used_moduli:
                continue
            # Check if (r, m) is disjoint from all chosen
            ok = True
            for r2, m2 in chosen:
                if not _is_disjoint(r, m, r2, m2):
                    ok = False
                    break
            if ok:
                chosen.append((r, m))
                used_moduli.add(m)
        return chosen
    
    # Try Order 1
    if time.time() < deadline - 10:
        res1 = greedy(order1)
        if len(res1) > best_count:
            best_count = len(res1)
            best_result = res1
            
    # Try Order 2
    if time.time() < deadline - 10:
        res2 = greedy(order2)
        if len(res2) > best_count:
            best_count = len(res2)
            best_result = res2
            
    # Try a smarter greedy:
    # Iterate moduli from 2 to 128. For each modulus, try all residues.
    # Pick the residue that is disjoint from all previously chosen.
    # If multiple, pick one that leaves the most options for future moduli?
    # That's hard to evaluate. Just pick the first valid one?
    # Let's try multiple residue orderings for each modulus.
    
    def greedy_moduli_first(residue_strategy):
        chosen = []
        for m in range(2, MAX_MOD + 1):
            found = False
            if residue_strategy == 'asc':
                residues = range(m)
            elif residue_strategy == 'desc':
                residues = range(m-1, -1, -1)
            else: # 'coprime_first'
                # Prefer residues coprime to m?
                residues = sorted(range(m), key=lambda r: (math.gcd(r, m), r))
            
            for r in residues:
                ok = True
                for r2, m2 in chosen:
                    if not _is_disjoint(r, m, r2, m2):
                        ok = False
                        break
                if ok:
                    chosen.append((r, m))
                    found = True
                    break
            # If no residue found for this modulus, skip it
        return chosen
    
    strategies = ['asc', 'desc', 'coprime_first']
    for strat in strategies:
        if time.time() > deadline - 5:
            break
        res = greedy_moduli_first(strat)
        if len(res) > best_count:
            best_count = len(res)
            best_result = res
            
    # Let's also try a local search / hill climbing on the result?
    # Or a more exhaustive search for small subsets?
    
    # The seed has 7. Can we do better?
    # Let's check if we can add more to the seed.
    # Seed: moduli {2,4,8,16,32,64,128}.
    # Residues: 1,3,7,15,31,63,127.
    # Let's see if we can add modulus 3.
    # We need r such that gcd(|r-1|, 3, 2) > 1, gcd(|r-3|, 3, 4) > 1, etc.
    
    # Let's run a quick check to see if we can improve the seed by adding moduli.
    if time.time() < deadline - 5:
        seed_pairs = [(1,2), (3,4), (7,8), (15,16), (31,32), (63,64), (127,128)]
        # Try to add any modulus from 3 to 128 not in seed
        used_mods = set(m for _, m in seed_pairs)
        improved = list(seed_pairs)
        for m in range(2, MAX_MOD + 1):
            if m in used_mods:
                continue
            for r in range(m):
                ok = True
                for r2, m2 in improved:
                    if not _is_disjoint(r, m, r2, m2):
                        ok = False
                        break
                if ok:
                    improved.append((r, m))
                    used_mods.add(m)
                    break
        
        if len(improved) > best_count:
            best_count = len(improved)
            best_result = improved
    
    # If best_result is still None or empty, use seed
    if best_result is None or len(best_result) == 0:
        best_result = [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
    
    return [[r, m] for r, m in best_result]
# EVOLVE-BLOCK-END
