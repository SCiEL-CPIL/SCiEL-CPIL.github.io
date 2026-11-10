"""Improved program: backtracking search for maximum pairwise disjoint residue classes."""
# EVOLVE-BLOCK-START
import time
import math

def are_disjoint_fast(r1, m1, r2, m2):
    """Check if two residue classes are disjoint using exact gcd criterion.
    Classes {x: x ≡ r1 mod m1} and {x: x ≡ r2 mod m2} are disjoint iff
    r1 not congruent r2 mod gcd(m1, m2).
    """
    g = math.gcd(m1, m2)
    return (r1 - r2) % g != 0

def verify_solution(solution):
    """Verify that all pairs in the solution are disjoint."""
    items = list(solution.items())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            m1, r1 = items[i]
            m2, r2 = items[j]
            if not are_disjoint_fast(r1, m1, r2, m2):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    MAX_MOD = 128
    
    best_count = 0
    best_result = None
    
    # Precompute gcd for all pairs
    gcd_table = [[0] * (MAX_MOD + 1) for _ in range(MAX_MOD + 1)]
    for i in range(2, MAX_MOD + 1):
        for j in range(2, MAX_MOD + 1):
            gcd_table[i][j] = math.gcd(i, j)
    
    # Constraint score: higher gcd sum means more constrained
    constraint_score = {}
    for m in range(2, MAX_MOD + 1):
        score = 0
        for m2 in range(2, MAX_MOD + 1):
            if m2 != m:
                score += gcd_table[m][m2]
        constraint_score[m] = score
    
    moduli_order = sorted(range(2, MAX_MOD + 1), key=lambda m: -constraint_score[m])
    
    # Greedy with multiple orderings
    def greedy_with_order(order):
        chosen = {}
        for m in order:
            for r in range(m):
                ok = True
                for m2, r2 in chosen.items():
                    if not are_disjoint_fast(r, m, r2, m2):
                        ok = False
                        break
                if ok:
                    chosen[m] = r
                    break
        return chosen
    
    orderings = [
        moduli_order,
        sorted(range(2, MAX_MOD + 1)),
        sorted(range(2, MAX_MOD + 1), reverse=True),
    ]
    
    for order in orderings:
        if time.time() > deadline - 10:
            break
        result = greedy_with_order(order)
        if len(result) > best_count:
            best_count = len(result)
            best_result = result
    
    # Local Search: Try to add missing moduli
    if best_result is not None and time.time() < deadline - 5:
        current = dict(best_result)
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for m in range(2, MAX_MOD + 1):
                if m in current:
                    continue
                for r in range(m):
                    ok = True
                    for m2, r2 in current.items():
                        if not are_disjoint_fast(r, m, r2, m2):
                            ok = False
                            break
                    if ok:
                        current[m] = r
                        improved = True
                        break
                if time.time() > deadline - 3:
                    break
        
        if len(current) > best_count:
            best_count = len(current)
            best_result = current
    
    # Fallback: A known valid set
    # Powers of 2 with specific residues to ensure disjointness
    # r_m for m=2^k is chosen such that r_m not congruent r_n mod 2^min(k,n)
    # Using r_m = 2^(k-1) - 1? No.
    # Simple valid set:
    # m=2, r=0
    # m=4, r=1 (1!=0 mod 2)
    # m=8, r=3 (3!=0 mod 2, 3!=1 mod 4)
    # m=16, r=7 (7!=0 mod 2, 7!=1 mod 4, 7!=3 mod 8)
    # Pattern: r_m = m/2 - 1?
    # 2: 0
    # 4: 1
    # 8: 3
    # 16: 7
    # 32: 15
    # 64: 31
    # 128: 63
    fallback = {2: 0, 4: 1, 8: 3, 16: 7, 32: 15, 64: 31, 128: 63}
    
    if best_result is None or len(best_result) == 0 or not verify_solution(best_result):
        best_result = fallback
    
    # Final verification
    if not verify_solution(best_result):
        # If even fallback fails (should not happen), return a single class
        best_result = {2: 0}
    
    return [[r, m] for m, r in sorted(best_result.items())]
# EVOLVE-BLOCK-END
