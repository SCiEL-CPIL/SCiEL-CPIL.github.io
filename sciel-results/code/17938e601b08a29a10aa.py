import random
import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5 second safety margin before 90s limit
    
    # The seed program gives 7 classes with moduli 2,4,8,16,32,64,128
    # Residues: 1,3,7,15,31,63,127 (i.e., 2^k - 1 mod 2^(k+1))
    
    # Let's verify the seed solution is valid:
    # Two residue classes (r1, m1) and (r2, m2) are disjoint iff gcd(r1-r2, m1, m2) != 1
    # Actually, two infinite residue classes {x : x ≡ r1 (mod m1)} and {x : x ≡ r2 (mod m2)} 
    # are disjoint iff the system x ≡ r1 (mod m1), x ≡ r2 (mod m2) has NO solution.
    # By CRT, this system has a solution iff gcd(m1, m2) divides (r1 - r2).
    # So they are disjoint iff gcd(m1, m2) does NOT divide (r1 - r2).
    
    def are_disjoint(r1, m1, r2, m2):
        g = math.gcd(m1, m2)
        return (r1 - r2) % g != 0
    
    # Build conflict graph: nodes are (modulus, residue) pairs
    # We want maximum independent set in the conflict graph,
    # which equals maximum clique in the complement graph.
    
    # Moduli range: 2 to 128
    # For each modulus m, residues are 0 to m-1
    
    # Total number of candidate nodes: sum(m for m in range(2, 129)) = sum(2..128) = (2+128)*127/2 = 8225
    # That's a lot. Let's think about how to make this tractable.
    
    # Actually, let's first try a greedy approach to get a good lower bound,
    # then try to improve with local search or branch-and-bound.
    
    # Precompute all candidates
    candidates = []
    for m in range(2, 129):
        for r in range(m):
            candidates.append((r, m))
    
    n = len(candidates)
    
    # Build adjacency for conflicts: two nodes conflict if they are NOT disjoint
    # i.e., gcd(m1,m2) divides (r1-r2)
    
    # This is O(n^2) to build, which is ~67 million pairs. Might be too slow in Python.
    # Let's be smarter.
    
    # Actually, let's first try a randomized greedy approach.
    
    # Greedy: randomly order candidates, add each if it's disjoint from all already chosen.
    
    def greedy_random():
        chosen = []
        idxs = list(range(n))
        random.shuffle(idxs)
        for i in idxs:
            r, m = candidates[i]
            ok = True
            for (r2, m2) in chosen:
                if not are_disjoint(r, m, r2, m2):
                    ok = False
                    break
            if ok:
                chosen.append((r, m))
        return chosen
    
    # Run multiple greedy attempts
    best = []
    for _ in range(200):
        if time.time() > deadline:
            break
        sol = greedy_random()
        if len(sol) > len(best):
            best = sol
    
    # Also try a smarter greedy: sort by modulus or some heuristic
    def greedy_by_modulus():
        # Try to pick smaller moduli first? Or larger?
        # Let's try picking in order of increasing modulus
        candidates_sorted = sorted(candidates, key=lambda x: x[1])
        chosen = []
        for (r, m) in candidates_sorted:
            ok = True
            for (r2, m2) in chosen:
                if not are_disjoint(r, m, r2, m2):
                    ok = False
                    break
            if ok:
                chosen.append((r, m))
        return chosen
    
    sol = greedy_by_modulus()
    if len(sol) > len(best):
        best = sol
    
    # Try decreasing modulus
    def greedy_by_modulus_desc():
        candidates_sorted = sorted(candidates, key=lambda x: -x[1])
        chosen = []
        for (r, m) in candidates_sorted:
            ok = True
            for (r2, m2) in chosen:
                if not are_disjoint(r, m, r2, m2):
                    ok = False
                    break
            if ok:
                chosen.append((r, m))
        return chosen
    
    sol = greedy_by_modulus_desc()
    if len(sol) > len(best):
        best = sol
    
    # Local search: try to improve the current best solution
    # Replace a chosen element with an unchosen one, or add if possible
    
    # First, let's check if we can add any more elements to the best solution
    def can_add(r, m, chosen):
        for (r2, m2) in chosen:
            if not are_disjoint(r, m, r2, m2):
                return False
        return True
    
    # Try adding elements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        chosen_set = set(best)
        for (r, m) in candidates:
            if (r, m) in chosen_set:
                continue
            if can_add(r, m, best):
                best.append((r, m))
                improved = True
                break
    
    # Now try local search: remove one element, try to add more
    for _ in range(50):
        if time.time() > deadline:
            break
        # Try removing each element and see if we can add more
        for i in range(len(best)):
            if time.time() > deadline:
                break
            saved = best[i]
            new_best = best[:i] + best[i+1:]
            # Try to add as many as possible
            for (r, m) in candidates:
                if (r, m) in new_best:
                    continue
                if can_add(r, m, new_best):
                    new_best.append((r, m))
            if len(new_best) > len(best):
                best = new_best
                break
    
    # Ensure we have at least the seed solution
    if len(best) < 7:
        best = [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
    
    # Convert to the required format: list of [residue, modulus] pairs
    result = [[r, m] for (r, m) in best]
    
    # Verify the result
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            r1, m1 = result[i]
            r2, m2 = result[j]
            if not are_disjoint(r1, m1, r2, m2):
                # Fallback to seed
                return [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
    
    # Verify moduli are distinct
    moduli = [m for (r, m) in result]
    if len(set(moduli)) != len(moduli):
        return [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
    
    # Verify residues are canonical
    for (r, m) in result:
        if not (0 <= r < m):
            return [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
    
    # Verify moduli are in range
    for (r, m) in result:
        if not (2 <= m <= 128):
            return [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
    
    return result
