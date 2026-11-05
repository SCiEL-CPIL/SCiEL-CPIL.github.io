"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def construct():
    # Numbers where 2a is a perfect square must be excluded
    excluded_self = set()
    for a in range(1, 257):
        if _is_square(2 * a):
            excluded_self.add(a)
    
    candidates = [a for a in range(1, 257) if a not in excluded_self]
    
    # Precompute conflicts: for each a, set of b > a such that a+b is square
    # We'll build a full conflict set
    conflict = {a: set() for a in candidates}
    cand_set = set(candidates)
    for a in candidates:
        for b in candidates:
            if a < b and _is_square(a + b):
                conflict[a].add(b)
                conflict[b].add(a)
    
    def greedy(ordering):
        current = set()
        for a in ordering:
            ok = True
            for b in current:
                if b in conflict[a]:
                    ok = False
                    break
            if ok:
                current.add(a)
        return current
    
    def local_search(s):
        s = set(s)
        improved = True
        while improved:
            improved = False
            # Try to add any candidate not in s
            for a in candidates:
                if a not in s:
                    ok = True
                    for b in s:
                        if b in conflict[a]:
                            ok = False
                            break
                    if ok:
                        s.add(a)
                        improved = True
                        break
            # Try swaps: remove one, add another
            for a in list(s):
                for b in candidates:
                    if b in s:
                        continue
                    # Try swapping a with b
                    new_s = s - {a}
                    ok = True
                    for c in new_s:
                        if c in conflict[b]:
                            ok = False
                            break
                    if ok and len(new_s | {b}) > len(s):
                        # This can't happen since we're swapping
                        pass
                    # Actually, swap doesn't increase size. Skip.
                    # Instead, try: remove a, then see if we can add b
                    if ok:
                        # We can add b after removing a, but size stays same
                        # Only useful if it enables further additions
                        pass
        return s
    
    # Multiple greedy with different orderings
    best = set()
    
    orderings = []
    # By degree
    orderings.append(sorted(candidates, key=lambda x: len(conflict[x])))
    # By value
    orderings.append(sorted(candidates))
    orderings.append(sorted(candidates, reverse=True))
    # Random
    random.seed(42)
    for _ in range(20):
        sh = candidates[:]
        random.shuffle(sh)
        orderings.append(sh)
    
    for ord_ in orderings:
        s = greedy(ord_)
        if len(s) > len(best):
            best = s
    
    # Local search on best
    best = local_search(best)
    
    return sorted(best)
# EVOLVE-BLOCK-END
