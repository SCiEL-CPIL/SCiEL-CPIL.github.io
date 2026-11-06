import time
import random
from itertools import combinations
from math import lcm

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    
    # Precompute all lcm pairs
    n = 128
    lcm_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            lcm_table[i][j] = lcm(i, j)
    
    # A triple {a,b,c} is bad if lcm(a,b)==lcm(a,c)==lcm(b,c)
    # For a given x, x "blocks" pair (a,b) if lcm(a,x)==lcm(b,x)==lcm(a,b)
    # Precompute for each x, the set of pairs (a,b) that x would block
    
    # For each x, compute which pairs (a,b) with a<b<x or a<b and both != x
    # Actually, let's think differently. For a set S, we need no triple with equal LCMs.
    # A triple {a,b,c} is bad iff lcm(a,b)==lcm(a,c)==lcm(b,c)
    
    # Precompute: for each pair (a,b), what values x would make {a,b,x} a bad triple?
    # lcm(a,x) == lcm(a,b) AND lcm(b,x) == lcm(a,b)
    # Let L = lcm(a,b). We need lcm(a,x)==L and lcm(b,x)==L.
    # lcm(a,x)==L means x divides L and lcm(a,x)=L, i.e., x contains exactly the prime powers of L not already in a.
    # More simply: x must divide L, and for each prime p, v_p(x) must be such that max(v_p(a),v_p(x))=v_p(L).
    # Since v_p(L)=max(v_p(a),v_p(b)), we need v_p(x) >= v_p(L)-v_p(a) if v_p(L)>v_p(a), else v_p(x) can be anything up to v_p(L).
    # Actually simpler: just check directly.
    
    # Let's precompute blocking: for each x in [1,128], which pairs (a,b) (a<b, a,b != x) get blocked by x
    # blocked_pairs[x] = set of (a,b) such that lcm(a,x)==lcm(b,x)==lcm(a,b)
    
    blocked_pairs = [set() for _ in range(n+1)]
    for x in range(1, n+1):
        for a in range(1, x):
            lax = lcm_table[a][x] if a < x else lcm_table[x][a]
            for b in range(a+1, n+1):
                if b == x:
                    continue
                lbx = lcm_table[b][x] if b < x else lcm_table[x][b]
                lab = lcm_table[a][b]
                if lax == lbx == lab:
                    blocked_pairs[x].add((a,b))
    
    # Now use randomized greedy with local search
    best_set = set()
    best_size = 0
    
    def is_valid(S):
        """Check if set S has no bad triple."""
        elems = sorted(S)
        for i in range(len(elems)):
            a = elems[i]
            for j in range(i+1, len(elems)):
                b = elems[j]
                lab = lcm_table[a][b]
                for k in range(j+1, len(elems)):
                    c = elems[k]
                    if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                        return False
        return True
    
    def greedy_random(order):
        """Greedy: add elements in given order if they don't create a bad triple."""
        S = set()
        for x in order:
            # Check if x can be added
            # x creates a bad triple with pair (a,b) in S iff (a,b) in blocked_pairs[x]
            # i.e., there exist a,b in S with a<b and (a,b) in blocked_pairs[x]
            can_add = True
            for a in S:
                for b in S:
                    if a == b:
                        continue
                    if a < b:
                        if (a,b) in blocked_pairs[x]:
                            can_add = False
                            break
                    else:
                        if (b,a) in blocked_pairs[x]:
                            can_add = False
                            break
                if not can_add:
                    break
            if can_add:
                S.add(x)
        return S
    
    def local_search(S):
        """Try to improve S by swapping out elements."""
        S = set(S)
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            improved = False
            elems = sorted(S)
            # Try removing one element and adding one not in S
            for rem in elems:
                if time.time() > deadline - 5:
                    break
                S_new = S - {rem}
                # Try adding each x not in S
                for x in range(1, n+1):
                    if x in S_new:
                        continue
                    # Check if x can be added to S_new
                    can_add = True
                    for a in S_new:
                        for b in S_new:
                            if a == b:
                                continue
                            if a < b:
                                if (a,b) in blocked_pairs[x]:
                                    can_add = False
                                    break
                            else:
                                if (b,a) in blocked_pairs[x]:
                                    can_add = False
                                    break
                        if not can_add:
                            break
                    if can_add:
                        S_new.add(x)
                        if len(S_new) > len(S):
                            S = S_new
                            improved = True
                            break
        return S
    
    # Try multiple random orderings
    base_order = list(range(1, n+1))
    
    # First, try the natural order (baseline)
    S = greedy_random(base_order)
    best_set = S
    best_size = len(S)
    
    # Random permutations
    for trial in range(5000):
        if time.time() > deadline - 10:
            break
        order = base_order[:]
        random.shuffle(order)
        S = greedy_random(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            # Do local search
            S_improved = local_search(S)
            if len(S_improved) > best_size:
                best_size = len(S_improved)
                best_set = S_improved
    
    # Try kick-and-refill: remove k random elements, refill greedily
    for trial in range(2000):
        if time.time() > deadline - 10:
            break
        S = set(best_set)
        if len(S) < 2:
            continue
        # Kick: remove 1-3 random elements
        k = random.randint(1, min(3, len(S)-1))
        to_remove = random.sample(sorted(S), k)
        for r in to_remove:
            S.discard(r)
        # Refill: try adding elements from remaining in random order
        remaining = [x for x in range(1, n+1) if x not in S]
        random.shuffle(remaining)
        for x in remaining:
            can_add = True
            for a in S:
                for b in S:
                    if a == b:
                        continue
                    if a < b:
                        if (a,b) in blocked_pairs[x]:
                            can_add = False
                            break
                    else:
                        if (b,a) in blocked_pairs[x]:
                            can_add = False
                            break
                if not can_add:
                    break
            if can_add:
                S.add(x)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            S_improved = local_search(S)
            if len(S_improved) > best_size:
                best_size = len(S_improved)
                best_set = S_improved
    
    # Final local search on best
    if time.time() < deadline - 5:
        S_improved = local_search(best_set)
        if len(S_improved) > best_size:
            best_size = len(S_improved)
            best_set = S_improved
    
    return list(best_set)
