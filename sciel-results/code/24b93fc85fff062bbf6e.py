"""Rewritten program: hypergraph independent set with aggressive local search."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = 128
    elements = list(range(1, n + 1))
    
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            lcm_table[i][j] = i * j // math.gcd(i, j)
    
    # Precompute conflict triples: sets {a,b,c} where lcm(a,b)==lcm(a,c)==lcm(b,c)
    # For each element, track which pairs conflict with it
    # conflict[a][b] = set of c such that {a,b,c} is a conflict triple
    # Actually, let's track: for a pair (a,b), which c makes {a,b,c} a conflict
    # Better: for each element x, track the set of "forbidden pairs" - pairs (a,b) 
    # such that adding x creates a conflict triple
    
    # Let's use: for each x, forbidden_pairs[x] = set of frozenset({a,b}) 
    # where a,b are in selected and lcm(a,b)==lcm(a,x)==lcm(b,x)
    # But this depends on selected set... 
    
    # Alternative: precompute all conflict triples as a set of frozensets
    conflict_triples = set()
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    conflict_triples.add(frozenset((a, b, c)))
    
    # For fast checking: for each pair (a,b), list of c that form conflict
    # pair_conflicts[(a,b)] = list of c
    pair_conflicts = {}
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            pair_conflicts[(a, b)] = []
    for triple in conflict_triples:
        t = tuple(sorted(triple))
        a, b, c = t
        pair_conflicts[(a, b)].append(c)
        pair_conflicts[(a, c)].append(b)
        pair_conflicts[(b, c)].append(a)
    
    def is_valid_set(selected_set):
        """Check if a set has no conflict triples."""
        selected_list = list(selected_set)
        for i in range(len(selected_list)):
            for j in range(i + 1, len(selected_list)):
                a, b = selected_list[i], selected_list[j]
                # Check if any c in selected_set forms conflict with a,b
                for c in pair_conflicts.get((a, b), []):
                    if c in selected_set:
                        return False
        return True
    
    def count_conflicts(selected_set):
        """Count number of conflict triples in selected_set."""
        count = 0
        selected_list = list(selected_set)
        for i in range(len(selected_list)):
            for j in range(i + 1, len(selected_list)):
                a, b = selected_list[i], selected_list[j]
                for c in pair_conflicts.get((a, b), []):
                    if c in selected_set:
                        count += 1
        return count
    
    def try_add(selected_set, x):
        """Try to add x to selected_set, return True if valid."""
        if x in selected_set:
            return True
        # Check all pairs in selected_set
        selected_list = list(selected_set)
        for i in range(len(selected_list)):
            for j in range(i + 1, len(selected_list)):
                a, b = selected_list[i], selected_list[j]
                for c in pair_conflicts.get((a, b), []):
                    if c == x:
                        return False
        # Also check pairs involving x with existing elements
        # Actually we need: for any a in selected_set, is there b in selected_set
        # such that lcm(a,b)==lcm(a,x)==lcm(b,x)?
        # That's exactly checking if x is in pair_conflicts for any pair (a,b) in selected_set
        # Which we just did above. But wait, we also need to check pairs (a,x) and (b,x)
        # No - the condition is that all THREE pairwise LCMs are equal.
        # So we need: exists a,b in selected_set such that lcm(a,b)==lcm(a,x)==lcm(b,x)
        # This is equivalent to: x is in the "conflict" set for pair (a,b)
        # Which is what pair_conflicts[(a,b)] stores.
        # So the above check is sufficient.
        return True
    
    def greedy_solve(order):
        """Greedy: add elements in given order if valid."""
        selected = set()
        for x in order:
            if try_add(selected, x):
                selected.add(x)
        return selected
    
    def local_search(selected, max_iterations=50000):
        """Local search with 1-swap and 2-swap (kicks)."""
        selected = set(selected)
        current_size = len(selected)
        no_improve = 0
        iteration = 0
        
        while iteration < max_iterations:
            if time.time() > deadline - 5:
                break
            iteration += 1
            improved = False
            
            # Try 1-swap: remove one, add one
            for x in list(selected):
                for y in range(1, n + 1):
                    if y in selected:
                        continue
                    new_selected = set(selected)
                    new_selected.discard(x)
                    new_selected.add(y)
                    if try_add(new_selected, y):  # x already removed, just check y
                        # Actually need to verify the whole set is valid
                        # Since we removed x and added y, we need to check y doesn't
                        # create conflicts with remaining pairs
                        # try_add checks if y can be added to new_selected (which has x removed)
                        # But we need to make sure the rest is still valid (it was before)
                        if len(new_selected) > current_size:
                            selected = new_selected
                            current_size = len(selected)
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                # Try 2-swap (kick): remove two, add two
                kicked = False
                selected_list = list(selected)
                for i in range(len(selected_list)):
                    for j in range(i + 1, len(selected_list)):
                        x1, x2 = selected_list[i], selected_list[j]
                        for y1 in range(1, n + 1):
                            if y1 in selected or y1 == x1 or y1 == x2:
                                continue
                            for y2 in range(y1 + 1, n + 1):
                                if y2 in selected or y2 == x1 or y2 == x2:
                                    continue
                                new_selected = set(selected)
                                new_selected.discard(x1)
                                new_selected.discard(x2)
                                new_selected.add(y1)
                                new_selected.add(y2)
                                # Check validity
                                if is_valid_set(new_selected):
                                    if len(new_selected) > current_size:
                                        selected = new_selected
                                        current_size = len(selected)
                                        improved = True
                                        kicked = True
                                        break
                            if kicked:
                                break
                        if kicked:
                            break
                    if kicked:
                        break
            
            if not improved:
                no_improve += 1
                if no_improve > 200:
                    break
            else:
                no_improve = 0
        
        return selected
    
    # Multiple greedy initializations with different orderings
    best = set()
    
    # Ordering 1: natural order
    best = greedy_solve(elements)
    best = local_search(best)
    
    # Ordering 2: reverse
    best2 = greedy_solve(elements[::-1])
    best2 = local_search(best2)
    if len(best2) > len(best):
        best = best2
    
    # Ordering 3: large numbers first [65,128] then rest
    order3 = list(range(65, 129)) + list(range(1, 65))
    best3 = greedy_solve(order3)
    best3 = local_search(best3)
    if len(best3) > len(best):
        best = best3
    
    # Ordering 4: primes+1 first, then others
    def is_prime(k):
        if k < 2:
            return False
        for i in range(2, int(math.isqrt(k)) + 1):
            if k % i == 0:
                return False
        return True
    
    primes_plus_1 = [p + 1 for p in range(2, 129) if is_prime(p)]
    primes_plus_1 = [x for x in primes_plus_1 if 1 <= x <= 128]
    others = [x for x in elements if x not in primes_plus_1]
    order4 = primes_plus_1 + others
    best4 = greedy_solve(order4)
    best4 = local_search(best4)
    if len(best4) > len(best):
        best = best4
    
    # Ordering 5: random
    for _ in range(5):
        if time.time() > deadline - 10:
            break
        random.shuffle(elements)
        best5 = greedy_solve(elements)
        best5 = local_search(best5)
        if len(best5) > len(best):
            best = best5
    
    # Final aggressive local search on best
    if time.time() < deadline - 5:
        best = local_search(best, max_iterations=100000)
    
    return list(best)
# EVOLVE-BLOCK-END
