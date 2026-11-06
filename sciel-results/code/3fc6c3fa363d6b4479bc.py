"""Direct structural approach for equal-LCM-triple-free subset of [1,128]."""
import time
import random
import math
from math import gcd
from itertools import combinations

def construct():
    deadline = time.time() + 85
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute LCM table
    lcm_table = {}
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            lcm_table[(i, j)] = i * j // gcd(i, j)
    
    # Find all bad triples
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[(a, b)]
            for c in range(b + 1, N + 1):
                if lcm_table[(a, c)] == lab and lcm_table[(b, c)] == lab:
                    bad_triples.append((a, b, c))
    
    # Build conflict structure
    conflict_pairs = {x: set() for x in range(1, N + 1)}
    for (a, b, c) in bad_triples:
        conflict_pairs[a].add((b, c))
        conflict_pairs[b].add((a, c))
        conflict_pairs[c].add((a, b))
    
    def can_add(x, S):
        """Check if x can be added to S without creating a bad triple."""
        for (y, z) in conflict_pairs[x]:
            if y in S and z in S:
                return False
        return True
    
    def count_bad_triples(S):
        """Count bad triples in S (should be 0 for valid sets)."""
        count = 0
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                for k in range(j+1, len(S_list)):
                    a, b, c = S_list[i], S_list[j], S_list[k]
                    if lcm_table[(a,b)] == lcm_table[(a,c)] == lcm_table[(b,c)]:
                        count += 1
        return count
    
    def greedy_fill(S, order=None):
        """Fill S by adding elements in given order."""
        if order is None:
            order = elements
        for x in order:
            if x not in S and can_add(x, S):
                S.add(x)
        return S
    
    def local_search(S, max_rounds=30):
        """Improve S via additions and swaps."""
        S = set(S)
        improved = True
        rounds = 0
        while improved and rounds < max_rounds and time.time() < deadline - 5:
            improved = False
            rounds += 1
            # Try additions
            for x in elements:
                if x not in S and can_add(x, S):
                    S.add(x)
                    improved = True
            # Try 1-out-2-in swaps
            for x in list(S):
                S.remove(x)
                best_move = None
                best_score = len(S)
                for y in elements:
                    if y not in S and can_add(y, S):
                        S.add(y)
                        # Try to add one more
                        for z in elements:
                            if z not in S and can_add(z, S):
                                new_size = len(S) + 1
                                if new_size > best_score:
                                    best_score = new_size
                                    best_move = (y, z)
                                S.add(z)
                                break
                        S.remove(y)
                if best_move is not None:
                    S.add(best_move[0])
                    if best_move[1] is not None and can_add(best_move[1], S):
                        S.add(best_move[1])
                    improved = True
                else:
                    S.add(x)
        return S
    
    def simulated_annealing(S, max_iters=5000):
        """SA over add/remove/swap moves."""
        S = set(S)
        # Ensure S is valid first
        S = set()
        greedy_fill(S)
        S = local_search(S, max_rounds=5)
        
        best_S = set(S)
        best_size = len(S)
        T = 5.0
        cooling = 0.999
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            # Random move: add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.3:
                # Try to add a random element
                x = random.choice(elements)
                if x not in S and can_add(x, S):
                    S.add(x)
                    delta = 1
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                    else:
                        # Accept with probability exp(-delta/T) if worsening (shouldn't happen for valid add)
                        if random.random() > math.exp(-delta/T):
                            S.remove(x)
                elif x not in S:
                    # Can't add, try anyway and accept if not too bad
                    S.add(x)
                    # This creates bad triples, so delta is negative
                    delta = -1  # simplified
                    if random.random() > math.exp(delta/T):
                        S.remove(x)
            elif move_type < 0.6:
                # Try to remove a random element
                if S:
                    x = random.choice(list(S))
                    S.remove(x)
                    delta = -1
                    if random.random() > math.exp(delta/T):
                        S.add(x)
            else:
                # Try a swap: remove x, add y
                if S:
                    x = random.choice(list(S))
                    y = random.choice(elements)
                    if x != y:
                        S.remove(x)
                        if can_add(y, S):
                            S.add(y)
                            delta = 0  # size unchanged
                            if random.random() > math.exp(delta/T):
                                S.remove(y)
                                S.add(x)
                        else:
                            S.add(x)
            
            T *= cooling
        
        # Final cleanup
        S = set()
        for x in best_S:
            if can_add(x, S):
                S.add(x)
        greedy_fill(S)
        S = local_search(S, max_rounds=10)
        
        return S
    
    best = set()
    best_size = 0
    
    # Strategy 1: Greedy with different orderings
    triple_count = {x: len(conflict_pairs[x]) for x in range(1, N + 1)}
    
    orderings = [
        sorted(elements, key=lambda x: (triple_count[x], x)),
        sorted(elements, key=lambda x: (triple_count[x], -x)),
        sorted(elements, key=lambda x: x),
        sorted(elements, key=lambda x: -x),
    ]
    
    for order in orderings:
        if time.time() > deadline - 40:
            break
        S = set()
        greedy_fill(S, order)
        S = local_search(S, max_rounds=20)
        if len(S) > best_size:
            best_size = len(S)
            best = set(S)
    
    # Strategy 2: Simulated annealing from multiple starts
    random.seed(42)
    num_sa_starts = 50
    for start in range(num_sa_starts):
        if time.time() > deadline - 20:
            break
        
        # Random initial solution
        order = elements[:]
        random.shuffle(order)
        S = set()
        greedy_fill(S, order)
        
        S = simulated_annealing(S, max_iters=3000)
        
        if len(S) > best_size:
            best_size = len(S)
            best = set(S)
    
    # Strategy 3: More local search on best
    if time.time() < deadline - 5:
        best = local_search(best, max_rounds=50)
    
    # Final fill
    if time.time() < deadline - 3:
        greedy_fill(best)
    
    if not best:
        best = {1}
    
    return list(best)
