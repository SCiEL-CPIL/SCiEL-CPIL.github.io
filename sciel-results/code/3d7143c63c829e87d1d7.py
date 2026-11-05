import time
from math import gcd
from itertools import combinations
import random

def construct():
    deadline = time.time() + 85
    N = 128
    # Precompute GCD table
    g = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            g[i][j] = g[j][i] = gcd(i, j)
    
    # Precompute for each pair (a,b), the set of c such that (a,b,c) is a bad triple
    # bad_pair[a][b] = bitmask of c values that form a bad triple with a and b
    # Actually, let's precompute: for each element x, and each pair (a,b) already in S,
    # check if (a,b,x) is bad. But that's still O(|S|²).
    # Better: precompute for each x, the set of pairs (a,b) that would conflict with x.
    # conflict_with[x] = set of (a,b) pairs where gcd(a,b)==gcd(a,x)==gcd(b,x)
    
    # Alternative: for each x, precompute a bitmask of elements that "conflict" with x in the sense
    # that there exists some y such that (x, y, z) is bad for some z.
    # This is getting complex. Let's use a different approach:
    
    # Precompute all bad triples
    bad_triples = []
    bad_triple_set = set()
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            gab = g[a][b]
            for c in range(b+1, N+1):
                if g[a][c] == gab and g[b][c] == gab:
                    bad_triples.append((a,b,c))
                    bad_triple_set.add((a,b,c))
    
    # For each element x, precompute the set of pairs (a,b) with a<b, a!=x, b!=x
    # such that (min(a,b,x), mid(a,b,x), max(a,b,x)) is a bad triple
    # Actually, let's precompute: for each x, a bitmask conflict_mask[x] where bit y is set
    # if there exists z such that (x,y,z) is a bad triple. But that's not quite right either.
    
    # The real constraint: when adding x to set S, we need no (a,b,x) in S∪{x} to be bad.
    # So for each x, and each pair (a,b) in S, check if (a,b,x) is bad.
    # We can precompute: for each x, a set of forbidden pairs. But pairs are O(N²) = 8000+.
    
    # Let's use a different representation: for each x, precompute a 128-bit integer
    # where bit y is set if there exists z (z != x, z != y) such that (x,y,z) is bad.
    # No wait, that's not the right constraint. The constraint is: for the specific set S,
    # no triple (a,b,c) with all three in S is bad.
    
    # Fast approach: for each x, precompute a list of (a,b) pairs that form a bad triple with x.
    # Then when checking if x can be added to S, we just check if any (a,b) in that list
    # has both a and b in S.
    
    # Precompute: for each x, list of (a,b) pairs (a < b, a != x, b != x) such that
    # {a,b,x} forms a bad triple
    forbidden_pairs = [[] for _ in range(N+1)]
    for (a,b,c) in bad_triples:
        # a < b < c
        forbidden_pairs[a].append((b,c))
        forbidden_pairs[b].append((a,c))
        forbidden_pairs[c].append((a,b))
    
    def can_add_fast(S_set, x):
        """Check if x can be added to S_set without creating a bad triple.
        S_set is a set of integers. x is not in S_set."""
        for (a, b) in forbidden_pairs[x]:
            if a in S_set and b in S_set:
                return False
        return True
    
    def greedy_build(order):
        S = []
        S_set = set()
        for x in order:
            if x not in S_set and can_add_fast(S_set, x):
                S.append(x)
                S_set.add(x)
        return S, S_set
    
    def local_search(S, S_set):
        """Try to improve S by adding elements or doing swaps."""
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try adding any element not in S
            for x in range(1, N+1):
                if x not in S_set:
                    if can_add_fast(S_set, x):
                        S.append(x)
                        S_set.add(x)
                        improved = True
                        break
            if improved:
                continue
            # Try swaps: remove one, add two
            for i in range(len(S)):
                if time.time() > deadline - 2:
                    break
                a = S[i]
                S.remove(a)
                S_set.discard(a)
                added = []
                for x in range(1, N+1):
                    if x not in S_set:
                        if can_add_fast(S_set, x):
                            S.append(x)
                            S_set.add(x)
                            added.append(x)
                            if len(added) >= 2:
                                break
                if len(added) >= 2:
                    improved = True
                else:
                    # Undo
                    for y in added:
                        S.remove(y)
                        S_set.discard(y)
                    S.append(a)
                    S_set.add(a)
        return S, S_set
    
    best_S = []
    best_set = set()
    
    base_order = list(range(1, N+1))
    
    # Strategy 1: ascending
    S, S_set = greedy_build(base_order)
    S, S_set = local_search(S, S_set)
    if len(S) > len(best_S):
        best_S = S[:]
        best_set = S_set.copy()
    
    # Strategy 2: descending
    S, S_set = greedy_build(base_order[::-1])
    S, S_set = local_search(S, S_set)
    if len(S) > len(best_S):
        best_S = S[:]
        best_set = S_set.copy()
    
    # Strategy 3: random restarts
    random.seed(42)
    for restart in range(200):
        if time.time() > deadline - 5:
            break
        order = base_order[:]
        random.shuffle(order)
        S, S_set = greedy_build(order)
        S, S_set = local_search(S, S_set)
        if len(S) > len(best_S):
            best_S = S[:]
            best_set = S_set.copy()
    
    # Strategy 4: conflict-aware ordering - prefer elements with fewer forbidden pairs
    conflict_count = [0]*(N+1)
    for x in range(1, N+1):
        conflict_count[x] = len(forbidden_pairs[x])
    order = sorted(range(1, N+1), key=lambda x: conflict_count[x])
    S, S_set = greedy_build(order)
    S, S_set = local_search(S, S_set)
    if len(S) > len(best_S):
        best_S = S[:]
        best_set = S_set.copy()
    
    # Strategy 5: Simulated annealing on the binary selection vector
    # State: bitmask of 128 bits, bit i set means element i+1 is selected
    # Energy: number of bad triples in the selected set
    # We want to maximize |S| subject to energy = 0
    
    # Precompute for each triple (a,b,c) a way to check if all three are selected
    # Using bit operations: if state has bits a,b,c all set, it's a bad triple
    
    def count_bad_triples(state):
        """Count bad triples in the given state (bitmask)."""
        count = 0
        for (a,b,c) in bad_triples:
            if (state >> a) & 1 and (state >> b) & 1 and (state >> c) & 1:
                count += 1
        return count
    
    def popcount(x):
        return bin(x).count('1')
    
    # SA: start from best_S, try to improve
    state = 0
    for x in best_S:
        state |= (1 << x)
    
    # Try SA
    if time.time() < deadline - 10:
        T = 5.0
        cooling = 0.999
        best_sa_state = state
        best_sa_size = popcount(state)
        
        # Precompute for each element, which bad triples it's part of
        element_triples = [[] for _ in range(N+1)]
        for idx, (a,b,c) in enumerate(bad_triples):
            element_triples[a].append(idx)
            element_triples[b].append(idx)
            element_triples[c].append(idx)
        
        # For faster energy computation, precompute a list of (a,b,c) triples
        # and for each element, which triples it's in
        
        # SA moves: add an element, remove an element, or swap
        for step in range(50000):
            if time.time() > deadline - 5:
                break
            if T < 0.01:
                break
            
            # Random move
            move_type = random.random()
            if move_type < 0.3:
                # Add a random element not in state
                candidates = [x for x in range(1, N+1) if not (state >> x) & 1]
                if not candidates:
                    continue
                x = random.choice(candidates)
                new_state = state | (1 << x)
            elif move_type < 0.6:
                # Remove a random element from state
                selected = [x for x in range(1, N+1) if (state >> x) & 1]
                if not selected:
                    continue
                x = random.choice(selected)
                new_state = state & ~(1 << x)
            else:
                # Swap: remove one, add one
                selected = [x for x in range(1, N+1) if (state >> x) & 1]
                unselected = [x for x in range(1, N+1) if not (state >> x) & 1]
                if not selected or not unselected:
                    continue
                x_out = random.choice(selected)
                x_in = random.choice(unselected)
                new_state = (state & ~(1 << x_out)) | (1 << x_in)
            
            # Compute energy difference
            # Old energy
            old_energy = count_bad_triples(state)
            new_energy = count_bad_triples(new_state)
            
            old_size = popcount(state)
            new_size = popcount(new_state)
            
            # Score: we want to maximize size, minimize energy
            # Combined objective: size - lambda * energy
            # Accept if new is better, or with probability exp(-delta/T)
            
            # Prefer valid solutions: if new_energy == 0 and old_energy > 0, accept
            # If both valid, prefer larger
            # If new is invalid, accept with lower probability
            
            delta = (new_size - old_size) - 10.0 * (new_energy - old_energy)
            
            if delta > 0 or random.random() < (2.718281828 ** (delta / T)):
                state = new_state
                if new_energy == 0 and new_size > best_sa_size:
                    best_sa_size = new_size
                    best_sa_state = state
            
            T *= cooling
        
        if best_sa_size > len(best_S):
            best_S = [x for x in range(1, N+1) if (best_sa_state >> x) & 1]
            best_set = set(best_S)
    
    return best_S
