"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations
import random

def _lcm(a, b):
    return a * b // gcd(a, b)

def _num_omega(n):
    """Count distinct prime factors."""
    count = 0
    d = 2
    while d * d <= n:
        if n % d == 0:
            count += 1
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        count += 1
    return count

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    n = 128
    universe = list(range(1, n + 1))
    universe_set = set(universe)

    # Precompute LCM table for all pairs
    lcm_table = {}
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[(i, j)] = _lcm(i, j)

    def get_lcm(a, b):
        if a <= b:
            return lcm_table[(a, b)]
        else:
            return lcm_table[(b, a)]

    # Find all bad triples: (a,b,c) where lcm(a,b)==lcm(a,c)==lcm(b,c)
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = get_lcm(a, b)
            for c in range(b + 1, n + 1):
                if get_lcm(a, c) == lab and get_lcm(b, c) == lab:
                    bad_triples.append((a, b, c))

    # Build conflict structure using bitsets
    # conflict[i] is a bitset of vertices that conflict with i (i.e., appear in a bad triple with i)
    # For a 3-uniform hypergraph, vertex i conflicts with j if there exists k such that (i,j,k) is a bad triple
    # But we need to track which specific pairs conflict given a chosen third element.
    # Actually for independent set in 3-uniform hypergraph: a set S is independent if no hyperedge is fully contained in S.
    # So we need: for each pair (i,j), the set of k such that (i,j,k) is a bad triple.
    
    # pair_conflicts[(i,j)] = set of k such that (i,j,k) is a bad triple (with i<j<k or any ordering)
    pair_conflicts = {}
    for (a, b, c) in bad_triples:
        for (x, y) in [(a,b), (a,c), (b,c)]:
            key = (min(x,y), max(x,y))
            if key not in pair_conflicts:
                pair_conflicts[key] = set()
            pair_conflicts[key].add(max(a,b,c) if max(a,b,c) not in (x,y) else min(a,b,c))
            # Actually let me redo this more carefully
            third = a + b + c - x - y
            pair_conflicts[key].add(third)

    # For branch and bound on 3-uniform hypergraph independent set:
    # State: chosen set, remaining candidates
    # A candidate x can be added if for all pairs (x,y) where y is already chosen,
    # there is no z in remaining such that (x,y,z) is a bad triple.
    # Wait, that's not quite right. Let me think again.
    #
    # We want the largest set S with no bad triple fully in S.
    # When building S incrementally, when we consider adding x to S:
    #   For each pair (x, y) where y is already in S, we need to ensure that
    #   there is no z (in S or to be added) such that (x,y,z) is a bad triple.
    #   But z could be in S already or added later.
    #
    # Better approach: maintain the set of "forbidden" vertices.
    # If we have chosen {a, b} so far, then any c such that (a,b,c) is bad is forbidden.
    # When we add a new vertex x, we must check that x is not forbidden by any pair in the chosen set.
    # Also, adding x creates new forbidden vertices: for each y in chosen, any z such that (x,y,z) is bad.
    
    # Let's use a different representation:
    # forbidden_mask: bitset of vertices that cannot be added
    # chosen_mask: bitset of vertices already chosen
    # remaining_mask: bitset of vertices not yet decided
    
    # When we add vertex x:
    #   new_forbidden = forbidden_mask
    #   For each y in chosen: for each z in pair_conflicts[(min(x,y),max(x,y))]:
    #       if z != x and z != y: new_forbidden |= (1 << z)
    #   Also x itself becomes chosen, so it's no longer in remaining.
    
    # For upper bound: |chosen| + |remaining & ~forbidden|
    # This is a valid upper bound since we can't add any forbidden vertex.
    
    # Let's implement branch and bound with this approach.
    
    # Precompute for each pair the third elements
    pair_thirds = {}
    for (a, b, c) in bad_triples:
        for (x, y) in [(a,b), (a,c), (b,c)]:
            key = (min(x,y), max(x,y))
            third = a + b + c - x - y
            if key not in pair_thirds:
                pair_thirds[key] = set()
            pair_thirds[key].add(third)
    
    # Convert to bitsets for speed
    pair_thirds_bits = {}
    for key, thirds in pair_thirds.items():
        bits = 0
        for t in thirds:
            bits |= (1 << t)
        pair_thirds_bits[key] = bits
    
    # Precompute for each vertex x, the list of (y, third_bits) for all y
    # such that (x,y) is a pair in pair_thirds
    vertex_pairs = [[] for _ in range(n + 1)]
    for (x, y), thirds in pair_thirds.items():
        bits = pair_thirds_bits[(x, y)]
        vertex_pairs[x].append((y, bits))
        vertex_pairs[y].append((x, bits))
    
    best_size = 0
    best_set = set()
    
    # Use iterative deepening or just branch and bound
    # Branch and bound:
    # def bnb(chosen_mask, forbidden_mask, remaining_mask):
    #     upper_bound = bit_count(chosen_mask) + bit_count(remaining_mask & ~forbidden_mask)
    #     if upper_bound <= best_size: return
    #     if remaining_mask == 0:
    #         if bit_count(chosen_mask) > best_size:
    #             best_size = bit_count(chosen_mask)
    #             best_set = ...
    #         return
    #     # Pick a vertex from remaining & ~forbidden
    #     # Branch: include or exclude
    
    # For performance, let's use a greedy coloring upper bound instead of just |remaining & ~forbidden|
    # Actually, for 3-uniform hypergraph, a simple bound is:
    #   ub = |chosen| + |remaining & ~forbidden|
    # But we can do better by noting that among the remaining non-forbidden vertices,
    # some pairs will forbid others.
    
    # Let me start with the simple bound and see if it's fast enough.
    
    def bit_count(x):
        return bin(x).count('1')
    
    # Precompute bit masks for vertices 1..128
    # Vertex i has bit (1 << i)
    
    def solve_bnb():
        nonlocal best_size, best_set
        chosen_mask = 0
        forbidden_mask = 0
        remaining_mask = 0
        for i in range(1, n + 1):
            remaining_mask |= (1 << i)
        
        # Order vertices by degree (number of conflicts) for better pruning
        # Higher degree vertices first tend to prune better
        degrees = [0] * (n + 1)
        for i in range(1, n + 1):
            for (y, bits) in vertex_pairs[i]:
                degrees[i] += bit_count(bits)
        
        # Sort vertices by degree descending
        order = sorted(range(1, n + 1), key=lambda x: -degrees[x])
        
        def bnb_recursive(chosen_mask, forbidden_mask, remaining_mask, depth):
            nonlocal best_size, best_set
            
            # Time check
            if time.time() - start_time > DEADLINE:
                return
            
            # Upper bound
            available = remaining_mask & ~forbidden_mask
            ub = bit_count(chosen_mask) + bit_count(available)
            if ub <= best_size:
                return
            
            if available == 0:
                if bit_count(chosen_mask) > best_size:
                    best_size = bit_count(chosen_mask)
                    best_set = set(i for i in range(1, n+1) if (chosen_mask >> i) & 1)
                return
            
            # Pick the first available vertex in our ordering
            # Find the vertex with highest degree in available
            best_v = -1
            best_deg = -1
            temp = available
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp ^= (1 << v)
                if v <= n:
                    deg = 0
                    for (y, bits) in vertex_pairs[v]:
                        if (available >> y) & 1:
                            deg += 1
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                if time.time() - start_time > DEADLINE:
                    return
            
            if best_v == -1:
                if bit_count(chosen_mask) > best_size:
                    best_size = bit_count(chosen_mask)
                    best_set = set(i for i in range(1, n+1) if (chosen_mask >> i) & 1)
                return
            
            v = best_v
            v_bit = 1 << v
            
            # Branch 1: Include v
            new_forbidden = forbidden_mask
            for (y, bits) in vertex_pairs[v]:
                if (chosen_mask >> y) & 1:
                    new_forbidden |= bits
            
            new_chosen = chosen_mask | v_bit
            new_remaining = remaining_mask & ~v_bit
            
            bnb_recursive(new_chosen, new_forbidden, new_remaining, depth + 1)
            
            # Branch 2: Exclude v
            bnb_recursive(chosen_mask, forbidden_mask, new_remaining, depth + 1)
        
        # Initial greedy solution for lower bound
        greedy_set = set()
        greedy_forbidden = set()
        for v in order:
            if v not in greedy_forbidden:
                greedy_set.add(v)
                for (y, bits) in vertex_pairs[v]:
                    if y in greedy_set:
                        greedy_forbidden |= set(i for i in range(1, n+1) if (bits >> i) & 1)
        
        best_size = len(greedy_set)
        best_set = greedy_set
        
        # Now run branch and bound
        bnb_recursive(0, 0, remaining_mask, 0)
    
    # Run a few greedy passes to get a good initial lower bound
    strategies = [0, 1, 3, 4]
    
    def make_order(seed, strategy):
        if strategy == 0:
            return list(universe)
        elif strategy == 1:
            return sorted(universe, key=lambda x: (_num_omega(x), x))
        elif strategy == 2:
            random.seed(seed)
            order = list(universe)
            random.shuffle(order)
            return order
        elif strategy == 3:
            return list(reversed(universe))
        elif strategy == 4:
            return sorted(universe, key=lambda x: (-_num_omega(x), x))
        elif strategy == 5:
            random.seed(seed)
            weights = [1.0 / (1 + _num_omega(x)) for x in universe]
            order = []
            remaining = list(universe)
            remaining_weights = weights[:]
            while remaining:
                total = sum(remaining_weights)
                r = random.random() * total
                cum = 0
                for idx, w in enumerate(remaining_weights):
                    cum += w
                    if cum >= r:
                        order.append(remaining[idx])
                        remaining.pop(idx)
                        remaining_weights.pop(idx)
                        break
            return order
        return list(universe)
    
    def greedy(order):
        selected = []
        selected_set = set()
        for x in order:
            # Check if x can be added
            can_add = True
            for (y, bits) in vertex_pairs[x]:
                if y in selected_set:
                    # Check if any third element is in selected_set
                    for z in range(1, n+1):
                        if (bits >> z) & 1 and z in selected_set:
                            can_add = False
                            break
            if can_add:
                selected.append(x)
                selected_set.add(x)
        return selected
    
    # Run greedy passes
    for s in strategies:
        if time.time() - start_time > DEADLINE - 5:
            break
        order = make_order(0, s)
        result = greedy(order)
        if len(result) > best_size:
            best_size = len(result)
            best_set = set(result)
    
    for i in range(30):
        if time.time() - start_time > DEADLINE - 5:
            break
        order = make_order(i * 12345 + 67890, 2)
        result = greedy(order)
        if len(result) > best_size:
            best_size = len(result)
            best_set = set(result)
    
    # Now run branch and bound to try to find the optimal
    if time.time() - start_time < DEADLINE - 3:
        solve_bnb()
    
    # Ensure nonempty
    if not best_set:
        best_set = {1}
    
    return list(best_set)
# EVOLVE-BLOCK-END
