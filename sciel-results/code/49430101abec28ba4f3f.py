import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    N = 256
    
    # Weights: 1/i
    weights = [0.0] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = 1.0 / i
    
    # Precompute for each pair (a,b), the third element c that would complete a 3-AP
    # If c is in [1,N] and c != a and c != b, then {a,b,c} is a 3-AP
    # We need: for each x, given current set S, check if adding x creates any 3-AP
    
    # Precompute: for each x, list of (a, b) pairs such that {x, a, b} is a 3-AP
    # Actually, more useful: for each x, which y in [1,N] would conflict with x
    # given a third element z. But the conflict depends on what's in S.
    
    # Better approach: for each x, precompute all y such that there exists z in [1,N]
    # with {x, y, z} being a 3-AP. Then when checking if x can be added to S,
    # we check if any such y is in S AND the corresponding z is also in S.
    
    # For each x, compute the set of (y, z) pairs where {x,y,z} is a 3-AP
    # This is: 
    #   Case 1: x is first: x, y, z with y-x = z-y => z = 2y - x. For each y in [x+1, N], z = 2y-x. If z <= N, then (y,z) is a conflict pair.
    #   Case 2: x is middle: y, x, z with x-y = z-x => y+z = 2x. For each y in [1, x-1], z = 2x-y. If z > x and z <= N, then (y,z) is a conflict pair.
    #   Case 3: x is last: y, z, x with z-y = x-z => y+z = 2x... wait no. y, z, x: z-y = x-z => 2z = y+x => z = (y+x)/2. For each y in [1, x-1] with (y+x) even, z = (y+x)//2. If 1 <= z < x, then (y,z) is a conflict pair.
    
    # Let me precompute: for each x, a list of (y, z) such that {x,y,z} is a 3-AP with y != z, y != x, z != x
    # When adding x to S, we need to check: for each (y,z) in conflicts[x], are BOTH y and z in S?
    # If yes, then adding x would create a 3-AP.
    
    # Actually, that's not quite right. The 3-AP is {x, y, z}. If y and z are both in S, then adding x creates the 3-AP.
    # But we also need to check: does x form a 3-AP with two elements already in S where x is one of the three?
    # Yes, that's exactly what we're checking.
    
    # Precompute conflict_pairs[x] = list of (y, z) such that {x, y, z} is a 3-AP, y != x, z != x, y != z
    conflict_pairs = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        seen = set()
        # x is first: x, y, z with z = 2y - x, y > x
        for y in range(x + 1, N + 1):
            z = 2 * y - x
            if z <= N and z != x and z != y:
                pair = (min(y, z), max(y, z))
                if pair not in seen:
                    seen.add(pair)
                    conflict_pairs[x].append((y, z))
        # x is middle: y, x, z with y < x < z, y + z = 2x
        for y in range(1, x):
            z = 2 * x - y
            if z > x and z <= N:
                pair = (min(y, z), max(y, z))
                if pair not in seen:
                    seen.add(pair)
                    conflict_pairs[x].append((y, z))
        # x is last: y, z, x with y < z < x, y + x = 2z => z = (y+x)/2
        for y in range(1, x):
            s = y + x
            if s % 2 == 0:
                z = s // 2
                if 1 <= z < x and z != y:
                    pair = (min(y, z), max(y, z))
                    if pair not in seen:
                        seen.add(pair)
                        conflict_pairs[x].append((y, z))
    
    def can_add_fast(x, in_S):
        """Check if x can be added to the set defined by in_S without creating a 3-AP."""
        for y, z in conflict_pairs[x]:
            if in_S[y] and in_S[z]:
                return False
        return True
    
    # Greedy initialization with multiple strategies
    best_set = set()
    best_score = 0.0
    
    # Strategy 1: Greedy by increasing index (highest weight first)
    in_S = [False] * (N + 1)
    S = set()
    for x in range(1, N + 1):
        if can_add_fast(x, in_S):
            S.add(x)
            in_S[x] = True
    score = sum(weights[x] for x in S)
    if score > best_score:
        best_score = score
        best_set = S.copy()
    
    # Strategy 2: Greedy with slight randomization - try a few different orderings
    import random
    rng = random.Random(42)
    for trial in range(5):
        if time.time() > deadline - 30:
            break
        in_S = [False] * (N + 1)
        S = set()
        # Shuffle but keep small numbers prioritized
        order = list(range(1, N + 1))
        # Bias towards smaller numbers
        for i in range(len(order) - 1):
            j = rng.randint(0, len(order) - 1)
            # With some probability, swap to put smaller number earlier
            if order[i] > order[j] and rng.random() < 0.3:
                order[i], order[j] = order[j], order[i]
        for x in order:
            if can_add_fast(x, in_S):
                S.add(x)
                in_S[x] = True
        score = sum(weights[x] for x in S)
        if score > best_score:
            best_score = score
            best_set = S.copy()
    
    # Local search on the best solution found
    S = best_set.copy()
    in_S = [False] * (N + 1)
    for x in S:
        in_S[x] = True
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        best_gain = 0
        best_move = None
        
        # Try single swaps: remove u, add v
        for v in range(1, N + 1):
            if in_S[v]:
                continue
            # Check if v can be added after removing some u
            # First, find all u in S that conflict with v (i.e., removing u might allow v)
            # v conflicts with u if there exists w such that {v, u, w} is a 3-AP and w is in S
            # Actually, we need: after removing u, can v be added?
            # This is expensive. Let's try: for each u in S, remove u, check if v can be added, then put u back.
            
            # Optimization: only try u that are "close" to v or known to conflict
            # For now, try all u in S
            for u in list(S):
                # Temporarily remove u
                in_S[u] = False
                if can_add_fast(v, in_S):
                    gain = weights[v] - weights[u]
                    if gain > best_gain:
                        best_gain = gain
                        best_move = ('swap', u, v)
                in_S[u] = True
        
        # Try adding v if valid (no removal needed)
        for v in range(1, N + 1):
            if not in_S[v] and can_add_fast(v, in_S):
                gain = weights[v]
                if gain > best_gain:
                    best_gain = gain
                    best_move = ('add', v)
        
        if best_move and best_gain > 0:
            if best_move[0] == 'swap':
                u, v = best_move[1], best_move[2]
                S.remove(u)
                in_S[u] = False
                S.add(v)
                in_S[v] = True
            elif best_move[0] == 'add':
                v = best_move[1]
                S.add(v)
                in_S[v] = True
            
            best_score = sum(weights[x] for x in S)
            best_set = S.copy()
            improved = True
        else:
            # Try 2-swap: remove u1, u2, add v1, v2
            # This is more expensive, only try if we have time
            if time.time() < deadline - 10:
                found_2swap = False
                # Sample some pairs to keep it fast
                candidates = list(range(1, N + 1))
                rng2 = random.Random(123)
                rng2.shuffle(candidates)
                candidates = candidates[:50]  # Limit candidates
                
                for v1 in candidates:
                    if in_S[v1]:
                        continue
                    for v2 in candidates:
                        if v2 <= v1 or in_S[v2]:
                            continue
                        # Try to find u1, u2 to remove
                        # This is very expensive, so we'll skip for now
                        # and rely on single swaps
                        pass
                
                # If no improvement from single moves, try a randomized restart
                if not improved:
                    # Try a fresh greedy with different seed
                    in_S2 = [False] * (N + 1)
                    S2 = set()
                    order = list(range(1, N + 1))
                    rng3 = random.Random(int(time.time() * 1000) % 10000)
                    rng3.shuffle(order)
                    # Bias towards small numbers
                    for i in range(len(order)):
                        if order[i] > 32:
                            # Swap with a random earlier position
                            j = rng3.randint(0, i - 1) if i > 0 else 0
                            if order[j] < order[i]:
                                order[i], order[j] = order[j], order[i]
                    for x in order:
                        if can_add_fast(x, in_S2):
                            S2.add(x)
                            in_S2[x] = True
                    score2 = sum(weights[x] for x in S2)
                    if score2 > best_score:
                        best_score = score2
                        best_set = S2.copy()
                        S = S2.copy()
                        in_S = in_S2.copy()
                        improved = True
    
    return list(best_set)
