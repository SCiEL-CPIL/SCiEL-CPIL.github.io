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
    
    # Precompute conflict pairs for each x
    # conflict_pairs[x] = list of (y, z) such that {x,y,z} is a 3-AP
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
    
    # Bitset representation for faster conflict checking
    # For each x, create a bitset where bit i is set if i conflicts with x
    # Then we can check if any two conflicting elements are both in S using bit operations
    
    # Actually, let's use a different approach: for each x, precompute the list of 
    # "witness" elements that would complete a 3-AP with x and some other element in S
    # 
    # Better: For each x, create a set of "blocking pairs" (y,z) where if both y and z 
    # are in S, then x cannot be added.
    # 
    # Even better: Use numpy for bitset operations
    
    import numpy as np
    
    # Create bitsets for each x: bit i is set if element i is part of a 3-AP with x
    # But we need pairs, not just individual elements.
    # 
    # Let's stick with the list approach but optimize the check
    
    def can_add_fast(x, in_S):
        """Check if x can be added to the set defined by in_S without creating a 3-AP."""
        for y, z in conflict_pairs[x]:
            if in_S[y] and in_S[z]:
                return False
        return True
    
    # Multi-start greedy with diverse strategies
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
    
    # Strategy 2: Multiple random restarts with different biases
    import random
    rng = random.Random(42)
    
    # Try different random seeds and bias strategies
    for trial in range(20):
        if time.time() > deadline - 40:
            break
        
        in_S = [False] * (N + 1)
        S = set()
        
        # Create a biased ordering
        order = list(range(1, N + 1))
        
        # Different bias strategies
        if trial % 4 == 0:
            # Uniform random
            rng.shuffle(order)
        elif trial % 4 == 1:
            # Bias towards small numbers (higher weights)
            for i in range(len(order) - 1):
                j = rng.randint(0, len(order) - 1)
                if order[i] > order[j] and rng.random() < 0.5:
                    order[i], order[j] = order[j], order[i]
        elif trial % 4 == 2:
            # Bias towards large numbers
            for i in range(len(order) - 1):
                j = rng.randint(0, len(order) - 1)
                if order[i] < order[j] and rng.random() < 0.3:
                    order[i], order[j] = order[j], order[i]
        else:
            # Weighted random based on 1/x
            # Sample without replacement with probability proportional to 1/x
            remaining = list(range(1, N + 1))
            for _ in range(N):
                total_weight = sum(weights[x] for x in remaining)
                r = rng.random() * total_weight
                cum = 0
                for idx, x in enumerate(remaining):
                    cum += weights[x]
                    if r <= cum:
                        order.append(x)
                        remaining.pop(idx)
                        break
        
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
    max_iter = 50
    iter_count = 0
    
    while improved and time.time() < deadline - 10 and iter_count < max_iter:
        improved = False
        best_gain = 0
        best_move = None
        iter_count += 1
        
        # Try single additions
        for v in range(1, N + 1):
            if not in_S[v] and can_add_fast(v, in_S):
                gain = weights[v]
                if gain > best_gain:
                    best_gain = gain
                    best_move = ('add', v)
        
        # Try single swaps: remove u, add v
        # Only consider v that has high weight and u that has low weight
        sorted_S = sorted(S, key=lambda x: weights[x])
        for u in sorted_S[:50]:  # Only try removing low-weight elements
            in_S[u] = False
            for v in range(1, N + 1):
                if not in_S[v] and can_add_fast(v, in_S):
                    gain = weights[v] - weights[u]
                    if gain > best_gain:
                        best_gain = gain
                        best_move = ('swap', u, v)
            in_S[u] = True
        
        if best_move and best_gain > 0:
            if best_move[0] == 'add':
                v = best_move[1]
                S.add(v)
                in_S[v] = True
            elif best_move[0] == 'swap':
                u, v = best_move[1], best_move[2]
                S.remove(u)
                in_S[u] = False
                S.add(v)
                in_S[v] = True
            
            best_score = sum(weights[x] for x in S)
            best_set = S.copy()
            improved = True
        else:
            # Try 2-swap: remove u1, u2, add v1, v2
            # This is more expensive, only try if we have time
            if time.time() < deadline - 15:
                found_2swap = False
                # Sample some pairs to keep it fast
                candidates_v = [v for v in range(1, N + 1) if not in_S[v]]
                candidates_u = list(S)
                
                # Limit candidates for speed
                rng2 = random.Random(iter_count)
                rng2.shuffle(candidates_v)
                candidates_v = candidates_v[:30]
                rng2.shuffle(candidates_u)
                candidates_u = candidates_u[:30]
                
                for v1 in candidates_v:
                    for v2 in candidates_v:
                        if v2 <= v1:
                            continue
                        for u1 in candidates_u:
                            in_S[u1] = False
                            for u2 in candidates_u:
                                if u2 == u1:
                                    continue
                                in_S[u2] = False
                                if can_add_fast(v1, in_S) and can_add_fast(v2, in_S):
                                    gain = weights[v1] + weights[v2] - weights[u1] - weights[u2]
                                    if gain > best_gain:
                                        best_gain = gain
                                        best_move = ('2swap', u1, u2, v1, v2)
                                in_S[u2] = True
                            in_S[u1] = True
                
                if best_move and best_move[0] == '2swap' and best_gain > 0:
                    u1, u2, v1, v2 = best_move[1], best_move[2], best_move[3], best_move[4]
                    S.remove(u1)
                    in_S[u1] = False
                    S.remove(u2)
                    in_S[u2] = False
                    S.add(v1)
                    in_S[v1] = True
                    S.add(v2)
                    in_S[v2] = True
                    
                    best_score = sum(weights[x] for x in S)
                    best_set = S.copy()
                    improved = True
    
    # Final polish: try to add any remaining elements that fit
    S = best_set.copy()
    in_S = [False] * (N + 1)
    for x in S:
        in_S[x] = True
    
    for v in range(1, N + 1):
        if not in_S[v] and can_add_fast(v, in_S):
            S.add(v)
            in_S[v] = True
    
    if S:
        final_score = sum(weights[x] for x in S)
        if final_score > best_score:
            best_score = final_score
            best_set = S.copy()
    
    return list(best_set)
