import math
import time
import random

def construct():
    start_time = time.time()
    deadline = 60.0  # Keep safety margin
    
    def get_score(S):
        return sum(1.0/x for x in S)
        
    def is_compatible(a, b):
        return math.lcm(a, b) > 256
    
    # Precompute compatibility matrix for speed
    n = 256
    nodes = list(range(2, 257))
    idx = {x: i for i, x in enumerate(nodes)}
    m = len(nodes)
    weights = [1.0/x for x in nodes]
    
    # compatibility[i][j] = True if nodes[i] and nodes[j] can coexist
    compat = [[True]*m for _ in range(m)]
    for i in range(m):
        for j in range(i+1, m):
            compat[i][j] = compat[j][i] = is_compatible(nodes[i], nodes[j])
    
    # Build conflict lists: conflicts[i] = set of j that conflict with i
    conflicts = [set() for _ in range(m)]
    for i in range(m):
        for j in range(m):
            if i != j and not compat[i][j]:
                conflicts[i].add(j)
    
    # Sort nodes by weight descending for better bounding
    order = sorted(range(m), key=lambda i: -weights[i])
    order_inv = [0]*m
    for pos, i in enumerate(order):
        order_inv[i] = pos
    
    # Reindex so that higher weight nodes come first
    new_weights = [weights[i] for i in order]
    new_nodes = [nodes[i] for i in order]
    new_conflicts = [set() for _ in range(m)]
    for i in range(m):
        for j in conflicts[order[i]]:
            new_conflicts[i].add(order_inv[j])
    
    # Branch and bound with fractional knapsack bound
    best_set = set()
    best_score = 0.0
    
    # Initial greedy solution: take nodes in weight order if compatible
    current = set()
    for i in range(m):
        if all(i not in conflicts[j] for j in current):
            current.add(i)
    if get_score({new_nodes[i] for i in current}) > best_score:
        best_score = get_score({new_nodes[i] for i in current})
        best_set = set(current)
    
    def fractional_bound(pos, current_set, current_score):
        """Upper bound using fractional knapsack on remaining nodes."""
        bound = current_score
        remaining = []
        for i in range(pos, m):
            if i in current_set:
                continue
            # Check if i is compatible with all in current_set
            if any(i in conflicts[j] for j in current_set):
                continue
            remaining.append(i)
        
        if not remaining:
            return bound
        
        # Sort by weight (already sorted by new_weights)
        # Fractional knapsack: take in order of weight, but need to check conflicts among themselves
        # Simple bound: just sum of all remaining compatible with current_set
        # Better: sort by weight and take greedily, but account for conflicts
        # For a tighter bound, use the fact that we can at most take a subset
        # Simple upper bound: sum of all remaining weights
        for i in remaining:
            bound += new_weights[i]
        
        return bound
    
    # Improved bound: use the maximum weight independent set approximation
    # on the remaining subgraph as a tighter bound
    def tighter_bound(pos, current_set, current_score):
        """Use greedy on remaining nodes as a lower bound on what's achievable,
           and sum of all compatible weights as upper bound."""
        bound = current_score
        # Upper bound: sum of weights of all nodes compatible with current_set
        for i in range(pos, m):
            if i in current_set:
                continue
            if any(i in conflicts[j] for j in current_set):
                continue
            bound += new_weights[i]
        return bound
    
    # Recursive branch and bound
    def branch_and_bound(pos, current_set, current_score):
        nonlocal best_set, best_score
        
        if time.time() > deadline:
            return
        
        # Update best if current set is valid and better
        if current_score > best_score:
            best_score = current_score
            best_set = set(current_set)
        
        if pos >= m:
            return
        
        # Bound check
        ub = tighter_bound(pos, current_set, current_score)
        if ub <= best_score:
            return
        
        # Try including node pos
        i = pos
        if i not in current_set:
            # Check if i is compatible with all in current_set
            if not any(i in conflicts[j] for j in current_set):
                current_set.add(i)
                branch_and_bound(pos + 1, current_set, current_score + new_weights[i])
                current_set.remove(i)
        
        # Try excluding node pos
        branch_and_bound(pos + 1, current_set, current_score)
    
    # Run B&B with time limit
    # Use iterative deepening or just run until time runs out
    # To avoid deep recursion, use a stack-based approach
    stack = [(0, set(), 0.0)]
    visited = 0
    
    while stack and time.time() < deadline:
        pos, current_set, current_score = stack.pop()
        visited += 1
        
        if current_score > best_score:
            best_score = current_score
            best_set = set(current_set)
        
        if pos >= m:
            continue
        
        ub = tighter_bound(pos, current_set, current_score)
        if ub <= best_score:
            continue
        
        i = pos
        # Exclude first (to keep best_score high early)
        stack.append((pos + 1, current_set, current_score))
        
        # Include if compatible
        if not any(i in conflicts[j] for j in current_set):
            new_set = current_set | {i}
            stack.append((pos + 1, new_set, current_score + new_weights[i]))
    
    # Convert back to original node values
    result = [new_nodes[i] for i in best_set]
    
    # Also try some random restarts with local search as backup
    if time.time() < deadline - 10:
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            random_set = set()
            shuffled = list(range(m))
            random.shuffle(shuffled)
            for i in shuffled:
                if not any(i in conflicts[j] for j in random_set):
                    random_set.add(i)
            rs_score = sum(new_weights[i] for i in random_set)
            if rs_score > best_score:
                best_score = rs_score
                best_set = set(random_set)
        
        # Local search on best_set with 1-for-1 and 2-for-2 swaps
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # 1-for-1 swap
            for i in range(m):
                if i in best_set:
                    continue
                if time.time() > deadline - 2:
                    break
                if not any(i in conflicts[j] for j in best_set):
                    # Can add i
                    best_set.add(i)
                    best_score += new_weights[i]
                    improved = True
            # Try removing and replacing
            for i in range(m):
                if i not in best_set:
                    continue
                if time.time() > deadline - 2:
                    break
                # Try removing i
                temp_set = best_set - {i}
                temp_score = best_score - new_weights[i]
                for j in range(m):
                    if j in best_set:
                        continue
                    if not any(j in conflicts[k] for k in temp_set):
                        if new_weights[j] > new_weights[i]:
                            best_set = temp_set | {j}
                            best_score = temp_score + new_weights[j]
                            improved = True
                            break
    
    result = [new_nodes[i] for i in best_set]
    
    # Ensure non-empty
    if not result:
        result = [2]
    
    return result

# EVOLVE-BLOCK-END
