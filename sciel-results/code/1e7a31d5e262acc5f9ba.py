import time
import random
from math import gcd

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin
    
    N = 256
    # Nodes are 2..256
    nodes = list(range(2, N + 1))
    
    # Precompute incompatibility sets
    # incompat[x] is a set of y such that lcm(x, y) <= 256
    incompat = {x: set() for x in nodes}
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a = nodes[i]
            b = nodes[j]
            # lcm(a, b) = a * b // gcd(a, b)
            g = gcd(a, b)
            if (a * b) // g <= 256:
                incompat[a].add(b)
                incompat[b].add(a)
                
    def score(S):
        return sum(1.0 / x for x in S)
        
    def is_compatible(x, S):
        # Check if x is compatible with all y in S
        # i.e., no y in S is in incompat[x]
        # Since incompat[x] is a set, we can check intersection
        # But S is a set too.
        # Faster: check if any y in incompat[x] is in S
        # Or check if incompat[x] & S is empty
        return len(incompat[x] & S) == 0
        
    def greedy_construct(order):
        S = set()
        for x in order:
            if is_compatible(x, S):
                S.add(x)
        return S
        
    def local_search(S):
        S = set(S)
        best_score = score(S)
        improved = True
        while improved:
            improved = False
            
            # Phase 1: Try to add any compatible node
            # Sort candidates by weight (descending) i.e., value (ascending)
            # Candidates are nodes not in S
            candidates = [x for x in nodes if x not in S]
            candidates.sort(key=lambda x: x) # Smaller x = higher weight
            
            for v in candidates:
                if is_compatible(v, S):
                    S.add(v)
                    best_score += 1.0 / v
                    improved = True
                    # After adding, other candidates might become incompatible
                    # But we are iterating a static list. 
                    # It's safer to re-evaluate or just break and restart loop.
                    # Since we want max weight, adding the smallest compatible is good.
                    # But adding one might block another.
                    # Greedy add is fine. We'll loop again.
                    break # Restart the while loop to re-evaluate candidates
                    
            if improved:
                continue
                
            # Phase 2: Try 1-swap
            # For each u in S, try to find best v in S^c such that 
            # v is compatible with S \ {u} and 1/v > 1/u
            best_gain = 0
            best_swap = None
            
            # Optimization: 
            # Instead of checking all v for all u, we can be smarter.
            # But N is small.
            S_list = list(S)
            S_set = S
            candidates_not_in_S = [x for x in nodes if x not in S_set]
            # Sort candidates by weight (small x first) to find best v quickly?
            # We want max (1/v - 1/u). Since 1/u is fixed for a given u, we want min v.
            # So for each u, we just need the smallest v compatible with S\{u}.
            
            for u in S_list:
                # S without u
                S_without_u = S_set - {u}
                # Find smallest v in candidates_not_in_S compatible with S_without_u
                best_v = None
                for v in candidates_not_in_S:
                    # v must be compatible with S_without_u
                    # Note: v might be incompatible with u, but u is removed.
                    # v must not be in S_without_u (it isn't, by definition of candidates)
                    if is_compatible(v, S_without_u):
                        # Since candidates are sorted by x (ascending), the first one found is the smallest (highest weight)
                        best_v = v
                        break
                
                if best_v is not None:
                    gain = (1.0 / best_v) - (1.0 / u)
                    if gain > best_gain + 1e-12:
                        best_gain = gain
                        best_swap = (u, best_v)
                        
            if best_swap is not None:
                u, v = best_swap
                S.remove(u)
                S.add(v)
                best_score += best_gain
                improved = True
                continue
                
            # Phase 3: Try 1-remove-greedy-add
            # For each u in S, remove it, then greedily add best compatible nodes from S^c
            # If the resulting set has higher score, accept.
            best_move = None
            best_move_score = best_score
            
            for u in S_list:
                S_temp = S_set - {u}
                current_score = best_score - (1.0 / u)
                
                # Greedily add compatible nodes from candidates_not_in_S
                # We should add the highest weight (smallest x) compatible ones first
                # candidates_not_in_S is already sorted by x (ascending)
                added_any = False
                for v in candidates_not_in_S:
                    if is_compatible(v, S_temp):
                        S_temp.add(v)
                        current_score += 1.0 / v
                        added_any = True
                        
                if current_score > best_move_score + 1e-12:
                    best_move_score = current_score
                    best_move = (u, S_temp)
                    
            if best_move is not None:
                u, S_new = best_move
                S = S_new
                best_score = best_move_score
                improved = True
                continue
                
            # Phase 4: Try 2-swaps? 
            # Might be too slow. Let's rely on restarts.
            
        return S

    # Main Search
    best_S = set()
    best_val = 0.0
    
    # Initial Solution 1: Greedy by smallest x
    S1 = greedy_construct(nodes)
    S1 = local_search(S1)
    val1 = score(S1)
    if val1 > best_val:
        best_val = val1
        best_S = S1
        
    # Initial Solution 2: Greedy by largest x (to see if it leads to different local optima)
    S2 = greedy_construct(sorted(nodes, reverse=True))
    S2 = local_search(S2)
    val2 = score(S2)
    if val2 > best_val:
        best_val = val2
        best_S = S2
        
    # Random Restarts
    # We have time. Let's run multiple random restarts.
    # Perturbation: Start with a random subset, or shuffle the greedy order.
    
    num_restarts = 50 # We will check time in loop
    
    for i in range(num_restarts):
        if time.time() - start_time > timeout:
            break
            
        # Random order biased towards small numbers?
        # Or just random shuffle.
        random_order = nodes[:]
        random.shuffle(random_order)
        
        S_rand = greedy_construct(random_order)
        S_rand = local_search(S_rand)
        val_rand = score(S_rand)
        
        if val_rand > best_val:
            best_val = val_rand
            best_S = S_rand
            
    # Final Polish on Best Solution
    # Run local search again on the best solution found, maybe with more patience or different order?
    # The local_search function is deterministic given the set.
    # But we can try to perturb the best solution and search again.
    
    # Perturbation: Remove a random node from best_S and re-search
    if len(best_S) > 1:
        S_perturb = set(best_S)
        # Remove a random node
        node_to_remove = random.choice(list(S_perturb))
        S_perturb.remove(node_to_remove)
        S_perturb = local_search(S_perturb)
        val_perturb = score(S_perturb)
        if val_perturb > best_val:
            best_val = val_perturb
            best_S = S_perturb

    return list(best_S)
