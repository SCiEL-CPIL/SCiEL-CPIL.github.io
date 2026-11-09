import time
from math import gcd

def construct():
    """
    Find a set of distinct pairwise coprime integers in 1..511 that maximizes
    the sum of 1/(512 - a) using maximum weight independent set formulation
    with Moon-Moser clique cover bound for branch-and-bound.
    """
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s limit
    
    N = 512
    
    # Get all primes < 512
    def get_primes(n):
        sieve = [True] * n
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = get_primes(N)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Precompute prime factor masks for all candidates 1..511
    def get_factor_mask(n):
        mask = 0
        temp = n
        for i, p in enumerate(primes):
            if p * p > temp:
                break
            if temp % p == 0:
                mask |= (1 << i)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            mask |= (1 << prime_to_idx[temp])
        return mask
    
    factor_masks = [0] * N
    for a in range(1, N):
        factor_masks[a] = get_factor_mask(a)
    
    # Precompute scores
    scores = [0.0] * N
    for a in range(1, N):
        scores[a] = 1.0 / (N - a)
    
    # Build conflict graph: two numbers conflict if they share a prime factor
    # Nodes are 1..511, indexed by their value
    # conflict[i] is a bitmask of nodes that conflict with node i
    conflict = [0] * N
    for i in range(1, N):
        for j in range(i+1, N):
            if (factor_masks[i] & factor_masks[j]) != 0:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    
    # Greedy initialization: pick highest score nodes that don't conflict
    def greedy_init():
        chosen = []
        chosen_mask = 0
        # Sort by score descending
        candidates = sorted(range(1, N), key=lambda x: -scores[x])
        for a in candidates:
            if (conflict[a] & chosen_mask) == 0:
                chosen.append(a)
                chosen_mask |= (1 << a)
        return chosen, chosen_mask, sum(scores[a] for a in chosen)
    
    best_chosen, best_mask, best_score = greedy_init()
    
    # Local search: try to improve by swapping
    def local_search(chosen, mask, score):
        chosen_set = set(chosen)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i, a in enumerate(list(chosen)):
                new_mask = mask & ~(1 << a)
                best_b = -1
                best_gain = 0.0
                for b in range(1, N):
                    if b in chosen_set:
                        continue
                    if (conflict[b] & new_mask) == 0:
                        gain = scores[b] - scores[a]
                        if gain > best_gain:
                            best_gain = gain
                            best_b = b
                if best_b != -1 and best_gain > 1e-12:
                    chosen[i] = best_b
                    chosen_set.discard(a)
                    chosen_set.add(best_b)
                    mask = new_mask | (1 << best_b)
                    score += best_gain
                    improved = True
        return chosen, mask, score
    
    best_chosen, best_mask, best_score = local_search(best_chosen, best_mask, best_score)
    
    # Maximum Weight Independent Set via Branch and Bound
    # Use Moon-Moser style bound: partition remaining vertices into cliques
    # Upper bound = sum of max weight in each clique
    
    best_solution = list(best_chosen)
    best_val = best_score
    
    def compute_clique_cover_bound(available_mask):
        """
        Partition available nodes into cliques and return upper bound on 
        maximum weight independent set.
        A clique can contribute at most one node to an independent set.
        """
        remaining = available_mask
        bound = 0.0
        
        while remaining:
            # Find a clique by greedily adding nodes that conflict with all others in the clique
            clique = 0
            clique_weight = 0.0
            candidate_mask = remaining
            
            # Greedy clique construction: pick highest weight node, then add neighbors
            # that conflict with all previously picked nodes in the clique
            while candidate_mask:
                # Pick highest weight node from candidate_mask
                best_node = -1
                best_w = -1.0
                m = candidate_mask
                while m:
                    if m & 1:
                        node = (m & -m).bit_length() - 1
                        if scores[node] > best_w:
                            best_w = scores[node]
                            best_node = node
                    m &= m - 1
                
                if best_node == -1:
                    break
                
                # Add to clique
                clique |= (1 << best_node)
                clique_weight = max(clique_weight, scores[best_node])
                
                # Remove from candidates
                candidate_mask &= ~(1 << best_node)
                
                # Only keep candidates that conflict with best_node (must be in same clique)
                candidate_mask &= conflict[best_node]
            
            bound += clique_weight
            remaining &= ~clique
        
        return bound
    
    def branch_and_bound(available_mask, current_score, chosen_list):
        nonlocal best_solution, best_val
        
        if time.time() > deadline:
            return
        
        # Update best if we have a valid solution
        if current_score > best_val:
            best_val = current_score
            best_solution = list(chosen_list)
        
        # If no available nodes, return
        if available_mask == 0:
            return
        
        # Compute upper bound using clique cover
        ub = compute_clique_cover_bound(available_mask) + current_score
        if ub <= best_val + 1e-12:
            return
        
        # Select branching node: highest weight node in available
        best_i = -1
        best_score_i = -1.0
        m = available_mask
        while m:
            if m & 1:
                node = (m & -m).bit_length() - 1
                if scores[node] > best_score_i:
                    best_score_i = scores[node]
                    best_i = node
            m &= m - 1
        
        if best_i == -1:
            return
        
        # Branch 1: include best_i
        new_available = available_mask & ~(1 << best_i)
        # Remove all nodes that conflict with best_i
        new_available &= ~conflict[best_i]
        
        branch_and_bound(new_available, current_score + best_score_i, 
                        chosen_list + [best_i])
        
        # Branch 2: exclude best_i
        new_available = available_mask & ~(1 << best_i)
        branch_and_bound(new_available, current_score, chosen_list)
    
    # Start branch and bound
    # Build initial available mask: all nodes not in chosen and not conflicting with chosen
    initial_mask = 0
    for a in range(1, N):
        if a not in best_chosen:
            if (conflict[a] & best_mask) == 0:
                initial_mask |= (1 << a)
    
    branch_and_bound(initial_mask, best_val, list(best_chosen))
    
    # Final local search on the best solution
    if time.time() < deadline:
        final_mask = 0
        for a in best_solution:
            final_mask |= (1 << a)
        best_solution, final_mask, best_val = local_search(best_solution, final_mask, best_val)
    
    return best_solution
