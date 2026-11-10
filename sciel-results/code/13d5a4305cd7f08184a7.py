import time
import sys
import random

def construct():
    deadline = time.time() + 85
    N = 256
    
    # 1. Build Graph
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # 2. Initial Solution (Greedy with odd-number base)
    # Start with odds
    S = set(range(1, N + 1, 2))
    # Add evens if possible
    for e in range(2, N + 1, 2):
        ok = True
        for y in S:
            if (adj[e] >> y) & 1:
                ok = False
                break
        if ok:
            S.add(e)
            
    best_size = len(S)
    best_set = S.copy()
    
    # 3. Local Search with 1-for-2 swaps
    def can_add(element, current_set, adj_list):
        for y in current_set:
            if (adj_list[element] >> y) & 1:
                return False
        return True
    
    def local_search(S, adj, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            S_list = list(S)
            random.shuffle(S_list)
            
            # Try 1-for-1 swaps: remove one element, add another
            for remove_elem in S_list:
                if time.time() > deadline:
                    return S
                S_without = S - {remove_elem}
                for add_elem in range(1, N + 1):
                    if add_elem in S_without:
                        continue
                    if can_add(add_elem, S_without, adj):
                        new_S = S_without | {add_elem}
                        if len(new_S) > len(S):
                            S = new_S
                            improved = True
                            break
                if improved:
                    break
            
            # Try 1-for-2 swaps: remove one element, add two others
            if not improved:
                for remove_elem in S_list:
                    if time.time() > deadline:
                        return S
                    S_without = S - {remove_elem}
                    # Find all elements that can be added to S_without
                    candidates = []
                    for add_elem in range(1, N + 1):
                        if add_elem in S_without:
                            continue
                        if can_add(add_elem, S_without, adj):
                            candidates.append(add_elem)
                    
                    # Try pairs from candidates
                    for i in range(len(candidates)):
                        if time.time() > deadline:
                            return S
                        for j in range(i + 1, len(candidates)):
                            c1, c2 = candidates[i], candidates[j]
                            # Check if c1 and c2 are compatible
                            if (adj[c1] >> c2) & 1:
                                continue
                            new_S = S_without | {c1, c2}
                            if len(new_S) > len(S):
                                S = new_S
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        return S
    
    # Run local search multiple times with different random seeds
    random.seed(42)
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
        
        # Start from current best or a perturbed version
        if attempt % 5 == 0 and attempt > 0:
            # Perturb: remove 2-3 random elements and re-add greedily
            S_temp = best_set.copy()
            to_remove = random.sample(list(S_temp), min(3, len(S_temp)))
            for elem in to_remove:
                S_temp.discard(elem)
            # Re-add greedily
            for e in range(1, N + 1):
                if e not in S_temp and can_add(e, S_temp, adj):
                    S_temp.add(e)
        else:
            S_temp = best_set.copy()
        
        S_result = local_search(S_temp, adj, deadline - 5)
        
        if len(S_result) > best_size:
            best_size = len(S_result)
            best_set = S_result
    
    # Final polish: try to add any remaining elements
    final_S = best_set.copy()
    for e in range(1, N + 1):
        if e not in final_S and can_add(e, final_S, adj):
            final_S.add(e)
    
    if len(final_S) > best_size:
        best_size = len(final_S)
        best_set = final_S
    
    return list(best_set)

construct()
