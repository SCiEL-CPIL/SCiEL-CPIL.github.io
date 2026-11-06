import time
from math import gcd
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    # Candidates: numbers in [1, 255] that share a factor with 255 (3, 5, 17)
    # Any number in the clique must have gcd > 1 with 255, so must be div by 3, 5, or 17.
    candidates = [x for x in range(1, 256) if x % 3 == 0 or x % 5 == 0 or x % 17 == 0]
    n = len(candidates)
    
    # Build adjacency list: edge if gcd > 1
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Pre-compute degrees for ordering
    degrees = [len(adj[i]) for i in range(n)]
    
    def greedy_clique(order):
        clique = []
        for v in order:
            if all(u in adj[v] for u in clique):
                clique.append(v)
        return clique

    best_clique = []
    best_size = 0
    
    # 1. Try simple greedy strategies
    # Order by degree descending
    order_desc = sorted(range(n), key=lambda i: -degrees[i])
    clique = greedy_clique(order_desc)
    if len(clique) > best_size:
        best_clique = clique
        best_size = len(clique)
        
    # Order by degree ascending
    order_asc = sorted(range(n), key=lambda i: degrees[i])
    clique = greedy_clique(order_asc)
    if len(clique) > best_size:
        best_clique = clique
        best_size = len(clique)
        
    # 2. Local Search with Random Restarts
    # We use a set for O(1) lookups in the current clique
    current_clique = set(best_clique)
    
    # To ensure 255 is in the result, we can force it or check at the end.
    # 255 is divisible by 3, 5, 17. It is compatible with all multiples of 3, 5, 17?
    # gcd(255, x) > 1 iff x is div by 3, 5, or 17.
    # All candidates are div by 3, 5, or 17.
    # So 255 is compatible with ALL candidates.
    # Thus, 255 should always be in the max clique if we start with a clique that can be extended to include it,
    # or we can just add it if it's not there (since it's compatible with everyone in the candidate pool).
    # Wait, if 255 is compatible with everyone in the candidate pool, then any clique in the candidate pool
    # can be extended to include 255.
    # So we can just find the max clique in the candidate pool, and 255 will be included if we ensure it.
    # Actually, if 255 is compatible with all candidates, it is a universal vertex in the induced subgraph.
    # The max clique will include it.
    
    idx_255 = candidates.index(255)
    
    # Local search loop
    random.seed(42)
    no_improvement_count = 0
    max_no_improvement = 500
    
    while time.time() < deadline - 5:
        # Try to add a random vertex
        not_in = [i for i in range(n) if i not in current_clique]
        if not_in:
            v = random.choice(not_in)
            if all(u in adj[v] for u in current_clique):
                current_clique.add(v)
                if len(current_clique) > best_size:
                    best_size = len(current_clique)
                    best_clique = list(current_clique)
                    no_improvement_count = 0
                else:
                    # If we added but didn't improve best, we might want to keep it for further exploration
                    # or revert. Standard local search keeps it.
                    no_improvement_count = 0
                continue
        
        # Try swap: remove one, add another
        if len(current_clique) > 1:
            to_remove = random.choice(list(current_clique))
            candidate_new = current_clique - {to_remove}
            not_in = [i for i in range(n) if i not in candidate_new]
            if not_in:
                v = random.choice(not_in)
                if all(u in adj[v] for u in candidate_new):
                    new_set = candidate_new | {v}
                    if len(new_set) > best_size:
                        best_size = len(new_set)
                        best_clique = list(new_set)
                        current_clique = new_set
                        no_improvement_count = 0
                    else:
                        # Accept with some probability or just update current to explore?
                        # For max clique, we usually only accept if it improves or is equal (to escape local mins).
                        # Here we only accept if it improves best_size to track the global best.
                        # But we update current_clique to explore the neighborhood.
                        current_clique = new_set
                        no_improvement_count += 1
        else:
            no_improvement_count += 1
            
        if no_improvement_count > max_no_improvement:
            # Restart with a different greedy ordering
            random.shuffle(order_desc)
            clique = greedy_clique(order_desc)
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique
            current_clique = set(best_clique)
            no_improvement_count = 0

    result = best_clique
    
    # Ensure 255 is in the result
    if idx_255 not in result:
        # Since 255 is compatible with all candidates, we can just add it if it's not there?
        # Wait, if it's not in the max clique found, it means the max clique found is not actually the max clique
        # because adding 255 would make it larger (since it's compatible with all).
        # So the max clique MUST contain 255.
        # If our search missed it, we can add it.
        # Check if 255 is compatible with all in result.
        if all(idx_255 in adj[v] for v in result):
            result.append(idx_255)
            best_size += 1
        else:
            # This should not happen if 255 is universal in the candidate set.
            # But just in case, replace an incompatible node.
            for i, v in enumerate(result):
                if idx_255 not in adj[v]:
                    result[i] = idx_255
                    break

    # Convert indices back to values
    final_set = [candidates[i] for i in result]
    
    # Final sanity check: is it a valid clique?
    # If not, fallback to multiples of 3
    is_valid = True
    for i in range(len(final_set)):
        for j in range(i+1, len(final_set)):
            if gcd(final_set[i], final_set[j]) == 1:
                is_valid = False
                break
        if not is_valid:
            break
            
    if not is_valid:
        final_set = [x for x in range(1, 256) if x % 3 == 0]
        
    # Ensure 255 is present
    if 255 not in final_set:
        final_set.append(255)

    return final_set
