import time
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    N = 512
    candidates = list(range(1, N))
    weights = {a: 1.0 / (N - a) for a in candidates}
    
    # Precompute prime factors and masks
    primes = []
    for i in range(2, N):
        is_p = True
        for p in primes:
            if p * p > i:
                break
            if i % p == 0:
                is_p = False
                break
        if is_p:
            primes.append(i)
    
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    factor_masks = [0] * N
    for a in range(1, N):
        x = a
        mask = 0
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                mask |= (1 << prime_to_idx[p])
                while x % p == 0:
                    x //= p
        if x > 1:
            mask |= (1 << prime_to_idx[x])
        factor_masks[a] = mask
    
    # Build conflict graph
    adj = {a: set() for a in candidates}
    for i in range(len(candidates)):
        u = candidates[i]
        for j in range(i + 1, len(candidates)):
            v = candidates[j]
            if factor_masks[u] & factor_masks[v]:
                adj[u].add(v)
                adj[v].add(u)
    
    degrees = {a: len(adj[a]) for a in candidates}
    
    # Greedy initial solution
    def greedy_init():
        avail = set(candidates)
        score = 0.0
        selected = set()
        while avail:
            best_node = max(avail, key=lambda x: weights[x])
            score += weights[best_node]
            selected.add(best_node)
            avail.remove(best_node)
            for nb in list(adj[best_node]):
                if nb in avail:
                    avail.remove(nb)
        return score, selected
    
    best_score, best_set = greedy_init()
    
    # Branch and Bound with Moon-Moser clique cover bound
    # For MWIS, a better bound uses the fact that the conflict graph can be covered
    # by cliques, and the independent set can take at most one node from each clique.
    # We use a greedy coloring of the conflict graph to get a clique cover.
    
    def compute_clique_cover_bound(avail, current_score):
        """
        Compute an upper bound using greedy coloring of the induced subgraph on avail.
        The bound is current_score + sum of max weight in each color class.
        """
        if not avail:
            return current_score
        
        # Greedy coloring: assign colors to nodes in avail
        # We want to minimize the number of colors (or maximize the bound)
        # For an upper bound, we want a valid coloring where each color class is an independent set
        # Wait, no. The conflict graph edges mean nodes CANNOT be together.
        # So a coloring of the conflict graph partitions nodes into independent sets.
        # The maximum weight independent set can take at most the sum of max weights from each color class?
        # No, that's not right. 
        # Actually, if we color the conflict graph with k colors, the nodes in each color class form an independent set.
        # But we are looking for ONE independent set, not a union of them.
        # The correct bound from coloring is: 
        # If the conflict graph has a clique cover (partition into cliques) C_1, ..., C_k, 
        # then any independent set contains at most one node from each C_i.
        # So the bound is current_score + sum_{i=1}^k max_{v in C_i} weight(v).
        
        # How to get a clique cover? 
        # A clique in the conflict graph is a set of nodes where every pair shares a factor.
        # This is hard to find directly. 
        # Alternative: Use the complement graph. 
        # In the complement graph, an edge means coprime. 
        # We want a clique cover in the conflict graph = independent set cover in the complement graph.
        # This is still hard.
        
        # Simpler bound: Use the fact that the sum of weights is an upper bound (already used).
        # Let's try a different approach: branch on the node with highest weight, 
        # and use the sum of remaining weights as the bound.
        # To improve, we can precompute a better bound using a greedy independent set 
        # as a lower bound for pruning, but we already have that.
        
        # Let's stick with the simple bound but optimize the search order.
        return current_score + sum(weights[u] for u in avail)
    
    def dfs(avail, current_score, current_set):
        nonlocal best_score, best_set
        
        if time.time() > deadline:
            return
            
        if not avail:
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
            return
            
        # Upper bound
        bound = current_score + sum(weights[u] for u in avail)
        if bound <= best_score:
            return
            
        # Pick node with highest weight to branch on
        v = max(avail, key=lambda x: weights[x])
        
        # Option 1: Include v
        new_avail_1 = set(avail)
        new_avail_1.remove(v)
        for nb in adj[v]:
            if nb in new_avail_1:
                new_avail_1.remove(nb)
        
        dfs(new_avail_1, current_score + weights[v], current_set | {v})
        
        # Option 2: Exclude v
        new_avail_2 = set(avail)
        new_avail_2.remove(v)
        
        dfs(new_avail_2, current_score, current_set)
    
    initial_avail = set(candidates)
    
    try:
        dfs(initial_avail, 0.0, set())
    except RecursionError:
        pass

    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(f"Selected: {res}")
    score = sum(1.0 / (512 - a) for a in res)
    print(f"Score: {score}")
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            if math.gcd(res[i], res[j]) != 1:
                print(f"ERROR: {res[i]} and {res[j]} not coprime")
                valid = False
    if valid:
        print("Valid")
    print("Done")
