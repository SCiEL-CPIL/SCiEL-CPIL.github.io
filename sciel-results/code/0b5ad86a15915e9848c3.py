import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    N = 512
    # Candidates are 1..511
    # We want to maximize sum 1/(512-a) for pairwise coprime a.
    # This is equivalent to finding a Maximum Weight Independent Set in the conflict graph
    # (edge if gcd > 1) or Maximum Weight Clique in the coprime graph.
    
    # Precompute weights as Fractions for exact comparison to avoid floating point issues
    # and to ensure we don't miss tiny improvements.
    weights = {a: Fraction(1, N - a) for a in range(1, N)}
    
    # Precompute coprime relationships
    # coprime_adj[a] is a set of b such that gcd(a, b) == 1
    coprime_adj = [set() for _ in range(N)]
    for i in range(1, N):
        for j in range(i + 1, N):
            if gcd(i, j) == 1:
                coprime_adj[i].add(j)
                coprime_adj[j].add(i)
                
    # Helper to check if a node is coprime to all nodes in a set
    def is_coprime_to_all(a, S_set):
        # a is coprime to all in S_set if S_set is a subset of coprime_adj[a]
        # Using set issubset is fast in C
        return S_set.issubset(coprime_adj[a])

    def get_score(S):
        return sum(weights[a] for a in S)

    # 1. Greedy Initialization
    # Sort candidates by weight descending (i.e., a descending)
    # High a -> small denominator -> high weight
    candidates = list(range(N - 1, 0, -1))
    
    S = []
    S_set = set()
    for a in candidates:
        if is_coprime_to_all(a, S_set):
            S.append(a)
            S_set.add(a)
            
    current_score = get_score(S)
    best_S = list(S)
    best_score = current_score

    # 2. Local Search with 1-for-1 and 2-for-1 moves
    # 1-for-1: Remove u, add v.
    # 2-for-1: Remove u, v, add w. (Or more generally, remove a set R, add a set A)
    # The key insight from the inspiration is that merging small primes into large composites
    # can be beneficial due to the convexity of 1/(512-a).
    # Specifically, replacing two smaller coprime numbers with one larger number that is coprime
    # to the rest of the set might yield a higher score if the weight of the larger number
    # exceeds the sum of the weights of the two smaller ones.
    
    # To keep it efficient, we will implement:
    # A. 1-for-1 swaps: Remove u, try to add v.
    # B. 2-for-1 swaps: Remove u, v, try to add w.
    
    # We will iterate through the current set and try these moves.
    # To handle the "greedy refill" aspect from the original code, we can also try
    # to fill the set with any other compatible nodes after a removal, but
    # since we start with a maximal set (greedy), removing a node might allow
    # adding multiple new nodes.
    
    # Let's refine the local search to be more robust.
    # Instead of just simple swaps, we can do:
    # For each u in S:
    #   Remove u.
    #   Try to add any v not in S that is coprime to S\{u}.
    #   If we add v, check if the score improved.
    #   Also, try to add pairs (v1, v2) that are coprime to S\{u} and to each other.
    
    # However, a more powerful move is the "2-for-1" identified in the findings:
    # Replace {u, v} with {w}.
    # This is equivalent to removing u and v, and adding w.
    # w must be coprime to S \ {u, v}.
    
    # Let's implement a loop that tries:
    # 1. 1-for-1: Remove u, add v.
    # 2. 2-for-1: Remove u, v, add w.
    
    # To make this efficient, we precompute the list of candidates.
    # We will use a "best improvement" strategy: find the best move in one pass, apply it, repeat.
    
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        best_move = None
        best_gain = Fraction(0)
        best_new_S = None
        best_new_S_set = None
        
        # 1-for-1 moves
        # For each u in S, for each v in candidates \ S:
        #   if is_coprime_to_all(v, S \ {u}):
        #       gain = weight(v) - weight(u)
        #       if gain > best_gain: record move
        
        # 2-for-1 moves
        # For each pair (u, v) in S:
        #   for each w in candidates \ S:
        #       if is_coprime_to_all(w, S \ {u, v}):
        #           gain = weight(w) - weight(u) - weight(v)
        #           if gain > best_gain: record move
        
        # Optimization:
        # Precompute S \ {u} and S \ {u, v} sets?
        # S is small (probably < 50 elements), so creating new sets is fast.
        
        S_list = list(S)
        
        # 1-for-1
        for i, u in enumerate(S_list):
            S_minus_u = S_set - {u}
            # Try to add any v
            for v in candidates:
                if v in S_set:
                    continue
                if is_coprime_to_all(v, S_minus_u):
                    gain = weights[v] - weights[u]
                    if gain > best_gain:
                        best_gain = gain
                        best_move = ('1for1', u, v)
                        # We don't need to construct the new set yet, just record the move
                        # But we need to check if it's valid (which we did)
                        
        # 2-for-1
        # Iterate over pairs in S
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                u = S_list[i]
                v = S_list[j]
                S_minus_uv = S_set - {u, v}
                # Try to add any w
                for w in candidates:
                    if w in S_set:
                        continue
                    if is_coprime_to_all(w, S_minus_uv):
                        gain = weights[w] - weights[u] - weights[v]
                        if gain > best_gain:
                            best_gain = gain
                            best_move = ('2for1', u, v, w)
                            
        if best_gain > 0:
            # Apply the best move
            if best_move[0] == '1for1':
                _, u, v = best_move
                S.remove(u)
                S_set.remove(u)
                S.append(v)
                S_set.add(v)
            elif best_move[0] == '2for1':
                _, u, v, w = best_move
                S.remove(u)
                S.remove(v)
                S_set.remove(u)
                S_set.remove(v)
                S.append(w)
                S_set.add(w)
                
            current_score += best_gain
            if current_score > best_score:
                best_score = current_score
                best_S = list(S)
            improved = True
            
    # Return the best set found
    return best_S
