import random
import time
import math

def construct():
    # Precompute perfect squares up to 256+256 = 512
    max_sum = 512
    squares = set()
    k = 1
    while k * k <= max_sum:
        squares.add(k * k)
        k += 1
    
    # Precompute conflict graph
    # adj[i] is a set of j such that i + j is a perfect square
    adj = [set() for _ in range(257)]
    for i in range(1, 257):
        for s in squares:
            j = s - i
            if 1 <= j <= 256:
                adj[i].add(j)
    
    def is_valid_add(v, S):
        """
        Check if v can be added to set S without creating a square sum.
        S is a set of integers.
        """
        # Check self-conflict: 2v is a square
        if v in adj[v]:
            return False
        # Check conflicts with existing elements
        for u in S:
            if u in adj[v]:
                return False
        return True

    def greedy_random():
        """
        Construct a maximal independent set using a random ordering.
        """
        S = set()
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        for v in candidates:
            if is_valid_add(v, S):
                S.add(v)
        return S

    def local_improve(S):
        """
        Try to add more elements to S until no more can be added.
        """
        current_S = set(S)
        improved = True
        while improved:
            improved = False
            # Find candidates not in current_S
            not_in_S = [v for v in range(1, 257) if v not in current_S]
            random.shuffle(not_in_S)
            for v in not_in_S:
                if is_valid_add(v, current_S):
                    current_S.add(v)
                    improved = True
        return current_S

    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s
    
    # Initial solution
    best_S = greedy_random()
    best_S = local_improve(best_S)
    best_size = len(best_S)
    
    # Local search with restarts
    while time.time() - start_time < deadline:
        S = greedy_random()
        S = local_improve(S)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
            
    return sorted(list(best_S))
