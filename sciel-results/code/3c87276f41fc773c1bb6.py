"""Rewritten program: Maximum Independent Set on conflict graph for Property P."""
import random
import sys

N = 256

def _build_conflict_graph():
    """
    Build a static conflict graph where vertices are elements 1..N.
    An edge exists between b and c (b < c) if there exists some a < b
    such that a divides (b+c).
    
    Note: This creates a sufficient condition for validity. If a set S is an
    independent set in this graph, then for any b, c in S, there is no a < b
    (in the entire universe [1, N]) such that a | (b+c). Consequently, there
    is no a in S such that a | (b+c). Thus S is a valid solution to the task.
    """
    # adjacency[i] is a set of neighbors of i
    adjacency = [set() for _ in range(N + 1)]
    
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            s = b + c
            # Check if there exists a < b such that a | s
            # a must be a divisor of s and a < b
            # Divisors of s are at most sqrt(s) to check efficiently, or just iterate a
            # Since N=256, b+c <= 512. Iterating a up to b is fine.
            conflict = False
            for a in range(1, b):
                if s % a == 0:
                    conflict = True
                    break
            if conflict:
                adjacency[b].add(c)
                adjacency[c].add(b)
                
    return adjacency

def _greedy_mis(adjacency, order):
    """
    Greedily construct an independent set from the given vertex order.
    """
    S = []
    S_set = set()
    for v in order:
        # Check if v has any neighbor in S
        is_conflict = False
        for u in adjacency[v]:
            if u in S_set:
                is_conflict = True
                break
        if not is_conflict:
            S.append(v)
            S_set.add(v)
    return S

def construct():
    """
    Find a large valid subset of [1, 256] using greedy MIS on the static conflict graph.
    """
    adjacency = _build_conflict_graph()
    
    # Fallback: The set of numbers > 128 is always valid?
    # If a, b, c > 128, then b+c > 256.
    # a < b. a divides b+c.
    # This is not trivially empty.
    # Let's just use a random greedy result as fallback if it fails (it shouldn't).
    best_set = []
    best_size = 0
    
    random.seed(42)
    
    # We will try multiple random orderings.
    # Also, sorting by degree (ascending) often yields larger independent sets in greedy MIS.
    # Let's try a mix of pure random and degree-sorted.
    
    # Calculate degrees
    degrees = [len(adjacency[i]) for i in range(1, N + 1)]
    
    for restart in range(100):
        if restart < 50:
            # Random order
            order = list(range(1, N + 1))
            random.shuffle(order)
        else:
            # Degree-sorted order with some noise
            # Sort by degree, then randomize ties or add noise
            vertices = list(range(1, N + 1))
            # Add random noise to degrees to break ties and create variations
            noisy_degrees = [(degrees[i] + random.uniform(0, 1), i) for i in vertices]
            noisy_degrees.sort()
            order = [i for _, i in noisy_degrees]
            
        S = _greedy_mis(adjacency, order)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
    # Ensure non-empty
    if not best_set:
        # 1 is never a conflict source for b,c > 1? 
        # If a=1, 1 divides everything. So if 1 is in S, it conflicts with all pairs b,c.
        # So 1 is a "bad" vertex.
        # The set [129, 256] is likely very large and valid.
        # Let's just return a known valid small set if greedy fails completely (unlikely)
        return list(range(129, 257))
        
    return best_set
