import random

def construct():
    N = 180
    
    # 1. Pre-compute forbidden triples
    # 1/a = 1/b + 1/c => a = bc / (b+c)
    # Iterate b, c. b != c.
    # a must be integer in [1, N].
    # Also a, b, c distinct.
    
    # To ensure we catch all, we can iterate a, b and calculate c.
    # 1/a = 1/b + 1/c => 1/c = 1/a - 1/b = (b-a)/(ab) => c = ab / (b-a)
    # This requires b > a.
    # c must be integer.
    # c != a, c != b.
    # c <= N.
    
    triples_by_vertex = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            # c = ab / (b-a)
            num = a * b
            den = b - a
            if num % den == 0:
                c = num // den
                if c <= N and c != a and c != b:
                    # Triple is (a, b, c)
                    # Add to adjacency lists
                    # For a: other are b, c
                    # For b: other are a, c
                    # For c: other are a, b
                    triples_by_vertex[a].append((b, c))
                    triples_by_vertex[b].append((a, c))
                    triples_by_vertex[c].append((a, b))
                    
    # 2. Randomized Greedy Independent Set
    best_set = set()
    best_size = 0
    
    # Number of restarts
    # N=180 is small. Let's do 200 restarts.
    # Each run is O(N * deg) which is very fast.
    
    for _ in range(200):
        vertices = list(range(1, N + 1))
        random.shuffle(vertices)
        
        current_set = set()
        
        for v in vertices:
            # Check if v can be added
            can_add = True
            for u, w in triples_by_vertex[v]:
                if u in current_set and w in current_set:
                    can_add = False
                    break
            
            if can_add:
                current_set.add(v)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    return list(best_set)
