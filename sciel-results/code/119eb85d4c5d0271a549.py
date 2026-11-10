import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin
    
    N = 180
    full_set = set(range(1, N+1))
    
    # Find all harmonic triples (a,b,c) with a<b<c, all in [1,180], 1/a = 1/b + 1/c
    # a = bc/(b+c), so (b+c) must divide bc
    triples = []
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            # 1/a = 1/b + 1/c => 1/c = 1/a - 1/b = (b-a)/(ab) => c = ab/(b-a)
            if b > a:
                num = a * b
                den = b - a
                if den > 0 and num % den == 0:
                    c = num // den
                    if c > b and c <= N:
                        triples.append((a, b, c))
    
    # Greedy: repeatedly remove element in most triples
    from collections import Counter
    
    current = set(range(1, N+1))
    triples_set = set(triples)
    
    def count_triples_in_set(triples, s):
        count = 0
        for t in triples:
            if all(x in s for x in t):
                count += 1
        return count
    
    # Greedy removal
    while True:
        if time.time() > deadline - 5:
            break
        # Find triples that are still "active" (all three in current set)
        active_triples = [t for t in triples if all(x in current for x in t)]
        if not active_triples:
            break
        
        # Count how many active triples each element is in
        elem_count = Counter()
        for t in active_triples:
            for x in t:
                elem_count[x] += 1
        
        # Remove element with highest count
        if elem_count:
            worst_elem = max(elem_count, key=lambda x: elem_count[x])
            current.remove(worst_elem)
    
    return sorted(current)
