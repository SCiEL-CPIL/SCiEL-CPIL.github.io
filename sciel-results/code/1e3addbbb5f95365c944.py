import random
import time
from itertools import combinations
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    N = 128
    # Precompute LCM for all pairs
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            lcm_table[i][j] = (i * j) // gcd(i, j)
            lcm_table[j][i] = lcm_table[i][j]
            
    # Identify all forbidden triples
    # We store them as sets of vertices involved
    forbidden_triples = []
    # To speed up removal, we can track for each vertex, how many forbidden triples it is in
    # And a set of active triples
    
    # Instead of listing all triples, we can generate them on the fly or precompute adjacency
    # Given N=128, total triples ~ 350k. This is small.
    triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            L_ab = lcm_table[a][b]
            for c in range(b + 1, N + 1):
                if lcm_table[a][c] == L_ab and lcm_table[b][c] == L_ab:
                    triples.append((a, b, c))
                    
    # Greedy removal
    active = set(range(1, N + 1))
    # Map vertex to set of triple indices it belongs to
    # Or simply, in each step, find a triple in active, and remove the vertex with highest degree in active triples.
    
    # To make it efficient, we can use a heuristic:
    # 1. Count degree of each vertex in the hypergraph of forbidden triples.
    # 2. Remove vertex with max degree.
    # 3. Update degrees and active triples.
    
    # Since N is small, we can just re-scan or maintain lists.
    # Let's maintain a list of active triples.
    active_triples = list(triples)
    deg = {i: 0 for i in range(1, N + 1)}
    for t in active_triples:
        for v in t:
            deg[v] += 1
            
    while active_triples and (time.time() - start_time) < deadline:
        # Pick a triple
        # To be more effective, pick a triple involving the vertex with max degree?
        # Or just pick the first one and remove max degree vertex from it.
        # Better: Find global max degree vertex, remove it, and update all triples containing it.
        
        max_deg_vertex = max(active, key=lambda x: deg[x])
        
        # Remove max_deg_vertex from active
        active.remove(max_deg_vertex)
        
        # Update active_triples: remove all triples containing max_deg_vertex
        new_active_triples = []
        for t in active_triples:
            if max_deg_vertex in t:
                # Decrease degree of other vertices in t
                for v in t:
                    if v != max_deg_vertex:
                        deg[v] -= 1
            else:
                new_active_triples.append(t)
        active_triples = new_active_triples
        
    return list(active)

# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    from math import gcd
    
    start_time = time.time()
    deadline = start_time + 50
    
    N = 128
    # Precompute LCM
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            lcm_table[i][j] = (i * j) // gcd(i, j)
            lcm_table[j][i] = lcm_table[i][j]
            
    # Find all forbidden triples
    triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            L_ab = lcm_table[a][b]
            for c in range(b + 1, N + 1):
                if lcm_table[a][c] == L_ab and lcm_table[b][c] == L_ab:
                    triples.append((a, b, c))
                    
    # Greedy removal with randomization to escape local optima
    best_set = None
    best_size = 0
    
    # Try a few random orderings of removal or random restarts
    for _ in range(100): # Limit restarts to ensure time
        if time.time() > deadline:
            break
            
        active = set(range(1, N + 1))
        deg = {i: 0 for i in range(1, N + 1)}
        active_triples = []
        
        # Shuffle triples to introduce randomness in tie-breaking or selection
        shuffled_triples = triples[:]
        random.shuffle(shuffled_triples)
        
        for t in shuffled_triples:
            active_triples.append(t)
            for v in t:
                deg[v] += 1
                
        while active_triples and (time.time() - start_time) < deadline:
            # Pick vertex with max degree
            # If tie, pick randomly among max
            max_deg = max(deg[v] for v in active)
            candidates = [v for v in active if deg[v] == max_deg]
            v_to_remove = random.choice(candidates)
            
            active.remove(v_to_remove)
            
            new_active_triples = []
            for t in active_triples:
                if v_to_remove in t:
                    for v in t:
                        if v != v_to_remove:
                            deg[v] -= 1
                else:
                    new_active_triples.append(t)
            active_triples = new_active_triples
            
        if len(active) > best_size:
            best_size = len(active)
            best_set = list(active)
            
    if best_set is None:
        # Fallback to seed
        selected = []
        for x in range(1, 129):
            if all(not (lcm_table[a][b]==lcm_table[a][x]==lcm_table[b][x]) for a,b in combinations(selected,2)):
                selected.append(x)
        return selected
        
    return best_set
# EVOLVE-BLOCK-END
