"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    N = 128
    # 1. Generate all "bad" triples (a, b, c) such that lcm(a,b) == lcm(a,c) == lcm(b,c)
    bad_triples = []
    # Optimization: Precompute LCMs for all pairs? 128*128 = 16384 pairs. 
    # Checking triples: 128^3 / 6 ~ 350k checks.
    # Let's just iterate and check.
    
    # To speed up, we can group by LCM? 
    # If lcm(a,b) = L, then a|L and b|L.
    # For a triple (a,b,c) to be bad, lcm(a,b)=lcm(a,c)=lcm(b,c)=L.
    # This implies a,b,c are divisors of L.
    # And for each pair, their LCM is L.
    
    # Let's identify all sets of divisors of L that form a "clique" in the graph 
    # where edges exist if LCM is L.
    # Actually, simpler: just find the triples first.
    
    # We will map each element to a bitmask of other elements it conflicts with?
    # No, the conflict is a HYPEREDGE of size 3.
    # A set S is valid if it does NOT contain any of the bad_triples.
    # This is the "3-Hitting Set" complement or "Independent Set in 3-Uniform Hypergraph".
    
    # Algorithm: Branch and Bound on the Hypergraph.
    # State: (available_elements, current_set)
    # We want to maximize len(current_set).
    
    # Preprocessing:
    # 1. Find all bad triples.
    # 2. For each element i, find which triples it belongs to.
    
    # Let's represent the problem as:
    # We have a set of vertices V = {1..128}.
    # We have a set of hyperedges H.
    # We want a subset S of V such that S does not contain any h in H.
    
    # Branching:
    # Pick an element v.
    # Branch 1: Include v. Then we must remove from 'available' any elements that would complete a bad triple with v and two other available elements?
    # No, if we include v, we must ensure that for ANY pair (a,b) in the remaining set, (v,a,b) is not a bad triple.
    # This means if (v,a,b) is a bad triple, we cannot include both a and b.
    # This reduces the problem to a standard graph problem for the next step?
    # If we fix v, the remaining elements cannot contain any pair (a,b) such that {v,a,b} is bad.
    # Let G_v be the graph on V \ {v} where (a,b) is an edge if {v,a,b} is a bad triple.
    # Then the remaining set must be an Independent Set in G_v.
    # So: Max Independent Set in Hypergraph H.
    # Branch on v:
    # 1. Include v: The rest of the set must be an Independent Set in the graph G_v (defined by triples containing v).
    # 2. Exclude v: The rest of the set is a valid set in H restricted to V \ {v}.
    
    # This seems complex to implement efficiently with standard MIS algorithms.
    
    # Alternative Strategy: 
    # Since N=128 is small, but 2^128 is huge.
    # However, the density of bad triples might be low or structured.
    
    # Let's try a simpler heuristic first: Randomized Local Search (Simulated Annealing or Hill Climbing).
    # Start with a random valid set or empty set, add elements, fix conflicts by removing one of the triple.
    # Or start with all elements, remove elements to break triples.
    
    # "Remove to break" strategy:
    # 1. Start with S = {1..128}.
    # 2. While there is a bad triple in S:
    #    - Pick a bad triple (a,b,c).
    #    - Remove the element among a,b,c that appears in the most other bad triples (greedy removal).
    # 3. Try to add back removed elements if possible.
    
    # Let's implement this greedy removal + local search.
    
    # Step 1: Find all bad triples.
    # Optimization: 
    # For each L in 1..128? No, LCM can be up to 128*127? No, elements are in [1,128].
    # Max LCM of two numbers in [1,128] is 128*127 = 16256.
    # But we only care about triples in [1,128].
    
    # Let's just iterate a < b < c.
    # lcm(a,b) == lcm(a,c) == lcm(b,c)
    
    bad_triples = []
    # To speed up, we can precompute lcms.
    lcms = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            lcms[i][j] = (i * j) // gcd(i, j)
            
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            lab = lcms[a][b]
            for c in range(b+1, N+1):
                if lcms[a][c] == lab and lcms[b][c] == lab:
                    bad_triples.append((a-1, b-1, c-1)) # 0-indexed for bitmasks
    
    # If no bad triples, return all.
    if not bad_triples:
        return list(range(1, N+1))
    
    # Map each vertex to the list of triples it is in.
    # And map each pair to the triples they form?
    # For local search, we need to check if adding x to S creates a bad triple.
    # A bad triple is created if there exists (y, z) in S such that {x,y,z} is a bad triple.
    
    # Let's build a structure: 
    # For each x, and each y, what is the set of z such that {x,y,z} is bad?
    # This is expensive.
    
    # Better: For each x, build a graph G_x where edges (y,z) exist if {x,y,z} is bad.
    # Then x can be added to S only if S \ {x} contains no edge from G_x.
    # i.e., S \ {x} is an independent set in G_x.
    
    # This is still heavy for dynamic updates.
    
    # Let's stick to the "Greedy Removal" which is O(1) to check validity of removal (just remove).
    # And "Greedy Addition" requires checking if adding x creates ANY bad triple.
    # To check if adding x creates a bad triple:
    # We need to know if there exists y, z in S such that {x,y,z} is bad.
    # We can precompute for each x, a list of pairs (y,z) that form a bad triple with x.
    # Then checking is: for each (y,z) in pairs[x], if y in S and z in S, then conflict.
    
    # Let's build pairs[x] = set of (y,z) tuples.
    # And also a reverse index: for a pair (y,z), which x's make it bad?
    # Actually, for addition, we just need to check if any (y,z) in pairs[x] is a subset of S.
    
    # Data Structure:
    # S is a set of ints.
    # pairs_by_x[x] = list of (y, z)
    
    pairs_by_x = [[] for _ in range(N)]
    for t in bad_triples:
        a, b, c = t
        pairs_by_x[a].append((b, c))
        pairs_by_x[b].append((a, c))
        pairs_by_x[c].append((a, b))
        
    # Greedy Removal Phase
    S = set(range(N))
    removed = []
    
    # To speed up finding a bad triple, we can maintain a list of active bad triples?
    # Or just scan? 
    # Scanning all bad_triples (size K) for each removal is O(K).
    # If we remove M elements, total O(M*K).
    # K can be large. Let's estimate K.
    # How many equal-lcm triples?
    # For L=120, divisors are 1,2,3,4,5,6,8,10,12,15,20,24,30,40,60,120.
    # Pairs with LCM 120...
    # It's likely K is in the thousands or tens of thousands.
    
    # Let's optimize the "find a bad triple" step.
    # We can iterate through bad_triples and check if all 3 are in S.
    # If yes, remove one.
    
    # Which one to remove?
    # Heuristic: Remove the one with the highest "degree" in the current conflict graph?
    # Or simply the one that is in the most bad triples (static degree).
    
    # Precompute static degree of each vertex in the hypergraph.
    degree = [0]*N
    for t in bad_triples:
        for v in t:
            degree[v] += 1
            
    # Sort vertices by degree descending? No, we remove dynamically.
    
    # Let's run the removal loop.
    # To avoid O(K) scan every time, we can just scan. If K is ~10k and removals ~50, 500k ops. Fast.
    
    current_S = set(range(N))
    
    while True:
        found = False
        for a, b, c in bad_triples:
            if a in current_S and b in current_S and c in current_S:
                # Conflict found
                # Pick the one to remove
                # Heuristic: Remove the one with highest degree among the three
                # Or random? Let's try highest degree first.
                va, vb, vc = a, b, c
                # Tie-breaker: smallest value?
                candidates = [(degree[va], va), (degree[vb], vb), (degree[vc], vc)]
                candidates.sort(reverse=True)
                to_remove = candidates[0][1]
                
                current_S.remove(to_remove)
                found = True
                break # Restart scan to find next conflict
        if not found:
            break
        if time.time() > deadline - 10: # 10s left for search
            break

    # Local Search Phase: Try to add back removed elements
    # We need to track which elements were removed.
    # Actually, we lost that info. Let's restart or track it.
    # Let's track removed elements.
    
    # Let's re-implement with tracking.
    current_S = set(range(N))
    removed_set = set()
    
    # We'll use a list of removed elements to try to add back.
    # To make it efficient, we'll just iterate through all 1..128 not in current_S.
    
    # To check if x can be added:
    # Check if any (y,z) in pairs_by_x[x] are both in current_S.
    
    # Optimization: 
    # If we add x, we might create conflicts. 
    # We can just try to add. If conflict, skip.
    
    # To improve, we can do multiple passes.
    
    # Let's do a simple hill climbing:
    # 1. Greedy removal as above.
    # 2. Try to add every non-selected element.
    # 3. Repeat 1-2 if improvement? No, removal breaks validity, addition restores.
    #    Actually, removal ensures validity. Addition must be careful.
    #    If we add x and it creates a conflict, we don't add it.
    #    So the set remains valid.
    
    # Let's refine the removal to be smarter.
    # Instead of removing one at a time, can we remove a minimal set to break all triples?
    # That's Hitting Set, NP-hard.
    
    # Let's stick to:
    # 1. Greedy Removal.
    # 2. Greedy Addition (try to add all non-selected).
    
    # Re-run removal with tracking
    current_S = set(range(N))
    
    # We need to be careful with time.
    # Let's limit the number of removals? No, must be valid.
    
    # Let's implement the loop with time check.
    
    # To speed up the "find conflict" scan, we can pre-filter?
    # No, just scan.
    
    # One optimization: If we remove a vertex, many triples become invalid.
    # We could maintain a count of "active" triples?
    # Too complex.
    
    # Let's just run it.
    
    # Improvement: Randomized restarts?
    # If we have time, we can try different removal orders.
    
    # Strategy:
    # 1. Run Greedy Removal + Addition.
    # 2. If time permits, run Randomized Greedy Removal (pick random vertex from conflict triple) + Addition.
    # 3. Keep the best result.
    
    best_set = set()
    best_size = 0
    
    def run_greedy(randomize=False):
        S = set(range(N))
        # Shuffle bad_triples? No, the order of finding conflicts matters.
        # If randomize, we can pick a random conflict triple from the available ones?
        # Or just shuffle the bad_triples list before scanning?
        # Shuffling K items is O(K).
        
        triples_to_check = bad_triples[:]
        if randomize:
            import random
            random.shuffle(triples_to_check)
            
        while True:
            if time.time() > deadline - 5:
                return S
            
            found = False
            for a, b, c in triples_to_check:
                if a in S and b in S and c in S:
                    # Conflict
                    if randomize:
                        import random
                        to_remove = random.choice([a, b, c])
                    else:
                        # Highest degree
                        candidates = [(degree[a], a), (degree[b], b), (degree[c], c)]
                        candidates.sort(reverse=True)
                        to_remove = candidates[0][1]
                    
                    S.remove(to_remove)
                    found = True
                    break
            if not found:
                break
        
        # Greedy Addition
        # Try to add elements not in S
        # Order: by degree ascending? (less likely to cause conflicts)
        # Or just 1..N?
        # Let's try adding in order of ascending degree.
        
        candidates_to_add = [x for x in range(N) if x not in S]
        candidates_to_add.sort(key=lambda x: degree[x])
        
        for x in candidates_to_add:
            if time.time() > deadline - 2:
                return S
            
            # Check if x can be added
            # Check pairs_by_x[x]
            can_add = True
            for y, z in pairs_by_x[x]:
                if y in S and z in S:
                    can_add = False
                    break
            
            if can_add:
                S.add(x)
                
        return S

    # Run deterministic first
    s1 = run_greedy(randomize=False)
    if len(s1) > best_size:
        best_size = len(s1)
        best_set = s1
        
    # Run randomized versions if time permits
    import random
    random.seed(42)
    
    num_restarts = 5
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        s = run_greedy(randomize=True)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
    # Convert to list and sort
    return sorted(list(best_set))

# EVOLVE-BLOCK-END
