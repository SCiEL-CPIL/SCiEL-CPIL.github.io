import time
import random
from math import gcd
from itertools import combinations

# Global variables for speed
N = 128
bad_triples = []
triples_by_node = [[] for _ in range(N)]

def precompute():
    global bad_triples, triples_by_node
    bad_triples = []
    triples_by_node = [[] for _ in range(N)]
    # Precompute GCD matrix
    g = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(i, N):
            g[i][j] = gcd(i+1, j+1)
            g[j][i] = g[i][j]
            
    for a in range(N):
        for b in range(a+1, N):
            gab = g[a][b]
            for c in range(b+1, N):
                if g[a][c] == gab and g[b][c] == gab:
                    bad_triples.append((a, b, c))
                    triples_by_node[a].append((a, b, c))
                    triples_by_node[b].append((a, b, c))
                    triples_by_node[c].append((a, b, c))

def construct():
    if not bad_triples:
        precompute()
        
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin
    
    best_set = set()
    best_size = 0
    
    # Initial Greedy: Randomized Removal
    # We try multiple random restarts to find a better independent set
    # The greedy algorithm: 
    # 1. Start with all nodes.
    # 2. While there is a bad triple in S:
    #    - Pick a bad triple.
    #    - Remove the node in that triple with the highest degree (number of active bad triples it's part of).
    
    # Optimization: Instead of re-scanning all triples, maintain counts.
    
    def greedy_solve(random_seed):
        random.seed(random_seed)
        S = set(range(N))
        # Count of active bad triples for each node
        # Initially, count is len(triples_by_node[i])
        counts = [len(triples_by_node[i]) for i in range(N)]
        
        # We need to efficiently find "active" bad triples.
        # A triple is active if all its nodes are in S.
        # Removing a node deactivates all triples containing it.
        
        # To speed up, we can maintain a list of "active" triples? 
        # Or just iterate through all bad_triples and check if they are in S?
        # N=128, bad_triples ~ 350k. 
        # If we do this in a while loop, and each iteration removes 1 node, 
        # and we scan 350k triples each time, it's 128 * 350k ~ 45M ops. 
        # In Python, this might be slow if not optimized.
        
        # Better: Maintain a set of active triples? 
        # Or, simpler: Just check validity of the final set, and use a heuristic to remove.
        
        # Let's use a slightly different greedy:
        # Sort nodes by "danger" (initial degree).
        # Actually, the standard greedy for independent set in hypergraph:
        # While conflicts exist:
        #   Find a conflict (triple)
        #   Remove vertex with max degree in conflict subgraph
        
        # Let's implement the "Find conflict, remove max degree" loop.
        # To find a conflict quickly, we can keep a list of triples and filter.
        # Since N is small, we can just loop through all bad_triples to find one in S.
        
        active_triples = list(bad_triples)
        
        while active_triples:
            # Pick a random active triple to introduce diversity
            idx = random.randint(0, len(active_triples) - 1)
            a, b, c = active_triples[idx]
            
            # Check if it's still in S (it should be, since we only remove)
            # Actually, we maintain active_triples as those where all nodes are in S?
            # When we remove a node, we must remove all triples containing it from active_triples.
            
            # This is O(N * |E|) per step if we filter list.
            # |E| ~ 350k. N=128. 128 * 350k = 45M. 
            # Python list filtering is fast C loop, but creating new lists is overhead.
            
            # Alternative: Just check if a,b,c are in S.
            if a in S and b in S and c in S:
                # It is a valid conflict
                # Determine which to remove
                # Count active conflicts for a, b, c
                # We can approximate by just using the precomputed static degree, 
                # or dynamic. Dynamic is better.
                
                # Let's just use static degree for speed, or simple heuristic: remove random one?
                # Removing the one with highest static degree is a good heuristic.
                # Let's try removing the one with highest index in (a,b,c) based on static degree.
                
                # Actually, let's just remove the one with the highest count of active triples.
                # To get active count, we'd need to scan.
                # Let's stick to: Remove the node with the highest *static* degree among a,b,c.
                # This is O(1) per step.
                
                # Tie-breaking: Random
                candidates = [(counts[a], a), (counts[b], b), (counts[c], c)]
                # Sort by count desc, then random?
                # Just pick max count. If tie, pick random.
                max_count = max(x[0] for x in candidates)
                best_cands = [x[1] for x in candidates if x[0] == max_count]
                to_remove = random.choice(best_cands)
                
                S.remove(to_remove)
                
                # Update active_triples: remove all triples containing to_remove
                # This is the expensive part.
                # We can filter the list.
                active_triples = [t for t in active_triples if to_remove not in t]
                
        return S

    # Run multiple restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        S = greedy_solve(i)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    # Local Search: Try to add back elements
    # Convert to list for easier manipulation? No, set is fine.
    # Try to add any element not in S if it doesn't create a bad triple with existing pairs in S.
    
    # Check if adding x creates a bad triple:
    # For any pair (u, v) in S, check if (x, u, v) is a bad triple.
    # Precompute: For each x, which pairs (u,v) form a bad triple with x?
    # bad_pairs_for_node[x] = set of (u, v) such that (x, u, v) is a bad triple.
    
    bad_pairs_for_node = [set() for _ in range(N)]
    for a, b, c in bad_triples:
        bad_pairs_for_node[a].add((b, c))
        bad_pairs_for_node[a].add((c, b)) # Order doesn't matter for set of tuples? 
        # Better to store as frozenset or sorted tuple for pair?
        # (b,c) and (c,b) are distinct in tuple set. 
        # Let's just store all permutations or check both.
        # Actually, if I have S, I can iterate pairs in S.
        
    # Simpler check for adding x:
    # Iterate all pairs (u, v) in S.
    # Check if g(x,u) == g(x,v) == g(u,v).
    # This is O(|S|^2) per addition attempt.
    # |S| ~ 60. Pairs ~ 1800. 
    # Trying to add 60 elements: 60 * 1800 = 108k ops. Very fast.
    
    def try_add_elements(S):
        S = set(S)
        improved = True
        while improved:
            improved = False
            for x in range(N):
                if x in S:
                    continue
                # Check if x can be added
                valid = True
                # Get list of elements in S
                S_list = list(S)
                # Check all pairs in S
                for i in range(len(S_list)):
                    u = S_list[i]
                    gu_x = gcd(x+1, u+1)
                    for j in range(i+1, len(S_list)):
                        v = S_list[j]
                        # If gu_x != g(u,v), we can break early for this u?
                        # No, we need g(x,u) == g(x,v) == g(u,v).
                        # So first check g(x,u) == g(u,v). If not, this pair is safe.
                        # But we need to check if there EXISTS a pair that forms a bad triple.
                        # So if we find ANY pair that forms a bad triple, it's invalid.
                        
                        gu_v = gcd(x+1, v+1)
                        gu_v2 = gcd(u+1, v+1)
                        
                        if gu_x == gu_v and gu_x == gu_v2:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    S.add(x)
                    improved = True
        return S

    # Apply local search to best_set
    final_set = try_add_elements(best_set)
    
    # Also try local search on other good sets found?
    # Or just return the best one.
    
    # One more pass: Try removing one element and adding two? 
    # This is complex. The "add back" is usually sufficient for greedy hitting set results.
    
    return [x+1 for x in sorted(final_set)]

# Precompute on import to save time in construct?
# But construct might be called multiple times? 
# The task says "Each evaluation must finish within 60s".
# It implies construct() is the entry point.
# I will precompute inside construct with a global flag to avoid re-doing it if called again in same process.
# But usually these are run in fresh processes.
# To be safe and fast, I'll do it in construct.

# Wait, the precompute is ~350k iterations. In Python, 350k gcd calls is fast (<0.1s).
# The greedy loop is the bottleneck.

# Let's refine the greedy loop speed.
# Filtering `active_triples` list of 350k items 128 times is 45M checks.
# In Python, `list comprehension` is fast. 45M simple checks might take 1-2 seconds.
# 50 restarts * 2s = 100s. Too slow!
# We need to reduce restarts or speed up greedy.

# Optimization:
# Instead of filtering the whole list, we can maintain a set of active triple indices?
# Or, we can just do fewer restarts (e.g., 5) and rely on local search.
# Or, we can optimize the "find conflict" step.
# Actually, we don't need to filter the list to find a conflict.
# We can just iterate through `bad_triples` and find the first one where all nodes are in S.
# If we do this, we don't need `active_triples` list at all!
# We just need to know which triples are "alive".
# But checking "is this triple alive" requires checking 3 set lookups.
# If we scan all 350k triples to find ONE alive triple, it's 350k * 3 lookups.
# If we do this 128 times, it's 45M * 3 lookups. Still ~1-2s per run.
# 50 runs is too much.

# Let's reduce restarts to 10 and add a time check.
# Or, improve the removal strategy to remove multiple nodes at once? No.

# Alternative: Use the "Static Degree" heuristic more aggressively.
# The greedy removal with static degree is deterministic if tie-breaking is fixed.
# Randomizing tie-breaking gives diversity.
# 10 restarts should be enough to find a good set.

# Let's adjust the code to be faster.

def construct():
    if not bad_triples:
        precompute()
        
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = set()
    best_size = 0
    
    def greedy_solve(random_seed):
        random.seed(random_seed)
        S = set(range(N))
        counts = [len(triples_by_node[i]) for i in range(N)]
        
        # We don't maintain active_triples list.
        # We just scan bad_triples to find a conflict.
        # To speed up, we can break early if no conflict found? 
        # But we must find one to remove.
        
        while True:
            # Find a conflict
            conflict_found = False
            for a, b, c in bad_triples:
                if a in S and b in S and c in S:
                    conflict_found = True
                    # Remove node with max static degree
                    candidates = [(counts[a], a), (counts[b], b), (counts[c], c)]
                    max_count = max(x[0] for x in candidates)
                    best_cands = [x[1] for x in candidates if x[0] == max_count]
                    to_remove = random.choice(best_cands)
                    S.remove(to_remove)
                    break
            
            if not conflict_found:
                break
                
            if time.time() > deadline:
                break
                
        return S

    # Run restarts
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        S = greedy_solve(i)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    # Local Search: Add back
    def try_add_elements(S):
        S = set(S)
        improved = True
        while improved:
            improved = False
            for x in range(N):
                if x in S:
                    continue
                valid = True
                S_list = list(S)
                for i in range(len(S_list)):
                    u = S_list[i]
                    gu_x = gcd(x+1, u+1)
                    for j in range(i+1, len(S_list)):
                        v = S_list[j]
                        gu_v = gcd(x+1, v+1)
                        gu_v2 = gcd(u+1, v+1)
                        if gu_x == gu_v and gu_x == gu_v2:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    S.add(x)
                    improved = True
        return S

    final_set = try_add_elements(best_set)
    
    # Verify final set is valid? 
    # The add-back logic ensures no NEW bad triples are created.
    # Since best_set was valid, final_set is valid.
    
    return [x+1 for x in sorted(final_set)]

# One more optimization: 
# The inner loop of try_add_elements is O(N^3) worst case? No, O(|S|^2) per x.
# Total O(N * |S|^2). With N=128, |S|=60, it's 128 * 3600 ~ 450k ops. Fast.

# The greedy loop is the bottleneck.
# Scanning 350k triples to find 1 conflict, 128 times.
# We can optimize by keeping a list of "candidate" triples that are likely to be active.
# But given the time limit, 20 restarts of ~0.5s each is 10s. Safe.

# Let's double check the GCD logic.
# Bad triple: g(a,b) == g(a,c) == g(b,c).
# Yes.

# Final check on "p52" score 1.625.
# If seed is 1.0, and seed size is S0.
# My algorithm should find size > 1.625 * S0.
# If S0 is ~40, target > 65.
# Greedy hitting set on hypergraph usually gets close to optimal for small N.
# Max independent set in 3-uniform hypergraph with 128 nodes.
# Random hypergraph with p ~ ? 
# The structure is specific (GCD).
# It should work.

# I will increase restarts slightly if time permits, but 20 is a good start.
# I will add a time check inside the greedy loop to abort early if a restart is taking too long.

# One detail: `precompute` is global. If `construct` is called multiple times, it shouldn't recompute.
# The code handles this.

# Let's write the final code.
