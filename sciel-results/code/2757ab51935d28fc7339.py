import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    N = 256
    # 1. Build the conflict graph
    # adj[i] is a set of j such that (i*j) % (i+j) == 0
    # We use 1-based indexing for the problem, 0-based for lists
    # Let's use a list of sets for adjacency
    adj = [set() for _ in range(N + 1)]
    
    # Precompute conflicts
    # (a*b) % (a+b) == 0  <=>  a*b = k*(a+b) for some integer k
    # This is equivalent to 1/a + 1/b = 1/k
    # Or (a-k)(b-k) = k^2
    # For a,b in [1, 256], we can just check directly.
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                adj[a].add(b)
                adj[b].add(a)
                
    best_size = 0
    best_set = set()
    
    # Heuristic: Greedy Independent Set
    # Strategy: Pick vertex v, add to IS, remove v and N(v) from graph.
    # To get different results, we can vary the selection rule.
    # Rule 1: Pick node with min degree
    # Rule 2: Pick node with max degree
    # Rule 3: Pick random node
    
    def run_greedy(rule):
        current_graph = [set(s) for s in adj] # Copy of adjacency
        # We need to track which nodes are still available
        available = set(range(1, N + 1))
        independent_set = set()
        
        while available:
            if time.time() > deadline:
                return independent_set
            
            # Select node based on rule
            if rule == 'min_degree':
                # Find node in available with min degree in current_graph
                # Note: degree in current_graph is len(current_graph[node] & available)
                # This is O(N^2) per step if done naively. N=256, steps<=256. 256^3 ~ 16M ops. Fast.
                min_deg = float('inf')
                selected = None
                for node in available:
                    deg = len(current_graph[node] & available)
                    if deg < min_deg:
                        min_deg = deg
                        selected = node
            elif rule == 'max_degree':
                max_deg = -1
                selected = None
                for node in available:
                    deg = len(current_graph[node] & available)
                    if deg > max_deg:
                        max_deg = deg
                        selected = node
            else: # random
                selected = random.choice(list(available))
            
            independent_set.add(selected)
            
            # Remove selected and its neighbors from available
            # Neighbors in the *original* graph that are still available
            neighbors_to_remove = current_graph[selected] & available
            available.difference_update(neighbors_to_remove)
            available.discard(selected)
            
            # We don't strictly need to update current_graph if we always check intersection with available
            # But for efficiency, we could. With N=256, intersection is fast.
            
        return independent_set

    # Run multiple instances with different rules and random seeds
    # We can also try a "shake" version: start with a random independent set, then try to improve?
    # Greedy from scratch is usually good enough for MIS on sparse graphs.
    # Conflict graph density: 
    # Number of solutions to 1/a + 1/b = 1/k.
    # For small k, many solutions. For large k, few.
    # The graph is likely sparse.
    
    for i in range(50): # 50 restarts
        if time.time() > deadline:
            break
        
        # Alternate between rules to explore different parts of the solution space
        if i % 3 == 0:
            rule = 'min_degree'
        elif i % 3 == 1:
            rule = 'max_degree'
        else:
            rule = 'random'
            
        # For random rule, we just pick random.
        # We can also add a slight bias to random: pick from bottom half of degrees?
        
        current_set = run_greedy(rule)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
            # Early exit if we hit a very high number?
            # Max possible is 256. 
            # If we get > 150, it's likely near optimal.
            if best_size > 160:
                # Check if we can do better with a quick local search?
                # For now, keep searching until deadline.
                pass

    # Final local search on the best set found
    # Try to add any node that was removed?
    # The greedy algorithm removes nodes. The complement of the IS in the graph is a vertex cover.
    # Any node in the vertex cover cannot be added.
    # So the greedy IS is maximal.
    # To improve, we need to swap.
    
    # Let's do a quick local search on the best set
    S = list(best_set)
    S_set = set(S)
    
    # Try swapping elements to increase size? 
    # Actually, greedy IS is maximal, so you can't just add.
    # You have to remove some to add more.
    # This is hard.
    
    # Alternative: Use the time remaining to run a more sophisticated heuristic like Simulated Annealing?
    # Given the time, let's just return the best greedy result.
    # The current program scored 1.46. 
    # If the seed (all odds) is size 128, and score is 1.46, that means 128/128 = 1.0? 
    # Wait, score = value/seed. Seed program returns all odds (size 128).
    # If current score is 1.46, then value = 1.46 * 128 = 186.88 -> 187?
    # So the current program finds a set of size ~187.
    # My greedy might find something better or worse.
    # Min-degree greedy is a standard heuristic for MIS.
    
    return list(best_set)

# Note: The seed program returns all odds.
# (a*b) % (a+b) for two odds a,b.
# a=3, b=5 -> 15 % 8 = 7 != 0.
# a=3, b=9 -> 27 % 12 = 3 != 0.
# a=5, b=15 -> 75 % 20 = 15 != 0.
# a=1, b=3 -> 3 % 4 = 3 != 0.
# It seems all odds is a valid set. Size 128.
# The current program adds evens.
# My greedy MIS approach should find a large independent set.

# Let's double check the "value" metric.
# "Maximize cardinality".
# "The evaluator measures the raw value".
# "search score is value/seed".
# Seed program: `return list(range(1, 257, 2))` -> Size 128.
# Current program score 1.4609 -> Value = 1.4609 * 128 = 186.9952.
# So the current program finds a set of size 187.
# Can we find larger?
# Max independent set in this graph.
# Let's estimate the size.
# If we pick all numbers not divisible by small primes?
# The constraint is 1/a + 1/b = 1/k.
# This is related to Egyptian fractions.
# If we avoid numbers that have many divisors?
# Min-degree greedy tends to pick numbers with few conflicts.
# Numbers with few conflicts are likely primes or numbers with large prime factors.
# Primes p: (p*q) % (p+q) = 0?
# p*q = k(p+q) => p(q-k) = kq => p = kq / (q-k).
# If p is prime, then p divides kq. Since p is prime, p|k or p|q.
# If p|q, q=mp. p(mp) = k(p+mp) => mp^2 = kp(1+m) => mp = k(1+m).
# k = mp/(1+m). For k to be integer, 1+m must divide mp.
# gcd(1+m, m) = 1, so 1+m must divide p.
# So p is a multiple of 1+m.
# Example: m=1, q=2p. k = p/2. If p is even, k integer. But p is odd prime.
# So if p is odd, m=1 doesn't work for integer k unless p is even?
# Wait, k = mp/(1+m).
# If p=3, m=1, q=6. k=3/2=1.5. No.
# If p=3, m=2, q=9. k=6/3=2. Yes. 3*9=27, 3+9=12. 27%12=3. No.
# Wait, 1/3 + 1/9 = 4/9. 1/k = 4/9 => k=9/4. Not integer.
# My derivation: p = kq/(q-k).
# Let's re-evaluate.
# (ab) % (a+b) == 0.
# If a=3, b=6. 18 % 9 = 0. Conflict.
# 3 is prime. 6 is not.
# So primes conflict with their multiples?
# If a=p (prime), b=kp.
# p(kp) % (p+kp) = k p^2 % (p(1+k)) = k p % (1+k).
# So we need (1+k) | kp.
# Since gcd(1+k, k)=1, we need (1+k) | p.
# So 1+k is a divisor of p.
# Divisors of prime p are 1 and p.
# 1+k=1 => k=0 (not allowed, b>=1).
# 1+k=p => k=p-1.
# So b = p(p-1).
# So prime p conflicts with b = p(p-1).
# Also, if b is a divisor?
# If a=3, b=3? Distinct pairs only.
# So each prime p conflicts with p(p-1) and potentially other numbers.
# If we pick all primes, do they conflict with each other?
# p, q distinct primes.
# pq % (p+q) = 0?
# pq = k(p+q) => p(q-k) = kq.
# p | kq. p is prime. p|k or p|q.
# p|q impossible (distinct primes).
# So p|k. Let k=mp.
# p q = mp(p+q) => q = m(p+q) => q(1-m) = mp.
# If m=1, q(0)=p => 0=p impossible.
# If m != 1, q = mp / (1-m).
# If m > 1, denom negative.
# If m < 1, m=0? k=0. No.
# So distinct primes do NOT conflict.
# Thus, the set of all primes in [1, 256] is an independent set.
# Number of primes <= 256 is 54.
# This is much smaller than 187.
# So we can add composite numbers.
# The current program finds 187.
# My greedy min-degree might find a larger set.
