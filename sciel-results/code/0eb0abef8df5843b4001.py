"""
Strategy: Model B2[2] set construction as a Maximum Independent Set (MIS) problem.
Vertices: 1..1000.
Edges: (u, v) exist if adding u and v to a set causes some sum to have > 2 representations.
       Note: The condition is "every sum has at most 2 representations".
       Representations are pairs (a,b) with a<=b from the set.
       If we have a set S, and we consider adding x, we must ensure that for all y in S,
       the sum x+y does not already have 2 representations, and 2x does not already have 2.
       
       However, the conflict between two numbers u and v is not static in the simplest sense 
       because it depends on the *other* numbers in the set. 
       Wait, the condition is global. 
       Let R(s) be the number of pairs (a,b) in S such that a+b=s and a<=b.
       We require R(s) <= 2 for all s.
       
       If we build S incrementally, adding x to S is valid if for all y in S, 
       R_{S}(x+y) < 2 and R_S(2x) < 2.
       
       This is not a simple graph independent set where edges are static. 
       The "conflict" between u and v depends on the rest of the set.
       Specifically, u and v "conflict" if there exists a third element w (or pair) 
       such that u+v is already represented twice by other elements.
       
       Actually, a simpler structural change is to use a different heuristic search:
       Simulated Annealing or Local Search (Greedy + Hill Climbing).
       
       Let's try a different angle: 
       The current program uses Greedy + Random Restarts + Local Fill.
       
       A "MATERIAL DIFFERENCE" could be:
       1. Use a specific constructive heuristic based on number theory (e.g., Sidon-like sequences but relaxed for B2[2]).
       2. Use a more sophisticated local search: Start with a random valid set, then perform "swap" moves (remove one, add another) to escape local optima, or "kick" moves (remove k, add k+1).
       
       Let's implement a robust Local Search with "Swap" and "Add" moves.
       1. Initialize with a greedy set.
       2. Loop:
          a. Try to add any missing number (Greedy Fill).
          b. If no add possible, try a "Swap": Remove a random element, try to add a different one.
          c. If no swap improves size, try a "Kicker": Remove 2 elements, add 3? (Too complex).
          
       Actually, the "conflict graph" idea in the prompt suggests a global view.
       Let's try to implement a Greedy Set Cover / Packing approach using a priority queue of "benefit".
       
       Alternative Structural Change: 
       Instead of iterating 1..1000, iterate based on "density" of sums.
       Or, use a different initialization: Start with a known good B2[2] structure (like a scaled Sidon set) and expand.
       
       Let's stick to the prompt's suggestion: "Model as MIS... greedy coloring heuristic with backtracking".
       Since true MIS is NP-hard and the graph is dynamic, we can approximate it by:
       1. Build a static conflict graph assuming the set is "dense" or just use the incremental check but change the search strategy to **Backtracking with Heuristics**.
       
       However, backtracking on 1000 nodes is too slow.
       
       Let's try **Simulated Annealing** on the set of selected numbers.
       State: Bitmask of 1000 bits (or list).
       Move: Flip a bit (add/remove a number).
       Energy: Penalty for violating B2[2] (number of sums with >2 reps) - Lambda * Size.
       We want to minimize Energy.
       
       This is a fundamentally different optimization landscape traversal compared to greedy construction.
       
       Implementation Plan:
       1. Represent set as a boolean array `in_set` of size 1001.
       2. Maintain `sum_counts` array of size 2001.
       3. Initial state: Greedy fill (fast, valid).
       4. Simulated Annealing:
          - T starts high, decreases.
          - At each step, pick a random index `i`.
          - If `in_set[i]` is True, try to remove it.
            - Calculate delta in violation count and size.
            - If it reduces violations or keeps them same and increases size (impossible, size decreases), accept.
            - Actually, we want to maximize size subject to 0 violations.
            - Let's define Score = Size - M * Violations, where M is large.
            - We want to maximize Score.
          - If `in_set[i]` is False, try to add it.
            - Calculate change in violations.
            - If violations become 0 (or stay 0), and size increases, accept.
            - If violations increase, accept with probability exp((new_score - old_score) / T).
            
       This allows escaping local optima where greedy gets stuck.
       
       Key Optimization:
       Fast update of sum_counts.
       Adding x:
         For all y in set, sum_counts[x+y] += 1.
         sum_counts[2x] += 1.
       Removing x:
         For all y in set, sum_counts[x+y] -= 1.
         sum_counts[2x] -= 1.
       
       To check violations efficiently:
       Maintain a count of how many sums have count > 2.
       Let `bad_sums` be the number of s such that sum_counts[s] > 2.
       
       When adding x:
       For each y in set:
         old_c = sum_counts[x+y]
         new_c = old_c + 1
         if old_c == 2: bad_sums += 1
         if old_c == 1: pass # 1->2 is ok
         if old_c == 0: pass # 0->1 is ok
         # if old_c >= 3, bad_sums stays same (already bad)
         sum_counts[x+y] = new_c
         
       Same for 2x.
       
       This makes the move evaluation O(|S|).
       With |S| ~ 300-400, and 1000 steps, this is fast enough for 120s.
       
       We will run multiple SA restarts.
"""

import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    N = 1000
    
    # Helper to calculate current state metrics
    def get_metrics(in_set, sum_counts):
        size = sum(in_set)
        bad_sums = 0
        for s in range(1, 2*N + 1):
            if sum_counts[s] > 2:
                bad_sums += 1
        return size, bad_sums

    def run_sa(initial_set, time_limit):
        """
        Run Simulated Annealing starting from initial_set.
        Returns the best valid set found (0 bad sums) with max size.
        """
        in_set = [False] * (N + 1)
        sum_counts = [0] * (2 * N + 1)
        
        # Initialize from initial_set
        for x in initial_set:
            in_set[x] = True
        
        # Build sum_counts
        # Optimization: Use list of active elements for iteration
        active_elements = list(initial_set)
        
        for i in range(len(active_elements)):
            for j in range(i, len(active_elements)):
                s = active_elements[i] + active_elements[j]
                sum_counts[s] += 1
                
        cur_size, cur_bad = get_metrics(in_set, sum_counts)
        # If initial is not valid, we need to fix it or start fresh.
        # Greedy init should be valid.
        
        best_size = cur_size
        best_set = list(initial_set)
        if cur_bad == 0:
            best_set = list(initial_set)
        
        T = 10.0
        T_min = 0.01
        cooling = 0.999
        steps = 0
        max_steps_per_sec = 5000 # Estimate
        
        while time.time() < time_limit:
            steps += 1
            if steps % 100 == 0 and T > T_min:
                T *= cooling
            
            # Pick random move
            # Biased towards adding? Or uniform?
            # Uniform over 1..N
            x = random.randint(1, N)
            
            if in_set[x]:
                # Try Remove
                # Calculate delta
                delta_bad = 0
                # 2x
                s_2x = 2 * x
                old_c = sum_counts[s_2x]
                new_c = old_c - 1
                if old_c == 3: delta_bad -= 1 # 3->2, good
                if old_c == 2: pass # 2->1, neutral
                if old_c <= 1: pass
                # Note: if old_c > 3, bad count doesn't change (still >2)
                # if old_c == 3, it becomes 2, so bad decreases.
                # if old_c == 2, it becomes 1, bad unchanged.
                # if old_c == 1, it becomes 0, bad unchanged.
                
                # x + y
                for y in active_elements:
                    if y == x: continue
                    s_xy = x + y
                    old_c = sum_counts[s_xy]
                    new_c = old_c - 1
                    if old_c == 3: delta_bad -= 1
                    # else no change to bad count
                    
                delta_size = -1
                new_bad = cur_bad + delta_bad
                new_size = cur_size + delta_size
                
                # Acceptance Criteria
                # We prefer valid sets (bad=0) with high size.
                # Score = size - 1000 * bad
                cur_score = cur_size - 1000 * cur_bad
                new_score = new_size - 1000 * new_bad
                
                accept = False
                if new_score > cur_score:
                    accept = True
                else:
                    delta = new_score - cur_score
                    if T > 0:
                        prob = math.exp(delta / T)
                        if random.random() < prob:
                            accept = True
                
                if accept:
                    # Perform Remove
                    in_set[x] = False
                    sum_counts[s_2x] -= 1
                    for y in active_elements:
                        if y == x: continue
                        s_xy = x + y
                        sum_counts[s_xy] -= 1
                    active_elements.remove(x)
                    cur_size = new_size
                    cur_bad = new_bad
                    
                    if cur_bad == 0 and cur_size > best_size:
                        best_size = cur_size
                        best_set = active_elements[:]
                        
            else:
                # Try Add
                delta_bad = 0
                # 2x
                s_2x = 2 * x
                old_c = sum_counts[s_2x]
                new_c = old_c + 1
                if old_c == 2: delta_bad += 1 # 2->3, bad
                # if old_c < 2, no new bad
                
                # x + y
                for y in active_elements:
                    s_xy = x + y
                    old_c = sum_counts[s_xy]
                    new_c = old_c + 1
                    if old_c == 2: delta_bad += 1
                    
                delta_size = 1
                new_bad = cur_bad + delta_bad
                new_size = cur_size + delta_size
                
                cur_score = cur_size - 1000 * cur_bad
                new_score = new_size - 1000 * new_bad
                
                accept = False
                if new_score > cur_score:
                    accept = True
                else:
                    delta = new_score - cur_score
                    if T > 0:
                        prob = math.exp(delta / T)
                        if random.random() < prob:
                            accept = True
                
                if accept:
                    # Perform Add
                    in_set[x] = True
                    sum_counts[s_2x] += 1
                    for y in active_elements:
                        s_xy = x + y
                        sum_counts[s_xy] += 1
                    active_elements.append(x)
                    cur_size = new_size
                    cur_bad = new_bad
                    
                    if cur_bad == 0 and cur_size > best_size:
                        best_size = cur_size
                        best_set = active_elements[:]
                        
        return best_set

    def greedy_init():
        chosen = []
        counts = {}
        for x in range(1, N + 1):
            valid = True
            # Check 2x
            if counts.get(2*x, 0) >= 2:
                valid = False
            else:
                for y in chosen:
                    if counts.get(x+y, 0) >= 2:
                        valid = False
                        break
            if valid:
                chosen.append(x)
                counts[2*x] = counts.get(2*x, 0) + 1
                for y in chosen:
                    if y == x: continue
                    counts[x+y] = counts.get(x+y, 0) + 1
        return chosen

    best_result = []
    
    # Initial greedy
    g1 = greedy_init()
    best_result = g1
    current_best_size = len(g1)
    
    # Run SA for a few iterations
    # Time budget for SA
    sa_deadline = time.time() + 100
    
    restarts = 0
    while time.time() < sa_deadline:
        restarts += 1
        # Initialize with a perturbed greedy or random valid set?
        # Let's try starting from the current best and perturbing it.
        # Or start from a fresh greedy.
        if restarts % 5 == 0:
            init_set = greedy_init()
        else:
            # Take best_result and remove some random elements to create space
            init_set = best_result[:]
            for _ in range(min(10, len(init_set)//5)):
                if init_set:
                    idx = random.randint(0, len(init_set)-1)
                    init_set.pop(idx)
                    
        # Run SA for this restart
        # Limit time per restart to avoid getting stuck in one
        single_run_time = 10
        res = run_sa(init_set, time.time() + single_run_time)
        
        if len(res) > current_best_size:
            current_best_size = len(res)
            best_result = res

    # Final local improvement (Greedy Fill) on the best result found
    if time.time() < deadline - 5:
        used = set(best_result)
        counts = {}
        for i in range(len(best_result)):
            for j in range(i, len(best_result)):
                s = best_result[i] + best_result[j]
                counts[s] = counts.get(s, 0) + 1
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for x in range(1, N + 1):
                if x in used:
                    continue
                valid = True
                if counts.get(2*x, 0) >= 2:
                    valid = False
                else:
                    for y in best_result:
                        if counts.get(x+y, 0) >= 2:
                            valid = False
                            break
                if valid:
                    best_result.append(x)
                    used.add(x)
                    counts[2*x] = counts.get(2*x, 0) + 1
                    for y in best_result:
                        if y == x: continue
                        counts[x+y] = counts.get(x+y, 0) + 1
                    improved = True

    return best_result
