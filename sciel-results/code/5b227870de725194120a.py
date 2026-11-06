import time
import random
import numpy as np
from collections import deque

N = 12

def _compute_threshold_fast(trans):
    """
    Compute the exact shortest reset word length using BFS on the subset automaton.
    Returns -1 if not synchronizing.
    """
    trans = np.array(trans, dtype=np.uint8)
    
    # Precompute subset transitions
    # 2^12 = 4096 subsets
    # For each subset (mask) and letter (0,1), compute next mask
    subset_trans = np.zeros((2, 1 << N), dtype=np.int32)
    
    # Optimization: Precompute bit contributions
    # For a mask M, next_mask(L) = OR_{s in M} (1 << trans[L][s])
    
    # We can compute this quickly using bit manipulation tricks or simple loops since 4096*12 is small.
    # But let's vectorize the inner loop slightly or just use a tight loop.
    
    for letter in range(2):
        # trans[letter] is an array of length N
        # We want subset_trans[letter][mask]
        
        # Trick: Iterate masks.
        # Actually, iterating masks and bits is O(2^N * N). 4096 * 12 = ~50k ops. Very fast.
        
        for mask in range(1 << N):
            new_mask = 0
            # Extract set bits
            m = mask
            while m:
                # Get lowest set bit
                lb = m & -m
                s = lb.bit_length() - 1
                new_mask |= (1 << trans[letter][s])
                m ^= lb
            subset_trans[letter][mask] = new_mask

    full_set_mask = (1 << N) - 1
    visited = np.full(1 << N, -1, dtype=np.int32)
    queue = deque()
    
    visited[full_set_mask] = 0
    queue.append(full_set_mask)
    
    while queue:
        current_mask = queue.popleft()
        dist = visited[current_mask]
        
        # Check if singleton
        if current_mask != 0 and (current_mask & (current_mask - 1)) == 0:
            return dist
            
        # Expand
        for letter in range(2):
            next_mask = subset_trans[letter][current_mask]
            if visited[next_mask] == -1:
                visited[next_mask] = dist + 1
                queue.append(next_mask)
                
    return -1

def _construct_from_path(path):
    """
    Given a path of subsets [S_0, S_1, ..., S_k] where S_0 is full set and S_k is singleton,
    attempt to construct a deterministic automaton that realizes this path as a reset sequence.
    
    This is a constraint satisfaction problem.
    For each step i, we need to choose a letter L_i such that delta(S_i, L_i) = S_{i+1}.
    This imposes constraints on the transition function for states in S_i.
    
    Specifically, if S_{i+1} has size 1, say {t}, then all states in S_i must map to t under L_i.
    If S_{i+1} has size > 1, the mapping must collapse S_i to exactly S_{i+1}.
    
    We will use a randomized backtracking or constraint propagation approach to fill the transition table.
    Since we only need *a* valid automaton, we can fix the letters used in the path (e.g., alternating 'a' and 'b' or just 'a') 
    and try to satisfy the constraints.
    
    However, a simpler approach for this specific "path" strategy:
    Let's assume the path uses a specific sequence of letters.
    Let the path be S_0 -> S_1 -> ... -> S_k.
    Let the letters be L_0, L_1, ..., L_{k-1}.
    
    For each state s, its transition under letter L is constrained by the steps where s is in S_i.
    If s is in S_i, then trans[L_i][s] must be in S_{i+1}.
    Moreover, the image of S_i under L_i must be EXACTLY S_{i+1}.
    
    This is complex. A simpler heuristic:
    Construct the automaton backwards or use the path to define "critical" transitions.
    
    Alternative Structural Strategy:
    Instead of arbitrary paths, use the known "Pavlov" or "Bisynchronizing" structures but optimize the "delay" parameters.
    Or, use the path to guide a local search.
    
    Let's try a direct construction based on a specific family:
    The "Longest Reset Word" for N states is bounded by N(N-1)/2.
    Known best for N=12 is likely around 40-60? 
    Actually, for N=12, the maximum known threshold is likely in the 30s-50s range.
    
    Let's use a **Greedy Path Extension** on the Subset Automaton itself to find a long chain of subsets, 
    then solve the CSP to realize it.
    """
    pass # Placeholder, we will implement the actual search in construct

def _find_long_subset_path():
    """
    Find a long path in the subset automaton (graph of subsets) from FullSet to Singleton.
    We treat the subset automaton as a graph where nodes are subsets.
    Edges exist if there is a letter mapping one subset to another.
    But we don't have the automaton yet. 
    We need to find a path S_0, S_1, ... S_k such that it is *realizable*.
    
    A subset S is "smaller" than T if S is a proper subset of T? No, size can vary.
    Usually reset words reduce the number of states.
    So we look for a path where |S_{i+1}| < |S_i| mostly, or at least eventually.
    
    Let's generate a random "target" path of subsets that shrinks slowly.
    S_0 = {0..11}
    S_1 = {0..10} (remove 1)
    S_2 = {0..9} (remove 1)
    ...
    This is just the standard N-1 path.
    
    To get longer, we need "detours".
    S_i -> S_{i+1} where |S_{i+1}| is close to |S_i| but not equal, or even increases?
    No, reset words are shortest paths to a singleton. If the path increases size, it's not "progressing" to sync unless it comes back.
    However, the shortest reset word length is the distance in the subset automaton.
    To maximize this distance, we want the "bottleneck" to be far.
    
    Strategy: 
    1. Generate a random sequence of subsets S_0, ..., S_k where S_0=Full, S_k=Singleton.
    2. Ensure |S_{i+1}| <= |S_i| + 1 (allow slight expansion? No, BFS distance counts steps. 
       If we can go Full -> A -> B -> ... -> Single, and each step is valid, the length is k.
       To make k large, we want many steps.
       Max steps is bounded by number of subsets? No, simple path length is bounded by 2^N.
       But we need it to be realizable by a *deterministic* automaton.
       
    Let's use a **Randomized CSP Solver** for the transition table, guided by a long random chain of subsets.
    """
    pass

def construct():
    """
    Search for a synchronizing automaton with N=12 states that maximizes
    the shortest reset word length.
    """
    start_time = time.time()
    deadline = 85.0  # Safety margin
    
    best_score = -1
    best_trans = None
    
    # Baseline
    baseline_a = [(i + 1) % N for i in range(N)]
    baseline_b = [0] * N
    for i in range(1, N):
        baseline_b[i] = i - 1
    baseline_trans = [baseline_a, baseline_b]
    score = _compute_threshold_fast(baseline_trans)
    if score > best_score:
        best_score = score
        best_trans = baseline_trans
        
    # Strategy: Simulated Annealing / Random Restart with "Long Path" Bias
    # We will generate candidates that are "near-permutations" for one letter and "collapsing" for the other,
    # but we will bias the collapsing letter to create "bottlenecks" that are hard to cross.
    
    rng = random.Random(123)
    
    # We will try to construct automata where one letter (say 'a') is a long cycle,
    # and the other letter ('b') has a specific structure that prevents immediate collapse.
    
    # Known good structure for high thresholds:
    # 'a' is a cycle (0->1->...->N-1->0)
    # 'b' maps i -> i-1 for most i, but has "jumps" or "stalls".
    # Specifically, if b(i) = i-1, it syncs in N-1.
    # If we modify b to map some i to i (self-loop) or i+1, we can increase the diameter.
    
    # Let's use a Random Walk on the space of "near-cycle" b-mappings.
    
    attempts = 0
    
    while time.time() < deadline:
        attempts += 1
        
        # Generate 'a' as a random permutation, preferably a single cycle
        trans_a = list(range(N))
        rng.shuffle(trans_a)
        # Ensure it's a single cycle? 
        # 99% of random permutations are single cycles for small N? No, probability is 1/N.
        # Let's force a single cycle:
        # Start with identity, swap random pairs? 
        # Easier: Generate a random permutation, then check/fix.
        # For now, just shuffle. If it's not a single cycle, it might sync faster or slower.
        # Actually, single cycle 'a' is good for "moving" states.
        
        # Let's enforce single cycle for 'a' to maximize the "spread" potential.
        # Algorithm for random single cycle:
        # 1. Start with 0.
        # 2. Pick random unused node, add to cycle.
        # 3. Continue until all used.
        # 4. Shuffle the order of nodes in the cycle to randomize the permutation structure?
        #    A cycle (0, p1, p2, ..., pN-1) is a specific permutation.
        #    Let's just generate a random permutation and check if it's a single cycle. 
        #    If not, retry. For N=12, prob of single cycle is 1/12 ~ 8%. Too slow.
        #    Let's construct it:
        nodes = list(range(N))
        rng.shuffle(nodes)
        # The cycle is nodes[0] -> nodes[1] -> ... -> nodes[N-1] -> nodes[0]
        # So trans_a[nodes[i]] = nodes[(i+1)%N]
        trans_a = [0] * N
        for i in range(N):
            trans_a[nodes[i]] = nodes[(i + 1) % N]
            
        # Generate 'b'
        # Base case: b(i) = i-1 (mod N) or similar.
        # Let's start with a "shift" and introduce random "defects".
        
        # Type 1: Shift with random stalls
        # b(i) = i - 1 with prob 0.8, else random
        trans_b = [0] * N
        for i in range(N):
            if rng.random() < 0.7:
                trans_b[i] = (i - 1) % N
            else:
                trans_b[i] = rng.randint(0, N - 1)
                
        # Type 2: Hub structure
        if rng.random() < 0.3:
             # Pick 2 hubs
             h1, h2 = rng.sample(range(N), 2)
             for i in range(N):
                 if rng.random() < 0.5:
                     trans_b[i] = h1
                 else:
                     trans_b[i] = h2
                     # Add some noise
                     if rng.random() < 0.2:
                         trans_b[i] = rng.randint(0, N-1)
                         
        candidate = [trans_a, trans_b]
        
        # Evaluate
        score = _compute_threshold_fast(candidate)
        
        if score > best_score:
            best_score = score
            best_trans = candidate
            # Optional: If we find a very good one, we could try local search around it.
            # But simple random restarts might be enough if the generation is smart.
            
            # If score is very high (> 30), we might want to bias towards this structure.
            # For now, just continue.
            
            # Check time
            if time.time() > deadline - 5:
                break

    # Local Search Refinement on the best candidate found
    if best_trans is not None and time.time() < deadline - 2:
        current_best = best_score
        current_trans = best_trans
        t = time.time()
        
        # Simple Hill Climbing: Change one transition, see if score improves.
        # To speed up, we can use the subset trans table from the last eval? 
        # No, we recompute. 
        # But we can just do a few iterations.
        
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            # Try mutating one random transition
            for _ in range(20): # 20 mutations per iter
                if time.time() > deadline - 1:
                    break
                letter = rng.randint(0, 1)
                state = rng.randint(0, N - 1)
                old_val = current_trans[letter][state]
                new_val = rng.randint(0, N - 1)
                
                current_trans[letter][state] = new_val
                score = _compute_threshold_fast(current_trans)
                
                if score > current_best:
                    current_best = score
                    best_score = score
                    best_trans = [list(x) for x in current_trans] # Deep copy
                    improved = True
                    # Restart mutation loop? Or continue?
                    # Continue to see if we can push further.
                else:
                    # Revert if not improved (Strict Hill Climbing)
                    current_trans[letter][state] = old_val
                    
    return best_trans

# EVOLVE-BLOCK-END
