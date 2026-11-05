import time
from collections import deque
import random

N = 12

def calculate_threshold(a_trans, b_trans):
    """
    Computes the exact shortest reset word length (threshold) for a binary automaton.
    Uses BFS on the subset automaton with bitmasks for efficiency.
    """
    start = (1 << N) - 1
    
    # If start is a single state, threshold is 0
    if start & (start - 1) == 0:
        return 0
    
    visited = {start: 0}
    q = deque([start])
    
    # Precompute transition masks for each letter and each state
    # a_mask[s] = bitmask of states reachable from state s via 'a'
    # Since each state goes to exactly one state, this is just 1 << a_trans[s]
    a_masks = [1 << a_trans[i] for i in range(N)]
    b_masks = [1 << b_trans[i] for i in range(N)]
    
    while q:
        curr = q.popleft()
        curr_dist = visited[curr]
        
        # Check if we have a single state (reset achieved)
        if curr & (curr - 1) == 0:
            return curr_dist
        
        # Apply letter 'a': for each state in curr, go to a_trans[state]
        next_a = 0
        temp = curr
        while temp:
            lsb = temp & (-temp)
            s = lsb.bit_length() - 1
            next_a |= a_masks[s]
            temp ^= lsb
        
        if next_a not in visited:
            visited[next_a] = curr_dist + 1
            q.append(next_a)
            
        # Apply letter 'b'
        next_b = 0
        temp = curr
        while temp:
            lsb = temp & (-temp)
            s = lsb.bit_length() - 1
            next_b |= b_masks[s]
            temp ^= lsb
            
        if next_b not in visited:
            visited[next_b] = curr_dist + 1
            q.append(next_b)
            
    return -1  # Not synchronizing

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a 12-state binary automaton that maximizes the shortest reset word length.
    Uses a hybrid approach:
    1. Start with the known Černý automaton (threshold 121).
    2. Perform a local search using bitmask-based threshold calculation (very fast).
    3. The local search perturbs individual transitions in 'a' and 'b' to see if a
       higher threshold can be achieved.
    4. If no improvement is found, it falls back to the initial best.
    
    The bitmask BFS is significantly faster than frozenset-based BFS, allowing
    for more iterations of local search within the time budget.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Initial construction: Černý automaton
    # a[i] = (i + 1) % N
    # b[0] = 0, b[i] = i - 1 for i > 0
    a = [(i + 1) % N for i in range(N)]
    b = [0] * N
    for i in range(1, N):
        b[i] = i - 1
        
    best_score = calculate_threshold(a, b)
    best_a = a[:]
    best_b = b[:]
    
    # If the initial score is already the theoretical max, return immediately
    # (Though for N=12, (N-1)^2 = 121 is the conjectured max, but we search anyway 
    #  in case there's a better non-standard construction or if the conjecture is wrong for small N)
    # Actually, Černý's conjecture holds for N up to 6. For N=12, it is unknown if 121 is max.
    # But the known best is 121. We search for > 121.
    
    # Local Search
    # We will try random perturbations of the transition functions.
    # A perturbation consists of changing one state's destination for either 'a' or 'b'.
    
    # To make the search more effective, we can use a "restart" strategy or 
    # simply iterate random single-step mutations.
    
    # Optimization: Pre-compute masks for the current best to speed up evaluation?
    # No, calculate_threshold recomputes masks internally. That's fine.
    
    # Let's run a loop of random local search steps.
    # Each step:
    # 1. Copy current best a, b
    # 2. Pick a random letter (a or b) and a random state i
    # 3. Change the destination of that state to a random value
    # 4. Calculate threshold
    # 5. If better, accept
    
    # We can also try "double mutations" if single steps are too slow to escape local optima.
    # But let's stick to single steps first, with many iterations.
    
    # To avoid getting stuck, we can occasionally restart from a random automaton 
    # or from the Černý automaton with a few random changes.
    
    # Let's use a simple random walk with acceptance if improved.
    # Since the space is huge, random single-step mutations from the Černý automaton
    # might not find a better one quickly.
    
    # Alternative strategy: Random restarts.
    # Generate random automata, check threshold. If > best, keep it.
    # Then do local search from that point.
    
    # Given the speed of bitmask BFS (likely < 1ms for N=12), we can do thousands of evaluations.
    
    # Let's combine: 
    # Phase 1: Random restarts to find a good starting point (if any exist > 121)
    # Phase 2: Local search from the best point found.
    
    # However, random restarts are unlikely to find > 121 by chance.
    # The structure of the Černý automaton is specific.
    # Maybe we should perturb the Černý automaton specifically.
    
    # Let's try a "simulated annealing" like approach or just hill climbing with random restarts 
    # from the Černý automaton.
    
    # Actually, let's just do a broad random search of the neighborhood of the Černý automaton.
    # We will mutate 1-3 transitions at a time.
    
    import sys
    # Ensure we don't use too much memory or time
    
    for step in range(100000):  # Max 100k steps, but we check time
        if time.time() > deadline:
            break
            
        # Decide on mutation strategy
        # 80% chance: mutate 1 transition
        # 20% chance: mutate 2 transitions
        num_mutations = 1 if random.random() < 0.8 else 2
        
        curr_a = best_a[:]
        curr_b = best_b[:]
        
        for _ in range(num_mutations):
            if random.random() < 0.5:
                # Mutate 'a'
                i = random.randint(0, N - 1)
                new_dest = random.randint(0, N - 1)
                if new_dest != curr_a[i]:
                    curr_a[i] = new_dest
            else:
                # Mutate 'b'
                i = random.randint(0, N - 1)
                new_dest = random.randint(0, N - 1)
                if new_dest != curr_b[i]:
                    curr_b[i] = new_dest
        
        # Evaluate
        score = calculate_threshold(curr_a, curr_b)
        
        if score > best_score:
            best_score = score
            best_a = curr_a
            best_b = curr_b
            # Optional: Reset some counters or change strategy if improved
            # For now, just continue hill climbing from the new best
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
