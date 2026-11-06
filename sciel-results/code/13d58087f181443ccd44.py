"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
import random
import time
import math
from collections import deque

def _reset_length_bitmask(a, b):
    """
    Compute shortest reset word length using bitmasks for efficiency.
    a, b: lists of ints 0..N-1
    Returns length or -1 if not synchronizing.
    """
    start_mask = (1 << N) - 1
    
    # Check if a or b is already a reset
    # If a is constant, length 1.
    if len(set(a)) == 1 or len(set(b)) == 1:
        return 1

    visited = {start_mask}
    queue = deque([(start_mask, 0)])
    
    while queue:
        mask, dist = queue.popleft()
        
        # Check if singleton
        if mask & (mask - 1) == 0:
            return dist
            
        # Apply a
        next_a = 0
        m = mask
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            next_a |= (1 << a[i])
            m -= lsb
        if next_a not in visited:
            visited.add(next_a)
            queue.append((next_a, dist + 1))
            
        # Apply b
        next_b = 0
        m = mask
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            next_b |= (1 << b[i])
            m -= lsb
        if next_b not in visited:
            visited.add(next_b)
            queue.append((next_b, dist + 1))
            
    return -1

def construct():
    # Simulated Annealing to maximize reset length
    # The space of automata is N^(2N). We want to find a synchronizing one with max threshold.
    
    rng = random.Random(42)
    
    # Start with a random automaton, or a heuristic one if we want to be safe
    # Let's start with a random one to explore broadly, but ensure we track best.
    # Actually, starting with a known synchronizing one might be safer to avoid getting stuck in non-syncing regions,
    # but SA can jump out. Let's start with a random one and accept non-syncing states?
    # No, the score for non-syncing is -1 (or 0 in some contexts). The task says "It must synchronize".
    # If it doesn't synchronize, the score is effectively 0 or invalid.
    # The evaluator returns -1 for non-syncing. We should treat -1 as very bad.
    
    # Let's initialize with a random automaton.
    a = [rng.randint(0, N-1) for _ in range(N)]
    b = [rng.randint(0, N-1) for _ in range(N)]
    
    best_a = a[:]
    best_b = b[:]
    best_score = _reset_length_bitmask(best_a, best_b)
    
    # If initial is not synchronizing, try a few randoms to find a sync one to start from
    # Or just let SA handle it. But -1 is a deep pit.
    # Let's force a start with a simple synchronizing automaton to be safe.
    # Standard Cerny: a is cycle, b maps all to 0.
    # a = [(i+1)%N for i in range(N)], b = [0]*N
    # This has threshold N-1.
    
    a = [(i + 1) % N for i in range(N)]
    b = [0] * N
    best_a = a[:]
    best_b = b[:]
    best_score = _reset_length_bitmask(best_a, best_b)
    
    current_a = a[:]
    current_b = b[:]
    current_score = best_score
    
    start_time = time.time()
    max_time = 55.0  # Leave 5s buffer
    
    # Simulated Annealing Parameters
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.001
    
    iterations = 0
    while time.time() - start_time < max_time:
        # Mutation: change one transition
        # Pick random state i and random letter L (0 for a, 1 for b)
        i = rng.randint(0, N-1)
        letter = rng.randint(0, 1)
        
        old_dest = current_a[i] if letter == 0 else current_b[i]
        new_dest = rng.randint(0, N-1)
        
        if old_dest == new_dest:
            continue
            
        # Apply mutation
        if letter == 0:
            current_a[i] = new_dest
        else:
            current_b[i] = new_dest
            
        new_score = _reset_length_bitmask(current_a, current_b)
        
        # Acceptance criterion
        # If new_score is -1, it's non-syncing. We treat it as very bad.
        # Let's map -1 to a large negative number for the delta calculation if needed,
        # or just reject if it drops from positive to -1 unless T is very high?
        # Actually, if we go to non-syncing, we lose the "synchronizing" property.
        # The task requires the returned automaton to synchronize.
        # Intermediate states can be non-syncing, but we must not return one.
        # We track best_score which is always from a sync automaton if we start with one.
        
        if new_score == -1:
            # Reject if we were previously sync and become non-sync, unless we want to explore that space.
            # Usually, it's better to stay in the sync manifold if possible, or jump back quickly.
            # Let's reject non-syncing states if current is sync.
            if current_score != -1:
                 # Revert
                 if letter == 0:
                     current_a[i] = old_dest
                 else:
                     current_b[i] = old_dest
                 continue
            else:
                 # Both non-syncing, accept with prob 1? Or just accept to move around.
                 # Let's accept to move towards sync.
                 pass
        
        delta = new_score - current_score
        
        if delta >= 0:
            # Accept
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_a = current_a[:]
                best_b = current_b[:]
        else:
            # Worse
            # Probability of acceptance
            # If current is -1 and new is positive, delta is positive, handled above.
            # If current is positive and new is -1, delta is negative.
            # exp(delta / T)
            try:
                prob = math.exp(delta / T)
            except OverflowError:
                prob = 0.0
                
            if rng.random() < prob:
                current_score = new_score
                # Note: we do NOT update best if new_score < best_score
            else:
                # Revert
                if letter == 0:
                    current_a[i] = old_dest
                else:
                    current_b[i] = old_dest
        
        # Cool down
        T = max(min_T, T * cooling_rate)
        iterations += 1
        
        # Occasionally restart if stuck? Or just let SA cool.
        # With N=12, the landscape is complex.
        
    return best_a, best_b
# EVOLVE-BLOCK-END
