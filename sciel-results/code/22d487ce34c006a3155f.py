import random
import time
import math

N = 512

# Precompute conflicts
# conflicts[i] contains lists of 3-tuples (a, b, c) such that {i, a, b, c} is a 4-AP
conflicts = [[] for _ in range(N + 1)]

for d in range(1, N // 3 + 1):
    for start in range(1, N - 3 * d + 1):
        a = start
        b = start + d
        c = start + 2 * d
        e = start + 3 * d
        
        # 4-AP: a, b, c, e
        conflicts[a].append((b, c, e))
        conflicts[b].append((a, c, e))
        conflicts[c].append((a, b, e))
        conflicts[e].append((a, b, c))

def can_add(i, in_set):
    for a, b, c in conflicts[i]:
        if in_set[a] and in_set[b] and in_set[c]:
            return False
    return True

def greedy_construction(order):
    in_set = [False] * (N + 1)
    size = 0
    for i in order:
        if can_add(i, in_set):
            in_set[i] = True
            size += 1
    return in_set, size

def local_polish(in_set, deadline):
    """
    Greedy add phase to clean up.
    """
    size = sum(in_set[1:])
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(1, N + 1):
            if not in_set[i]:
                if can_add(i, in_set):
                    in_set[i] = True
                    size += 1
                    improved = True
                    break
    return in_set, size

def simulated_annealing(in_set, deadline):
    """
    Simulated Annealing to maximize the size of the set.
    """
    size = sum(in_set[1:])
    best_size = size
    best_in_set = in_set[:]
    
    T = 1.0
    # Cooling: we want to cool down over the duration of this SA call.
    # Estimate steps per second.
    # Let's just use a fixed cooling rate per step.
    # If we run for ~5s, and ~500k steps, rate = (0.01)**(1/500k) ~ 0.999988
    cooling_rate = 0.999988
    
    current_size = size
    steps_since_improve = 0
    max_stagnation = 50000 # Restart logic inside SA? No, let outer loop handle restarts.
    # But if we stagnate, we can force a "shake" (remove random element) to escape.
    
    while time.time() < deadline:
        # Pick random element
        i = random.randint(1, N)
        
        if in_set[i]:
            # Propose removal
            in_set[i] = False
            new_size = current_size - 1
            
            # Acceptance for worsening move
            # Delta size = -1
            # Prob = exp(1/T) ? No, exp(-DeltaE/T). 
            # If E = -size, DeltaE = - (new_size - old_size) = -(-1) = 1.
            # Prob = exp(-1/T).
            if random.random() < math.exp(-1.0 / T):
                current_size = new_size
                steps_since_improve += 1
            else:
                # Reject
                in_set[i] = True
                # current_size unchanged
                steps_since_improve += 1
        else:
            # Propose addition
            if can_add(i, in_set):
                in_set[i] = True
                new_size = current_size + 1
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_in_set = in_set[:]
                    steps_since_improve = 0
                else:
                    steps_since_improve += 1
            else:
                steps_since_improve += 1
        
        # Cool down
        T *= cooling_rate
        
        # If stuck for too long, force a removal to break symmetry?
        # The random removals already do this.
        
    return best_in_set, best_size

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s margin
    
    best_in_set = None
    best_size = 0
    
    # 1. Deterministic Greedy
    for order in [list(range(1, N + 1)), list(range(N, 0, -1))]:
        in_set, size = greedy_construction(order)
        in_set, size = local_polish(in_set, deadline)
        if size > best_size:
            best_size = size
            best_in_set = in_set[:]
            
    # 2. Randomized Greedy + SA
    random.seed(42)
    
    # Run several SA instances
    while time.time() < deadline - 5.0: # Leave 5s for final polish
        # Generate a random start
        order = list(range(1, N + 1))
        random.shuffle(order)
        in_set, size = greedy_construction(order)
        
        # Polish start
        in_set, size = local_polish(in_set, deadline)
        
        # Run SA
        # Dynamic SA time:
        # If we have > 20s left, run for 15s.
        # If we have < 20s left, run for remaining - 5s.
        time_left = deadline - time.time()
        if time_left > 20.0:
            sa_time = 15.0
        else:
            sa_time = max(1.0, time_left - 5.0)
            
        sa_deadline = time.time() + sa_time
        
        in_set, size = simulated_annealing(in_set, sa_deadline)
        
        # Final polish on the SA result
        in_set, size = local_polish(in_set, deadline)
        
        if size > best_size:
            best_size = size
            best_in_set = in_set[:]
            
    # Final polish on the best result found
    if best_in_set:
        best_in_set, best_size = local_polish(best_in_set, deadline)
        
    if best_in_set is None:
        return [1]
        
    return [i for i in range(1, N + 1) if best_in_set[i]]
