import time
import random
import math

def construct():
    N = 512
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit

    # Precompute 4-AP conflicts
    # conflicts[x] is a list of tuples (y, z, w) such that {x, y, z, w} is a 4-AP
    # We store the other 3 elements.
    conflicts = [[] for _ in range(N + 1)]
    
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # AP is (a, b, c, e)
            conflicts[a].append((b, c, e))
            conflicts[b].append((a, c, e))
            conflicts[c].append((a, b, e))
            conflicts[e].append((a, b, c))

    def is_valid_add(S, x):
        """Check if adding x to set S creates a 4-AP."""
        # S is a list of booleans or a set. For speed, use a list of booleans indexed by element.
        for y, z, w in conflicts[x]:
            if S[y] and S[z] and S[w]:
                return False
        return True

    def greedy_construct(order):
        """Build a set greedily given an order of elements."""
        S = [False] * (N + 1)
        size = 0
        for x in order:
            if is_valid_add(S, x):
                S[x] = True
                size += 1
        return S, size

    # 1. Initial Greedy Construction (Sorted)
    S, size = greedy_construct(range(1, N + 1))
    best_S = S[:]
    best_size = size
    
    # 2. Simulated Annealing
    current_S = S[:]
    current_size = size
    T = 10.0  # Initial temperature
    T_min = 0.01
    alpha = 0.99  # Cooling factor
    # We will run until time runs out
    
    # Precompute list of elements for random selection
    all_elements = list(range(1, N + 1))
    
    while time.time() - start_time < time_limit:
        # Choose a move type
        # We want to bias towards adding elements if possible, but also allow swaps/removals
        move_type = random.random()
        
        new_S = current_S[:]
        new_size = current_size
        accepted = False
        
        if move_type < 0.4:
            # Try to add a random element not in S
            # Efficiently pick a candidate not in S
            # Since N is small, we can just pick random x and check if in S
            # Or maintain a list of non-S elements? That's expensive to update.
            # Let's just pick random x.
            x = random.randint(1, N)
            if not current_S[x]:
                if is_valid_add(current_S, x):
                    new_S[x] = True
                    new_size = current_size + 1
                    accepted = True # Always accept improvement
        elif move_type < 0.7:
            # Try to swap: remove i from S, add j not in S
            # Pick i from S
            # We need a list of elements in S to pick from
            # Maintaining a list is O(N) to update. 
            # Alternative: Pick random x in [1,N]. If in S, use as i. Else, pick random y in S?
            # Let's just iterate a few times to find an element in S.
            i_found = False
            for _ in range(10):
                i = random.randint(1, N)
                if current_S[i]:
                    i_found = True
                    break
            if i_found:
                # Pick j not in S
                j_found = False
                for _ in range(10):
                    j = random.randint(1, N)
                    if not current_S[j]:
                        j_found = True
                        break
                if j_found:
                    # Check if swap is valid
                    # Remove i, check if j can be added
                    # We need to check conflicts of j with S \ {i}
                    valid = True
                    for y, z, w in conflicts[j]:
                        if y == i:
                            if current_S[z] and current_S[w]:
                                valid = False
                                break
                        elif z == i:
                            if current_S[y] and current_S[w]:
                                valid = False
                                break
                        elif w == i:
                            if current_S[y] and current_S[z]:
                                valid = False
                                break
                        else:
                            if current_S[y] and current_S[z] and current_S[w]:
                                valid = False
                                break
                    
                    if valid:
                        new_S[i] = False
                        new_S[j] = True
                        # Size remains same
                        new_size = current_size
                        # Accept based on temperature? 
                        # Since size is same, we might accept to explore.
                        # Let's accept with prob 0.5 or similar, or always if we want to explore.
                        # Actually, let's only accept if we are "stuck" or with low prob.
                        # For now, let's accept with probability 0.1 to allow exploration.
                        if random.random() < 0.1:
                            accepted = True
        else:
            # Try to remove a random element (to escape local optima)
            i_found = False
            for _ in range(10):
                i = random.randint(1, N)
                if current_S[i]:
                    i_found = True
                    break
            if i_found:
                new_S[i] = False
                new_size = current_size - 1
                # Accept with probability exp(-1/T)
                if random.random() < math.exp(-1.0 / T):
                    accepted = True

        if accepted:
            current_S = new_S
            current_size = new_size
            
            if current_size > best_size:
                best_size = current_size
                best_S = current_S[:]
            
            # Cool down
            T *= alpha
            if T < T_min:
                T = T_min
        else:
            # If no move was accepted or possible, cool down slowly?
            # Or just cool down anyway to progress
            T *= alpha
            if T < T_min:
                T = T_min

    # Return the best set found
    result = []
    for i in range(1, N + 1):
        if best_S[i]:
            result.append(i)
    return result
