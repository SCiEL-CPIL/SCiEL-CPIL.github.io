import time
import random

N = 256
TIME_LIMIT = 55.0

def construct():
    start_time = time.time()
    
    # 1. Precomputation
    triples = []
    triples_by_elem = [[] for _ in range(N + 1)]
    for d in range(1, (N // 2) + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            triples.append((a, b, c))
            triples_by_elem[a].append((b, c))
            triples_by_elem[b].append((a, c))
            triples_by_elem[c].append((a, b))
            
    def is_4ap_free_with(A_bool, x):
        # Check if adding x to set represented by A_bool creates a 4-AP
        # x as 1st
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            if A_bool[x + d] and A_bool[x + 2 * d] and A_bool[x + 3 * d]:
                return False
        # x as 2nd
        max_d = (N - x) // 2
        for d in range(1, max_d + 1):
            if A_bool[x - d] and A_bool[x + d] and A_bool[x + 2 * d]:
                return False
        # x as 3rd
        max_d = (x - 1) // 2
        for d in range(1, max_d + 1):
            if A_bool[x - 2 * d] and A_bool[x - d] and A_bool[x + d]:
                return False
        # x as 4th
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if A_bool[x - 3 * d] and A_bool[x - 2 * d] and A_bool[x - d]:
                return False
        return True

    def count_3aps(A_bool):
        score = 0
        for a, b, c in triples:
            if A_bool[a] and A_bool[b] and A_bool[c]:
                score += 1
        return score

    def greedy_construct(order):
        A_bool = [False] * (N + 1)
        A_list = []
        for x in order:
            if is_4ap_free_with(A_bool, x):
                A_bool[x] = True
                A_list.append(x)
        return A_bool, A_list

    # 2. Greedy Initialization with multiple random seeds
    best_A_bool = None
    best_A_list = None
    best_score = -1
    
    for _ in range(10):
        order = list(range(1, N + 1))
        random.shuffle(order)
        A_bool, A_list = greedy_construct(order)
        score = count_3aps(A_bool)
        if score > best_score:
            best_score = score
            best_A_bool = A_bool
            best_A_list = A_list
            
    # 3. Local Search
    A_bool = best_A_bool
    A_list = list(best_A_list) # Copy
    current_score = best_score
    
    # Maintain a set for O(1) lookup of membership for random selection
    A_set = set(A_list)
    
    while time.time() - start_time < TIME_LIMIT:
        # Try additions
        added = False
        for _ in range(100):
            x = random.randint(1, N)
            if not A_bool[x]:
                if is_4ap_free_with(A_bool, x):
                    # Calculate gain
                    gain = 0
                    for b, c in triples_by_elem[x]:
                        if A_bool[b] and A_bool[c]:
                            gain += 1
                    A_bool[x] = True
                    A_list.append(x)
                    A_set.add(x)
                    current_score += gain
                    added = True
                    break
        
        if not added:
            # Try swaps
            swapped = False
            for _ in range(100):
                if not A_list: break
                x_out = random.choice(A_list)
                x_in = random.randint(1, N)
                if A_bool[x_in]:
                    continue
                
                # Temporarily remove x_out
                A_bool[x_out] = False
                A_list.remove(x_out)
                A_set.discard(x_out)
                
                if is_4ap_free_with(A_bool, x_in):
                    # Calculate delta
                    # Loss from x_out
                    loss = 0
                    for b, c in triples_by_elem[x_out]:
                        if A_bool[b] and A_bool[c]:
                            loss += 1
                    # Gain from x_in
                    gain = 0
                    for b, c in triples_by_elem[x_in]:
                        if A_bool[b] and A_bool[c]:
                            gain += 1
                    delta = gain - loss
                    if delta >= 0:
                        A_bool[x_in] = True
                        A_list.append(x_in)
                        A_set.add(x_in)
                        current_score += delta
                        swapped = True
                        break
                    else:
                        # Reject
                        A_bool[x_out] = True
                        A_list.append(x_out)
                        A_set.add(x_out)
                else:
                    # Reject
                    A_bool[x_out] = True
                    A_list.append(x_out)
                    A_set.add(x_out)

    return sorted(A_list)
