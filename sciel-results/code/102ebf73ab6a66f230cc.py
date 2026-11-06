import time
import random
import numpy as np

N = 512

def construct():
    start_time = time.time()
    # Safety margin: 5 seconds
    deadline = start_time + 55.0
    
    # Precompute 4-APs containing each x
    # ap[x] = list of tuples of 3 other elements
    ap = [[] for _ in range(N+1)]
    for a in range(1, N-2):
        for d in range(1, (N-a)//3 + 1):
            b = a+d
            c = a+2*d
            e = a+3*d
            if e > N: break
            ap[a].append((b,c,e))
            ap[b].append((a,c,e))
            ap[c].append((a,b,e))
            ap[e].append((a,b,c))
            
    def can_add(x, in_S):
        for triple in ap[x]:
            # Check if all 3 are in S
            if in_S[triple[0]] and in_S[triple[1]] and in_S[triple[2]]:
                return False
        return True

    def greedy_fill(in_S, S):
        """Adds any valid elements not in S. Returns number added."""
        added = 0
        candidates = list(range(1, N+1))
        random.shuffle(candidates)
        for x in candidates:
            if not in_S[x]:
                if can_add(x, in_S):
                    in_S[x] = True
                    S.add(x)
                    added += 1
        return added

    # Initial solution
    in_S = [False]*(N+1)
    S = set()
    greedy_fill(in_S, S)
    
    best_S = S.copy()
    best_size = len(S)
    
    # SA Parameters
    T = 1.0
    T_min = 0.01
    cooling = 0.99
    
    # List of all elements for random selection
    all_elements = list(range(1, N+1))
    
    while T > T_min and time.time() < deadline:
        # Check time more frequently
        if time.time() > deadline - 2.0:
            break
            
        r = random.random()
        
        if r < 0.3:
            # Kick + Repair
            # Remove k random elements
            k = random.randint(1, 5)
            if len(S) <= k:
                continue
                
            # Select k elements to remove
            # Using sample on list(S) is O(len(S))
            to_remove = random.sample(list(S), k)
            
            for x in to_remove:
                S.remove(x)
                in_S[x] = False
            
            # Greedy Repair
            greedy_fill(in_S, S)
            
            if len(S) > best_size:
                best_size = len(S)
                best_S = S.copy()
                
        else:
            # Local Move
            move_type = random.random()
            if move_type < 0.6:
                # Add
                x = random.choice(all_elements)
                if not in_S[x]:
                    if can_add(x, in_S):
                        S.add(x)
                        in_S[x] = True
                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = S.copy()
            elif move_type < 0.8:
                # Remove
                if S:
                    x = random.choice(list(S))
                    # Accept removal with prob exp(-1/T)
                    if random.random() < np.exp(-1.0/T):
                        S.remove(x)
                        in_S[x] = False
            else:
                # Swap
                if S:
                    x = random.choice(list(S))
                    y = random.choice(all_elements)
                    if y != x and not in_S[y]:
                        # Try swap
                        S.remove(x)
                        in_S[x] = False
                        if can_add(y, in_S):
                            S.add(y)
                            in_S[y] = True
                            # Accept swap with prob 0.5 (neutral move)
                            if random.random() > 0.5:
                                # Reject swap, revert
                                S.remove(y)
                                in_S[y] = False
                                S.add(x)
                                in_S[x] = True
                        else:
                            # y not valid, revert
                            S.add(x)
                            in_S[x] = True
                            
        T *= cooling
        
    return sorted(best_S)
