import time
import numpy as np
import random

def construct():
    N = 512
    start_time = time.time()
    time_limit = 55.0  # Leave 5s margin for evaluator overhead

    # 1. Precompute Conflicts
    # For each element x, we need to know which "triples" (a,b,c) are blocked if x is present.
    # A 4-AP is (x, x+d, x+2d, x+3d).
    # If we add x, we block any triple (a,b,c) such that {x,a,b,c} forms a 4-AP.
    # We can precompute for each x, a list of indices into a flat array of all possible triples?
    # No, better: Precompute a boolean matrix M of shape (N, N) or similar?
    # Actually, the check is: Can we add x?
    # Condition: No 4-AP containing x is fully contained in S.
    # A 4-AP containing x is determined by x's position in the AP.
    # x can be x0, x1, x2, or x3 of the AP.
    
    # Let's precompute for each x, the list of "forbidden triples" (t1, t2, t3) 
    # such that if all t1, t2, t3 are in S, then x cannot be added.
    
    # Optimization: Represent S as a numpy boolean array of length N.
    # For a candidate x, we need to check if any forbidden triple is fully in S.
    # We can precompute for each x, a flat array of indices of the triples?
    # Or better, precompute a list of tuples (a,b,c) for each x.
    
    forbidden = [ [] for _ in range(N) ]
    
    for a in range(N):
        for d in range(1, (N - a) // 3 + 1):
            x0 = a
            x1 = a + d
            x2 = a + 2 * d
            x3 = a + 3 * d
            
            if x3 >= N: continue
            
            # If x0 is added, (x1, x2, x3) must not be all present
            forbidden[x0].append((x1, x2, x3))
            # If x1 is added, (x0, x2, x3) must not be all present
            forbidden[x1].append((x0, x2, x3))
            # If x2 is added, (x0, x1, x3) must not be all present
            forbidden[x2].append((x0, x1, x3))
            # If x3 is added, (x0, x1, x2) must not be all present
            forbidden[x3].append((x0, x1, x2))
            
    # Convert forbidden lists to numpy arrays for faster indexing?
    # Actually, iterating over a list of tuples in Python is fast enough if the list is short.
    # Max forbidden per x: 
    # x can be start: ~N/3
    # x can be end: ~N/3
    # Total ~ N/2 = 256 checks per x.
    # Vectorizing this specific check is tricky without precomputing a huge matrix.
    # Let's stick to Python loop for the check but optimize the rest.
    # Wait, the prompt says "different representation".
    # Let's try to vectorize the check.
    
    # Alternative Vectorized Check:
    # Precompute for each x, a numpy array of shape (num_forbidden, 3) containing the indices.
    # Then check: S[indices[:, 0]] & S[indices[:, 1]] & S[indices[:, 2]]
    # If any is True, x cannot be added.
    
    forbidden_np = [None] * N
    for x in range(N):
        if forbidden[x]:
            forbidden_np[x] = np.array(forbidden[x])
        else:
            forbidden_np[x] = np.empty((0, 3), dtype=np.int32)

    def can_add(S, x):
        """
        S: numpy boolean array of length N
        x: int index
        Returns True if x can be added to S without forming a 4-AP.
        """
        triples = forbidden_np[x]
        if len(triples) == 0:
            return True
        # Check if any row has all 3 elements present in S
        # S[triples] gives an array of shape (num_triples, 3)
        # We want to know if any row is all True.
        # np.all(S[triples], axis=1).any()
        # This is vectorized in C.
        return not np.any(np.all(S[triples], axis=1))

    # 2. Initial Construction: Greedy with Randomization
    # We will run multiple greedy passes with different orderings to find a good starting point.
    
    def greedy_fill(order):
        S = np.zeros(N, dtype=bool)
        for x in order:
            if can_add(S, x):
                S[x] = True
        return S

    best_S = np.zeros(N, dtype=bool)
    best_size = 0
    
    # Try a few simple orders
    orders = [
        list(range(N)),
        list(range(N-1, -1, -1)),
        sorted(range(N), key=lambda x: len(forbidden[x]))
    ]
    
    # Add a random order
    random.seed(42)
    rand_order = list(range(N))
    random.shuffle(rand_order)
    orders.append(rand_order)

    for order in orders:
        S = greedy_fill(order)
        size = np.sum(S)
        if size > best_size:
            best_size = size
            best_S = S.copy()

    # 3. Simulated Annealing
    # State: S (boolean array)
    # Move: Pick a random x. If in S, try to remove. If not in S, try to add.
    # Energy: -size (we want to maximize size, so minimize -size)
    # Or just track size.
    
    current_S = best_S.copy()
    current_size = best_size
    best_S_sa = best_S.copy()
    best_size_sa = best_size
    
    T = 1.0  # Temperature
    T_min = 0.01
    cooling = 0.995
    
    # Precompute list of all elements for random selection
    all_elements = list(range(N))
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        # Pick a random element to toggle
        x = random.choice(all_elements)
        
        if current_S[x]:
            # Try to remove x
            # New size = current_size - 1
            # Always accept removal if it allows adding more? 
            # In SA, we evaluate the new state.
            # Let's do a "toggle" move: flip x.
            # But simple flip might just reduce size.
            # Better move: 
            # 1. Try to add a random x not in S.
            # 2. Try to remove a random x in S, then greedily fill? (Too slow)
            # 3. Standard SA on the set: 
            #    Propose a move: Add x (if possible) or Remove x.
            
            # Let's try: Pick random x.
            # If x not in S:
            #   If can_add: Accept (size+1)
            #   Else: Reject? Or maybe this is a "blocked" state.
            # If x in S:
            #   Remove it. (size-1). 
            #   Then maybe try to add something else?
            
            # Simpler SA: 
            # Move: Swap. Pick x in S, y not in S.
            # Remove x, add y.
            # Check if valid.
            
            # Let's stick to Add/Remove.
            # To avoid getting stuck in small sets, we bias towards adding.
            
            pass # Placeholder logic, see below for actual implementation

    # Actual SA Loop
    T = 1.0
    T_min = 0.005
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        # Propose a move
        # 80% chance: Try to add a random element
        # 20% chance: Try to remove a random element (to escape local minima)
        
        if random.random() < 0.8:
            # Try to add
            x = random.choice(all_elements)
            if not current_S[x]:
                if can_add(current_S, x):
                    # Always accept improvement
                    current_S[x] = True
                    current_size += 1
                    if current_size > best_size_sa:
                        best_size_sa = current_size
                        best_S_sa = current_S.copy()
                else:
                    # Cannot add. 
                    # In SA, we might accept a "bad" move if we allow temporary invalidity? 
                    # No, constraint must hold.
                    # So we just reject.
                    pass
        else:
            # Try to remove
            # Pick a random element in S
            # We need a list of elements in S for fast sampling.
            # Maintaining a list is O(1) for add/remove if we use a dict/set.
            # Let's maintain a set of indices in S.
            pass
            
    # The above SA is weak because "Cannot add" is a dead end.
    # We need a better move set: "Swap".
    # Pick x in S, y not in S.
    # Remove x. Check if y can be added.
    # If yes, accept (size same).
    # If no, reject.
    # This allows exploration of the same-size manifold.
    
    # Let's restart with a better SA implementation.
    
    # Reset time check
    if time.time() - start_time > time_limit - 5:
         # Not enough time for complex SA, return best so far
         result = [x + 1 for x in range(N) if best_S[x]]
         return sorted(result)

    # Re-initialize SA state with best found so far
    current_S = best_S.copy()
    current_size = best_size
    best_S_sa = best_S.copy()
    best_size_sa = best_size
    
    # Maintain a list of elements in S for fast removal sampling
    current_indices = [i for i in range(N) if current_S[i]]
    
    T = 0.5
    T_min = 0.01
    cooling = 0.9995
    
    while time.time() - start_time < time_limit - 2:
        if T < T_min:
            break
            
        move_type = random.random()
        
        if move_type < 0.7:
            # Try to add a random element
            x = random.choice(all_elements)
            if not current_S[x]:
                if can_add(current_S, x):
                    current_S[x] = True
                    current_size += 1
                    current_indices.append(x)
                    if current_size > best_size_sa:
                        best_size_sa = current_size
                        best_S_sa = current_S.copy()
                # else: reject
        elif move_type < 0.9:
            # Try a swap: Remove x from S, Add y to S
            if not current_indices:
                continue
            x = random.choice(current_indices)
            y = random.choice(all_elements)
            if y == x or current_S[y]:
                continue
                
            # Tentatively remove x
            current_S[x] = False
            # Check if y can be added
            if can_add(current_S, y):
                current_S[y] = True
                # Update indices
                # Remove x, add y
                # O(N) to remove from list, but N=512 is small.
                # Optimize: Use a set for membership, list for iteration?
                # For now, list is fine.
                try:
                    current_indices.remove(x)
                except ValueError:
                    pass
                current_indices.append(y)
                # Size remains same. 
                # We accept this move to explore the landscape.
                # We don't update best_size_sa unless we find a larger set later.
            else:
                # Revert
                current_S[x] = True
                # Reject
        else:
            # Try to remove x (to escape local optima where no adds/swaps work)
            if not current_indices:
                continue
            x = random.choice(current_indices)
            current_S[x] = False
            current_size -= 1
            try:
                current_indices.remove(x)
            except ValueError:
                pass
            
            # Accept removal if it allows us to add more later?
            # In SA, we accept with probability exp(-DeltaE / T)
            # Here DeltaE = -1 (size decreased).
            # Prob = exp(1/T). Since T is small, this is exp(large) > 1? 
            # Wait, Energy = -Size.
            # Old Energy = -Size.
            # New Energy = -(Size - 1) = -Size + 1.
            # DeltaE = 1.
            # Prob = exp(-1/T).
            # If T is small, this is very low.
            # We should only remove if we are stuck or T is high.
            # Let's just accept removals with a small probability if T is high.
            if random.random() > exp(-1/T):
                # Revert
                current_S[x] = True
                current_size += 1
                current_indices.append(x)
            else:
                # Keep removal
                pass

        T *= cooling

    # Final Polish: Greedy fill on the best state found
    # The SA might have left some addable elements.
    S_final = best_S_sa.copy()
    size_final = best_size_sa
    changed = True
    while changed:
        changed = False
        for x in range(N):
            if not S_final[x]:
                if can_add(S_final, x):
                    S_final[x] = True
                    size_final += 1
                    changed = True
                    if size_final > best_size_sa:
                        best_size_sa = size_final
                        best_S_sa = S_final.copy()

    result = [x + 1 for x in range(N) if best_S_sa[x]]
    return sorted(result)

import math
def exp(x):
    return math.exp(x)

# EVOLVE-BLOCK-END
