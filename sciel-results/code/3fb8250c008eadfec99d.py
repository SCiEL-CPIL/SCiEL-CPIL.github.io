# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square.
    Uses simulated annealing for optimization.
    """
    m = 205
    k = 2
    deadline = time.time() + 580.0  # 580s safety margin before 630s kill

    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        val = pow(x, k, m)
        if val != 0:
            forbidden.add(val)
    
    # For fast lookup, also create a boolean array
    forbidden_arr = np.zeros(m, dtype=bool)
    for f in forbidden:
        forbidden_arr[f] = True
    
    # Precompute conflict neighbors: for each residue, which other residues conflict with it
    # conflict_neighbors[i] = set of j such that (i-j) % m is in forbidden
    conflict_neighbors = [set() for _ in range(m)]
    for i in range(m):
        for f in forbidden:
            j = (i - f) % m
            conflict_neighbors[i].add(j)
            j2 = (i + f) % m
            conflict_neighbors[i].add(j2)
            
    # Remove self if added (i-i=0 not in forbidden, so i not in conflict_neighbors[i] usually, 
    # but let's ensure. 0 is not in forbidden, so (i-i)%m=0 not in forbidden. So i not in conflict_neighbors[i].)

    def is_valid_addition(candidate, selected_set):
        """Check if candidate can be added to selected_set without creating a forbidden pair."""
        for s in selected_set:
            if s in conflict_neighbors[candidate]:
                return False
        return True

    def greedy_construct(order):
        """Greedy construction using a specific order of candidates."""
        selected = []
        selected_set = set()
        for candidate in order:
            if candidate not in selected_set:
                valid = True
                for s in selected_set:
                    if s in conflict_neighbors[candidate]:
                        valid = False
                        break
                if valid:
                    selected.append(candidate)
                    selected_set.add(candidate)
        return selected

    def simulated_annealing(initial_set, time_limit):
        """
        Simulated annealing to maximize the size of the independent set.
        """
        S = set(initial_set)
        best_S = set(S)
        best_size = len(S)
        
        # Temperature schedule
        T = 1.0
        cooling = 0.99995
        
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            # Generate a move
            move_type = random.random()
            
            if move_type < 0.4:
                # Try to add a random element
                candidate = random.randint(0, m - 1)
                if candidate in S:
                    continue
                if is_valid_addition(candidate, S):
                    S.add(candidate)
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                # If invalid, we could "reject" or just skip. 
                # In SA for set construction, adding invalid elements is not allowed.
                # So we just skip.
                
            elif move_type < 0.7:
                # Try to remove a random element
                if S:
                    to_remove = random.choice(list(S))
                    S.discard(to_remove)
                    # We don't track best here as size decreases, 
                    # but it might enable future additions.
                    
            else:
                # Try a swap: remove one, add another
                if S:
                    to_remove = random.choice(list(S))
                    to_add = random.randint(0, m - 1)
                    if to_add in S:
                        continue
                    # Check if to_add is valid with S \ {to_remove}
                    S.discard(to_remove)
                    if is_valid_addition(to_add, S):
                        S.add(to_add)
                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = set(S)
                    else:
                        # Revert
                        S.add(to_remove)
                        # Optionally, we could accept this "bad" move with some probability 
                        # to escape local minima, but since it doesn't change size 
                        # and just swaps, it's only useful if it leads to a better 
                        # configuration. Simple SA on size doesn't handle this well 
                        # without a secondary objective. 
                        # Let's stick to size-only for now, but allow "neutral" swaps 
                        # with high probability to explore.
                        # Actually, if we just revert, we are stuck.
                        # Let's accept neutral swaps with prob 0.5 to shuffle.
                        if random.random() < 0.5:
                             # We already discarded to_remove and failed to add to_add.
                             # We need to revert to the state BEFORE the swap attempt?
                             # No, the state was S with to_remove.
                             # We removed to_remove. We tried to add to_add. Failed.
                             # So S is now missing to_remove.
                             # We should put it back.
                             S.add(to_remove)
                             # And we didn't add to_add.
                             # So we are back to original state.
                             # To explore, we should just do nothing?
                             # Or we can try a different to_add?
                             # Let's just revert and continue.
                             pass
            
            T *= cooling
            if T < 0.01:
                T = 0.01

        return list(best_S)

    # Phase 1: Deterministic greedy with natural order
    best_result = greedy_construct(list(range(m)))
    best_size = len(best_result)
    
    # Phase 2: Randomized greedy (more restarts)
    random.seed(42)
    for _ in range(100):
        if time.time() > deadline - 100: # Leave time for SA
            break
        order = list(range(m))
        random.shuffle(order)
        result = greedy_construct(order)
        if len(result) > best_size:
            best_result = result
            best_size = len(result)
            
    # Phase 3: Simulated Annealing from best result
    if time.time() < deadline - 10:
        # Use a significant portion of time for SA
        sa_time = min(deadline - time.time() - 5, 300)
        if sa_time > 10:
            result = simulated_annealing(best_result, sa_time)
            if len(result) > best_size:
                best_result = result
                best_size = len(result)
                
    # Phase 4: More restarts with SA if time permits
    random.seed(123)
    while time.time() < deadline - 10:
        order = list(range(m))
        random.shuffle(order)
        result = greedy_construct(order)
        # Only run SA if this greedy result is promising
        if len(result) >= best_size - 3:
            sa_time = min(deadline - time.time() - 5, 30)
            if sa_time > 5:
                result = simulated_annealing(result, sa_time)
                if len(result) > best_size:
                    best_result = result
                    best_size = len(result)
                    
    # Final validation
    for i, a in enumerate(best_result):
        for b in best_result[i+1:]:
            d1 = (a - b) % m
            d2 = (b - a) % m
            if forbidden_arr[d1] or forbidden_arr[d2]:
                return np.array([], dtype=np.int64)
    
    return np.array(best_result, dtype=np.int64)


# EVOLVE-BLOCK-END
