import itertools
import random
import time
import math

def construct(q, n, radius):
    # Constants for the specific task instance
    # q=3, n=6, radius=1
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other instances, though task implies specific
        # Just return a trivial valid code (all words) if not the target case
        # This is inefficient but valid
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Precompute neighbors for all 729 words
    # Index: v[0]*3^5 + v[1]*3^4 + ... + v[5]
    # Convert index to tuple
    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n):
            t[n - 1 - i] = idx % q
            idx //= q
        return tuple(t)
    
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Precompute balls
    all_indices = list(range(q**n))
    neighbors = [[] for _ in range(q**n)]
    
    # Generate all words
    words = [tuple(t) for t in itertools.product(range(q), repeat=n)]
    
    for idx, w in enumerate(words):
        neighbors[idx].append(idx)
        for i in range(n):
            for d in range(1, q):
                new_w = list(w)
                new_w[i] = (new_w[i] + d) % q
                neighbors[idx].append(tuple_to_idx(tuple(new_w)))
        # Remove duplicates just in case, though logic above is unique
        neighbors[idx] = list(set(neighbors[idx]))
    
    # Convert neighbors to sets for faster membership testing
    neighbor_sets = [set(nb) for nb in neighbors]
    
    # Precompute "uncovered" check helper: for a given set of selected codewords,
    # we maintain cover_count.
    
    start_time = time.time()
    deadline = 140.0 # Safety margin (150s limit, 10s buffer)

    def greedy_cover(seed=None):
        """
        Greedy set cover with optional randomization.
        Returns list of codeword indices.
        """
        if seed is not None:
            random.seed(seed)
        
        uncovered = set(range(q**n))
        code = []
        
        while uncovered and time.time() < deadline:
            best_idx = -1
            best_count = 0
            
            # Shuffle candidates to break ties and allow randomized restarts
            candidates = list(range(q**n))
            if seed is not None:
                random.shuffle(candidates)
            else:
                # Deterministic order for the first run
                pass
                
            for c_idx in candidates:
                # Count how many uncovered words this codeword covers
                # Optimization: If we already found a count equal to len(uncovered), break
                count = 0
                for nb in neighbors[c_idx]:
                    if nb in uncovered:
                        count += 1
                        if count + (len(neighbors[c_idx]) - 1) <= best_count:
                             # Even if all remaining were uncovered, we wouldn't beat best
                             # This is a loose bound, skip complex pruning for now
                             pass
                if count > best_count:
                    best_count = count
                    best_idx = c_idx
                    if best_count == len(uncovered):
                        break
            
            if best_idx == -1 or best_count == 0:
                break
            
            code.append(best_idx)
            for nb in neighbors[best_idx]:
                uncovered.discard(nb)
                
        return code

    def walk_sat_optimize(initial_code_indices, max_time):
        """
        WalkSAT-style local search to minimize code size.
        State: set of selected codeword indices.
        Objective: Minimize |code| subject to all words covered.
        """
        current_code = set(initial_code_indices)
        if not current_code:
             return current_code
             
        # Initialize cover counts
        cover_count = [0] * (q**n)
        for c_idx in current_code:
            for nb in neighbors[c_idx]:
                cover_count[nb] += 1
        
        # Find uncovered words
        uncovered = [i for i in range(q**n) if cover_count[i] == 0]
        
        best_code = set(current_code)
        best_size = len(current_code)
        
        # Temperature for random moves
        temp = 1.0
        cooling_rate = 0.995
        min_temp = 0.01
        
        # Precompute for each word, the list of codewords that cover it (inverse neighbors)
        # This is useful for finding swaps.
        # inv_neighbors[w] = list of c such that w in neighbors[c]
        # Since neighbors are symmetric in Hamming space? No, but for radius 1, 
        # if w is in ball(c), c is in ball(w). Yes, Hamming distance is symmetric.
        # So inv_neighbors is just neighbors.
        
        while time.time() < max_time:
            if not uncovered:
                # Valid solution. Try to remove redundant codewords.
                # Find a codeword c such that removing it leaves no word uncovered.
                # i.e., for all nb in neighbors[c], cover_count[nb] > 1.
                # We can iterate through current_code.
                removable = None
                # Randomize order to avoid bias
                code_list = list(current_code)
                random.shuffle(code_list)
                
                for c_idx in code_list:
                    can_remove = True
                    for nb in neighbors[c_idx]:
                        if cover_count[nb] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        removable = c_idx
                        break
                        
                if removable is not None:
                    # Remove it
                    current_code.remove(removable)
                    for nb in neighbors[removable]:
                        cover_count[nb] -= 1
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = set(current_code)
                else:
                    # Cannot remove. Try to swap to potentially enable removal later or just stay.
                    # WalkSAT move: Pick a random codeword c in current_code.
                    # Pick a random candidate c_new NOT in current_code.
                    # If swapping c -> c_new keeps all words covered (or improves uncovered count if we allow invalid states? No, we stick to valid states for simplicity first, or use penalty).
                    # Let's stick to valid states: Swap is valid if all words covered by c are also covered by c_new OR covered by others.
                    # Actually, simpler: 
                    # 1. Remove c.
                    # 2. Identify words that become uncovered (count == 0).
                    # 3. If none, we could have just removed c. We handled that above.
                    # 4. If some are uncovered, we need a c_new that covers ALL of them.
                    
                    # To be efficient, we can look for a c_new that covers the most uncovered words after removing c.
                    # But checking all c_new is expensive.
                    
                    # Alternative: Random swap.
                    # Pick random c from current_code.
                    # Pick random c_new from all_indices.
                    # Check if swap is valid.
                    
                    if current_code:
                        c_old = random.choice(list(current_code))
                        c_new = random.randint(0, q**n - 1)
                        
                        if c_new not in current_code:
                            # Tentatively remove c_old
                            for nb in neighbors[c_old]:
                                cover_count[nb] -= 1
                            
                            # Check if any word is uncovered
                            new_uncovered = [nb for nb in neighbors[c_old] if cover_count[nb] == 0]
                            
                            if not new_uncovered:
                                # c_old was redundant? But we checked for removable above.
                                # This might happen if the removable check missed it due to shuffle? No, logic is same.
                                # Or if we are in a state where we *can* remove, but we chose swap?
                                # If new_uncovered is empty, we can just remove c_old and keep size same? No, we want to reduce size.
                                # If we just removed, size drops by 1.
                                # But we are in the "else" of "if removable is not None". So no single codeword is removable.
                                # So new_uncovered should not be empty if we pick a random c_old?
                                # Wait, if no codeword is removable, then for EVERY c_old, there exists some nb with count 1.
                                # So new_uncovered will always be non-empty.
                                
                                # So we need c_new to cover new_uncovered.
                                # Check if c_new covers all new_uncovered
                                valid = True
                                for u in new_uncovered:
                                    if u not in neighbor_sets[c_new]:
                                        valid = False
                                        break
                                
                                if valid:
                                    # Accept swap
                                    current_code.remove(c_old)
                                    current_code.add(c_new)
                                    for nb in neighbors[c_new]:
                                        cover_count[nb] += 1
                                    # Note: cover_count for new_uncovered is now 1 (from c_new) + others.
                                    # Words covered only by c_old and not c_new? 
                                    # If u in new_uncovered, count was 0 after removing c_old.
                                    # If c_new covers u, count becomes 1.
                                    # If c_new does NOT cover u, count remains 0 (Invalid).
                                    # We checked valid, so all u are covered.
                                    
                                    # Check if this swap allows a removal? 
                                    # The loop will handle it in the next iteration.
                                else:
                                    # Restore c_old
                                    for nb in neighbors[c_old]:
                                        cover_count[nb] += 1
                            else:
                                # Restore c_old
                                for nb in neighbors[c_old]:
                                    cover_count[nb] += 1
                            
            else:
                # Invalid state (uncovered words exist). 
                # This shouldn't happen if we only accept valid moves.
                # But if we start from a greedy cover, it is valid.
                # So we always start valid and stay valid.
                pass
            
            temp *= cooling_rate
            if temp < min_temp:
                temp = min_temp

        return best_code

    # 1. Initial Greedy
    greedy_code = greedy_cover(seed=None)
    best_code_indices = set(greedy_code)
    best_size = len(best_code_indices)
    
    # 2. Run WalkSAT on the greedy solution
    if time.time() < deadline - 5:
        optimized = walk_sat_optimize(best_code_indices, deadline - 5)
        if len(optimized) < best_size:
            best_code_indices = optimized
            best_size = len(best_code_indices)
            
    # 3. Randomized Restart with WalkSAT
    # Try a few random initializations or randomized greedy
    restarts = 5
    for i in range(restarts):
        if time.time() > deadline - 10:
            break
            
        # Randomized Greedy
        seed = i * 1000 + 42
        r_greedy = greedy_cover(seed=seed)
        
        if len(r_greedy) >= best_size:
            # Even if greedy is worse, WalkSAT might fix it? 
            # Usually greedy is a good starting point. If it's already worse, 
            # WalkSAT needs to remove many words.
            # But sometimes a different structure is better.
            # Let's try if it's close or if we have time.
            if len(r_greedy) <= best_size + 5:
                 pass
            else:
                 continue # Skip if too far

        if time.time() > deadline - 10:
            break
            
        optimized = walk_sat_optimize(set(r_greedy), deadline - 10)
        if len(optimized) < best_size:
            best_size = len(optimized)
            best_code_indices = optimized
            
    # 4. Final Local Search (Greedy Removal)
    # Ensure we squeeze out any last redundancies
    current_code = set(best_code_indices)
    cover_count = [0] * (q**n)
    for c_idx in current_code:
        for nb in neighbors[c_idx]:
            cover_count[nb] += 1
            
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        code_list = list(current_code)
        random.shuffle(code_list)
        for c_idx in code_list:
            can_remove = True
            for nb in neighbors[c_idx]:
                if cover_count[nb] == 1:
                    can_remove = False
                    break
            if can_remove:
                current_code.remove(c_idx)
                for nb in neighbors[c_idx]:
                    cover_count[nb] -= 1
                improved = True
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code_indices = current_code

    # Convert back to lists of tuples/lists
    result = [idx_to_tuple(c_idx) for c_idx in best_code_indices]
    return [list(c) for c in result]
